import codecs
import re
import requests
import os
from datetime import date, timedelta, datetime
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from xml.etree import ElementTree
from urllib.parse import urlparse


class URLFilter:

    def __init__(self, wanted_keywords: list[str], unwanted_keywords: list[str], lang_exceptions: list[str]):
        self.lang_pattern = re.compile(r"/([a-z]{2})/")
        self.lang_exceptions = lang_exceptions
        self._wanted_keywords = [k.lower() for k in wanted_keywords]
        self._unwanted_keywords = [k.lower() for k in unwanted_keywords]

    def _filter_unwanted_languages(self, urls: list[str], min_pages) -> list[str]:
        """Removes URLs that indicate a foreign language except for languages
        that are specified in the lang_exceptions list."""
        filtered_urls = []
        for url in urls:
            match = self.lang_pattern.search(url.lower())
            if match:
                if match.group(1) in self.lang_exceptions:
                    filtered_urls.append(url)
            else:
                filtered_urls.append(url)
        return urls if len(filtered_urls) < min_pages else filtered_urls

    def _score_url(self, url: str):
        path = urlparse(url).path
        depth = len([p for p in path.split('/') if p])
        scaling_factor = 1 / depth if depth else 1  # Prioritize shallow paths

        wanted_matches = any(k in path for k in self._wanted_keywords)
        unwanted_matches = any(k in path for k in self._unwanted_keywords)

        if wanted_matches and unwanted_matches:
            return 0.25 * scaling_factor
        if unwanted_matches:
            return 0.0
        if wanted_matches:
            return 1.0 * scaling_factor
        return 0.5 * scaling_factor

    def _rank_urls(self, urls: list[str]):
        """Sort URLs by relevance score in descending order."""
        sorted_urls = sorted(((url, self._score_url(url)) for url in urls), key=lambda x: x[1], reverse=True)
        return [url for url, score in sorted_urls if score > 0]

    def filter_urls(self, urls: list[str], min_pages: int = 10, max_pages: int = 15) -> list[str]:
        """Filters unwanted URLs based on language and specific keywords.

        Args:
            urls (list[str]): A list of URLs to be filtered.
            min_pages (int): If less than min_pages remain after filtering pages 
                with an explicitly set language in the URL, no pages are removed.
            max_pages (int): The maximum number of pages that are downloaded.

        Returns:
            list[str]: A filtered list of URLs with unwanted entries removed.
        """
        urls = [url for url in urls if '#' not in url]
        filtered_urls = self._filter_unwanted_languages(urls, min_pages)
        ranked_urls = self._rank_urls(filtered_urls)
        return ranked_urls[:max_pages]


class CompanyWebCrawler:
    """A web crawler for extracting URLs from sitemaps and crawling web pages.

    This crawler retrieves URLs from standard sitemaps and sitemap index files,
    filters them based on language and unwanted keywords, and asynchronously
    crawls the extracted pages.

    Features:
    - Supports both standard sitemaps (`<urlset>`) and sitemap index files (`<sitemapindex>`).
    - Filters URLs based on unwanted words and language-specific paths.
    - Uses an asynchronous web crawler for efficient page fetching.
    - Handles network errors and XML parsing failures gracefully.

    Args:
        unwanted_words (list[str]): If a URL contains any of these words, it will be excluded from the crawl.
        lang_exceptions (list[str], optional): The languages that should not lead to an exclusion of the URL from the crawl. 
            Defaults to `['de', 'en']`.

    Example:
        >>> crawler = CompanyWebCrawler(unwanted_words=['news', 'terms-of-use'])
        >>> urls = crawler.get_urls("https://example.com")
        >>> filtered_urls = crawler.filter_urls(urls)
        >>> results = asyncio.run(crawler.crawl(filtered_urls))
    """
    def __init__(
        self,
        wanted_keywords: list[str] = [],
        unwanted_keywords: list[str] = [],
        lang_exceptions: list[str] = ['en']
    ):
        self.filter = URLFilter(wanted_keywords, unwanted_keywords, lang_exceptions)
        self.current_date = datetime.today().strftime('%Y-%m-%d')

    @staticmethod
    def _create_sitemap_url(base_url: str) -> str:
        """Adds sitemap subdomain to a base url."""
        return f"{base_url.rstrip('/')}/sitemap.xml"

    @staticmethod
    def create_wayback_urls(urls: list[str], timestamp: str):
        """Creates wayback url for internal links when using the wayback machine."""
        return [f'http://web.archive.org/web/{timestamp}/{url}' for url in urls]

    @staticmethod
    def _detect_namespace(root: ElementTree.Element) -> dict[str, str]:
        """Extracts the namespace dynamically from the root element."""
        if root.tag.startswith("{"):
            return {"ns": root.tag.split("}")[0].strip("{")}
        return {}

    @staticmethod
    def _create_url_list(root: ElementTree.Element, namespace: dict[str, str]) -> list[str]:
        """Creates a list of urls from the root of the sitemap.xml."""
        return [loc.text.replace('\n', '').strip() for loc in root.findall('.//ns:loc', namespace)]

    def _crawl_sitemap(self, sitemap_url: str) -> tuple[ElementTree.Element, dict[str, str]]:
        """Tries to fetch the sitemap.xml for a URL and extract its root and namespace."""
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()

            root = ElementTree.fromstring(response.content)
            namespace = self._detect_namespace(root)
            return root, namespace

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch sitemap at {sitemap_url}: {e}") from e
        except ElementTree.ParseError as e:
            raise ValueError(f"Failed to parse XML at {sitemap_url}: {e}") from e

    def get_urls(self, base_url: str) -> list[str]:
        """Retrieves all URLs from the sitemap of the given base URL.

        This method attempts to fetch and parse the sitemap located at the base URL. 
        It supports both regular sitemaps and sitemap index files, retrieving URLs 
        recursively when necessary.

        Args:
            base_url (str): The base URL of the website.

        Returns:
            list[str]: A list of URLs extracted from the sitemap. If an error occurs, an empty list is returned.

        Raises:
            RuntimeError: If the sitemap cannot be fetched due to network issues.
            ValueError: If the XML parsing fails due to an invalid format.
        """
        sitemap_url = self._create_sitemap_url(base_url)
        try:
            root, namespace = self._crawl_sitemap(sitemap_url)

            if root.tag.endswith('sitemapindex'):
                sitemaps = self._create_url_list(root, namespace)

                urls = []
                for sitemap in sitemaps:
                    root, namespace = self._crawl_sitemap(sitemap)
                    urls.extend(self._create_url_list(root, namespace))

                return urls

            return self._create_url_list(root, namespace)

        except (RuntimeError, ValueError) as e:
            print(f"Error occured during XML parsing for {base_url}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error occured for {base_url}: {e}")
            return []

    def get_closest_snapshot(url: str, year: int, month: int, day: int, window_days: int = 730) -> dict | None:
        """
        Finds the closest available snapshot to the given date using the CDX API.

        Args:
            url (str): The URL of the website to look up.
            year, month, day (int): The target date to find the closest snapshot to.
            window_days (int): Number of days before and after the target date to search.

        Returns:
            dict or None: Closest snapshot info or None if not found.
        """
        from_date = date(year, month, day)
        to_date = (from_date + timedelta(days=window_days)).strftime('%Y%m%d')

        url = (
            f"https://web.archive.org/cdx/search/cdx?"
            f"url={url}&matchType=exact&limit=100&filter=statuscode:200"
            f"&from={from_date}&to={to_date}&output=json"
        )

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if len(data) <= 1:
            return None  # No snapshots found

        # Parse snapshots and find the closest one to the target date
        snapshots = data[1:]  # Skip header
        snapshots_with_distance = []

        for row in snapshots:
            ts = row[1]  # timestamp string, e.g. "20210417012345"
            snapshot_date = datetime.strptime(ts, "%Y%m%d%H%M%S").date()
            distance = abs((snapshot_date - from_date).days)
            snapshots_with_distance.append((distance, row))

        # Pick the closest snapshot
        _, closest_row = min(snapshots_with_distance, key=lambda x: x[0])
        return {
            "timestamp": closest_row[1],
            "original": closest_row[2],
            "url": f"https://web.archive.org/web/{closest_row[1]}/{closest_row[2]}"
        }

    def filter_urls(self, urls: list[str], min_pages: int = 10, max_pages: int = 20):
        return self.filter.filter_urls(urls, min_pages, max_pages)

    async def crawl(self, crawler: AsyncWebCrawler, urls: list[str]):
        """Asynchronously crawls the given list of URLs.

        Args:
            crawler (AsyncWebCrawler): Crawler object initialized outside of the method.
            urls (list[str]): The landing page url of the company.

        Returns:
            dict[str, dict]: A dictionary containing crawl results, including extracted content and links.
        """
        successful = {}

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=10,
        )

        config = CrawlerRunConfig(
            excluded_tags=["nav", "footer", "aside"],
            excluded_selector="""
                .cookie,
                .cookies,
                .cookie-banner,
                .cookie-consent,
                [id*="cookie"],
                [class*="cookie"]
            """,
            remove_forms=True,
            remove_overlay_elements=True,  # Any popup should be excluded!
            scan_full_page=True,
            check_robots_txt=True,
            semaphore_count=3,
            stream=False,
        )

        results = await crawler.arun_many(
            urls,
            config=config,
            dispatcher=dispatcher,
            headers={
                'Accept-Language': 'en,de;q=0.8,fr;q=0.6,it;q=0.4',  # Prefer English version of website but accept German, French, or Italian as an alternative
                'Cookie': 'CookieConsent=true; consent=all',  # Accept all cookies
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        for result in results:
            if result.success:
                successful[result.url] = {
                    'status_code': result.status_code,
                    'markdown': result.markdown,
                    'links': result.links,
                    'date': self.current_date,
                }
            else:
                successful[result.url] = {
                    'status_code': result.status_code,
                    'error_message': result.error_message
                }

        return successful



class WaybackCrawler:

    def __init__(
        self,
        website,
        output_folder="../../out",
        year_folder: bool = False,
        wanted_keywords: list[str] = [],
        unwanted_keywords: list[str] = [],
        lang_exceptions: list[str] = ['en']
    ):
        self.filter = URLFilter(wanted_keywords, unwanted_keywords, lang_exceptions)

        self.website = website
        self.output_folder = output_folder
        self.year_folder = year_folder

    def split_wayback_url(self, wayback_url):
        original_url = re.sub(r'http://web.archive.org/web/\d+/', '', wayback_url)
        website_piece = re.sub(r"http(s?)\://", '', original_url)

        try:
            (domain, address) = website_piece.split("/", 1)
        except ValueError:
            domain  = website_piece
            address = ''

        domain = data_reader.clean_domain_url(domain)

        return (domain, address)

    def store_page(self, wayback_url, html):
        (domain, address) = self.split_wayback_url(wayback_url)

        if self.year_folder:
            base_directory = "{0}/{1}/{2}".format(self.output_folder, domain, self.crawled_year)
        else:
            base_directory = self.output_folder + "/" + domain

        if not os.path.exists(base_directory):
                os.makedirs(base_directory)

        if address == "":
            address = "homepage.html"

        file_path = base_directory +  "/" + address.replace("/","_")
        outfile = codecs.open(file_path, "w",'utf-8')
        outfile.write(html)
        outfile.close()
        print ("\t .Stored in: {0}".format(file_path))

    def is_valid_url(self, url):
        if url.endswith(".pdf") or url.contains("godaddy") or url.contains("bobparsons"):
            return False

        if url == "." or url == "..":
            return False

        return True

    def crawl(self, wayback_url, levels=1, done_urls={}):
        # Recursive algorithm
        print ("\t .Crawl [L={0}].. {1}".format(levels, wayback_url))

        clean_url = re.sub(r"\?.*", '', wayback_url)
        clean_url = re.sub(r"\#.*", '', clean_url)

        try:
            response  = requests.get(clean_url)
            html = response.text
            self.store_page(clean_url, html)
        except ConnectionError as e:
            print ("Connection Error: Skipping")
            return done_urls

        done_urls = self.add_done_url(clean_url, done_urls)

        counter = 0

        if levels > 0:

            (domain, address) = self.split_wayback_url(clean_url)

            soup = BeautifulSoup(html, features="html.parser")
            for link in soup.findAll('a', attrs={'href': re.compile(domain)}):
                url = link['href']
                print('\t' + url)

                # Skipping Conditions: Begin

                if not self.is_valid_url(url):
                    print("\t .Skipped (not a valid url)")
                    continue

                if not url.startswith('http'):
                    url = 'http://web.archive.org' + url

                # if url not done. Scrape it.
                if self.url_done(url, done_urls):
                    print("\t .Skipped (already done)")

                counter += 1
                if counter >= 9:
                    print("\t .10 links donwloaded for website. Done.")
                    break

                # Skipping Conditions: End
                done_urls = self.crawl(url, levels-1, done_urls)

        return done_urls


    # Notes: If no year.. then stored under key value 0
    def add_done_url(self, wayback_url, done_urls):

        if self.year_folder is True and self.crawled_year not in done_urls:
            done_urls[self.crawled_year] = []

        elif self.year_folder is False and done_urls == {}:
            done_urls[0] = []

        ix = self.crawled_year if self.year_folder is True else 0

        done_urls[ix].append(wayback_url)

        return done_urls

    def url_done(self, url, done_urls):
        ix = self.crawled_year if self.year_folder is True else 0

        if url in done_urls[ix]:
            return True

        if url.replace('www.', '') in done_urls[ix]:
            return True

        return False



    def is_valid_url(self, url):
        if 'mailto' in url:
            return False

        if len(url) > 200:
            return False

        return True


