import re
import requests
from requests.exceptions import RequestException, ConnectionError
from datetime import date, timedelta, datetime
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from playwright.async_api import Error as PlaywrightError

from xml.etree import ElementTree


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
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()

            root = ElementTree.fromstring(response.content)
            namespace = self._detect_namespace(root)
            return root, namespace

        except TimeoutError as e:
            raise TimeoutError(f"Timeout fetching sitemap") from e
        except ConnectionError as e:
            raise TimeoutError(f"Connection error fetching sitemap") from e
        except RequestException as e:
            raise RuntimeError(f"Request error fetching sitemap") from e
        except ElementTree.ParseError as e:
            raise ValueError(f"Failed to parse XML") from e

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

        except Exception:
            raise

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

        try:
            config = CrawlerRunConfig(
                excluded_tags=["nav", "footer", "aside"],
                excluded_selector="""
                    [class*="cookie"],
                    [id*="cookie"],
                    [class*="consent"],
                    [id*="consent"],
                    [class*="privacy"],
                    [id*="privacy"],
                    [class*="gdpr"],
                    [id*="gdpr"],
                    [class*="ccpa"],
                    [id*="ccpa"],
                    [class^="ch2-"],
                    [id^="ch2-"],
                    #onetrust-banner-sdk,
                    .ot-sdk-container,
                    .ot-pc-header,
                    #CybotCookiebotDialog,
                    .CybotCookiebotDialog,
                    .qc-cmp2-container,
                    #qc-cmp2-ui,
                    .truste-message,
                    .truste-overlay,
                    .iubenda-cookie-policy,
                    .usercentrics-root
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

        except PlaywrightError as e:
            crawler.logger.error(message=e, tag='PLAYWRIGHT ERROR')

        except Exception as e:
            crawler.logger.error(message=e, tag='UNEXPECTED ERROR')
