import re
import requests
from requests.exceptions import RequestException, ConnectionError, JSONDecodeError
from datetime import date, timedelta, datetime
from urllib.parse import urlparse
from crawl4ai import (
    AsyncWebCrawler, CrawlerRunConfig, WebScrapingStrategy, LXMLWebScrapingStrategy)
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from playwright.async_api import Error as PlaywrightError
from wayback import WaybackClient, Mode
from wayback.exceptions import MementoPlaybackError
from xml.etree import ElementTree


class CrawlError(Exception):
    def __init__(self, url, original_exception):
        super().__init__(f"Error for URL {url}: {original_exception}")
        self.url = url
        self.original_exception = original_exception


class URLFilter:

    def __init__(
        self,
        wanted_keywords: list[str],
        unwanted_keywords: list[str],
        lang_exceptions: list[str]
    ):
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
        >>> crawler = CompanyWebCrawler(
                wanted_words=['product', 'about-us'],
                unwanted_words=['news', 'terms-of-use'])
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
        self.wayback_client = WaybackClient()

    @staticmethod
    def _create_sitemap_url(base_url: str) -> str:
        """Adds sitemap subdomain to a base url."""
        return f"{base_url.rstrip('/')}/sitemap.xml" if isinstance(base_url, str) else ''

    @staticmethod
    def create_wayback_urls(urls: list[str], timestamp: str):
        """Creates wayback url for internal links when using the wayback machine."""
        return [f'http://web.archive.org/web/{timestamp}/{url}' for url in urls if isinstance(url, str)]

    @staticmethod
    def _detect_namespace(root: ElementTree.Element) -> dict[str, str]:
        """Extracts the namespace dynamically from the root element."""
        if root.tag.startswith("{"):
            return {"ns": root.tag.split("}")[0].strip("{")}
        return {}

    @staticmethod
    def _create_url_list(root: ElementTree.Element, namespace: dict[str, str]) -> list[str]:
        """Creates a list of urls from the root of the sitemap.xml."""
        return [loc.text.replace('\n', '').strip() for loc in root.findall('.//ns:loc', namespace)] if namespace else []

    def _crawl_sitemap(self, sitemap_url: str) -> tuple[ElementTree.Element, dict[str, str]]:
        """Tries to fetch the sitemap.xml for a URL and extract its root and namespace."""
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()

            if 'xml' not in response.headers.get('Content-Type', ''):
                raise ValueError('Sitemap is not XML')

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
        if not sitemap_url:
            return []
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

    def get_closest_snapshot(
        self,
        url: str,
        year: int,
        month: int,
        day: int,
        window_days: int = 365
    ) -> dict | None:
        """
        Finds the closest available snapshot to the given date using the CDX API.

        Args:
            url (str): The URL of the website to look up.
            year, month, day (int): The target date to find the closest snapshot to.
            window_days (int): Number of days before and after the target date to search.

        Returns:
            dict or None: Closest snapshot info or None if not found.
        """
        try:
            from_date = date(year=year, month=month, day=day)
            to_date = (from_date + timedelta(days=window_days)).strftime('%Y%m%d')

            results = self.wayback_client.search(url, from_date=from_date, to_date=to_date)

            if not results:
                return None

            closest_record = next(results)
            return closest_record.timestamp.strftime("%Y%m%d%H%M%S")

        except Exception as e:
            print(f"Unexpected error in get_closest_snapshot for {url}: {e}")

        return None

    def filter_urls(
        self,
        urls: list[str],
        min_pages: int = 10,
        max_pages: int = 20
    ) -> list[str]:
        return self.filter.filter_urls(urls, min_pages, max_pages)

    async def extract_wayback_content(
        self,
        url: str,
        scraping_strategy: WebScrapingStrategy,
        md_generator: DefaultMarkdownGenerator,
        year: int,
        month: int,
        day: int,
        window_days: int = 365,
        strategy_config: dict = {
            "excluded_tags": ["nav", "footer", "aside"],
            "excluded_selector": """
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
        }
    ):
        try:
            from_date = date(year=year, month=month, day=day)
            to_date = (from_date + timedelta(days=window_days))
            response = list(self.wayback_client.search(url, from_date=from_date, to_date=to_date, limit=20))

            if not response:
                raise ValueError("No archive entries found")
            oldest_record = min(response, key=lambda r: r.timestamp)
            content = self.wayback_client.get_memento(oldest_record, Mode.view, exact=False, target_window=365*24*60*60)  # Allow fetching of the same page within the first year if broken archive
            parsed_content = scraping_strategy.scrap(url, content.text, kwargs=strategy_config)
            markdown = md_generator(parsed_content.cleaned_html)
            links = {'internal': [link.__dict__ for link in parsed_content.links.internal],
                     'external': [link.__dict__ for link in parsed_content.links.external]}
            # Extract closest content from memento that is a non-broken archived page
            return {
                'status_code': 200,
                'markdown': markdown,
                'links': links,
                'date': oldest_record.timestamp.strftime('%Y-%m-%d'),
            }
        except MementoPlaybackError as e:
            return {
                'status_code': 404,
                'error_message': 'MementoPlaybackError'
            }
        except Exception as e:
            return {
                'status_code': 400,
                'error_message': 'Unexpected Error'
            }

    async def wayback_crawl(
        self,
        base_url: str,
        year: int,
        month: int,
        day: int,
        window_days: int = 365,
        max_depth: int = 1
    ):
        if isinstance(base_url, str) and not base_url.endswith('/'):
            base_url += '/'

        scraping_strategy = WebScrapingStrategy()
        md_generator = DefaultMarkdownGenerator()

        successful = {}

        urls, scraped = [base_url], set()
        for _ in range(max_depth):
            to_scrape = [u for u in urls if u not in scraped]
            if not to_scrape:
                break

            for url in to_scrape:
                result = await self.extract_wayback_content(
                    url,
                    scraping_strategy,
                    md_generator,
                    year,
                    month,
                    day,
                    window_days
                )

                if result['status_code'] != 200:
                    successful[url] = result
                    continue

                successful[url] = result

                for link in result.get('links', {}).get('internal', []):
                    href = link.get('href').rstrip('/')
                    if href and ('www.' in href) and (href not in urls):
                        urls.append(href)

            scraped.update(to_scrape)

        filtered_urls = self.filter_urls(urls, min_pages=10, max_pages=20)
        return {key: successful[key] for key in filtered_urls}

    async def crawl(
        self,
        crawler: AsyncWebCrawler,
        urls: list[str],
    ) -> dict[str, dict]:
        """Asynchronously crawls the given list of URLs.

        Args:
            crawler (AsyncWebCrawler): Crawler object initialized outside of the method.
            urls (list[str]): The landing page url of the company.

        Returns:
            dict[str, dict]: A dictionary containing crawl results, including extracted content and links.
        """
        successful = {}

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=75.0,
            check_interval=1.0,
            max_session_permit=10,
        )

        try:
            config = CrawlerRunConfig(
                scraping_strategy=LXMLWebScrapingStrategy(),
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
                process_iframes=True,
                check_robots_txt=True,
                stream=False,
                wait_until="networkidle",
                page_timeout=100_000,
                delay_before_return_html=0.2,
                mean_delay=0.2,
                max_range=0.5,
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
            # Add context about all attempted URLs
            crawler.logger.error(
                message=f"PlaywrightError while crawling URLs: {urls}\n{e}",
                tag='PLAYWRIGHT ERROR'
            )
            return {url: {'status_code': None, 'error_message': str(e)} for url in urls}

        except Exception as e:
            # Add context for better debugging
            crawler.logger.error(
                message=f"Unexpected error while crawling URLs: {urls}\n{e}",
                tag='UNEXPECTED ERROR'
            )
            return {url: {'status_code': None, 'error_message': str(e)} for url in urls}

from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator