import re
import requests
from datetime import datetime
from functools import lru_cache
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, LXMLWebScrapingStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.scorers import URLScorer
from xml.etree import ElementTree
from urllib.parse import urlparse


class WantedUnwantedRelevanceScorer(URLScorer):
    __slots__ = ('_weight', '_stats', '_wanted_keywords', '_unwanted_keywords', '_case_sensitive')

    def __init__(
            self,
            wanted_keywords: list[str],
            unwanted_keywords: list[str],
            weight: float = 1.0,
            case_sensitive: bool = False
        ):
        super().__init__(weight=weight)
        self._case_sensitive = case_sensitive

        # Pre-process keywords once
        self._wanted_keywords = [k if case_sensitive else k.lower() for k in wanted_keywords]
        self._unwanted_keywords = [k if case_sensitive else k.lower() for k in unwanted_keywords]

    @lru_cache(maxsize=10000)
    def _url_bytes(self, url: str) -> bytes:
        """Cache decoded URL bytes"""
        return url.encode('utf-8') if self._case_sensitive else url.lower().encode('utf-8')

    def _calculate_score(self, url: str) -> float:
        """Fast string matching without regex or byte conversion"""
        parsed = urlparse(url)
        path = parsed.path

        if not self._case_sensitive:
            path = path.lower()

        wanted_matches = sum(1 for k in self._wanted_keywords if k in path)
        unwanted_matches = sum(1 for k in self._unwanted_keywords if k in path)

        if wanted_matches and unwanted_matches:
            return 0.25
        if unwanted_matches:
            return 0.0
        if wanted_matches:
            return 1.0
        return 0.5


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
    def __init__(self, wanted_keywords: list[str], unwanted_keywords: list[str] = [], lang_exceptions: list[str] = ['de', 'fr', 'it', 'en']):
        self.unwanted_words_pattern = re.compile(rf"/({'|'.join(map(re.escape, unwanted_keywords))})(?:/|$)")
        self.lang_pattern = re.compile(r"/([a-z]{2})/")
        self.lang_exceptions = lang_exceptions
        self.current_date = datetime.today().strftime('%Y-%m-%d')

        scorer = WantedUnwantedRelevanceScorer(
            wanted_keywords=wanted_keywords,
            unwanted_keywords=unwanted_keywords
        )

        strategy = BestFirstCrawlingStrategy(
            max_depth=2,
            include_external=False,
            url_scorer=scorer,
            max_pages=15,
        )

        self.dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=10,
        )

        self.config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            scraping_strategy=LXMLWebScrapingStrategy(),
            markdown_generator=DefaultMarkdownGenerator(),
            stream=True,
            scan_full_page=True,
            check_robots_txt=True,
        )

    @staticmethod
    def _create_sitemap_url(base_url: str) -> str:
        """Adds sitemap subdomain to a base url."""
        return f"{base_url}/sitemap.xml"

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

    def _filter_unwanted_languages(self, urls: list[str]) -> list[str]:
        """Removes URLs that indicate a foreign language except for languages
        that are specified in the lang_exceptions list."""
        filtered_urls = []
        for url in urls:
            match = self.lang_pattern.search(url)
            if match:
                if match.group(1) in self.lang_exceptions:
                    filtered_urls.append(url)
            else:
                filtered_urls.append(url)
        return filtered_urls

    def _filter_unwanted_words(self, urls: list[str]) -> list[str]:
        """Removes URLs that contain unwanted words as specified in the unwanted_words list."""
        return [url for url in urls if not self.unwanted_words_pattern.search(url)]

    def filter_urls(self, urls: list[str]) -> list[str]:
        """Filters unwanted URLs based on language and specific keywords.

        Args:
            urls (list[str]): A list of URLs to be filtered.

        Returns:
            list[str]: A filtered list of URLs with unwanted entries removed.
        """
        filtered_urls = self._filter_unwanted_languages(urls)
        filtered_urls = self._filter_unwanted_words(filtered_urls)
        return filtered_urls

    async def crawl(self, base_url: str):
        """Asynchronously crawls the given list of URLs.

        Args:
            base_url (str): The landing page url of the company.

        Returns:
            dict[str, dict]: A dictionary containing crawl results, including extracted content and links.
        """
        successful = {}
        async with AsyncWebCrawler() as crawler:
            async for result in await crawler.arun(
                base_url,
                config=self.config,
                dispatcher=self.dispatcher,
                headers={
                    # 'Accept-Language': 'en,de;q=0.5',  # Prefer English version of website but accept German as an alternative
                    'Cookie': 'CookieConsent=true; consent=all',  # Accept all cookies
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
            ):
                if result.success:
                    score = result.metadata.get('score', 0)
                    if score > 0:
                        successful[result.url] = {
                            'markdown': result.markdown,
                            'links': result.links,
                            'date': self.current_date,
                            'score': score,
                            'depth': result.metadata.get('depth', 0)
                        }
                elif result.status_code == 403 and "robots.txt" in result.error_message:
                    successful[result.url] = {'status_code': result.status_code, 'error_message': result.error_message}
                else:
                    successful[result.url] = {'status_code': result.status_code, 'error_message': result.error_message}

        return successful
