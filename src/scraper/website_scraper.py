import re
import requests
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from xml.etree import ElementTree


class CompanyWebCrawler:

    def __init__(self, lang_exceptions: list[str] = ['de', 'en']):
        self.lang_pattern = re.compile(r"/([a-z]{2})/")
        self.lang_exceptions = lang_exceptions

    @staticmethod
    def _create_sitemap_url(base_url: str) -> str:
        return f"{base_url}/sitemap.xml"

    def detect_namespace(root):
        """Extracts the namespace dynamically from the root element."""
        if root.tag.startswith("{"):
            return {"ns": root.tag.split("}")[0].strip("{")}
        return {}

    def get_urls(self, base_url: str) -> list[str]:
        sitemap_url = self._create_sitemap_url(base_url)
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()

            root = ElementTree.fromstring(response.content)

            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [loc.text.replace('\n', '').strip() for loc in root.findall('.//ns:loc', namespace)]

            return urls
        except Exception as e:
            print(f"Error fetching sitemap: {e}")
            return []

    def _filter_unwanted_languages(self, urls: list[str]) -> list[str]:        
        filtered_urls = []
        for url in urls:
            match = self.lang_pattern.search(url)
            if match:
                if match.group(1) in self.lang_exceptions:
                    filtered_urls.append(url)
            else:
                filtered_urls.append(url)
        return filtered_urls

    def filter_urls(self, urls: list[str]) -> list[str]:
        filtered_urls = self._filter_unwanted_languages(urls)
        return filtered_urls

    async def crawl(self, urls: list[str]):

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=10,
        )

        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            scan_full_page=True,
            check_robots_txt=True,
            semaphore_count=3,
            stream=False
        )

        successful = {}

        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun_many(
                urls,
                config=config,
                dispatcher=dispatcher,
                headers={'Accept-Language': 'en,de;q=0.5'}  # Prefer English version of website but accept German as an alternative
            )
            for result in results:
                if result.success:
                    successful[result.url] = {'markdown': result.markdown, 'links': result.links}
                elif result.status_code == 403 and "robots.txt" in result.error_message:
                    successful[result.url] = {'error_message': result.error_message}
                else:
                    successful[result.url] = {'error_message': result.error_message}

        return successful
