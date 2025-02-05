import asyncio
import json
from scraper import CompanyWebCrawler
from config import RAW_DATA_DIR

"""
def load_data():
    return {'CHE152876230': 'https://www.chiron-services.ch'}


async def main():
    cwc = CompanyWebCrawler()
    uid2url = load_data()

    for uid, base_url in uid2url.items():
        urls = cwc.get_urls(base_url)
        filtered_urls = cwc.filter_urls(urls)
        results = await cwc.crawl(filtered_urls)
        final_results = {uid: results}
        with open(RAW_DATA_DIR / f'{uid}.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
"""

def load_data():
    return {'CHE152876230': 'https://www.chiron-services.ch', 'CHE111222333': 'https://www.adresta.ch'}


def save_json(uid: str, results: dict):
    with open(RAW_DATA_DIR / f'{uid}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


async def process_base_url(cwc, uid, base_url):
    """Process a single base URL asynchronously."""
    urls = cwc.get_urls(base_url)

    if not urls:
        # Scrape the base url and find all internal links
        temporary_results = await cwc.crawl([base_url])
        internal_links = temporary_results.get('links', {}).get('internal', [])
        urls = [link.get('href') for link in internal_links if link.get('href')]

    filtered_urls = cwc.filter_urls(urls)
    results = await cwc.crawl(filtered_urls)

    final_results = {uid: results}

    # Save results asynchronously
    save_json(uid, final_results)


async def main():
    cwc = CompanyWebCrawler()
    uid2url = load_data()

    # Create async tasks for all base URLs
    tasks = [process_base_url(cwc, uid, base_url) for uid, base_url in uid2url.items()]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
