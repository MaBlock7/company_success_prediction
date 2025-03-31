import asyncio
import json
from scraper import CompanyWebCrawler
from config import RAW_DATA_DIR


UNWANTED_WORDS = [
    'news',
    'neuigkeiten',
    'terms-and-conditions',
    'terms-of-use',
    'imprint',
    'impressum',
    'contact',
    'kontakt',
    'blog',
    'privacy',
    'privacy-policy',
    'datenschutz',
    'datenschutzbestimmungen',
    'disclosure',
    'shop',
    'store',
]


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
        internal_links = temporary_results[base_url].get('links', {}).get('internal', [])
        urls = [base_url]
        for link in internal_links:
            href = link.get('href')
            if href:
                if 'www.' in href and href not in urls:
                    urls.append(href)

    filtered_urls = cwc.filter_urls(urls)
    results = await cwc.crawl(filtered_urls)

    final_results = {uid: results}

    # Save results asynchronously
    await asyncio.to_thread(save_json, uid, final_results)


async def main():
    cwc = CompanyWebCrawler(unwanted_words=UNWANTED_WORDS)
    uid2url = load_data()

    # Create async tasks for all base URLs
    tasks = [process_base_url(cwc, uid, base_url) for uid, base_url in uid2url.items()]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
