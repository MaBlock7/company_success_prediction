import argparse
import asyncio
import gzip
import json
import traceback
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from crawl4ai import AsyncWebCrawler, BrowserConfig
from scraper import CompanyWebCrawler
from scraper.crawler_config import WANTED_KEYWORDS, UNWANTED_KEYWORDS
from config import RAW_DATA_DIR


def save_compressed_json(idx: int, file: dict, wayback: bool = False):
    """Save a dictionary as a compressed JSON (.json.gz) file.

    Args:
        idx (int): Index used to name the output file.
        file (dict): The dictionary to save.
        wayback (bool, optional): Whether to save as a Wayback Machine snapshot. Defaults to False.
    """
    folder = 'wayback' if wayback else 'current'
    with gzip.open(RAW_DATA_DIR / 'company_websites' / folder / f'{idx}_websites.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=2)


def save_completed_ehraids(ehraids: list[int], wayback: bool = False):
    folder = 'wayback' if wayback else 'current'
    with open(RAW_DATA_DIR / 'company_websites' / folder / 'completed_ehraids.txt', 'a', encoding='utf-8') as f:
        f.writelines([str(ehraid) + '\n' for ehraid in ehraids])


def save_urls(urls: list[str]) -> None:
    """Append a list of URLs to a text file.

    Args:
        urls (list[str]): List of URLs to append.
    """
    with open(RAW_DATA_DIR / 'company_urls' / 'internal_urls.txt', 'a') as f:
        for url in urls:
            f.write(f'{url}\n')


def split_dataframe(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Splits a DataFrame into `n` approximately equal-sized chunks.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        n (int): Number of chunks to create.

    Returns:
        List[pd.DataFrame]: A list of DataFrames.
    """
    chunk_size = (len(df) + n - 1) // n  # ceil division
    return [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n) if i * chunk_size < len(df)]


async def bfs_search_urls(crawler: AsyncWebCrawler, cwc: CompanyWebCrawler, base_url: str, max_depth: int) -> list[str]:
    """Perform a breadth-first search to discover internal URLs from a base URL.

    Args:
        crawler (AsyncWebCrawler): The asynchronous web crawler instance.
        cwc (CompanyWebCrawler): The company-specific crawler logic.
        base_url (str): The base URL to start crawling from.
        max_depth (int): Maximum BFS depth.

    Returns:
        list[str]: A list of discovered internal URLs.
    """
    urls, scraped = [base_url], set()
    for _ in range(max_depth):
        to_scrape = [url for url in urls if url not in scraped]
        if not to_scrape:
            break
        temp_results = await cwc.crawl(crawler, to_scrape)

    scraped.update(to_scrape)

    for content in temp_results.values():
        for link in content.get('links', {}).get('internal', []):
            href = link.get('href').rstrip('/')
            if href and ('www.' in href) and (href not in urls):
                urls.append(href)

    return urls


async def fetch_urls_from_landing_page(
    crawler: AsyncWebCrawler,
    cwc: CompanyWebCrawler,
    base_url: str,
    max_depth: int = 1,
    semaphore: asyncio.Semaphore = None,
) -> None:
    if semaphore:
        async with semaphore:
            return await _fetch_urls(crawler, cwc, base_url, max_depth)
    return await _fetch_urls(crawler, cwc, base_url, max_depth)


async def _fetch_urls(
    crawler: AsyncWebCrawler,
    cwc: CompanyWebCrawler,
    base_url: str,
    max_depth: int = 1
) -> None:
    """Fetches the internal links present on the landing page of a website.

    Args:
        crawler (AsyncWebCrawler): The web crawler.
        cwc (CompanyWebCrawler): The company-specific web crawler.
        base_url (str): The base URL to process.
        max_depth (int, optional): Depth of link crawling. Defaults to 1.

    Returns:
        dict: A dictionary mapping the ehraid to the crawl result.
    """
    try:
        crawler.logger.info(message=base_url, tag='FETCH SITEMAP')
        urls = await asyncio.to_thread(cwc.get_urls, base_url)

    except (TimeoutError) as e:
        # Website (base_url) is unresponsive, therefore we don't need to try further
        crawler.logger.error(
            message=f'Error occured for {base_url}: {e}',
            tag='UNRESPONSIVE ERROR'
        )
        return

    except (RuntimeError, ValueError) as e:
        # Sitemap for base_url doesn't exist, but website is responsive
        crawler.logger.error(
            message=f'Error occured for {base_url}: {e}',
            tag='NO SITEMAP ERROR'
        )
        urls = []

    if not urls:
        urls = await bfs_search_urls(crawler, cwc, base_url, max_depth)
    await asyncio.to_thread(save_urls, urls)


async def process_base_url(
    crawler: AsyncWebCrawler,
    cwc: CompanyWebCrawler,
    ehraid: int,
    base_url: str,
    max_depth: int = 1,
    semaphore: asyncio.Semaphore = None,
) -> dict:
    if semaphore:
        async with semaphore:
            return await _process_base_url(crawler, cwc, ehraid, base_url, max_depth)
    return await _process_base_url(crawler, cwc, ehraid, base_url, max_depth)


async def _process_base_url(
    crawler: AsyncWebCrawler,
    cwc: CompanyWebCrawler,
    ehraid: int,
    base_url: str,
    max_depth: int = 1,
) -> dict:
    """Process a base URL to either save URLs or crawl filtered content.

    Args:
        crawler (AsyncWebCrawler): The web crawler.
        cwc (CompanyWebCrawler): The company-specific web crawler.
        ehraid (int): Unique identifier for the company.
        base_url (str): The base URL to process.
        max_depth (int, optional): Depth of link crawling. Defaults to 1.

    Returns:
        dict: A dictionary mapping the ehraid to the crawl result.
    """
    try:
        crawler.logger.info(message=base_url, tag='FETCH SITEMAP')
        urls = await asyncio.to_thread(cwc.get_urls, base_url)

    except (TimeoutError) as e:
        # Website (base_url) is unresponsive, therefore we don't need to try further
        crawler.logger.error(
            message=f'Error occured for {base_url}: {e}',
            tag='UNRESPONSIVE ERROR'
        )
        return {ehraid: {base_url: {'status_code': 408, 'error_message': str(e)}}}

    except (RuntimeError, ValueError) as e:
        # Sitemap for base_url doesn't exist, but website is responsive
        crawler.logger.error(
            message=f'Error occured for {base_url}: {e}',
            tag='NO SITEMAP ERROR'
        )
        urls = []

    if not urls:
        urls = await bfs_search_urls(crawler, cwc, base_url, max_depth)

    filtered_urls = cwc.filter_urls(urls, min_pages=10, max_pages=20)
    results = await cwc.crawl(crawler, filtered_urls)
    return {ehraid: results}


async def wayback_process_base_url(
    cwc: CompanyWebCrawler,
    ehraid: int,
    base_url: str,
    year: int,
    month: int,
    day: int,
    max_depth: int = 1,
    semaphore: asyncio.Semaphore = None,
) -> dict:
    if semaphore:
        async with semaphore:
            return await _wayback_process(cwc, ehraid, base_url, year, month, day, max_depth)
    return await _wayback_process(cwc, ehraid, base_url, year, month, day, max_depth)


async def _wayback_process(
    cwc: CompanyWebCrawler,
    ehraid: int,
    base_url: str,
    year: int,
    month: int,
    day: int,
    max_depth: int = 1
) -> dict:
    """Process a base URL using the Wayback Machine for a specific date to retrieve
    historical versions of a website.

    Args:
        cwc (CompanyWebCrawler): The company-specific web crawler.
        ehraid (int): Unique identifier for the company.
        base_url (str): The base URL to process.
        year (int): Year of snapshot.
        month (int): Month of snapshot.
        day (int): Day of snapshot.
        max_depth (int, optional): Depth of link crawling. Defaults to 1.

    Returns:
        dict: A dictionary mapping the ehraid to the crawl result.
    """
    try:
        results = await cwc.wayback_crawl(base_url, year, month, day, max_depth=max_depth)
        return {ehraid: results}

    except Exception as e:
        print(f"Error occurred during processing of {base_url}: {e}")


async def main(args):

    folder = 'wayback' if args.wayback else 'current'
    completed_ehraids_path = RAW_DATA_DIR / 'company_websites' / folder / 'completed_ehraids.txt'
    completed_ehraids = []
    if completed_ehraids_path.exists():
        with open(completed_ehraids_path, 'r') as f:
            completed_ehraids = [int(line.replace('\n', '')) for line in f.readlines()]

    cwc = CompanyWebCrawler(
        wanted_keywords=WANTED_KEYWORDS,
        unwanted_keywords=UNWANTED_KEYWORDS
    )

    crawler = AsyncWebCrawler(
        config=BrowserConfig(
            headers={'Accept-Language': 'en,de;q=0.8,fr;q=0.6,it;q=0.4'},  # Prefer English version of website but accept German, French, or Italian as an alternative
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
    )

    semaphore = asyncio.Semaphore(int(args.semaphore))

    try:
        chunk_size = 500
        with pd.read_csv(
            RAW_DATA_DIR / 'company_sample' / 'company_sample_website.csv',
            parse_dates=['founding_date'],
            chunksize=chunk_size,
            usecols=['ehraid', 'company_url', 'founding_date']
        ) as reader:

            for i, subset_df in enumerate(reader):

                subset_df = subset_df[(~subset_df['ehraid'].isin(completed_ehraids)) & (~subset_df['company_url'].isna())]
                print(f'At index {i}: {len(subset_df)} entries to scrape...')

                if subset_df.empty:
                    continue

                storage_file = {}
                for chunk_df in split_dataframe(subset_df, n=10):

                    await crawler.start()

                    if args.wayback:
                        tasks = [
                            wayback_process_base_url(
                                cwc,
                                ehraid=data['ehraid'],
                                base_url=data['company_url'],
                                year=data['founding_date'].year,
                                month=data['founding_date'].month,
                                day=data['founding_date'].day,
                                semaphore=semaphore
                            ) for _, data in chunk_df.iterrows()
                        ]
                    elif args.urls_only:
                        tasks = [
                            fetch_urls_from_landing_page(
                                crawler,
                                cwc,
                                base_url=data['company_url'],
                                semaphore=semaphore
                            ) for _, data in chunk_df.iterrows()
                        ]
                    else:
                        tasks = [
                            process_base_url(
                                crawler,
                                cwc,
                                ehraid=data['ehraid'],
                                base_url=data['company_url'],
                                semaphore=semaphore
                            ) for _, data in chunk_df.iterrows()
                        ]

                    try:
                        results = await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=900  # 15 minutes in seconds
                        )
                        if args.urls_only:
                            continue

                        for res in results:
                            if isinstance(res, Exception):
                                crawler.logger.error(f"task failed: {res}")
                                continue
                            elif isinstance(res, dict):
                                storage_file.update(res)

                        await crawler.close()

                    except asyncio.TimeoutError:
                        crawler.logger.error("Chunk timed out after 15 minutes.")
                        await crawler.close()
                        continue

                save_compressed_json(idx=i, file=storage_file, wayback=args.wayback)
                save_completed_ehraids(subset_df['ehraid'].tolist(), wayback=args.wayback)

    except Exception:
        print('Unexpected error occured:')
        traceback.print_exc()

    finally:
        await crawler.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CompanyCrawler',
        description='Crawls the websites for given company urls',
    )
    parser.add_argument('--wayback', action='store_true')
    parser.add_argument('--urls_only', action='store_true')
    parser.add_argument('--semaphore', default=5)
    args = parser.parse_args()

    asyncio.run(main(args))
