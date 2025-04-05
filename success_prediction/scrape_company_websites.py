import asyncio
import gzip
import json
from crawl4ai import AsyncWebCrawler
from scraper import CompanyWebCrawler
from config import RAW_DATA_DIR


UNWANTED_KEYWORDS = [
    'news',
    'terms-and-conditions',
    'terms-of-use',
    'imprint',
    'blog',
    'privacy',
    'disclosure',
    'legal',
    'shop',
    'store',
    'career',
    'jobs',

    'neuigkeiten',
    'impressum',
    'datenschutz',
    'datenschutzbestimmungen',
    'karriere',

    'nouvelles',               # news
    'conditions-generales',    # terms and conditions
    'mentions-legales',        # legal notice
    'blog',
    'confidentialite',         # privacy
    'politique-de-confidentialite',
    'divulgation',             # disclosure
    'boutique',                # shop
    'magasin',                 # store
    'carriere',

    'notizie',                 # news
    'termini-e-condizioni',
    'termini-di-utilizzo',
    'informazioni-legali',     # legal notice / imprint
    'blog',
    'privacy',
    'politica-sulla-privacy',
    'divulgazione',
    'negozio',
    'store',
    'carriera',
]

PRODUCT_KEYWORDS = [
    "product", "service", "solution", "offerings", "platform", "features", "tools",
    "application", "technology", "catalog", "portfolio", "what-we-offer", "what-we-do",

    "produkte", "dienstleistung", "loesung", "angebot", "plattform", "funktionen",
    "anwendung", "technologie", "katalog", "portfolio", "was-wir-anbieten", "was-wir-tun",

    "produit", "service", "solution", "offre", "plateforme", "fonctionnalites", "outils",
    "application", "technologie", "catalogue", "portfolio", "ce-que-nous-offrons", "ce-que-nous-faisons",

    "prodotto", "servizio", "soluzione", "offerta", "piattaforma", "funzionalita", "strumenti",
    "applicazione", "tecnologia", "catalogo", "portfolio", "cosa-offriamo", "cosa-facciamo"
]

ABOUT_TEAM_KEYWORDS = [
    "about", "team", "founder", "people", "staff", "who-we-are", "company", "history",

    "ueber-uns", "uber-uns", "gruender", "menschen", "mitarbeiter", "wer-wir-sind", "unternehmen", "geschichte",

    "a-propos", "equipe", "fondateur", "personnes", "personnel", "qui-nous-sommes", "entreprise", "histoire",

    "chi-siamo", "fondator", "persone", "staff", "azienda", "storia"
]

CONTACT_KEYWORDS = [
    "contact", "kontakt", "contactez", "contatto"
]

VALUES_KEYWORDS = [
    "values", "goal", "mission", "vision", "strategy", "purpose", "what-we-believe", "culture",
    "principles", "commitment", "beliefs",

    "werte", "ziel", "zweck", "glaubenssaetze", "strategie", "kultur",
    "prinzipien", "engagement", "ueberzeugungen",

    "valeurs", "but", "ce-que-nous-croyons", "culture",
    "principes", "engagement", "convictions",

    "valori", "obiettivo", "missione", "visione", "strategia", "scopo", "cosa-crediamo", "cultura",
    "principi", "impegno", "credenze"
]


def load_data():
    return {
        '1433629': {'url': 'https://www.chiron-services.ch', 'founding_date': (2020, 4, 17)},
        '1417133': {'url': 'https://www.adresta.ch', 'founding_date': (2019, 12, 4)}
    }

"""
def save_json(ehraid: str, results: dict, wayback: bool = False):
    name = f'{ehraid}_wb.json.gz' if wayback else f'{ehraid}.json.gz'
    with gzip.open(RAW_DATA_DIR / name, 'wt', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
"""

def save_json(ehraid: str, results: dict, wayback: bool = False):
    name = f'{ehraid}_wb.json' if wayback else f'{ehraid}.json'
    with open(RAW_DATA_DIR / name, 'wt', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

async def process_base_url(cwc, ehraid, base_url, max_depth: int = 1):
    try:
        try:
            urls = await asyncio.wait_for(asyncio.to_thread(cwc.get_urls, base_url), timeout=10)
        except asyncio.TimeoutError:
            print(f"[Timeout] Sitemap fetch for {base_url} took too long.")
            urls = []

        async with AsyncWebCrawler() as crawler:
            if not urls:
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

            filtered_urls = cwc.filter_urls(urls, min_pages=10, max_pages=20)
            results = await cwc.crawl(crawler, filtered_urls)
            await asyncio.to_thread(save_json, ehraid, {ehraid: results}, True)

    except Exception as e:
        print(f"Error occurred during processing of {base_url}: {e}")


async def wayback_process_base_url(
    cwc,
    ehraid,
    base_url,
    year,
    month,
    day,
    max_depth: int = 1
):
    wayback_base_url, timestamp = cwc.get_closest_snapshot(base_url, year, month, day)
    if not wayback_base_url:
        print(f"No snapshot found for {base_url}")
        return
    try:
        async with AsyncWebCrawler() as crawler:
            urls, scraped = [base_url], set()
            for _ in range(max_depth):
                to_scrape = [url for url in urls if url not in scraped]

                if not to_scrape:
                    break
                temp_results = await cwc.crawl(crawler, cwc.create_wayback_urls(to_scrape, timestamp))

            scraped.update(to_scrape)

            for content in temp_results.values():
                for link in content.get('links', {}).get('internal', []):
                    href = link.get('href').rstrip('/')
                    if href and ('www.' in href) and (href not in urls):
                        urls.append(href)

            filtered_urls = cwc.filter_urls(urls, min_pages=10, max_pages=20)
            results = await cwc.crawl(crawler, filtered_urls)
            await asyncio.to_thread(save_json, ehraid, {ehraid: results})

    except Exception as e:
        print(f"Error occurred during processing of {base_url}: {e}")


async def main():
    unique_wanted_keywords = list({k for k in PRODUCT_KEYWORDS+ABOUT_TEAM_KEYWORDS+CONTACT_KEYWORDS+VALUES_KEYWORDS})
    unique_unwanted_keywords = list({k for k in UNWANTED_KEYWORDS})
    cwc = CompanyWebCrawler(
        wanted_keywords=unique_wanted_keywords,
        unwanted_keywords=unique_unwanted_keywords
    )
    ehraid2data = load_data()

    # tasks = [process_base_url(cwc, ehraid, data['url']) for ehraid, data in ehraid2data.items()]
    tasks = [wayback_process_base_url(cwc, ehraid, data['url'], data['founding_date'][0], data['founding_date'][1], data['founding_date'][2]) for ehraid, data in ehraid2data.items()]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
