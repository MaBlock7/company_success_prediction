import asyncio
import json
from scraper import CompanyWebCrawler
from config import RAW_DATA_DIR


UNWANTED_KEYWORDS = [
    'news',
    'terms-and-conditions',
    'terms-of-use',
    'imprint',
    'blog',
    'privacy',
    'privacy-policy',
    'disclosure',
    'shop',
    'store',

    'neuigkeiten',
    'impressum',
    'datenschutz',
    'datenschutzbestimmungen',

    'nouvelles',               # news
    'conditions-generales',    # terms and conditions
    'mentions-legales',        # legal notice
    'blog',
    'confidentialite',         # privacy
    'politique-de-confidentialite',
    'divulgation',             # disclosure
    'boutique',                # shop
    'magasin',                 # store

    'notizie',                 # news
    'termini-e-condizioni',
    'termini-di-utilizzo',
    'informazioni-legali',     # legal notice / imprint
    'blog',
    'privacy',
    'politica-sulla-privacy',
    'divulgazione',
    'negozio',
    'store'
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
    "about", "team", "founders", "people", "staff", "who-we-are", "company", "history",

    "ueber-uns", "gruender", "menschen", "mitarbeiter", "wer-wir-sind", "unternehmen", "geschichte",

    "a-propos", "equipe", "fondateurs", "personnes", "personnel", "qui-nous-sommes", "entreprise", "histoire",

    "chi-siamo", "fondatori", "persone", "staff", "azienda", "storia"
]

CONTACT_KEYWORDS = [
    "contact", "kontakt", "contactez", "contatto"
]

VALUES_KEYWORDS = [
    "values", "ethics", "mission", "vision", "purpose", "what-we-believe", "culture",
    "principles", "commitment", "beliefs",

    "werte", "ethik", "zweck", "glaubenssaetze", "kultur",
    "prinzipien", "engagement", "ueberzeugungen",

    "valeurs", "ethique", "but", "ce-que-nous-croyons", "culture",
    "principes", "engagement", "convictions",

    "valori", "etica", "missione", "visione", "scopo", "cosa-crediamo", "cultura",
    "principi", "impegno", "credenze"
]


def load_data():
    return {'1433629': 'https://www.chiron-services.ch/', '1417133': 'https://www.adresta.ch/'}


def save_json(ehraid: str, results: dict):
    with open(RAW_DATA_DIR / f'{ehraid}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


async def process_base_url(cwc, ehraid, base_url):
    """Process a single base URL asynchronously."""
    results = await cwc.crawl(base_url)

    final_results = {ehraid: results}

    # Save results asynchronously
    await asyncio.to_thread(save_json, ehraid, final_results)


async def main():
    unique_wanted_keywords = list({k for k in PRODUCT_KEYWORDS+ABOUT_TEAM_KEYWORDS+CONTACT_KEYWORDS+VALUES_KEYWORDS})
    unique_unwanted_keywords = list({k for k in UNWANTED_KEYWORDS})
    cwc = CompanyWebCrawler(
        wanted_keywords=unique_wanted_keywords,
        unwanted_keywords=unique_unwanted_keywords
    )
    ehraid2url = load_data()

    # Create async tasks for all base URLs
    tasks = [process_base_url(cwc, ehraid, base_url) for ehraid, base_url in ehraid2url.items()]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
