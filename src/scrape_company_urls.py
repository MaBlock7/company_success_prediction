import os
import dotenv
from config import PROJ_ROOT
from scraper import GoogleMapsScraper

dotenv_path = os.path.join(PROJ_ROOT, '.env')
dotenv.load_dotenv(dotenv_path)


def parse(company_name: str, town: str, street: str, json_data: dict):
    organic_results = json_data.get('organic')
    if organic_results:
        lines = []
        for match in organic_results:
            line = {
                'company_name': company_name,
                'town': town,
                'street': street,
                'matched_name': match.get('title'),
                'matched_address': match.get('address'),
                'website': match.get('link'),
                'categories': [category.get('id') for category in match.get('category', [])],
                'rating': match.get('rating'),
                'reviews_cnt': match.get('reviews_cnt'),
                'latitude': match.get('latitude'),
                'longitude': match.get('longitude')
            }
            lines.append(line)
        return lines
    return []


def main():
    scraper = GoogleMapsScraper(auth_token=os.getenv('SERP_API_KEY'))

    # Load data
    pass

    # Scrape Google Maps
    json_lines = []
    for company_name, town, street in []:
        json_data = scraper.search(company_name, town, street)
        parsed_data = parse(company_name, town, street, json_data)
        json_lines.extend(parsed_data)

    # Save data
    pass


if __name__ == '__main__':
    main()
