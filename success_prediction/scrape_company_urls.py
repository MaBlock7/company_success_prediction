import os
import dotenv
import pandas as pd
from pocketknife.database import (
    connect_database, read_from_database)
from config import PROJ_ROOT, RAW_DATA_DIR
from scraper import GoogleMapsScraper

dotenv_path = os.path.join(PROJ_ROOT, '.env')
dotenv.load_dotenv(dotenv_path)


def load_data():
    """
    Loads the company data containing unique identifiers, names, towns, and streets.

    Returns:
        A list of tuples containing company data.
    """
    # This query gets a all firms (except branches) that existed between 2016 and current
    query_all_active_firms = """
        SELECT
            base.name,
            base.ehraid,
            base.uid,
            address.street,
            address.town,
        FROM zefix.base base
        -- Get only companies where we have the full history from founding (2016-present)
        LEFT JOIN (
            SELECT s.ehraid, s.shab_id, s.shab_date
            FROM zefix.shab s
            WHERE s.shab_id IN (
                SELECT shab_id
                FROM zefix.shab_mutation
                WHERE description = 'status.neu'
            )
        ) AS shab
        ON base.ehraid = shab.ehraid
        -- Join the addresses of the firms
        LEFT JOIN zefix.address address
        ON base.ehraid = address.ehraid
        --Exclude all kind of branches
        WHERE
            (NOT base.delete_date < '2016-01-01' OR base.delete_date IS NULL)
            AND NOT base.legal_form_id IN (9, 11, 13, 14, 18, 19)
            AND NOT base.is_branch
            AND LOWER(base.name) NOT LIKE '%zweigniederlassung%'
            AND LOWER(base.name) NOT LIKE '%succursale%';
    """
    with connect_database() as con:
        company_data = read_from_database(con, query=query_all_active_firms)
    return [(row['ehraid'], row['name'], row['town'], row['street'],) for _, row in company_data.iterrows()]


def check_if_scraped(scraped_results: dict = None, potential_matches: list[dict] | dict = None):
    """
    Checks if results have been scraped, otherwise returns potential matches.

    Args:
        scraped_results (dict, optional): The scraped Google Maps results.
        potential_matches (list[dict] | dict, optional): A list or dictionary of potential matches.

    Returns:
        A list of match dictionaries.
    """
    if scraped_results:
        return scraped_results.get('organic')
    return potential_matches if isinstance(potential_matches, list) else [potential_matches]


def evaluate_conditions(company_name: str, street: str, town: str, match: dict) -> tuple[bool]:
    """
    Evaluates conditions for determining if a match is perfect or partial.

    Args:
        company_name (str): The company name.
        street (str): The street name.
        town (str): The town name.
        match (dict): A match dictionary containing search results.

    Returns:
        A tuple of boolean values indicating which conditions are True/False.
    """
    perfect_name_match = company_name.lower() == match.get('title', '<no_match>').lower()
    partial_name_match = match.get('title', '<no_match>').lower() in company_name.lower()
    street_name_match = street.lower() in match.get('address', '<no_match>').lower()
    town_name_match = town.lower() in match.get('address', '<no_match>').lower()
    has_no_address = match.get('address') is None
    return perfect_name_match, partial_name_match, street_name_match, town_name_match, has_no_address


def parse(ehraid: str, company_name: str, town: str, street: str, scraped_results: dict = None, potential_matches: list[dict] | dict = None):
    """
    Parses scraped data and identifies perfect or close matches.

    Args:
        ehraid (str): Unique identifier for the company.
        company_name (str): The company name.
        town (str): The town name.
        street (str): The street name.
        scraped_results (dict, optional): Scraped Google Maps results.
        potential_matches (list[dict] | dict, optional): List or dictionary of potential matches.

    Returns:
        A tuple of lists of perfect and close matches. Both are of the following format:

        {'query': {
            'ehraid': '123',
            'company_name': 'Bitcoin Suisse AG',
            'town': 'Zug',
            'street': 'Grafenauweg'},
        'match': {
            'title': 'Bitcoin Suisse AG',
            'address': None,
            'link': 'https://bitcoinsuisse.com/',
            'category': [{'id': 'financial_institution', 'title': 'Finanzinstitut'},
                         {'id': 'establishment_service', 'title': 'Service establishment'}],
            'rating': 5,
            'reviews_cnt': 3,
            'latitude': 46.813187299999996,
            'longitude': 8.224119}}

        All the main and sub-keys are always given but can be None.
    """
    results = check_if_scraped(scraped_results, potential_matches)

    if results:
        perfect_matches, close_matches = [], []
        for match in results:
            line = {
                'query': {
                    'ehraid': ehraid,
                    'company_name': company_name,
                    'town': town,
                    'street': street,
                },
                'match': {
                    'title': match.get('title'),
                    'address': match.get('address'),
                    'link': match.get('link'),
                    'category': match.get('category'),
                    'rating': match.get('rating'),
                    'reviews_cnt': match.get('reviews_cnt'),
                    'latitude': match.get('latitude'),
                    'longitude': match.get('longitude')
                }
            }
            perfect_name_match, partial_name_match, street_name_match, town_name_match, has_no_address = evaluate_conditions(company_name, street, town, match)

            if perfect_name_match and (street_name_match or town_name_match):
                perfect_matches.append(line)
            elif partial_name_match and street_name_match:
                perfect_matches.append(line)
            elif perfect_name_match and has_no_address:
                perfect_matches.append(line)
            else:
                close_matches.append(line)

        return perfect_matches, close_matches

    return [], []


def find_perfect_match(ehraid: str, company_name: str, town: str, street: str, search_dict: dict) -> list[dict] | None:
    """
    Finds a perfect match for the company name in the search dictionary.

    Args:
        ehraid (str): Unique identifier for the company.
        company_name (str): The company name.
        town (str): The town name.
        street (str): The street name.
        search_dict (dict): Dictionary of previous Google Maps search results that
            were no perfect matches with previously searched companies.

    Returns:
        List of perfect matches or None if no match is found.
    """
    company_name_lower = company_name.lower()
    match = search_dict.get(company_name_lower)

    if match:
        perfect_match, _ = parse(ehraid, company_name, town, street, potential_matches=match)
        if perfect_match:
            return perfect_match
    return None


def find_fuzzy_match(ehraid: str, company_name: str, town: str, street: str, search_dict: dict)  -> list[dict] | None :
    """
    Finds a fuzzy match for the company name by checking partial matches in the search dictionary.

    Args:
        ehraid (str): Unique identifier for the company.
        company_name (str): The company name.
        town (str): The town name.
        street (str): The street name.
        search_dict (dict): Dictionary of previous Google Maps search results that
            were no perfect matches with previously searched companies.

    Returns:
        List of fuzzy matches or None if no match is found.
    """
    company_name_lower = company_name.lower()
    fuzzy_matches = [match for matched_name, match in search_dict.items() if matched_name.lower() in company_name_lower]

    if fuzzy_matches:
        perfect_match, _ = parse(ehraid, company_name, town, street, potential_matches=fuzzy_matches)
        if perfect_match:
            return perfect_match
    return None


def update_search_dict(search_dict: dict, perfect_matches: list[dict]) -> None:
    """
    Removes matched company names from the search dict.

    Args:
        search_dict (dict): Dictionary of previous Google Maps search results that
            were no perfect matches with previously searched companies.
        perfect_matches (list): The matched data from the search_dict.
    """
    for match in perfect_matches:
        search_dict.pop(match['match']['title'].lower(), None)


def main():
    scraper = GoogleMapsScraper(auth_token=os.getenv('SERP_API_KEY'))

    # load data
    data = load_data()
    perfect_matches = []
    search_dict = {}

    for ehraid, company_name, town, street in data:

        # Check if there is a perfect match for the company name already in the search dict
        match = find_perfect_match(ehraid, company_name, town, street, search_dict)
        if match:
            perfect_matches.extend(match)
            update_search_dict(search_dict, match)
            continue

        # Check if there is a close match for the company already in the search dict keys
        fuzzy_match = find_fuzzy_match(ehraid, company_name, town, street, search_dict)
        if fuzzy_match:
            perfect_matches.extend(fuzzy_match)
            update_search_dict(search_dict, fuzzy_match)
            continue

        # Else scrape the data from Google Maps to find the url
        json_data = scraper.search(company_name, town)
        perfect_match, potential_matches = parse(ehraid, company_name, town, street, scraped_results=json_data)
        perfect_matches.extend(perfect_match)

        for match in potential_matches:
            title = match['match'].get('title')
            if title:
                search_dict[title.lower()] = match['match']

    return perfect_matches, search_dict


if __name__ == '__main__':
    perfect_matches, search_dict = main()
    output_data = [{
        'ehraid': line['query']['ehraid'],
        'company_name': line['query']['company_name'],
        'company_url': line['match']['link']
    } for line in perfect_matches]

    pd.DataFrame(output_data).to_csv(RAW_DATA_DIR / 'company_urls' / 'urls.csv', index=False)
