import os
import dotenv
import pandas as pd
from config import PROJ_ROOT
from scraper import GoogleMapsScraper

dotenv_path = os.path.join(PROJ_ROOT, '.env')
dotenv.load_dotenv(dotenv_path)


def load_data():
    """
    Loads the company data containing unique identifiers, names, towns, and streets.

    Returns:
        A list of tuples containing company data.
    """
    return [
        ('CHE472481853', 'Bitcoin Suisse AG', 'Zug', 'Grafenauweg'),
        ('CHE312574485', 'Bitcoin Capital AG', 'Zug', 'Gubelstrasse'),
        ('CHE152876230', 'Chiron Services GmbH', 'Zürich', 'Badenerstrasse'),
        ('CHE152876230', 'Chiron Services GmbH', 'Basel', 'Starenstrasse'),
    ]


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


def parse(uid: str, company_name: str, town: str, street: str, scraped_results: dict = None, potential_matches: list[dict] | dict = None):
    """
    Parses scraped data and identifies perfect or close matches.

    Args:
        uid (str): Unique identifier for the company.
        company_name (str): The company name.
        town (str): The town name.
        street (str): The street name.
        scraped_results (dict, optional): Scraped Google Maps results.
        potential_matches (list[dict] | dict, optional): List or dictionary of potential matches.

    Returns:
        A tuple of lists of perfect and close matches. Both are of the following format:

        {'query': {
            'uid': 'CHE472481853',
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
                    'uid': uid,
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


def find_perfect_match(uid: str, company_name: str, town: str, street: str, search_dict: dict) -> list[dict] | None:
    """
    Finds a perfect match for the company name in the search dictionary.

    Args:
        uid (str): Unique identifier for the company.
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
        perfect_match, _ = parse(uid, company_name, town, street, potential_matches=match)
        if perfect_match:
            return perfect_match
    return None


def find_fuzzy_match(uid: str, company_name: str, town: str, street: str, search_dict: dict)  -> list[dict] | None :
    """
    Finds a fuzzy match for the company name by checking partial matches in the search dictionary.

    Args:
        uid (str): Unique identifier for the company.
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
        perfect_match, _ = parse(uid, company_name, town, street, potential_matches=fuzzy_matches)
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

    for uid, company_name, town, street in data:

        # Check if there is a perfect match for the company name already in the search dict
        match = find_perfect_match(uid, company_name, town, street, search_dict)
        if match:
            perfect_matches.extend(match)
            update_search_dict(search_dict, match)
            continue

        # Check if there is a close match for the company already in the search dict keys
        fuzzy_match = find_fuzzy_match(uid, company_name, town, street, search_dict)
        if fuzzy_match:
            perfect_matches.extend(fuzzy_match)
            update_search_dict(search_dict, fuzzy_match)
            continue

        # Else scrape the data from Google Maps to find the url
        json_data = scraper.search(company_name, town)
        perfect_match, potential_matches = parse(uid, company_name, town, street, scraped_results=json_data)
        perfect_matches.extend(perfect_match)

        for match in potential_matches:
            title = match['match'].get('title')
            if title:
                search_dict[title.lower()] = match['match']

    return perfect_matches, search_dict


if __name__ == '__main__':
    perfect_matches, search_dict = main()
