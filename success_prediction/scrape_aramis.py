import re
import requests
import time
from datetime import datetime, date
from tqdm import tqdm
import pandas as pd
from config import RAW_DATA_DIR



def to_aramis_date(dt: datetime) -> str:
    """Converts a datetime object to the ARAMIS timestamp string format.

    Args:
        dt (datetime): The datetime object to convert.

    Returns:
        str: The ARAMIS timestamp string in the format /Date(XXXXXXXX+0200)/.
    """
    timestamp = int(time.mktime(dt.timetuple()) * 1000)
    return f"/Date({timestamp}+0200)/"


def parse_aramis_date(date_str: str) -> date | None:
    """Parses an ARAMIS date string into a Python date object.

    Args:
        date_str (str): The ARAMIS date string (e.g., '/Date(1748728800000+0200)/').

    Returns:
        datetime.date | None: The parsed date or None if the input is invalid.
    """
    if not date_str:
        return None
    match = re.search(r'/Date\((\d+)', date_str)
    if match:
        timestamp = int(match.group(1)) / 1000  # convert ms to seconds
        return datetime.fromtimestamp(timestamp).date()
    return None


def parse_response(data: dict) -> dict:
    """Parses the detailed ARAMIS project response data into a structured dictionary.

    Args:
        data (dict): The response dictionary from ARAMIS API for a single project.

    Returns:
        dict: A dictionary with structured project information including metadata and participants.
    """
    contact_person = []
    scientific_management = []
    implementation_partner = []
    for participant in data.get('Participants', []):        
        base_data = {
            'first_name': participant.get('Prename'),
            'last_name': participant.get('FamilyName'),
            'canton': participant.get('Canton'),
            'city': participant.get('City'),
            'zip_code': participant.get('PostCode'),
            'street': participant.get('Street'),
            'short_view_name': participant.get('ShortViewName'),
            'company_1': participant.get('Company1'),
            'company_2': participant.get('Company2'),
            'company_3': participant.get('Company3'),
            'company_4': participant.get('Company4'),
            'url_1': f"https://{participant['URL1']}" if participant.get('URL1') else None,
            'url_2': f"https://{participant['URL1']}" if participant.get('URL1') else None,
        }

        uids = set()
        for entry in base_data.values():
            if isinstance(entry, str):
                uid_match = re.search(r"(CHE\d{9})", entry.replace('.', '').replace('-', '').upper())
                if uid_match:
                    uids.add(uid_match.groups()[0])
        uids_dict = {f'uid_{i}': uid for i, uid in enumerate(uids, start=1)}

        kind = participant['KindOfParticipation']['Text'].strip()

        if kind == 'Contact person':
            contact_person.append({**base_data, **uids_dict})
        elif kind == 'Implementation Partner':
            implementation_partner.append({**base_data, **uids_dict})
        elif kind == 'Scientific management':
            scientific_management.append({**base_data, **uids_dict})
        else:
            print(kind)

    return {
        'department': 'INNOSUISSE',
        'nabs_policy_domain': data.get('NABSPolicyDomain', {}).get('Text'),
        'project_id': data.get('ProjectId'),
        'project_number': data.get('ProjectNumber'),
        'project_title': data.get('ProjectTitle', {}).get('Text'),
        'project_url': f"https://www.aramis.admin.ch/Beteiligte/?ProjectID={data.get('ProjectId')}",
        'granted_total_costs': data.get('GrantedTotalCosts'),
        'abstract': next((t['Text'] for t in data.get('Textes', []) if t['Category']['Text'] == 'Abstract' and t['LanguageCode'] == 'EN'), None),
        'start_date': parse_aramis_date(data.get('StartDate')),
        'end_date': parse_aramis_date(data.get('EndDate')),
        'contact_person': contact_person,
        'scientific_management': scientific_management,
        'implementation_partner': implementation_partner,
    }


def fetch_project_data(project_id: int) -> dict:
    """Fetches project data from ARAMIS project endpoint.

    Args:
        project_id (int): The project id for the https argument.

    Returns:
        dict: A dictionary with structured project information including metadata and participants.
    """
    if not project_id:
        return None

    project_url = f"https://www.webservice.aramis.admin.ch/Public/Service.svc/project/?aramisId={project_id}&Language=EN"
    get_response = requests.get(project_url, headers={"Accept": "application/json"})

    if get_response.status_code == 200:
        return parse_response(get_response.json())
    else:
        print(f"Request failed with status code {get_response.status_code}")
        return None


def main(initial_skip: int) -> None:
    """Fetches and stores INNOSUISSE project data from ARAMIS API.

    Args:
        initial_skip (int): Starting offset for paginated API calls.

    Returns:
        None
    """
    project_list_url = "https://www.webservice.aramis.admin.ch/Public/Service.svc/projectlist"

    for skip in range(0, 100_000, 100):
        payload = {
            "Language": "EN",
            "Skip": skip,
            "Count": 100,
        }
        post_response = requests.post(project_list_url, json=payload, headers={"Content-Type": "application/json", "Accept": "application/json"})

        if post_response.status_code == 200:
            post_json = post_response.json()

            if not post_json['Projects']:
                break

            data = []
            for project in tqdm(post_json['Projects']):
                department = project['Department']['Text']

                if department == 'INNOSUISSE':
                    row = fetch_project_data(project['Id'])
                    if row is not None:
                        data.append(row)

            pd.DataFrame(data).to_csv(
                RAW_DATA_DIR / 'funding_data' / 'innosuisse_grants.csv',
                index=False,
                mode='a',
                header=not (RAW_DATA_DIR / 'funding_data' / 'innosuisse_grants.csv').exists()
            )

        else:
            print(f"Failed with status code: {post_response.status_code}")
            break
    return


if __name__ == '__main__':
    main()
