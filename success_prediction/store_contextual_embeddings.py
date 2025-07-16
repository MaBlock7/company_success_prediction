import gzip
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from ftlangdetect import detect
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from rag_components.embeddings import EmbeddingHandler
from rag_components.cleanup import MarkdownCleaner

from config import DATA_DIR, RAW_DATA_DIR


@dataclass
class Clients:
    md_cleaner: MarkdownCleaner
    embedding_creator: EmbeddingHandler
    db_client: MilvusClient


def load_raw_file(file_path: Path) -> dict:
    """
    Loads a gzipped JSON file and returns its content as a dictionary.

    Args:
        file_path (Path): Path to the gzipped JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with gzip.open(file_path, 'r') as f:
        return json.load(f)


def store_links(file_path: Path, data: dict) -> None:
    """
    Stores a dictionary as a formatted JSON file.

    Args:
        file_path (Path): Destination file path.
        data (dict): Dictionary to save.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        return json.dump(data, f, ensure_ascii=False, indent=4)


def structure_links(
    ehraid: int,
    links: list[dict],
    email_addresses: set,
    social_media: dict
) -> tuple[dict, dict]:
    """
    Organizes links by identifying emails and social media handles and storing them per company ID.

    Args:
        ehraid (int): Unique company identifier.
        links (List[dict]): List of extracted link dictionaries.
        email_addresses (Dict[int, Dict[str, Set[str]]]): Storage for emails.
        social_media (Dict[int, Dict[str, Set[str]]]): Storage for social links.

    Returns:
        Tuple containing updated email_addresses and social_media.
    """
    for link in links:
        base_domain = link.get('base_domain')
        if '@' in link.get('text'):
            email_addresses[ehraid]['emails'].add(link['text'])
        elif base_domain == "linkedin.com":
            social_media[ehraid]['linkedin'].add(link['href'])
        elif base_domain == "instagram.com":
            social_media[ehraid]['instagram'].add(link['href'])
        elif base_domain == "facebook.com":
            social_media[ehraid]['facebook'].add(link['href'])
        elif base_domain == "tiktok.com":
            social_media[ehraid]['tiktok'].add(link['href'])
        elif base_domain == "youtube.com":
            social_media[ehraid]['youtube'].add(link['href'])
        elif base_domain == "x.com" or base_domain == "twitter.com":
            social_media[ehraid]['x'].add(link['href'])
    return email_addresses, social_media


def run_pipeline(clients: Clients, idx: int, file_path: Path, **kwargs) -> None:
    """
    Processes raw company website data:
    - Cleans and chunks content
    - Embeds it
    - Extracts contact and social media links
    - Stores results in a Milvus database and contact info files

    Args:
        clients (Clients): Wrapper containing the database, embedding, and cleaning tools.
        idx (int): Index of the file being processed.
        file_path (Path): Path to the raw JSON file.
        **kwargs: Additional options, expects 'collection_name'.
    """
    raw_json = load_raw_file(file_path)
    processed_files = []
    email_addresses, social_media = {}, {}

    for ehraid, urls2attributes in tqdm(raw_json.items(), desc=f'Process websites file {idx}'):
        email_addresses[ehraid] = {'emails': set()}
        social_media[ehraid] = {k: set() for k in ['linkedin', 'instagram', 'facebook', 'tiktok', 'youtube', 'x']}

        for url, attributes in urls2attributes.items():
            markdown = attributes.get('markdown')
            if not markdown:
                continue

            date = attributes['date']
            internal_links = [link['href'] for link in attributes['links']['internal']]
            external_links = [link['href'] for link in attributes['links']['external']]

            email_addresses, social_media = structure_links(
                ehraid, attributes['links']['external'], email_addresses, social_media)

            markdown_clean = clients.md_cleaner.clean(markdown, internal_links, external_links)
            markdown_no_links = clients.md_cleaner.remove_nested_brackets(markdown_clean).replace('\n', ' ')
            if len(markdown_no_links) <= 300:
                continue

            # Detect language using the text without bracket content, since it includes
            # English tokens such as INTERNAL_LINKS that might confuse the model
            language = detect(text=markdown_no_links)

            # Split the text into smaller chunks to fit into the model context + normalize whitespace per chunk
            markdown_chunks = clients.embedding_creator.chunk(markdown_no_links)
            markdown_chunks_clean = [
                clients.md_cleaner.normalize_whitespace(doc.page_content)
                for doc in markdown_chunks
            ]

            passage_embeddings = clients.embedding_creator.embed(
                markdown_chunks_clean, prefix='passage:')

            query_embeddings = clients.embedding_creator.embed(
                markdown_chunks_clean, prefix='query:')

            processed_files.extend([
                {
                    'ehraid': int(ehraid),
                    'url': str(url),
                    'date': date,
                    'language': language.get('lang'),
                    'text': md,
                    'text_length': len(md),
                    'embedding_passage': p_emb,
                    'embedding_query': q_emb
                }
                for md, p_emb, q_emb in zip(markdown_chunks_clean, passage_embeddings, query_embeddings)
            ])

        email_addresses[ehraid] = {k: list(v) for k, v in email_addresses[ehraid].items()}
        social_media[ehraid] = {k: list(v) for k, v in social_media[ehraid].items()}

    clients.db_client.insert(collection_name=kwargs.get('collection_name'), data=processed_files)

    store_links(RAW_DATA_DIR / 'company_websites' / 'current' / 'contact_info' / f'emails_{idx}.json', email_addresses)
    store_links(RAW_DATA_DIR / 'company_websites' / 'current' / 'contact_info' / f'social_media_{idx}.json', social_media)


def setup_database(client: MilvusClient, collection_name: str, schema: CollectionSchema, replace: bool) -> None:
    """
    Sets up a Milvus collection for storing embedded documents.

    Args:
        client (MilvusClient): Initialized Milvus client.
        collection_name (str): Name of the collection to use/create.
        schema (CollectionSchema): Schema of the collection.
        replace (bool): Whether to drop and recreate the collection if it exists.
    """
    if replace and client.has_collection(collection_name):
        client.drop_collection(collection_name)

    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema=schema)
    else:
        print(f"{collection_name} already exists!")


def main(args: argparse.Namespace):

    clients = Clients(
        md_cleaner=MarkdownCleaner(),
        embedding_creator=EmbeddingHandler(),
        db_client=MilvusClient(uri=str(DATA_DIR / 'database' / 'websites.db'))
    )

    target_collection_name = f'{args.source_zipped_websites}_websites'
    print(f'Setting up database {target_collection_name}...')

    website_schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="ehraid", dtype=DataType.INT64),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=5),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=64_000),
        FieldSchema(name="text_length", dtype=DataType.INT64),
        FieldSchema(name="embedding_passage", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="embedding_query", dtype=DataType.FLOAT_VECTOR, dim=768),
    ])
    setup_database(clients.db_client, collection_name=target_collection_name, schema=website_schema, replace=args.replace or False)
    print('Database setup successful')

    raw_files = [file for file in Path(RAW_DATA_DIR / 'company_websites' / args.source_zipped_websites).iterdir() if str(file).endswith('.json.gz')]

    for i, file in enumerate(raw_files):
        run_pipeline(clients, idx=i, file_path=file, collection_name=target_collection_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ContextualEmbeddingPipeline',
        description='Creates chunked, contextual embeddings of the websites',
    )
    parser.add_argument(
        '--source_zipped_websites',
        type=str,
        choices=['current', 'wayback'],
        help='Name of the folder where the zipped scraped website content lives.'
    )
    parser.add_argument(
        '--replace',
        action='store_true',
        help='If set, replaces the existing target database.'
    )
    args = parser.parse_args()
    main(args)
