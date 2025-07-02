import gzip
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from ftlangdetect import detect
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from rag_components.embeddings import Doc2VecHandler
from rag_components.cleanup import MarkdownCleaner

from config import DATA_DIR, RAW_DATA_DIR, MODELS_DIR


@dataclass
class Clients:
    md_cleaner: MarkdownCleaner
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


def collect_training_data(clients: Clients, file_path: Path, idx: int) -> dict[str, dict[int, dict]]:
    """
    Collects the training data for the language specific doc2vec models.

    Args:
        clients (Clients): Wrapper containing the database, embedding, and cleaning tools.
        file_path (Path): Path to the raw JSON file.
    """
    raw_json = load_raw_file(file_path)

    training_data = {'de': {}, 'fr': {}, 'it': {}, 'en': {}}
    for ehraid, urls2attributes in tqdm(raw_json.items(), desc=f'Process websites file {idx}'):

        for _, attributes in urls2attributes.items():
            markdown = attributes.get('markdown')
            if not markdown:
                continue

            date = attributes['date']
            internal_links = [link['href'] for link in attributes['links']['internal']]
            external_links = [link['href'] for link in attributes['links']['external']]

            markdown_clean = clients.md_cleaner.clean(markdown, internal_links, external_links)
            markdown_no_links = clients.md_cleaner.remove_nested_brackets(markdown_clean).replace('\n', ' ')
            if len(markdown_no_links) <= 300:
                continue

            # Detect language using the text without bracket content, since it includes
            # English tokens such as INTERNAL_LINKS that might confuse the model
            language = detect(text=markdown_no_links).get('lang')

            if language in training_data:
                if ehraid not in training_data[language]:
                    training_data[language][ehraid] = {'date': date, 'text': [markdown_no_links]}
                else:
                    training_data[language][ehraid]['text'].append(markdown_no_links)
    return training_data


def save_to_collection(clients: Clients, training_data: Path, **kwargs) -> None:

    processed_files = []
    for lang_code, ehraid2data in training_data.items():
        code2lang = {'de': 'german', 'fr': 'french', 'it': 'italian', 'en': 'english'}
        doc2vec = Doc2VecHandler(ehraid2data, language=code2lang[lang_code])
        doc2vec.train_doc2vec()
        doc2vec.save_doc2vec_model(folder_path=kwargs.get('out_folder'))
        output = doc2vec.create_normalized_vectors()

        processed_files.extend([
            {
                'ehraid': int(ehraid),
                'date': date,
                'language': lang_code,
                'doc2vec_embedding': emb.tolist()
            }
            for ehraid, date, emb in output
        ])

    clients.db_client.insert(collection_name=kwargs.get('collection_name'), data=processed_files)


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
        db_client=MilvusClient(uri=str(DATA_DIR / 'database' / 'websites.db'))
    )

    target_collection_name = f'{args.source_zipped_websites}_doc2vec'
    print(f'Setting up database {target_collection_name}...')

    website_schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="ehraid", dtype=DataType.INT64),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=5),
        FieldSchema(name="doc2vec_embedding", dtype=DataType.FLOAT_VECTOR, dim=300),
    ])
    setup_database(clients.db_client, collection_name=target_collection_name, schema=website_schema, replace=args.replace or False)
    print('Database setup successful')

    raw_files = [file for file in Path(RAW_DATA_DIR / 'company_websites' / args.source_zipped_websites).iterdir() if str(file).endswith('.json.gz')]

    all_data = {'de': {}, 'fr': {}, 'it': {}, 'en': {}}
    for i, file in enumerate(raw_files):
        data = collect_training_data(clients, file_path=file, idx=i)
        for lang in all_data.keys():
            all_data[lang].update(data[lang])

    save_to_collection(clients, training_data=all_data, collection_name=target_collection_name, out_folder=MODELS_DIR / 'doc2vec' / args.source_zipped_websites)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Doc2VecPipeline',
        description='Creates doc2vec embeddings of the full websites',
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
