import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from pymilvus.client.types import ExtraList
from pymilvus.exceptions import MilvusException
from rag_components.embeddings import EmbeddingHandler
from rag_components.config import DIM2QUERY
from utils.helper_functions import cosine_sim, angular_distance_from_cosine
from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR


@dataclass
class Clients:
    embedding_creator: EmbeddingHandler
    db_client: MilvusClient


def ensemble_top_passages(
    company_data: list[dict],
    query_embeddings: list[np.ndarray],
    top_k_per_query: int = 15,
    final_top_k: int = 15
) -> list[dict]:
    """
    Returns the passages that appear most frequently in the top-k across a query ensemble.

    Args:
        company_data (list of dict): List of passages, each dict must have 'id', 'embedding_passage', etc.
        query_embeddings (list of np.ndarray): List of embedded queries for the dimension.
        top_k_per_query (int): Number of top results to keep for each query.
        final_top_k (int): Final number of consensus passages to return.

    Returns:
        list of dict: Top passages by consensus frequency.
    """
    passage_scores = defaultdict(list)

    for query_vec in query_embeddings:
        query_vec = np.array(query_vec[0])
        scored_entries = []

        for entry in company_data:
            score = np.dot(query_vec, entry['embedding_passage'])  # Use passage embedding for RAG
            scored_entries.append({**entry, 'score': score})

        top_k = sorted(scored_entries, key=lambda x: x['score'], reverse=True)[:top_k_per_query]

        for passage in top_k:
            passage_scores[passage['id']].append((passage['score'], passage))

    frequency_counter = Counter({pid: len(scores) for pid, scores in passage_scores.items()})
    # Sort by frequency (how often in top 15), then by best score (descending)
    sorted_passages = sorted(
        passage_scores.items(),
        key=lambda x: (frequency_counter[x[0]], max(s[0] for s in x[1])),
        reverse=True
    )

    final_passages = [x[1][0][1] for x in sorted_passages[:final_top_k]]
    return final_passages


def ensemble_rerank(
    clients: Clients,
    top_n_entries: list[dict],
    query_texts: list[str]
) -> list[dict]:
    """
    Reranks entries using a cross-encoder, assigning an attention score based on relevance to queries.

    Args:
        clients (Clients): Object with embedding_creator and db_client.
        top_n_entries (list of dict): Passages to rerank.
        query_texts (list of str): List of queries.

    Returns:
        list of dict: Entries with attention_score, filtered to z-score >= 0 and sorted descending.
    """
    for entry in top_n_entries:
        pairs = [(query, entry['text']) for query in query_texts]
        relevancy_score = clients.embedding_creator.calculate_relevancy_scores(sentence_pairs=pairs).median()
        entry.update({'attention_score': relevancy_score})

    sorted_entries = sorted(
        top_n_entries,
        key=lambda entry: (float(entry['attention_score'])),
        reverse=True
    )
    scores = np.array([entry['attention_score'] for entry in sorted_entries])
    std = np.std(scores)
    z_scores = (scores - np.mean(scores)) / std if std != 0 else np.zeros_like(scores)
    return [entry for z_score, entry in zip(z_scores, sorted_entries) if z_score >= -1e-5]


def get_dimension_vec(
    clients: Clients,
    dimension: str,
    company_data: ExtraList,
    dim2embedding: dict,
    dim2query: dict
) -> tuple[list[dict], torch.Tensor]:
    """
    For a company and a strategy dimension, finds most relevant passages and computes an aggregate vector.

    Args:
        clients (Clients): Object holding embedding creator and DB client.
        dimension (str): The strategy dimension to query.
        company_data (ExtraList): List of company passages with embeddings.
        dim2embedding (dict): Dictionary mapping dimensions to query embeddings.
        dim2query (dict): Dictionary mapping dimensions to queries.

    Returns:
        Tuple:
            - most_relevant (list of dict): Most relevant passages for the dimension.
            - dim_vec (torch.Tensor): Aggregated vector for the dimension.
    """
    # Get top 15 based on cosine / IP similarity
    top_15 = ensemble_top_passages(
        company_data=company_data,
        query_embeddings=dim2embedding[dimension]
    )

    # Get the most relevant by reranking them via cross encoder
    most_relevant = ensemble_rerank(
        clients,
        top_n_entries=top_15,
        query_texts=dim2query[dimension]
    )

    if not most_relevant:
        return [], torch.zeros(768)

    # combine the remaining into one vector by using the quasi attention score from the ensemble rerank
    # Use the embedding with 'query:' prefix since it is better for similarity comparisons
    dim_vec = clients.embedding_creator.waggregate_embeddings(
        [torch.tensor(entry['embedding_query']) for entry in most_relevant],
        [entry['attention_score'] for entry in most_relevant]
    )
    return most_relevant, dim_vec


def safe_query(client, collection_name, ehraid, output_fields):
    try:
        # First attempt: no limit
        return client.query(
            collection_name=collection_name,
            filter=f"ehraid == {ehraid}",
            output_fields=output_fields
        )
    except MilvusException as e:
        if "query results exceed the limit size" in str(e):
            print(f"[Retry] Query too large for ehraid={ehraid}. Retrying with limit=100.")
            try:
                return client.query(
                    collection_name=collection_name,
                    filter=f"ehraid == {ehraid}",
                    output_fields=output_fields,
                    limit=100
                )
            except MilvusException as e2:
                print(f"[Fail] Even limited query failed for ehraid={ehraid}: {e2}")
                return []
        else:
            print(f"[Error] Milvus query failed for ehraid={ehraid}: {e}")
            return []


def safe_batch_insert(client: Clients, collection_name: str, data, batch_size=500, max_retries=3, sleep_seconds=5):
    """
    Inserts data into Milvus in safe batches to avoid exceeding gRPC message size.

    Args:
        client (MilvusClient): Initialized client.
        collection_name (str): Name of the collection.
        data (list of dict): Data to insert.
        batch_size (int): Number of rows per batch.
        max_retries (int): Number of retry attempts for each batch.
        sleep_seconds (int): Wait time between retries.
    """
    total_batches = ceil(len(data) / batch_size)
    print(f"Inserting data in {total_batches} batches...")

    for i in range(total_batches):
        batch = data[i * batch_size: (i + 1) * batch_size]
        success = False
        attempt = 0
        while not success and attempt < max_retries:
            try:
                client.insert(collection_name=collection_name, data=batch)
                success = True
                print(f"[SUCCESS] Batch {i + 1}/{total_batches} inserted successfully.")
            except Exception as e:
                attempt += 1
                print(f"[RETRY] Failed to insert batch {i + 1}/{total_batches} (attempt {attempt}): {e}")
                if attempt < max_retries:
                    time.sleep(sleep_seconds)
        if not success:
            print(f"[FAIL] Skipping batch {i + 1}/{total_batches} after {max_retries} failed attempts.")


def create_dimension_vecs(
    clients: Clients,
    ehraids: list[int],
    dim2query: dict,
    dim2embedding: dict,
    source_collection_name: str,
    target_collection_name: str,
    **kwargs
) -> None:
    """
    Computes, aggregates, and stores dimension vectors for all companies.

    Args:
        clients (Clients): Object holding embedding creator and DB client.
        ehraids (list of int): List of company IDs.
        dim2query (dict): Mapping of dimension to queries.
        dim2embedding (dict): Mapping of dimension to embedded queries.
        source_collection_name (str): DB collection to read the embeddings from.
        target_collection_name (str): DB collection to store the dimension vectors in.
        **kwargs: Optional keyword arguments.

    Side effects:
        Saves results to disk and inserts embeddings into the database.
    """
    print(f"Reading company data from {source_collection_name}...")

    vec_results = []
    dates = []
    completed_ehraids = []
    for ehraid in tqdm(ehraids, desc='Aggregating embeddings'):
        try:
            company_data = safe_query(
                client=clients.db_client,
                collection_name=source_collection_name,
                ehraid=ehraid,
                output_fields=["ehraid", "date", "text", "embedding_passage", "embedding_query"]  # adjust as needed
            )

            if not company_data:
                continue

            dim_vectors = {}
            for dim in dim2query.keys():
                dim_vectors[dim] = {}
                most_relevant, dim_vec = get_dimension_vec(clients, dim, company_data, dim2embedding, dim2query)
                dim_vectors[dim]['n_vecs'] = len(most_relevant)
                dim_vectors[dim]['vectors'] = dim_vec
            dates.append(company_data[0]['date'])
            completed_ehraids.append(ehraid)
            vec_results.append({ehraid: dim_vectors})
        except Exception as e:
            print(e)

    sdg_references = pd.read_excel(RAW_DATA_DIR / 'synthetic_examples' / 'synthetic_corporate_responsibility.xlsx')
    sdg_embeddings = [clients.embedding_creator.embed([q], prefix='query:') for q in sdg_references['content']]
    sdg_vec = clients.embedding_creator.waggregate_embeddings(
        [torch.tensor(embed[0]) for embed in sdg_embeddings],
        [1 / len(sdg_embeddings) for _ in sdg_embeddings]  # Apply uniform weights for all embeddings
    )

    vp_n_vecs = [values['Value Proposition & Innovation']['n_vecs'] for entry in vec_results for values in entry.values()]
    pr_n_vecs = [values['Purpose & Responsibility']['n_vecs'] for entry in vec_results for values in entry.values()]
    lp_n_vecs = [values['Leadership & People']['n_vecs'] for entry in vec_results for values in entry.values()]

    vp_embeddings = torch.stack([values['Value Proposition & Innovation']['vectors'] for entry in vec_results for values in entry.values()])
    pr_embeddings = torch.stack([values['Purpose & Responsibility']['vectors'] for entry in vec_results for values in entry.values()])
    lp_embeddings = torch.stack([values['Leadership & People']['vectors'] for entry in vec_results for values in entry.values()])

    # whitening without dimensionality reduction
    vp_whitened, _ = clients.embedding_creator.whitening_k(embeddings=vp_embeddings)
    lp_whitened, _ = clients.embedding_creator.whitening_k(embeddings=lp_embeddings)

    # For pr, whiten with reference
    pr_whitened, sdg_vec_whitened = clients.embedding_creator.whitening_k(embeddings=pr_embeddings, reference=sdg_vec)

    # whitening with dimensionality reduction
    vp_whitened_red, _ = clients.embedding_creator.whitening_k(embeddings=vp_embeddings, k=300)
    lp_whitened_red, _ = clients.embedding_creator.whitening_k(embeddings=lp_embeddings, k=300)
    pr_whitened_red, sdg_vec_whitened_red = clients.embedding_creator.whitening_k(embeddings=pr_embeddings, k=300, reference=sdg_vec)

    # Calculate the responsibility scores right here
    pr_sim = 1 - angular_distance_from_cosine(cosine_sim(pr_embeddings.cpu().numpy(), sdg_vec.numpy()))
    pr_w_sim = 1 - angular_distance_from_cosine(cosine_sim(pr_whitened.cpu().numpy(), sdg_vec_whitened.cpu().numpy()))
    pr_w_red_sim = 1 - angular_distance_from_cosine(cosine_sim(pr_whitened_red.cpu().numpy(), sdg_vec_whitened_red.cpu().numpy()))

    sim_df = pd.DataFrame({
        'ehraid': [int(ehraid) for ehraid in completed_ehraids],
        'date': [str(date) for date in dates],
        'pr_sdg_similarity': pr_sim,
        'pr_w_sdg_similarity': pr_w_sim,
        'pr_w_red_sdg_similarity': pr_w_red_sim,
    })
    sim_df.to_csv(PROCESSED_DATA_DIR / f'{source_collection_name}_responsibility_scores.csv', index=False)
    print("Saved similarity scores")

    results = []

    for i in range(len(completed_ehraids)):
        results.append({
            'ehraid': int(completed_ehraids[i]),
            'date': str(dates[i]),
            'n_vecs_vp': vp_n_vecs[i],
            'n_vecs_lp': lp_n_vecs[i],
            'n_vecs_pr': pr_n_vecs[i],
            'vp': vp_embeddings[i],
            'lp': lp_embeddings[i],
            'pr': pr_embeddings[i],
            'vp_w': vp_whitened[i],
            'lp_w': lp_whitened[i],
            'pr_w': pr_whitened[i],
            'vp_w_red': vp_whitened_red[i],
            'lp_w_red': lp_whitened_red[i],
            'pr_w_red': pr_whitened_red[i],
        })

    safe_batch_insert(
        client=clients.db_client,
        collection_name=target_collection_name,
        batch_size=10,
        data=results
    )
    print(f"Saved embeddings to {target_collection_name} database")


def setup_database(
    client: MilvusClient,
    collection_name: str,
    schema: CollectionSchema,
    replace: bool
) -> None:
    """
    Sets up a Milvus collection for storing embedded documents.

    Args:
        client (MilvusClient): Initialized Milvus client.
        collection_name (str): Name of the collection to use/create.
        schema (CollectionSchema): Schema of the collection.
        replace (bool): Whether to drop and recreate the collection if it exists.

    Side effects:
        Creates or replaces a collection in the Milvus database.
    """
    if replace and client.has_collection(collection_name):
        client.drop_collection(collection_name)

    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema=schema)
    else:
        print(f"{collection_name} already exists!")


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for running the contextual embedding pipeline.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Side effects:
        Loads data, creates database schema, computes embeddings, stores results.
    """
    clients = Clients(
        embedding_creator=EmbeddingHandler(),
        db_client=MilvusClient(uri=str(DATA_DIR / 'database' / 'websites.db'))
    )

    target_collection_name = f'{str(args.source_collection_name).split('_')[0]}_strat2vec'
    print(f'Setting up database {target_collection_name}...')

    # Convert to list
    dim2embedding = {dimension: [clients.embedding_creator.embed([q], prefix='query:') for q in queries] for dimension, queries in DIM2QUERY.items()}
    print('Successfully embedded dimension queries')
    clients.db_client.load_collection(collection_name=args.source_collection_name)

    website_schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="ehraid", dtype=DataType.INT64),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="n_vecs_vp", dtype=DataType.INT8),
        FieldSchema(name="n_vecs_lp", dtype=DataType.INT8),
        FieldSchema(name="n_vecs_pr", dtype=DataType.INT8),
        FieldSchema(name="vp", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="lp", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="pr", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="vp_w", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="lp_w", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="pr_w", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="vp_w_red", dtype=DataType.FLOAT_VECTOR, dim=300),
        FieldSchema(name="lp_w_red", dtype=DataType.FLOAT_VECTOR, dim=300),
        FieldSchema(name="pr_w_red", dtype=DataType.FLOAT_VECTOR, dim=300),
    ])
    setup_database(clients.db_client, collection_name=target_collection_name, schema=website_schema, replace=args.replace or False)
    print('Database setup successful')

    training_data = pd.read_csv(RAW_DATA_DIR / 'company_sample' / args.source_file)  # Load all ehraids of the sample
    print(f'Company sample size: {training_data['ehraid'].nunique()}')

    create_dimension_vecs(
        clients,
        ehraids=training_data['ehraid'].tolist(),
        dim2query=DIM2QUERY,
        dim2embedding=dim2embedding,
        source_collection_name=args.source_collection_name,
        target_collection_name=target_collection_name
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ContextualEmbeddingPipeline',
        description='Creates chunked, contextual embeddings of the websites',
    )
    parser.add_argument(
        '--source_file',
        type=str,
        default='until_2020/2020_sample_base_data.csv',
        help='File name of the CSV company sample to create the dimension embeddings.'
    )
    parser.add_argument(
        '--source_collection_name',
        type=str,
        choices=['current_websites', 'wayback_websites'],
        help='Name of the collection to extract the contextualized embeddings from.'
    )
    parser.add_argument(
        '--replace',
        action='store_true',
        help='If set, replaces the existing target database.'
    )
    args = parser.parse_args()
    main(args)
