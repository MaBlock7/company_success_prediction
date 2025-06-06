import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from pymilvus.client.types import ExtraList
from success_prediction.rag_components.embeddings import EmbeddingHandler
from success_prediction.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR


@dataclass
class Clients:
    embedding_creator: EmbeddingHandler
    db_client: MilvusClient


DIM2QUERY = {
    "Value Proposition & Innovation": [
        "What solutions, services, or products does the company provide to customers?",
        "What products and services does the company advertise on its website?",
        "Which innovative features or technologies are highlighted in the company's offerings?",
        "What benefits or outcomes does the company promise or deliver through its solutions, services, products, or platforms?",
        "How does the company differentiate its products or services from competitors, and what specific customer needs are addressed?"
    ],
    "Purpose & Responsibility": [
        "What is the stated mission, purpose, or long-term vision of the company?",
        "Which ethical, social, or environmental commitments does the company emphasize?",
        "Does the company value sustainability, diversity, inclusion, or in general ESG-related goals?",
        "What values or principles guide the company's operations and decisions?",
        "Does the company participate in any charitable initiatives, community outreach, or global impact programs?"
    ],
    "Leadership & People": [
        "Who are the founders or key leaders of the company, and what roles do they hold?",
        "What are the professional backgrounds or credentials of the company's executive team?",
        "Who makes up the leadership team, and how is the company structured in terms of people and roles?",
        "What experience or expertise does the management bring to the company?",
        "Are there biographies or personal stories of team members or executives available on the website?"
    ]
}


def cosine_sim(a, b):
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)


def ensemble_top_passages(company_data, query_embeddings, top_k_per_query=15, final_top_k=15):
    """
    Returns the passages that appear most frequently in the top-k across query ensemble.

    Parameters:
        company_data: List of dicts with fields 'id', 'embedding_passage', 'embedding_query', 'text', 'url'
        query_embeddings: List of embedded queries for the dimension
        top_k_per_query: Number of top results to keep for each query
        final_top_k: Final number of consensus passages to return

    Returns:
        List of dicts: top passages by consensus frequency
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


def ensemble_rerank(clients: Clients, top_n_entries, query_texts):
    """
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
    z_scores = (scores - np.mean(scores)) / np.std(scores)
    return [entry for z_score, entry in zip(z_scores, sorted_entries) if z_score >= 0]

def get_dimension_vec(clients: Clients, dimension: str, company_data: ExtraList, dim2embedding: dict, dim2query: dict):
    """
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
    
    # combine the remaining into one vector by using the quasi attention score from the ensemble rerank
    # Use the embedding with 'query:' prefix since it is better for similarity comparisons
    dim_vec = clients.embedding_creator.waggregate_embeddings([torch.tensor(entry['embedding_query']) for entry in most_relevant], [entry['attention_score'] for entry in most_relevant])
    return most_relevant, dim_vec


def create_dimension_vecs(clients: Clients, ehraids: list, dim2query: dict, dim2embedding: dict, **kwargs) -> None:
    """
    """
    vec_results = []
    dates = []
    for ehraid in tqdm(ehraids):

        company_data = clients.db_client.query(collection_name='current_websites', filter=f"ehraid == {ehraid}")
        if not company_data:
            continue

        dim_vectors = {}
        for dim in dim2query.keys():
            dim_vectors[dim] = {}
            most_relevant, dim_vec = get_dimension_vec(clients, dim, company_data, dim2embedding, dim2query)
            dim_vectors[dim]['entries'] = most_relevant
            dim_vectors[dim]['vectors'] = dim_vec
        dates.append{}
        vec_results.append({ehraid: dim_vectors})

    sdg_references = pd.read_excel(RAW_DATA_DIR / 'synthetic_examples' / 'synthetic_corporate_responsibility.xlsx')
    sdg_embeddings = clients.embedding_creator.embed([sdg_references['content']], prefix='query:')
    sdg_vec = clients.embedding_creator.waggregate_embeddings(
        [torch.tensor(entry['embedding_query']) for entry in sdg_embeddings],
        [1/len(sdg_embeddings) for _ in sdg_embeddings]  # Apply uniform weights for all embeddings
    )

    vp_embeddings = torch.stack([values['Value Proposition & Innovation']['vectors'] for entry in vec_results for values in entry.values()])
    pr_embeddings = torch.stack([values['Purpose & Responsibility']['vectors'] for entry in vec_results for values in entry.values()])
    lp_embeddings = torch.stack([values['Leadership & People']['vectors'] for entry in vec_results for values in entry.values()])

    # PR embeddings: add sdg reference vector
    pr_plus_ref = torch.cat([pr_embeddings, sdg_vec.unsqueeze(0)], dim=0)

    # whitening without dimensionality reduction
    vp_whitened = clients.embedding_creator.whitening_k(embeddings=vp_embeddings)
    lp_whitened = clients.embedding_creator.whitening_k(embeddings=lp_embeddings)

    # For pr, whiten with reference, then split
    pr_plus_ref_whitened = clients.embedding_creator.whitening_k(embeddings=pr_plus_ref)
    pr_whitened = pr_plus_ref_whitened[:-1]
    sdg_vec_whitened = pr_plus_ref_whitened[-1]

    # whitening with dimensionality reduction
    vp_whitened_red = clients.embedding_creator.whitening_k(embeddings=vp_embeddings, k=300)
    lp_whitened_red = clients.embedding_creator.whitening_k(embeddings=lp_embeddings, k=300)
    pr_plus_ref_whitened_red = clients.embedding_creator.whitening_k(embeddings=pr_plus_ref, k=300)
    pr_whitened_red = pr_plus_ref_whitened_red[:-1]
    sdg_vec_whitened_red = pr_plus_ref_whitened_red[-1]

    # Calculate the responsibility scores right here
    pr_sim = cosine_sim(pr_embeddings.numpy(), sdg_vec.numpy())
    pr_w_sim = cosine_sim(pr_whitened.numpy(), sdg_vec_whitened.numpy())
    pr_w_red_sim = cosine_sim(pr_whitened_red.numpy(), sdg_vec_whitened_red.numpy())

    sim_df = pd.DataFrame({
        'ehraid': [int(ehraid) for ehraid in ehraids],
        'date': [str(date) for date in dates],
        'pr_sdg_similarity': pr_sim,
        'pr_w_sdg_similarity': pr_w_sim,
        'pr_w_red_sdg_similarity': pr_w_red_sim,
    })
    sim_df.to_csv(PROCESSED_DATA_DIR / 'responsibility_scores.csv', index=False)
    print(f"Saved similarity scores")

    results = [
        {
            'ehraid': int(ehraid),
            'date': date,
            'vp': vp,
            'lp': lp,
            'pr': pr,
            'vp_w': vp_w,
            'lp_w': lp_w,
            'pr_w': pr_w,
            'vp_w_red': vp_w_red,
            'lp_w_red': lp_w_red,
            'pr_w_red': pr_w_red,
        }
        for ehraid, date, vp, lp, pr, vp_w, lp_w, pr_w, vp_w_red, lp_w_red, pr_w_red in zip(
            ehraids, dates, 
            vp_embeddings, pr_embeddings, lp_embeddings,
            vp_whitened, pr_whitened, lp_whitened,
            vp_whitened_red, pr_whitened_red, lp_whitened_red
        )
    ]

    clients.db_client.insert(collection_name=kwargs.get('collection_name'), data=results)
    print(f"Saved embeddings to database")


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
        embedding_creator=EmbeddingHandler(),
        db_client=MilvusClient(uri=str(DATA_DIR / 'database' / 'websites.db'))
    )
    
    # Convert to list
    dim2embedding = {dimension: [clients.embedding_creator.embed([q], prefix='query:') for q in queries] for dimension, queries in DIM2QUERY.items()}
    clients.db_client.load_collection(collection_name='current_websites')


    website_schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="ehraid", dtype=DataType.INT64),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10),
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
    setup_database(clients.db_client, collection_name=args.collection_name, schema=website_schema, replace=args.replace or False)

    training_data = pd.read_csv(RAW_DATA_DIR / 'company_sample' / 'company_sample_website.csv')  # Load all ehraids and dates from the Milvus db

    create_dimension_vecs(
        clients,
        ehraids=training_data['ehraid'].tolist(),
        dim2query=DIM2QUERY,
        dim2embedding=dim2embedding,
        collection_name=args.collection_name
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ContextualEmbeddingPipeline',
        description='Creates chunked, contextual embeddings of the websites',
    )
    parser.add_argument('--collection_name', default='strategy_dimensions')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    main(args)
    