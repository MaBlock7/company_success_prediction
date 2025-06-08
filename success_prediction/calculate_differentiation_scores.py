import argparse
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from rag_components.embeddings import EmbeddingHandler
from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.helper_functions import cosine_sim


@dataclass
class Clients:
    embedding_creator: EmbeddingHandler
    db_client: MilvusClient


def calculate_diff_scores(
    clients: Clients,
    collection_name: str,
    output_fields: list[str],
    ehraid: int,
    top_k: int = 5
) -> tuple[dict[str, list[dict]], dict[str, dict[str, float | None]]]:
    """
    Calculates differentiation scores between a target company and its top competitors
    for selected embedding fields.

    Args:
        clients (Clients): The embedding and DB client wrapper.
        collection_name (str): Name of the collection in Milvus to search.
        output_fields (list of str): Fields for which to compute differentiation (e.g., embeddings).
        ehraid (int): Unique identifier for the target company.
        top_k (int, optional): Number of top competitors to consider (default: 5).

    Returns:
        tuple:
            - competitors (dict): For each output field, a list of top competitor documents (as dicts).
            - diff_scores (dict): For 'value_proposition' and 'leadership', a dict mapping field to mean diff score.
              Leadership differentiation is None if not applicable for a field.
        If the target company is not found, returns (None, None).
    """
    additional_fields = [f.replace('vp', 'lp') for f in output_fields if f.startswith('vp')] if 'vp' in output_fields else []
    query_result = clients.db_client.query(
        collection_name=collection_name,
        filter=f"ehraid == {ehraid}",
        output_fields=output_fields + additional_fields
    )
    if not query_result:
        return None, None

    target = query_result[0]
    competitors = {}
    diff_scores = {'value_proposition': {}, 'leadership': {}}

    for field in output_fields:
        query_vec = target[field]
        # If this is a value prop field, also get corresponding leadership vector for competitors
        leadership_field = field.replace('vp', 'lp') if field.startswith('vp') else None

        k_closest = clients.db_client.search(
            collection_name=collection_name,
            data=[query_vec],
            anns_field=field,
            params={"metric_type": "COSINE"},
            limit=top_k,
            output_fields = ['ehraid'] + ([leadership_field] if leadership_field else []),
            filter=f"ehraid != {ehraid}"
        )
        competitors[field] = k_closest[0]

        # Value proposition differentiation
        diff_scores['value_proposition'][field] = np.mean([1 - entry['distance'] for entry in k_closest[0]])

        # Leadership differentiation only if applicable
        if leadership_field:
            lp_target_vec = np.array(target[leadership_field])
            lp_diff = 0
            for entry in k_closest[0]:
                lp_comp_vec = np.array(entry['entity'][leadership_field])
                lp_diff += 1 - cosine_sim(lp_target_vec, lp_comp_vec)
            diff_scores['leadership'][field] = lp_diff / top_k
        else:
            diff_scores['leadership'][field] = None

    return competitors, diff_scores


def main(args: argparse.Namespace) -> None:
    """
    Main pipeline for calculating differentiation scores for all companies.

    Loads data, creates index on embedding fields, and computes competitor
    differentiation metrics for each company in the dataset.

    Args:
        args (argparse.Namespace): Command-line arguments including score_type.

    Side effects:
        Saves differentiation scores to a csv file.
    """
    clients = Clients(
        embedding_creator=EmbeddingHandler(),
        db_client=MilvusClient(uri=str(DATA_DIR / 'database' / 'websites.db'))
    )
    clients.db_client.load_collection(collection_name=args.score_type)

    training_data = pd.read_csv(RAW_DATA_DIR / 'company_sample' / 'sample_2022-04-01_website.csv')  # Load all ehraids and dates from the Milvus db

    index_fields = ['vp', 'vp_w', 'vp_w_red'] if args.score_type == 'strategy_dimensions' else ['doc2vec_embeddings']
    for field in index_fields:
        index_params = [{
            'field_name': field,
            'metric_type': 'COSINE',
            'index_type': 'FLAT',
        }]
        clients.db_client.create_index(
            collection_name=args.score_type,
            index_params=index_params,
        )

    results = []
    for ehraid in tqdm(training_data['ehraid']):
        competitors, diff_scores = calculate_diff_scores(clients, args.score_type, index_fields, ehraid)
        if competitors is None or diff_scores is None:
            continue

        for score_type in diff_scores:  # 'value_proposition' or 'leadership'
            for field, score in diff_scores[score_type].items():
                if score is None:
                    continue
                results.append({
                    "ehraid": ehraid,
                    "competitors": competitors,
                    "score_type": score_type,
                    "field": field,
                    "score": score
                })

    df_scores = pd.DataFrame(results)
    df_scores.to_csv(PROCESSED_DATA_DIR / f'differentiation_scores_{args.score_type}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ContextualEmbeddingPipeline',
        description='Creates chunked, contextual embeddings of the websites',
    )
    parser.add_argument('--score_type', default='strategy_dimensions', choices=['strategy_dimensions', 'doc2vec'])
    args = parser.parse_args()
    main(args)
