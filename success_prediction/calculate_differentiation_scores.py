import argparse
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from success_prediction.rag_components.embeddings import EmbeddingHandler
from success_prediction.config import DATA_DIR, RAW_DATA_DIR


@dataclass
class Clients:
    embedding_creator: EmbeddingHandler
    db_client: MilvusClient


def cosine_sim(a, b):
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)


def calculate_diff_scores(clients: Clients, collection_name: str, output_fields: list[str], ehraid: int, top_k: int = 5) -> tuple[dict, dict]:
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
            param={'metric_type': 'COSINE', 'params': {'nprobe': 16}},
            limit=top_k + 1,
            output_fields=['ehraid'] + [leadership_field] if leadership_field else ['ehraid'],
            filter=f"ehraid != {ehraid}"
        )
        top_comps = k_closest[0][:top_k]
        competitors[field] = top_comps

        # Value proposition differentiation
        diff_scores['value_proposition'][field] = np.mean([1 - entry['score'] for entry in top_comps])

        # Leadership differentiation only if applicable
        if leadership_field:
            lp_target_vec = np.array(target[leadership_field])
            lp_diff = 0
            for entry in top_comps:
                lp_comp_vec = np.array(entry[leadership_field])
                lp_diff += 1 - cosine_sim(lp_target_vec, lp_comp_vec)
            diff_scores['leadership'][field] = lp_diff / top_k
        else:
            diff_scores['leadership'][field] = None 

    return competitors, diff_scores


def main(args: argparse.Namespace):

    clients = Clients(
        embedding_creator=EmbeddingHandler(),
        db_client=MilvusClient(uri=str(DATA_DIR / 'database' / 'websites.db'))
    )
    clients.db_client.load_collection(collection_name=args.score_type)

    training_data = pd.read_csv(RAW_DATA_DIR / 'company_sample' / 'company_sample_website.csv')  # Load all ehraids and dates from the Milvus db

    index_fields = ['vp', 'vp_w', 'vp_w_red'] if args.score_type == 'strategy_dimensions' else ['doc2vec_embeddings']
    for field in index_fields:
        index_params = [{
            'field_name': field,
            'metric_type': 'COSINE',
            'index_type': 'FLAT',
        }]
        clients.db_client.create_index(
            collection_name='dimension_vectors',
            index_params=index_params,
        )

    for ehraid in tqdm(training_data['ehraid']):
        competitors, diff_scores = calculate_diff_scores(clients, args.score_type, index_fields, ehraid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ContextualEmbeddingPipeline',
        description='Creates chunked, contextual embeddings of the websites',
    )
    parser.add_argument('--score_type', default='strategy_dimensions', choices=['strategy_dimensions', 'doc2vec'])
    args = parser.parse_args()
    main(args)
