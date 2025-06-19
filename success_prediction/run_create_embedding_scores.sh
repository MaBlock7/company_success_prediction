#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# Exit immediately if a command exits with a non-zero status
set -e

echo "Start processing of raw website content..."


# Step 1: Store contextual embeddings for current websites
echo "Running store_contextual_embeddings.py for current websites..."
python store_contextual_embeddings.py --source_zipped_websites current --replace


# Step 2: Store contextual embeddings for wayback (founding) websites
echo "Running store_contextual_embeddings.py for founding websites..."
python store_contextual_embeddings.py --source_zipped_websites wayback --replace


# Step 3: Create doc2vec embeddings for current websites
echo "Running store_doc2vec_embeddings.py for current websites..."
python store_doc2vec_embeddings.py --source_zipped_websites current --replace


# Step 4: Create doc2vec embeddings for wayback (founding) websites
echo "Running store_doc2vec_embeddings.py for founding websites..."
python store_doc2vec_embeddings.py --source_zipped_websites wayback --replace


# Step 5: Create dimension embeddings for current websites
echo "Running store_dimension_embeddings.py for current websites..."
python store_dimension_embeddings.py --source_collection_name current_websites --replace


# Step 6: Create dimension embeddings for wayback (founding) websites
echo "Running store_dimension_embeddings.py for founding websites..."
python store_dimension_embeddings.py --source_collection_name wayback_websites --replace


# Step 7: Calculate differentiation scores based on doc2vec
echo "Running calculate_differentiation_scores.py for doc2vec..."
python calculate_differentiation_scores.py --score_type current_doc2vec
python calculate_differentiation_scores.py --score_type wayback_doc2vec


# Step 8: Calculate differentiation scores based on strategy dimension vectors
echo "Running calculate_differentiation_scores.py for strat2vec..."
python calculate_differentiation_scores.py --score_type current_strat2vec
python calculate_differentiation_scores.py --score_type wayback_strat2vec


echo "Embedding scores stored successfully."