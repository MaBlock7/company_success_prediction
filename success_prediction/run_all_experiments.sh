#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# Exit immediately if a command exits with a non-zero status
set -e

echo "Start model training pipeline..."


# Step 1: Store contextual embeddings for current websites


echo "Model training and evaluation completed successfully."