#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# Exit immediately if a command exits with a non-zero status
set -e

echo "Start experiments..."

# Step 1: Run experiment A
echo "Running experiment_a.py to evaluate models on baseline administrative features..."
python experiment_a.py

# Step 2: Run experiment B
echo "Running experiment_b.py to estimate the coefficients using logistic regression..."
python experiment_b.py

# Step 3: Run experiment C
echo "Running experiment_c.py to estimate the predictive value of website features and strategy scores..."
python experiment_c.py

echo "Experiments successfully run."