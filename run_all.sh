#!/usr/bin/env bash
set -euo pipefail

echo "=== Running data generation ==="
python3 pipeline_src/data_generator.py

echo "=== Running feature engineering ==="
python3 pipeline_src/feature_engineering.py

echo "=== Running rule scoring ==="
python3 pipeline_src/rule_scoring.py

echo "=== Running training and evaluation ==="
python3 pipeline_src/train_evaluate.py

echo "=== Running additional analyses ==="
python3 pipeline_src/additional_analyses.py | tee outputs/additional_analyses_output.txt

echo "=== All pipeline steps completed ==="