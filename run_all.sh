#!/usr/bin/env bash
set -euo pipefail

echo "=== Running data generation ==="
python pipeline_src/data_generator.py

echo "=== Running feature engineering ==="
python pipeline_src/feature_engineering.py

echo "=== Running rule scoring ==="
python pipeline_src/rule_scoring.py

echo "=== Running training and evaluation ==="
python pipeline_src/train_evaluate.py

echo "=== Running additional analyses ==="
python pipeline_src/additional_analyses.py | tee outputs/additional_analyses_output.txt

echo "=== All pipeline steps completed ==="