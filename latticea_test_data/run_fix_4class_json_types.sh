#!/usr/bin/env bash
set -euo pipefail

python3 latticea_test_data/fix_4class_json_types.py \
  --json-dir "/mnt/j/HandE/results/latticea_test_data/converted_4class_json" \
  --out-json-dir "/mnt/j/HandE/results/latticea_test_data/converted_4class_json_fixed"
