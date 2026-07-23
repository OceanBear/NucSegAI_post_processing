#!/usr/bin/env bash
# Evaluate CellViT++ Lizard predictions against old 4-class GT JSON.
#
# GT JSONs use the HoVer-Net-style {"nuc": {...}} layout (types 0-3).
# Predictions use CellViT++ {"cells": [...]} layout (Lizard types 0-5).
#
# Adjust --gt-dir / --pred-dir if your GT files are copied into the CellViT++
# output folder; basename matching strips "_cells" from prediction filenames.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/evaluate_lizard_to_old4.py" \
  --gt-dir "/mnt/j/HandE/results/latticea_test_data/converted_4class_json_fixed" \
  --pred-dir "/mnt/j/HandE/new_training_set/cellvitpp_lizard_3in1/cellvitpp_lizard_lattice-a" \
  --gt-glob "*.json" \
  --pred-glob "*_cells.json" \
  --iou-threshold 0.2 \
  --output-dir "${SCRIPT_DIR}/eval_results_lattice-a"