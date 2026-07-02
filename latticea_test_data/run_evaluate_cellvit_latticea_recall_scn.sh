#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINMENT_RADIUS="${1:-0}"

python3 "${SCRIPT_DIR}/../cellvit_inspection/evaluate_lizard_latticea_recall.py" \
  --pred-dir "/mnt/j/HandE/results/latticea_test_data/cellvitpp_lattice-a" \
  --gt-csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915" \
  --pred-glob "*_cells.json" \
  --output-dir "/mnt/j/HandE/results/latticea_test_data/cellvitpp_lizard/recall_results_scn" \
  --containment-radius "${CONTAINMENT_RADIUS}"
