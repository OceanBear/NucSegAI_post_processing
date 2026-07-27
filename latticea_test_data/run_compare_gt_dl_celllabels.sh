#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/compare_gt_dl_celllabels.py" \
  --gt-csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915" \
  --dl-csv-dir "/mnt/j/HandE/results/latticea_test_data/dl_celllabels_mpp025_from04915" \
  --out-dir "/mnt/j/HandE/results/latticea_test_data/matched_gt_vs_dl" \
  --max-distance 3
