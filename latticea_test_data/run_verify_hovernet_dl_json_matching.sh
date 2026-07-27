#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/verify_hovernet_gt_json_matching.py" \
  --json-dir "/mnt/j/HandE/results/latticea_test_data/pred_scn/json" \
  --gt-csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915_filtered_t0.5_f0.5_r4" \
  --out-dir "/mnt/j/HandE/results/latticea_test_data/matched_dl_hovernet_scn"
