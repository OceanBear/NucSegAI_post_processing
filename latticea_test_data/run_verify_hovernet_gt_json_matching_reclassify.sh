#!/usr/bin/env bash
set -euo pipefail

CONTAINMENT_RADIUS="${1:-4}"

python3 latticea_test_data/verify_hovernet_gt_json_matching.py \
  --json-dir "/mnt/j/HandE/results/latticea_test_data/pred_scn/json_reclass" \
  --gt-csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915" \
  --out-dir "/mnt/j/HandE/results/latticea_test_data/pred_scn/json_reclass_matched" \
  --containment-radius "${CONTAINMENT_RADIUS}"