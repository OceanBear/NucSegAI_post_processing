#!/usr/bin/env bash
set -euo pipefail

MATCH_STRATEGY="${1:-loose_tlo}"

python3 /home/qxiong/projects/nucsegai_post_processing/latticea_test_data/verify_nucsegai_gt_json_matching.py \
  --json-dir "/mnt/j/HandE/results/latticea_test_data/pred_scn/json" \
  --gt-csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915" \
  --out-dir "/mnt/j/HandE/results/latticea_test_data/matched_gt_nucsegai_scn_${MATCH_STRATEGY}" \
  --match-strategy "${MATCH_STRATEGY}"
