#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATCH_STRATEGY="${1:-loose_tlo}"
CONTAINMENT_RADIUS="${2:-4}"

python3 "${SCRIPT_DIR}/verify_hovernet_gt_json_matching.py" \
  --json-dir "/mnt/j/HandE/results/latticea_test_data/pred_scn/json" \
  --gt-csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915" \
  --out-dir "/mnt/j/HandE/results/latticea_test_data/matched_gt_hovernet_scn_${MATCH_STRATEGY}" \
  --match-strategy "${MATCH_STRATEGY}" \
  --containment-radius "${CONTAINMENT_RADIUS}"
