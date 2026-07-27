#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_filter_csv_with_ilastik_mask.sh [pixel_threshold] [fraction_threshold] [radius] [channel] [h5_layout]
# Example:
#   ./run_filter_csv_with_ilastik_mask.sh 0.5 0.5 4 0 yxc

PIXEL_THRESHOLD="${1:-0.5}"
FRACTION_THRESHOLD="${2:-0.5}"
RADIUS="${3:-4}"
CHANNEL="${4:-0}"
H5_LAYOUT="${5:-yxc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/filter_csv_with_ilastik_mask.py" \
  --csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915" \
  --h5-dir "/mnt/j/HandE/results/latticea_test_data/ilastik_h5/Artifacts" \
  --out-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915_filtered_t${PIXEL_THRESHOLD}_f${FRACTION_THRESHOLD}_r${RADIUS}" \
  --dataset-path "/exported_data" \
  --h5-layout "${H5_LAYOUT}" \
  --channel "${CHANNEL}" \
  --pixel-threshold "${PIXEL_THRESHOLD}" \
  --fraction-threshold "${FRACTION_THRESHOLD}" \
  --radius "${RADIUS}"
