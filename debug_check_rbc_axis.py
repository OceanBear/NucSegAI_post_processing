#!/usr/bin/env python
"""
Debug script to check whether ilastik probability maps are indexed as
  (y, x, c)  or  (x, y, c)
for a given tile, by comparing values sampled at JSON centroids under both
indexing conventions.

This script does NOT modify any existing files.

Usage example (adjust paths and stem as needed):

  python debug_check_rbc_axis.py \
    --h5-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_post_processing/RBC/Probabilities" \
    --json-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_wsi/json" \
    --stem "JN_TS_001_bg_tile_12883_7423" \
    --channel 0 \
    --num-samples 50
"""

import argparse
import json
import os
from typing import Dict, Tuple

import h5py  # type: ignore
import numpy as np  # type: ignore


def load_nuclei(json_path: str) -> Dict[str, dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    nuc = data.get("nuc", {})
    if not isinstance(nuc, dict):
        raise ValueError(f"'nuc' dict missing or not a dict in {json_path}")
    return nuc


def sample_centroid_values(
    arr: np.ndarray,
    nuclei: Dict[str, dict],
    channel: int,
    num_samples: int,
) -> None:
    """
    For up to num_samples nuclei, print:
      - centroid (y, x)
      - value assuming indexing arr[y, x, c]
      - value assuming indexing arr[x, y, c]
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (H,W,C), got shape {arr.shape}")
    H, W, C = arr.shape
    if channel < 0 or channel >= C:
        raise ValueError(f"channel={channel} out of bounds for shape {arr.shape}")

    ids = list(nuclei.keys())
    if not ids:
        print("No nuclei found in JSON.")
        return

    # Spread samples across the ID list
    step = max(1, len(ids) // max(1, num_samples))
    chosen_ids = ids[::step][:num_samples]

    vals_yx = []
    vals_xy = []

    print(f"Sampling up to {len(chosen_ids)} nuclei (channel={channel})")
    for nid in chosen_ids:
        info = nuclei[nid]
        cy, cx = info.get("centroid", [None, None])
        if cy is None or cx is None:
            continue

        iy = int(round(cy))
        ix = int(round(cx))
        val_yx = np.nan
        val_xy = np.nan

        if 0 <= iy < H and 0 <= ix < W:
            val_yx = float(arr[iy, ix, channel])
        if 0 <= ix < H and 0 <= iy < W:
            val_xy = float(arr[ix, iy, channel])

        vals_yx.append(val_yx)
        vals_xy.append(val_xy)

        print(
            f"nuc {nid:>6} centroid (y={cy:.2f}, x={cx:.2f}) "
            f"-> (iy={iy}, ix={ix}) "
            f"val[y,x]={val_yx:.4f}  val[x,y]={val_xy:.4f}"
        )

    # Aggregate statistics ignoring NaNs
    arr_yx = np.array(vals_yx, dtype=float)
    arr_xy = np.array(vals_xy, dtype=float)
    mask_yx = ~np.isnan(arr_yx)
    mask_xy = ~np.isnan(arr_xy)

    if mask_yx.any():
        print(
            f"\nSummary for val[y,x,channel]: "
            f"min={np.nanmin(arr_yx):.4f}, max={np.nanmax(arr_yx):.4f}, "
            f"mean={np.nanmean(arr_yx):.4f}"
        )
    if mask_xy.any():
        print(
            f"Summary for val[x,y,channel]: "
            f"min={np.nanmin(arr_xy):.4f}, max={np.nanmax(arr_xy):.4f}, "
            f"mean={np.nanmean(arr_xy):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check ilastik probability axis ordering by comparing values at "
            "JSON centroids under (y,x,c) vs (x,y,c) indexing."
        )
    )
    parser.add_argument(
        "--h5-dir",
        required=True,
        help="Directory containing *_Probabilities.h5 files.",
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory containing nuclei JSON files.",
    )
    parser.add_argument(
        "--stem",
        required=True,
        help=(
            "Common stem of files, e.g. 'JN_TS_001_bg_tile_12883_7423' "
            "for files '..._Probabilities.h5' and '.json'."
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to test (default: 0).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Maximum number of nuclei to sample (default: 50).",
    )

    args = parser.parse_args()

    h5_path = os.path.join(args.h5_dir, args.stem + "_Probabilities.h5")
    json_path = os.path.join(args.json_dir, args.stem + ".json")

    print(f"H5 path   : {h5_path}")
    print(f"JSON path : {json_path}")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    nuclei = load_nuclei(json_path)

    with h5py.File(h5_path, "r") as f:
        if "/exported_data" not in f:
            raise KeyError(f"/exported_data not found in {h5_path}")
        arr = f["/exported_data"][...]

    print(f"H5 /exported_data shape: {arr.shape}")

    sample_centroid_values(arr, nuclei, args.channel, args.num_samples)


if __name__ == "__main__":
    main()

