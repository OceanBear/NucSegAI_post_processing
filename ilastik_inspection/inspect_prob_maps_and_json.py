#!/usr/bin/env python
"""
Inspect ilastik probability maps and HoVer-Net JSON nuclei to understand:

1. Which HDF5 dataset likely holds the probability volumes.
2. The shape/layout of the probability arrays (channels-first vs channels-last).
3. Whether the probability map spatial dimensions match the JSON coordinate system.

This script is meant as an exploratory tool only – it does NOT modify any JSONs.

Requirements (install in your conda env if needed):
    pip install h5py numpy

Usage (adjust paths as needed):
    python inspect_prob_maps_and_json.py \\
        --h5-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_post_processing/artifacts/Probabilities" \\
        --json-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_test/pred_SN_WSI/json" \\
        --max-files 5
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import h5py  # type: ignore


def find_common_stems(
    json_dir: str,
    h5_dir: str,
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """
    Build stem -> path mappings for JSON and H5 files and return the common stems.

    JSON filenames are expected like:
        JN_TS_004_bg_tile_26418_6375_wsi4.json
    H5 filenames are expected like:
        JN_TS_004_bg_tile_26418_6375_Probabilities.h5
    """

    def json_stem(fn: str) -> str:
        # Drop suffix: _wsi<digits>.json
        return re.sub(r"_wsi\d+\.json$", "", fn)

    def h5_stem(fn: str) -> str:
        # Drop suffix: _Probabilities.h5
        return re.sub(r"_Probabilities\.h5$", "", fn)

    json_map: Dict[str, str] = {}
    h5_map: Dict[str, str] = {}

    for name in os.listdir(json_dir):
        if not name.lower().endswith(".json"):
            continue
        stem = json_stem(name)
        json_map[stem] = os.path.join(json_dir, name)

    for name in os.listdir(h5_dir):
        if not name.lower().endswith(".h5") and not name.lower().endswith(".hdf5"):
            continue
        stem = h5_stem(name)
        h5_map[stem] = os.path.join(h5_dir, name)

    common = sorted(set(json_map.keys()) & set(h5_map.keys()))
    return json_map, h5_map, common


def load_json_nuclei(json_path: str) -> Dict:
    """Load the JSON and return the nuclei dict under key 'nuc'."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if "nuc" not in data or not isinstance(data["nuc"], dict):
        raise ValueError(f"'nuc' dict not found in JSON: {json_path}")
    return data["nuc"]


def estimate_json_extent_from_bbox(nuclei: Dict) -> Tuple[int, int]:
    """
    Estimate image spatial extent (H, W) from nuclei 'bbox' fields.

    Assumes bbox format:
        bbox = [[x_min, y_min], [x_max, y_max]]
    """
    max_x = 0
    max_y = 0

    for nid, info in nuclei.items():
        bbox = info.get("bbox")
        if (
            not isinstance(bbox, Sequence)
            or len(bbox) != 2
            or not isinstance(bbox[0], Sequence)
            or not isinstance(bbox[1], Sequence)
            or len(bbox[0]) != 2
            or len(bbox[1]) != 2
        ):
            continue
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        try:
            max_x = max(max_x, int(round(x_max)))
            max_y = max(max_y, int(round(y_max)))
        except Exception:
            continue

    # Convert from max index to approximate image size
    return max_y + 1, max_x + 1


def collect_h5_datasets(h5_path: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """Return list of (dataset_path, shape) for all datasets inside an HDF5 file."""
    out: List[Tuple[str, Tuple[int, ...]]] = []

    def _visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            try:
                shape = tuple(int(d) for d in obj.shape)
            except Exception:
                shape = tuple(obj.shape)  # best effort
            out.append((f"/{name}", shape))

    with h5py.File(h5_path, "r") as f:
        f.visititems(_visitor)
    return out


def score_probability_candidate(shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int]]:
    """
    Heuristic to decide if a dataset shape looks like a 2-channel probability map.

    Returns (channels, height, width) if it looks plausible; otherwise None.
    """
    if len(shape) != 3:
        return None

    # channels-first: (C, H, W)
    if shape[0] == 2:
        c, h, w = shape
        return c, h, w

    # channels-last: (H, W, C)
    if shape[-1] == 2:
        h, w, c = shape
        return c, h, w

    return None


def pick_probability_dataset(
    h5_path: str,
) -> Optional[Tuple[str, Tuple[int, ...], Tuple[int, int, int]]]:
    """
    Among all datasets, pick the one most likely to be the probability volume.

    Returns:
        (dataset_path, raw_shape, (channels, height, width))
    or None if nothing plausible is found.
    """
    candidates: List[Tuple[str, Tuple[int, ...], Tuple[int, int, int]]] = []

    all_dsets = collect_h5_datasets(h5_path)
    for dset_path, shape in all_dsets:
        chw = score_probability_candidate(shape)
        if chw is None:
            continue
        c, h, w = chw
        # Heuristic: prefer "larger" spatial maps, as probabilities should cover the full tile
        spatial_size = h * w
        candidates.append((dset_path, shape, (c, h, w, spatial_size)))

    if not candidates:
        return None

    # Sort by spatial area descending, then by path name for determinism
    candidates.sort(key=lambda x: (x[2][3], x[0]), reverse=True)

    best_path, best_shape, (c, h, w, _) = candidates[0]
    return best_path, best_shape, (c, h, w)


def read_basic_stats(h5_path: str, dset_path: str) -> Tuple[float, float]:
    """Read min/max values of the selected dataset (on the whole array)."""
    with h5py.File(h5_path, "r") as f:
        arr = f[dset_path][...]
    arr = np.asarray(arr)
    return float(np.min(arr)), float(np.max(arr))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ilastik probability HDF5 files and matching HoVer-Net JSON files."
    )
    parser.add_argument(
        "--h5-dir",
        required=True,
        help="Directory containing ilastik probability .h5 files.",
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory containing HoVer-Net nuclei JSON files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Maximum number of matching stems to inspect (for brevity). Use 0 for all.",
    )
    parser.add_argument(
        "--artifact-channel",
        type=int,
        default=0,
        help="Index of the Artifact class channel (default: 0).",
    )

    args = parser.parse_args()

    json_map, h5_map, common = find_common_stems(args.json_dir, args.h5_dir)

    print(f"Found {len(json_map)} JSON files, {len(h5_map)} H5 files.")
    print(f"Common stems (JSON + H5): {len(common)}")
    for stem in common:
        print(f"  - {stem}")
    print()

    if not common:
        print("No matching stems found between JSON and H5. Nothing to inspect.")
        return

    to_inspect = common if args.max_files <= 0 else common[: args.max_files]
    print(f"Inspecting up to {len(to_inspect)} matching files.\n")

    for stem in to_inspect:
        json_path = json_map[stem]
        h5_path = h5_map[stem]

        print("=" * 80)
        print(f"STEM: {stem}")
        print(f"JSON: {json_path}")
        print(f"H5  : {h5_path}")

        # JSON side: nuclei and estimated spatial extent
        try:
            nuclei = load_json_nuclei(json_path)
        except Exception as e:
            print(f"[JSON] ERROR reading nuclei: {e}")
            continue

        n_nuclei = len(nuclei)
        json_H, json_W = estimate_json_extent_from_bbox(nuclei)
        print(f"[JSON] nuclei count: {n_nuclei}")
        print(f"[JSON] estimated spatial extent from bbox: H={json_H}, W={json_W}")

        # HDF5 side: dataset discovery
        try:
            all_dsets = collect_h5_datasets(h5_path)
        except Exception as e:
            print(f"[H5] ERROR reading datasets: {e}")
            continue

        print(f"[H5] total datasets: {len(all_dsets)}")
        for dset_path, shape in all_dsets:
            print(f"    dataset {dset_path}: shape={shape}")

        best = pick_probability_dataset(h5_path)
        if best is None:
            print("[H5] No plausible 2-channel probability dataset found.")
            continue

        prob_path, prob_shape, (c, h, w) = best
        print(f"[H5] LIKELY probability dataset: {prob_path} with shape={prob_shape}")

        # Determine layout and artifact channel axis
        layout: str
        if len(prob_shape) == 3 and prob_shape[0] == c and c == 2:
            layout = "channels_first (C,H,W)"
            prob_H, prob_W = prob_shape[1], prob_shape[2]
            channel_axis = 0
        elif len(prob_shape) == 3 and prob_shape[-1] == c and c == 2:
            layout = "channels_last (H,W,C)"
            prob_H, prob_W = prob_shape[0], prob_shape[1]
            channel_axis = 2
        else:
            layout = "unknown_3D_layout"
            # Fallback: assume second and third dims are spatial
            prob_H, prob_W = h, w
            channel_axis = -1

        print(f"[H5] interpreted layout: {layout}")
        print(f"[H5] interpreted spatial dims: H={prob_H}, W={prob_W}, channels={c}")

        # Compare JSON extent vs probability map size
        dH = prob_H - json_H
        dW = prob_W - json_W
        print(
            f"[ALIGNMENT] prob_map (H,W)=({prob_H},{prob_W}) vs JSON (~H,W)=({json_H},{json_W})"
        )
        print(f"[ALIGNMENT] delta (prob - json): dH={dH}, dW={dW}")

        # Basic stats on the probability values
        try:
            vmin, vmax = read_basic_stats(h5_path, prob_path)
            print(f"[H5] value range: min={vmin:.4f}, max={vmax:.4f}")
            if vmax > 1.01:
                print(
                    "[H5] NOTE: max > 1.0 – probabilities may be stored in 0..255 or 0..65535.\n"
                    "      You will likely need to normalize by 255 or 65535 in the main script."
                )
        except Exception as e:
            print(f"[H5] ERROR computing value stats: {e}")

        # Optional: sample a few centroids to check that indices fall inside the prob map
        sample_ids = list(nuclei.keys())[: min(5, n_nuclei)]
        print(f"[CHECK] sampling up to {len(sample_ids)} centroids to verify in-bounds:")
        try:
            with h5py.File(h5_path, "r") as f:
                arr = f[prob_path]
                for nid in sample_ids:
                    info = nuclei[nid]
                    cy, cx = info.get("centroid", [None, None])
                    if cy is None or cx is None:
                        print(f"    nucleus {nid}: missing centroid")
                        continue
                    iy = int(round(cy))
                    ix = int(round(cx))
                    in_bounds = 0 <= iy < prob_H and 0 <= ix < prob_W
                    msg = f"    nucleus {nid}: centroid (y={cy:.2f}, x={cx:.2f}) -> index (iy={iy}, ix={ix}), in_bounds={in_bounds}"
                    if in_bounds:
                        # Fetch artifact channel probability
                        if channel_axis == 0:
                            # (C,H,W)
                            val = float(arr[args.artifact_channel, iy, ix])
                        elif channel_axis == 2:
                            # (H,W,C)
                            val = float(arr[iy, ix, args.artifact_channel])
                        else:
                            # Fallback, just read scalar at (iy,ix) if possible
                            try:
                                val = float(arr[iy, ix])
                            except Exception:
                                val = float("nan")
                        msg += f", artifact_prob≈{val:.4f}"
                    print(msg)
        except Exception as e:
            print(f"[CHECK] ERROR sampling centroids: {e}")

        print()  # blank line between stems


if __name__ == "__main__":
    main()

