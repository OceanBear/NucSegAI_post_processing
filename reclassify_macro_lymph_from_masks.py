#!/usr/bin/env python
"""
Reclassify nuclei as Macrophage / Lymphocyte using ilastik probability masks
and NucSegAI type probabilities (typeprob JSON).

This script is intended to be run AFTER you have already produced filtered
JSONs (e.g., after Artifact/RBC removal + reindexing). It does NOT remove nuclei;
it only updates the nucleus 'type' field in the base JSON.

Inputs:
  --json-dir         : base nuclei JSONs (top-level key 'nuc')
  --typeprob-dir     : typeprob JSONs (top-level keys are nucleus IDs, values are
                       lists/tuples of length >= 7 giving class probabilities)
  --macro-h5-dir     : Macrophage ilastik Probabilities (*.h5)
  --lymph-h5-dir     : Lymphocytes ilastik Probabilities (*.h5)

Outputs:
  --out-json-dir     : output directory for reclassified base JSONs
  --out-typeprob-dir : output directory for typeprob JSONs (copied through, unchanged)
  reclassify_summary.csv : written to --out-json-dir (one row per stem + TOTAL row),
    same style as filter_summary.csv from filter_nuclei_with_ilastik_mask.py

Reclassification logic (per nucleus):
  - Compute macro_flag if fraction of contour pixels with Macro_prob > macro-threshold
    exceeds macro-fraction-threshold.
  - Compute lymph_flag similarly for Lymphocytes.
  - If neither flag: keep original type.
  - Else compute top-K classes by type probability excluding the current class.
    - If only macro_flag: reclassify to Macrophage (class 3) iff class 3 is in top-K.
    - If only lymph_flag: reclassify to Lymphocyte (class 4) iff class 4 is in top-K.
    - If both flags:
        - If both (3 and 4) are in top-K, pick the higher probability between probs[3] and probs[4].
        - Else if only one is in top-K, pick that one.
        - Else keep original.

Class mapping (NucSegAI):
  0: Undefined
  1: Epithelium (PD-L1 low and Ki67 low)
  2: Epithelium (PD-L1 hi or Ki67 hi)
  3: Macrophage
  4: Lymphocyte
  5: Vascular
  6: Fibroblast/Stroma

Important coordinate note (per your data spec):
  - H5 /exported_data is indexed as (Y, X, C), i.e. arr[y, x, c]
  - JSON contour points are [x, y] (col, row)
  - JSON bbox is [[y_min, x_min], [y_max, x_max]]
"""

import argparse
import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import h5py  # type: ignore
import numpy as np  # type: ignore


MACRO_CLASS = 3
LYMPH_CLASS = 4


def build_base_json_map(json_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for name in os.listdir(json_dir):
        if not name.lower().endswith(".json"):
            continue
        stem = os.path.splitext(name)[0]
        mapping[stem] = os.path.join(json_dir, name)
    return mapping


def build_typeprob_json_map(typeprob_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for name in os.listdir(typeprob_dir):
        if not name.lower().endswith(".json"):
            continue
        stem_full = os.path.splitext(name)[0]
        stem = re.sub(r"_typeprob$", "", stem_full)
        mapping[stem] = os.path.join(typeprob_dir, name)
    return mapping


def build_h5_map(h5_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for name in os.listdir(h5_dir):
        if not (name.lower().endswith(".h5") or name.lower().endswith(".hdf5")):
            continue
        stem = re.sub(r"_Probabilities\.h5$", "", name)
        mapping[stem] = os.path.join(h5_dir, name)
    return mapping


def load_base_json(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    if "nuc" not in data or not isinstance(data["nuc"], dict):
        raise ValueError(f"'nuc' dict not found in base JSON: {path}")
    return data


def load_typeprob_json(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"typeprob JSON is not a dict at top level: {path}")
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_prob_map(
    h5_path: str,
    dataset_path: str = "/exported_data",
    channel: int = 0,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if dataset_path not in f:
            raise KeyError(f"Dataset '{dataset_path}' not found in {h5_path}")
        ds = f[dataset_path][...]

    arr = np.asarray(ds)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array at {dataset_path} in {h5_path}, got {arr.shape}")
    if channel < 0 or channel >= arr.shape[2]:
        raise ValueError(f"channel={channel} out of range for shape {arr.shape} in {h5_path}")

    # Stored as (Y, X, C); keep as-is so downstream indexes as [y, x]
    prob_map = arr[:, :, channel].astype(np.float32)
    return prob_map


def _points_in_polygon(
    x: np.ndarray,
    y: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    poly_x = np.asarray(poly_x, dtype=np.float32)
    poly_y = np.asarray(poly_y, dtype=np.float32)

    inside = np.zeros(x.shape, dtype=bool)
    n = len(poly_x)
    if n < 3:
        return inside

    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= intersect
        j = i
    return inside


def compute_fraction_above_threshold_in_contour(
    prob_map: np.ndarray,
    contour: Sequence[Sequence[float]],
    threshold: float,
) -> float:
    if not contour:
        return 0.0
    height, width = prob_map.shape

    # Contour points are [x, y] (col, row)
    xs = np.array([pt[0] for pt in contour], dtype=np.float32)
    ys = np.array([pt[1] for pt in contour], dtype=np.float32)
    if ys.size == 0 or xs.size == 0:
        return 0.0

    y_min = max(0, int(np.floor(ys.min())))
    y_max = min(height - 1, int(np.ceil(ys.max())))
    x_min = max(0, int(np.floor(xs.min())))
    x_max = min(width - 1, int(np.ceil(xs.max())))
    if y_min > y_max or x_min > x_max:
        return 0.0

    region = prob_map[y_min : y_max + 1, x_min : x_max + 1]
    if region.size == 0:
        return 0.0

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    inside = _points_in_polygon(xx + 0.5, yy + 0.5, xs, ys)
    if not inside.any():
        return 0.0
    vals = region[inside]
    if vals.size == 0:
        return 0.0
    return float((vals > threshold).mean())


def top_k_classes_excluding_current(
    probs: Sequence[float],
    current_type,
    k: int,
) -> List[int]:
    try:
        current_int = int(current_type)
    except Exception:
        current_int = None

    candidates = []
    for i, p in enumerate(probs):
        if current_int is not None and i == current_int:
            continue
        candidates.append((i, float(p)))
    candidates.sort(key=lambda t: t[1], reverse=True)
    return [cls for cls, _ in candidates[:k]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reclassify nuclei as Macrophage/Lymphocyte using masks + typeprob JSON."
    )
    parser.add_argument("--json-dir", required=True, help="Directory with base nuclei JSON files.")
    parser.add_argument(
        "--typeprob-dir",
        required=True,
        help="Directory with *_typeprob.json files.",
    )
    parser.add_argument(
        "--macro-h5-dir",
        required=True,
        help="Directory with Macrophage *_Probabilities.h5 files.",
    )
    parser.add_argument(
        "--lymph-h5-dir",
        required=True,
        help="Directory with Lymphocytes *_Probabilities.h5 files.",
    )
    parser.add_argument(
        "--out-json-dir",
        required=True,
        help="Output directory for reclassified base JSON files.",
    )
    parser.add_argument(
        "--out-typeprob-dir",
        required=True,
        help="Output directory for typeprob JSON files (copied through unchanged).",
    )
    parser.add_argument(
        "--dataset-path",
        default="/exported_data",
        help="Dataset path inside H5 files (default: /exported_data).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Class channel index to use from each H5 dataset (default: 0).",
    )
    parser.add_argument("--macro-threshold", type=float, default=0.6)
    parser.add_argument("--macro-fraction-threshold", type=float, default=0.3)
    parser.add_argument("--lymph-threshold", type=float, default=0.6)
    parser.add_argument("--lymph-fraction-threshold", type=float, default=0.3)
    parser.add_argument(
        "--reclass-top-k",
        type=int,
        default=3,
        help="Top-K (excluding current type) required for reclassification (default: 3).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, process only the first N stems (for quick tests).",
    )

    args = parser.parse_args()

    ensure_dir(args.out_json_dir)
    ensure_dir(args.out_typeprob_dir)

    base_map = build_base_json_map(args.json_dir)
    typeprob_map = build_typeprob_json_map(args.typeprob_dir)
    macro_map_paths = build_h5_map(args.macro_h5_dir)
    lymph_map_paths = build_h5_map(args.lymph_h5_dir)

    stems = sorted(set(base_map.keys()) & set(typeprob_map.keys()) & set(macro_map_paths.keys()) & set(lymph_map_paths.keys()))
    if args.max_files and args.max_files > 0:
        stems = stems[: args.max_files]

    print(f"Base JSON files      : {len(base_map)}")
    print(f"Typeprob JSON files  : {len(typeprob_map)}")
    print(f"Macro H5 files       : {len(macro_map_paths)}")
    print(f"Lymph H5 files       : {len(lymph_map_paths)}")
    print(f"Stems with all inputs: {len(stems)}")
    print(f"Using reclass-top-k={args.reclass_top_k}")
    print()

    # Summary CSV (same pattern as filter_nuclei_with_ilastik_mask.py -> filter_summary.csv)
    summary_rows: List[str] = []
    summary_rows.append(
        "stem,total_nuclei,n_macro_flag,n_lymph_flag,n_both_flags,n_reclass_to_macro,n_reclass_to_lymph"
    )
    tot_n = tot_mf = tot_lf = tot_both = tot_rm = tot_rl = 0

    for stem in stems:
        base_path = base_map[stem]
        tp_path = typeprob_map[stem]
        macro_h5_path = macro_map_paths[stem]
        lymph_h5_path = lymph_map_paths[stem]

        print("=" * 80)
        print(f"STEM: {stem}")

        try:
            base_data = load_base_json(base_path)
            tp_data = load_typeprob_json(tp_path)
            macro_prob = load_prob_map(
                macro_h5_path, dataset_path=args.dataset_path, channel=args.channel
            )
            lymph_prob = load_prob_map(
                lymph_h5_path, dataset_path=args.dataset_path, channel=args.channel
            )
        except Exception as e:
            print(f"  [ERROR] Skipping stem due to load error: {e}")
            continue

        nuc = base_data["nuc"]

        n_macro_flag = 0
        n_lymph_flag = 0
        n_both_flags = 0
        n_reclass_macro = 0
        n_reclass_lymph = 0

        k = max(1, int(args.reclass_top_k))

        for nid, info in nuc.items():
            contour = info.get("contour")
            if contour is None:
                continue

            probs = tp_data.get(nid)
            if not isinstance(probs, (list, tuple)) or len(probs) < 7:
                continue

            macro_fraction = compute_fraction_above_threshold_in_contour(
                macro_prob, contour, args.macro_threshold
            )
            lymph_fraction = compute_fraction_above_threshold_in_contour(
                lymph_prob, contour, args.lymph_threshold
            )

            macro_flag = macro_fraction > args.macro_fraction_threshold
            lymph_flag = lymph_fraction > args.lymph_fraction_threshold

            if macro_flag:
                n_macro_flag += 1
            if lymph_flag:
                n_lymph_flag += 1
            if macro_flag and lymph_flag:
                n_both_flags += 1

            if not (macro_flag or lymph_flag):
                continue

            current_type = info.get("type")
            topk = top_k_classes_excluding_current(probs, current_type, k)

            in_top_macro = MACRO_CLASS in topk
            in_top_lymph = LYMPH_CLASS in topk

            new_type = current_type
            if macro_flag and not lymph_flag:
                if in_top_macro:
                    new_type = MACRO_CLASS
            elif lymph_flag and not macro_flag:
                if in_top_lymph:
                    new_type = LYMPH_CLASS
            else:
                # both flags
                if in_top_macro and in_top_lymph:
                    if float(probs[LYMPH_CLASS]) > float(probs[MACRO_CLASS]):
                        new_type = LYMPH_CLASS
                    else:
                        new_type = MACRO_CLASS
                elif in_top_macro:
                    new_type = MACRO_CLASS
                elif in_top_lymph:
                    new_type = LYMPH_CLASS

            if new_type != current_type:
                info["type"] = new_type
                if new_type == MACRO_CLASS:
                    n_reclass_macro += 1
                elif new_type == LYMPH_CLASS:
                    n_reclass_lymph += 1

        # Write outputs
        out_base_path = os.path.join(args.out_json_dir, os.path.basename(base_path))
        out_tp_path = os.path.join(args.out_typeprob_dir, os.path.basename(tp_path))

        with open(out_base_path, "w") as f:
            json.dump(base_data, f, indent=2)

        # copy typeprob unchanged (but into output dir)
        with open(out_tp_path, "w") as f:
            json.dump(tp_data, f, indent=2)

        print(
            f"n_nuclei={len(nuc)}, macro_flag={n_macro_flag}, lymph_flag={n_lymph_flag}, "
            f"both={n_both_flags}, reclass_macro={n_reclass_macro}, reclass_lymph={n_reclass_lymph}"
        )
        print(f"wrote -> {out_base_path}")

        nn = int(len(nuc))
        summary_rows.append(
            f"{stem},{nn},{n_macro_flag},{n_lymph_flag},{n_both_flags},"
            f"{n_reclass_macro},{n_reclass_lymph}"
        )
        tot_n += nn
        tot_mf += n_macro_flag
        tot_lf += n_lymph_flag
        tot_both += n_both_flags
        tot_rm += n_reclass_macro
        tot_rl += n_reclass_lymph

    if len(summary_rows) > 1:
        summary_rows.append(
            f"TOTAL,{tot_n},{tot_mf},{tot_lf},{tot_both},{tot_rm},{tot_rl}"
        )

    summary_path = os.path.join(args.out_json_dir, "reclassify_summary.csv")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_rows))

    print()
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
