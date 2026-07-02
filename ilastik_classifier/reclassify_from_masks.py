#!/usr/bin/env python
"""
Reclassify nuclei as Tumor / Macrophage / Lymphocyte / Fibroblast using ilastik probability
masks and NucSegAI type probabilities (typeprob JSON).

This script is intended to be run AFTER you have already produced filtered
JSONs (e.g., after Artifact/RBC removal + reindexing). It does NOT remove nuclei;
it only updates the nucleus 'type' field in the base JSON.

Inputs:
  --json-dir         : base nuclei JSONs (top-level key 'nuc')
  --typeprob-dir     : typeprob JSONs (top-level keys are nucleus IDs, values are
                       lists/tuples of length >= 7 giving class probabilities)
  --tumor-h5-dir     : Tumor ilastik Probabilities (*.h5)
  --macro-h5-dir     : Macrophage ilastik Probabilities (*.h5)
  --lymph-h5-dir     : Lymphocytes ilastik Probabilities (*.h5)
  --fibro-h5-dir     : Fibroblast/Stroma ilastik Probabilities (*.h5)

Outputs:
  --out-json-dir     : output directory for reclassified base JSONs
  --out-typeprob-dir : output directory for typeprob JSONs (copied through, unchanged)
  reclassify_summary.csv : written to --out-json-dir (one row per stem + TOTAL row),
    same style as filter_summary.csv from filter_nuclei_with_ilastik_mask.py

Reclassification logic (per nucleus):
  - For each class (Tumor/Macro/Lymph/Fibro):
      - compute fraction of contour pixels with Prob > class-threshold
      - flag if fraction exceeds class-fraction-threshold
  - If no flags: keep original type.
  - Else compute top-K classes by type probability excluding the current class.
  - Candidate classes are those that are (flagged) AND (in top-K).
  - If there are candidates: reclassify to the candidate with max mask fraction (ties: typeprob, then class id).
  - Else: keep original type.

Class mapping (NucSegAI):
  0: Undefined
  1: Tumor/Epithelium (PD-L1 low and Ki67 low)
  2: Tumor/Epithelium (PD-L1 hi or Ki67 hi)
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

TUMOR_CLASS = 1
MACRO_CLASS = 3
LYMPH_CLASS = 4
FIBRO_CLASS = 6

CLASS_IDS = (TUMOR_CLASS, MACRO_CLASS, LYMPH_CLASS, FIBRO_CLASS)
TUMOR_CLASSES = (1, 2)

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


def pick_new_type_from_candidates(
    *,
    probs: Sequence[float],
    current_type,
    topk: Sequence[int],
    class_id_to_fraction: Dict[int, float],
) -> Tuple[object, List[int]]:
    """
    Returns (new_type, considered_class_ids).
    considered_class_ids are those that were flagged and passed top-K gating.
    """
    try:
        current_int = int(current_type)
    except Exception:
        current_int = None

    candidates: List[int] = []
    for cls_id, frac in class_id_to_fraction.items():
        if frac <= 0:
            continue
        if current_int is not None and cls_id == current_int:
            continue
        if cls_id in topk:
            candidates.append(cls_id)

    if not candidates:
        return current_type, []

    def sort_key(cls_id: int) -> Tuple[float, float, int]:
        # Primary: highest mask fraction
        frac = float(class_id_to_fraction.get(cls_id, 0.0))
        # Secondary: highest typeprob
        tp = float(probs[cls_id]) if cls_id < len(probs) else 0.0
        # Tertiary: stable determinism (lower class id wins)
        return (frac, tp, -cls_id)

    winner = max(candidates, key=sort_key)
    return winner, candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reclassify nuclei as Tumor/Macro/Lymph/Fibro using masks + typeprob JSON."
    )
    parser.add_argument("--json-dir", required=True, help="Directory with base nuclei JSON files.")
    parser.add_argument(
        "--typeprob-dir",
        required=True,
        help="Directory with *_typeprob.json files.",
    )
    parser.add_argument(
        "--tumor-h5-dir",
        required=True,
        help="Directory with Tumor *_Probabilities.h5 files.",
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
        "--fibro-h5-dir",
        required=True,
        help="Directory with Fibroblast/Stroma *_Probabilities.h5 files.",
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
    parser.add_argument("--tumor-threshold", type=float, default=0.6)
    parser.add_argument("--tumor-fraction-threshold", type=float, default=0.3)
    parser.add_argument("--macro-threshold", type=float, default=0.6)
    parser.add_argument("--macro-fraction-threshold", type=float, default=0.3)
    parser.add_argument("--lymph-threshold", type=float, default=0.6)
    parser.add_argument("--lymph-fraction-threshold", type=float, default=0.3)
    parser.add_argument("--fibro-threshold", type=float, default=0.6)
    parser.add_argument("--fibro-fraction-threshold", type=float, default=0.3)
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
    tumor_map_paths = build_h5_map(args.tumor_h5_dir)
    macro_map_paths = build_h5_map(args.macro_h5_dir)
    lymph_map_paths = build_h5_map(args.lymph_h5_dir)
    fibro_map_paths = build_h5_map(args.fibro_h5_dir)

    stems = sorted(
        set(base_map.keys())
        & set(typeprob_map.keys())
        & set(tumor_map_paths.keys())
        & set(macro_map_paths.keys())
        & set(lymph_map_paths.keys())
        & set(fibro_map_paths.keys())
    )
    if args.max_files and args.max_files > 0:
        stems = stems[: args.max_files]

    print(f"Base JSON files      : {len(base_map)}")
    print(f"Typeprob JSON files  : {len(typeprob_map)}")
    print(f"Tumor H5 files       : {len(tumor_map_paths)}")
    print(f"Macro H5 files       : {len(macro_map_paths)}")
    print(f"Lymph H5 files       : {len(lymph_map_paths)}")
    print(f"Fibro H5 files       : {len(fibro_map_paths)}")
    print(f"Stems with all inputs: {len(stems)}")
    print(f"Using reclass-top-k={args.reclass_top_k}")
    print()

    # Summary CSV (same pattern as filter_nuclei_with_ilastik_mask.py -> filter_summary.csv)
    summary_rows: List[str] = []
    summary_rows.append(
        "stem,total_nuclei,"
        "n_tumor_flag,n_macro_flag,n_lymph_flag,n_fibro_flag,"
        "n_multi_flag,"
        "n_reclass_to_tumor,n_reclass_to_macro,n_reclass_to_lymph,n_reclass_to_fibro"
    )
    tot_n = 0
    tot_tf = tot_mf = tot_lf = tot_ff = 0
    tot_multi = 0
    tot_rt = tot_rm = tot_rl = tot_rf = 0

    for stem in stems:
        base_path = base_map[stem]
        tp_path = typeprob_map[stem]
        tumor_h5_path = tumor_map_paths[stem]
        macro_h5_path = macro_map_paths[stem]
        lymph_h5_path = lymph_map_paths[stem]
        fibro_h5_path = fibro_map_paths[stem]

        print("=" * 80)
        print(f"STEM: {stem}")

        try:
            base_data = load_base_json(base_path)
            tp_data = load_typeprob_json(tp_path)
            tumor_prob = load_prob_map(
                tumor_h5_path, dataset_path=args.dataset_path, channel=args.channel
            )
            macro_prob = load_prob_map(
                macro_h5_path, dataset_path=args.dataset_path, channel=args.channel
            )
            lymph_prob = load_prob_map(
                lymph_h5_path, dataset_path=args.dataset_path, channel=args.channel
            )
            fibro_prob = load_prob_map(
                fibro_h5_path, dataset_path=args.dataset_path, channel=args.channel
            )
        except Exception as e:
            print(f"  [ERROR] Skipping stem due to load error: {e}")
            continue

        nuc = base_data["nuc"]

        n_tumor_flag = 0
        n_macro_flag = 0
        n_lymph_flag = 0
        n_fibro_flag = 0
        n_multi_flag = 0
        n_reclass_tumor = 0
        n_reclass_macro = 0
        n_reclass_lymph = 0
        n_reclass_fibro = 0

        k = max(1, int(args.reclass_top_k))

        for nid, info in nuc.items():
            contour = info.get("contour")
            if contour is None:
                continue

            probs = tp_data.get(nid)
            if not isinstance(probs, (list, tuple)) or len(probs) < 7:
                continue

            current_type = info.get("type")
            topk = top_k_classes_excluding_current(probs, current_type, k)

            tumor_fraction = compute_fraction_above_threshold_in_contour(
                tumor_prob, contour, args.tumor_threshold
            )
            macro_fraction = compute_fraction_above_threshold_in_contour(
                macro_prob, contour, args.macro_threshold
            )
            lymph_fraction = compute_fraction_above_threshold_in_contour(
                lymph_prob, contour, args.lymph_threshold
            )
            fibro_fraction = compute_fraction_above_threshold_in_contour(
                fibro_prob, contour, args.fibro_threshold
            )

            tumor_flag = tumor_fraction > args.tumor_fraction_threshold
            macro_flag = macro_fraction > args.macro_fraction_threshold
            lymph_flag = lymph_fraction > args.lymph_fraction_threshold
            fibro_flag = fibro_fraction > args.fibro_fraction_threshold

            flag_count = int(tumor_flag) + int(macro_flag) + int(lymph_flag) + int(fibro_flag)
            if tumor_flag:
                n_tumor_flag += 1
            if macro_flag:
                n_macro_flag += 1
            if lymph_flag:
                n_lymph_flag += 1
            if fibro_flag:
                n_fibro_flag += 1
            if flag_count >= 2:
                n_multi_flag += 1

            if flag_count == 0:
                continue

            # Special rule: NucSegAI has two tumor-like classes (1 and 2). If the nucleus is
            # already 1 or 2 and ONLY the tumor mask flag is raised, do not reclassify.
            if tumor_flag and flag_count == 1:
                try:
                    cur_int = int(current_type)
                except Exception:
                    cur_int = None
                if cur_int in TUMOR_CLASSES:
                    continue

            class_id_to_fraction: Dict[int, float] = {}
            if tumor_flag:
                class_id_to_fraction[TUMOR_CLASS] = tumor_fraction
            if macro_flag:
                class_id_to_fraction[MACRO_CLASS] = macro_fraction
            if lymph_flag:
                class_id_to_fraction[LYMPH_CLASS] = lymph_fraction
            if fibro_flag:
                class_id_to_fraction[FIBRO_CLASS] = fibro_fraction

            new_type, _considered = pick_new_type_from_candidates(
                probs=probs,
                current_type=current_type,
                topk=topk,
                class_id_to_fraction=class_id_to_fraction,
            )

            if new_type != current_type:
                info["type"] = new_type
                if new_type == TUMOR_CLASS:
                    n_reclass_tumor += 1
                elif new_type == MACRO_CLASS:
                    n_reclass_macro += 1
                elif new_type == LYMPH_CLASS:
                    n_reclass_lymph += 1
                elif new_type == FIBRO_CLASS:
                    n_reclass_fibro += 1

        # Write outputs
        out_base_path = os.path.join(args.out_json_dir, os.path.basename(base_path))
        out_tp_path = os.path.join(args.out_typeprob_dir, os.path.basename(tp_path))

        with open(out_base_path, "w") as f:
            json.dump(base_data, f, indent=2)

        # copy typeprob unchanged (but into output dir)
        with open(out_tp_path, "w") as f:
            json.dump(tp_data, f, indent=2)

        print(
            f"n_nuclei={len(nuc)}, tumor_flag={n_tumor_flag}, macro_flag={n_macro_flag}, "
            f"lymph_flag={n_lymph_flag}, fibro_flag={n_fibro_flag}, multi_flag={n_multi_flag}, "
            f"reclass_tumor={n_reclass_tumor}, reclass_macro={n_reclass_macro}, "
            f"reclass_lymph={n_reclass_lymph}, reclass_fibro={n_reclass_fibro}"
        )
        print(f"wrote -> {out_base_path}")

        nn = int(len(nuc))
        summary_rows.append(
            f"{stem},{nn},"
            f"{n_tumor_flag},{n_macro_flag},{n_lymph_flag},{n_fibro_flag},"
            f"{n_multi_flag},"
            f"{n_reclass_tumor},{n_reclass_macro},{n_reclass_lymph},{n_reclass_fibro}"
        )
        tot_n += nn
        tot_tf += n_tumor_flag
        tot_mf += n_macro_flag
        tot_lf += n_lymph_flag
        tot_ff += n_fibro_flag
        tot_multi += n_multi_flag
        tot_rt += n_reclass_tumor
        tot_rm += n_reclass_macro
        tot_rl += n_reclass_lymph
        tot_rf += n_reclass_fibro

    if len(summary_rows) > 1:
        summary_rows.append(
            f"TOTAL,{tot_n},"
            f"{tot_tf},{tot_mf},{tot_lf},{tot_ff},"
            f"{tot_multi},"
            f"{tot_rt},{tot_rm},{tot_rl},{tot_rf}"
        )

    summary_path = os.path.join(args.out_json_dir, "reclassify_summary.csv")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_rows))

    print()
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
