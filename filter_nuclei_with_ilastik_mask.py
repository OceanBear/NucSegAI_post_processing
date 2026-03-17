#!/usr/bin/env python
"""
Filter nuclei in base JSONs using ilastik probability maps, and keep
base/typeprob JSONs in sync by deleting the same nucleus IDs in both.

Inputs (directories):
  --json-dir       : base nuclei JSONs (with top-level key 'nuc')
  --typeprob-dir   : type-probability JSONs (top-level keys are nucleus IDs)
  --h5-dir         : ilastik HDF5 probability maps (*_Probabilities.h5)

Outputs (directories, created if needed):
  --out-json-dir       : cleaned base JSONs
  --out-typeprob-dir   : cleaned typeprob JSONs

Only stems that have ALL THREE of:
  - base JSON
  - typeprob JSON
  - H5 Probabilities
are processed. Others are skipped.

Filtering rule (bbox-mean, channel-last layout):
  - From H5, use dataset '/exported_data' by default.
  - Interpret shape as (H, W, C) when C == 2; artifact channel is index 0.
  - For each nucleus, compute the mean Artifact probability over its bbox
    (bbox is [[x_min, y_min], [x_max, y_max]] in pixel coordinates).
  - If mean_artifact_prob > THRESHOLD, the nucleus is removed.

The same nucleus ID is removed from:
  - base_json['nuc'][id]
  - typeprob_json[id]

Thresholds are configurable via:
  --artifact-threshold (default: 0.9)
  --rbc-threshold (default: 0.9)
"""

import argparse
import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import h5py  # type: ignore
import numpy as np  # type: ignore


def build_base_json_map(json_dir: str) -> Dict[str, str]:
    """Map STEM -> base JSON path from json_dir (STEM is basename without extension)."""
    mapping: Dict[str, str] = {}
    for name in os.listdir(json_dir):
        if not name.lower().endswith(".json"):
            continue
        stem = os.path.splitext(name)[0]
        mapping[stem] = os.path.join(json_dir, name)
    return mapping


def build_typeprob_json_map(typeprob_dir: str) -> Dict[str, str]:
    """
    Map STEM -> typeprob JSON path from typeprob_dir.

    Expected filenames, e.g.:
        JN_TS_004_bg_tile_26418_6375_typeprob.json

    STEM is the part before the '_typeprob' suffix.
    """
    mapping: Dict[str, str] = {}
    for name in os.listdir(typeprob_dir):
        if not name.lower().endswith(".json"):
            continue
        stem_full = os.path.splitext(name)[0]
        stem = re.sub(r"_typeprob$", "", stem_full)
        mapping[stem] = os.path.join(typeprob_dir, name)
    return mapping


def build_h5_map(h5_dir: str) -> Dict[str, str]:
    """
    Map STEM -> H5 path from h5_dir.

    Expected filenames, e.g.:
        JN_TS_004_bg_tile_26418_6375_Probabilities.h5

    STEM is the part before '_Probabilities.h5'.
    """
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
    """
    Load typeprob JSON. We assume top-level keys are nucleus IDs.
    The internal structure is not interpreted; we only drop IDs.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"typeprob JSON is not a dict at top level: {path}")
    return data


def load_prob_map(
    h5_path: str,
    dataset_path: str = "/exported_data",
    channel: int = 0,
) -> np.ndarray:
    """
    Load a single-channel probability map as a 2D numpy array with shape (H, W).

    The ilastik /exported_data dataset appears to be stored as (X, Y, C),
    where X is the horizontal axis and Y is the vertical axis, while the
    JSON coordinates and the rest of this script assume (Y, X) ordering.
    To reconcile this, we transpose the first two axes so that the returned
    array is indexed as [y, x].
    """
    with h5py.File(h5_path, "r") as f:
        if dataset_path not in f:
            raise KeyError(f"Dataset '{dataset_path}' not found in {h5_path}")
        ds = f[dataset_path][...]

    arr = np.asarray(ds)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array at {dataset_path} in {h5_path}, got {arr.shape}")

    # channels-last (X, Y, C); transpose to (Y, X, C) so that downstream
    # code can index as [y, x].
    if arr.shape[2] <= channel:
        raise ValueError(
            f"channel={channel} out of range for shape {arr.shape} in {h5_path}"
        )
    arr_yx = np.transpose(arr, (1, 0, 2))
    prob_map = arr_yx[:, :, channel]

    # Ensure float probabilities
    prob_map = prob_map.astype(np.float32)
    return prob_map


def bbox_to_indices(
    bbox: Sequence[Sequence[float]],
    height: int,
    width: int,
) -> Tuple[int, int, int, int]:
    """
    Convert bbox [[x_min, y_min], [x_max, y_max]] to clipped integer indices.

    Returns (y_min, y_max_inclusive, x_min, x_max_inclusive).
    If the bbox lies completely outside the image, the caller can detect
    this because y_min > y_max or x_min > x_max after clipping.
    """
    if (
        not isinstance(bbox, Sequence)
        or len(bbox) != 2
        or not isinstance(bbox[0], Sequence)
        or not isinstance(bbox[1], Sequence)
        or len(bbox[0]) != 2
        or len(bbox[1]) != 2
    ):
        raise ValueError(f"Unexpected bbox format: {bbox}")

    x_min_f, y_min_f = bbox[0]
    x_max_f, y_max_f = bbox[1]

    x_min = int(round(x_min_f))
    y_min = int(round(y_min_f))
    x_max = int(round(x_max_f))
    y_max = int(round(y_max_f))

    # Clip to image bounds
    x_min = max(0, min(x_min, width - 1))
    x_max = max(0, min(x_max, width - 1))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(0, min(y_max, height - 1))

    return y_min, y_max, x_min, x_max


def compute_bbox_mean(
    prob_map: np.ndarray,
    bbox: Sequence[Sequence[float]],
) -> float:
    """
    Compute mean probability over the bbox region.

    prob_map: 2D array (H, W)
    bbox: [[x_min, y_min], [x_max, y_max]] (float coordinates)
    """
    height, width = prob_map.shape
    try:
        y_min, y_max, x_min, x_max = bbox_to_indices(bbox, height, width)
    except ValueError:
        return 0.0

    if y_min > y_max or x_min > x_max:
        # Completely out-of-bounds or degenerate
        return 0.0

    region = prob_map[y_min : y_max + 1, x_min : x_max + 1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def _points_in_polygon(
    x: np.ndarray,
    y: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
) -> np.ndarray:
    """
    Vectorized point-in-polygon test using the ray casting algorithm.

    x, y: arrays of point coordinates (same shape)
    poly_x, poly_y: 1D arrays of polygon vertex coordinates (closed or open)

    Returns a boolean array with the same shape as x/y indicating points inside.
    """
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
            x
            < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= intersect
        j = i

    return inside


def compute_fraction_above_threshold_in_contour(
    prob_map: np.ndarray,
    contour: Sequence[Sequence[float]],
    threshold: float,
) -> float:
    """
    Compute the fraction of pixels inside a nucleus contour whose probability
    exceeds the given threshold.

    - prob_map: 2D array (H, W) of probabilities.
    - contour: list of [y, x] points defining the nucleus polygon in pixel coords.
    - threshold: per-pixel probability cutoff (e.g., 0.6).
    """
    if not contour:
        return 0.0

    height, width = prob_map.shape

    ys = np.array([pt[0] for pt in contour], dtype=np.float32)
    xs = np.array([pt[1] for pt in contour], dtype=np.float32)

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
    # Use pixel centers for the point-in-polygon test
    inside = _points_in_polygon(xx + 0.5, yy + 0.5, xs, ys)
    if not inside.any():
        return 0.0

    vals = region[inside]
    if vals.size == 0:
        return 0.0

    return float((vals > threshold).mean())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter nuclei using ilastik Artifact probability maps (bbox-mean, channel-last)."
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory with base nuclei JSON files.",
    )
    parser.add_argument(
        "--typeprob-dir",
        required=True,
        help="Directory with *_typeprob.json files.",
    )
    parser.add_argument(
        "--h5-dir",
        required=True,
        help="Directory with Artifact *_Probabilities.h5 files.",
    )
    parser.add_argument(
        "--rbc-h5-dir",
        default=None,
        help="Optional directory with RBC *_Probabilities.h5 files (same stems as Artifact).",
    )
    parser.add_argument(
        "--macro-h5-dir",
        default=None,
        help="Optional directory with Macrophage *_Probabilities.h5 files (same stems as Artifact).",
    )
    parser.add_argument(
        "--lymph-h5-dir",
        default=None,
        help="Optional directory with Lymphocytes *_Probabilities.h5 files (same stems as Artifact).",
    )
    parser.add_argument(
        "--out-json-dir",
        required=True,
        help="Output directory for cleaned base JSON files.",
    )
    parser.add_argument(
        "--out-typeprob-dir",
        required=True,
        help="Output directory for cleaned typeprob JSON files.",
    )
    parser.add_argument(
        "--artifact-threshold",
        dest="artifact_threshold",
        type=float,
        default=0.9,
        help="Artifact probability threshold for bbox-mean filtering (default: 0.9).",
    )
    # Backwards-compatible alias (deprecated)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="DEPRECATED: use --artifact-threshold instead.",
    )
    parser.add_argument(
        "--artifact-channel",
        type=int,
        default=0,
        help="Index of the Artifact class channel in the H5 dataset (default: 0).",
    )
    parser.add_argument(
        "--artifact-fraction-threshold",
        type=float,
        default=0.3,
        help=(
            "Minimum fraction of contour pixels that must exceed the Artifact "
            "probability threshold for a nucleus to be removed (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="/exported_data",
        help="Path of the probability dataset inside each H5 file (default: /exported_data).",
    )
    parser.add_argument(
        "--rbc-dataset-path",
        default="/exported_data",
        help="Path of the RBC probability dataset inside each RBC H5 file (default: /exported_data).",
    )
    parser.add_argument(
        "--rbc-channel",
        type=int,
        default=0,
        help="Index of the RBC class channel in the RBC H5 dataset (default: 0).",
    )
    parser.add_argument(
        "--rbc-threshold",
        type=float,
        default=0.9,
        help="RBC probability threshold for bbox-mean filtering (default: 0.9).",
    )
    parser.add_argument(
        "--rbc-fraction-threshold",
        type=float,
        default=0.3,
        help=(
            "Minimum fraction of contour pixels that must exceed the RBC "
            "probability threshold for a nucleus to be removed (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--macro-threshold",
        type=float,
        default=0.6,
        help="Per-pixel probability threshold for Macrophage mask (default: 0.6).",
    )
    parser.add_argument(
        "--macro-fraction-threshold",
        type=float,
        default=0.3,
        help=(
            "Minimum fraction of contour pixels that must exceed the Macrophage "
            "probability threshold to mark as Macrophage candidate (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--lymph-threshold",
        type=float,
        default=0.6,
        help="Per-pixel probability threshold for Lymphocyte mask (default: 0.6).",
    )
    parser.add_argument(
        "--lymph-fraction-threshold",
        type=float,
        default=0.3,
        help=(
            "Minimum fraction of contour pixels that must exceed the Lymphocyte "
            "probability threshold to mark as Lymphocyte candidate (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--reclass-top-k",
        type=int,
        default=3,
        help=(
            "Number of highest-probability classes (excluding current type) "
            "to consider when deciding Macrophage/Lymphocyte reclassification "
            "(default: 3)."
        ),
    )

    args = parser.parse_args()

    # If the deprecated --threshold is provided, prefer it unless the new flag is also set.
    # argparse does not tell us if --artifact-threshold was explicitly set, so we use:
    # - if --threshold is not None: override artifact_threshold
    if args.threshold is not None:
        args.artifact_threshold = float(args.threshold)

    ensure_dir(args.out_json_dir)
    ensure_dir(args.out_typeprob_dir)

    base_map = build_base_json_map(args.json_dir)
    typeprob_map = build_typeprob_json_map(args.typeprob_dir)
    artifact_h5_map = build_h5_map(args.h5_dir)
    rbc_h5_map: Dict[str, str] = {}
    if args.rbc_h5_dir:
        rbc_h5_map = build_h5_map(args.rbc_h5_dir)
    macro_h5_map: Dict[str, str] = {}
    if getattr(args, "macro_h5_dir", None):
        macro_h5_map = build_h5_map(args.macro_h5_dir)
    lymph_h5_map: Dict[str, str] = {}
    if getattr(args, "lymph_h5_dir", None):
        lymph_h5_map = build_h5_map(args.lymph_h5_dir)

    base_stems = set(base_map.keys())
    typeprob_stems = set(typeprob_map.keys())
    artifact_h5_stems = set(artifact_h5_map.keys())

    common_stems = sorted(base_stems & typeprob_stems & artifact_h5_stems)

    print(f"Base JSON count       : {len(base_stems)}")
    print(f"Typeprob JSON count   : {len(typeprob_stems)}")
    print(f"Artifact H5 count      : {len(artifact_h5_stems)}")
    if args.rbc_h5_dir:
        print(f"RBC H5 directory       : {args.rbc_h5_dir}")
        print(f"RBC threshold          : {args.rbc_threshold}")
    print(f"Stems with all three  : {len(common_stems)}")
    print(f"Artifact threshold     : {args.artifact_threshold}")
    print()

    summary_rows: List[str] = []
    summary_rows.append(
        "stem,total_nuclei,removed_nuclei,kept_nuclei,removed_nuclei_artifacts,removed_nuclei_rbc,removed_nuclei_both"
    )

    for stem in common_stems:
        base_path = base_map[stem]
        typeprob_path = typeprob_map[stem]
        artifact_h5_path = artifact_h5_map[stem]
        rbc_h5_path = rbc_h5_map.get(stem) if rbc_h5_map else None
        macro_h5_path = macro_h5_map.get(stem) if macro_h5_map else None
        lymph_h5_path = lymph_h5_map.get(stem) if lymph_h5_map else None

        print("=" * 80)
        print(f"STEM: {stem}")
        print(f"  base JSON   : {base_path}")
        print(f"  typeprob JSON: {typeprob_path}")
        print(f"  Artifact H5 : {artifact_h5_path}")
        if rbc_h5_path:
            print(f"  RBC H5      : {rbc_h5_path}")
        else:
            print("  RBC H5      : (none for this stem)")
        if macro_h5_path:
            print(f"  Macro H5    : {macro_h5_path}")
        if lymph_h5_path:
            print(f"  Lymph H5    : {lymph_h5_path}")

        try:
            base_data = load_base_json(base_path)
            typeprob_data = load_typeprob_json(typeprob_path)
            artifact_map = load_prob_map(
                artifact_h5_path,
                dataset_path=args.dataset_path,
                channel=args.artifact_channel,
            )
            rbc_map = None
            if rbc_h5_path:
                rbc_map = load_prob_map(
                    rbc_h5_path,
                    dataset_path=args.rbc_dataset_path,
                    channel=args.rbc_channel,
                )
            macro_map = None
            if macro_h5_path:
                macro_map = load_prob_map(
                    macro_h5_path,
                    dataset_path=args.dataset_path,
                    channel=args.artifact_channel,
                )
            lymph_map = None
            if lymph_h5_path:
                lymph_map = load_prob_map(
                    lymph_h5_path,
                    dataset_path=args.dataset_path,
                    channel=args.artifact_channel,
                )
        except Exception as e:
            print(f"  [ERROR] Skipping stem due to load error: {e}")
            continue

        nuc_dict: Dict[str, dict] = base_data["nuc"]
        # Work with nucleus IDs in numeric order so that reindexing after
        # filtering produces a clean sequence 1..N with preserved ordering.
        try:
            all_ids = sorted(nuc_dict.keys(), key=lambda s: int(s))
        except Exception:
            # Fallback to insertion order if IDs are not numeric strings
            all_ids = list(nuc_dict.keys())
        total_nuclei = len(all_ids)
        removed_ids: List[str] = []
        removed_by_artifact: List[str] = []
        removed_by_rbc: List[str] = []
        removed_by_both: List[str] = []

        print(f"  total nuclei in base JSON: {total_nuclei}")

        for nid in all_ids:
            info = nuc_dict[nid]
            bbox = info.get("bbox")
            contour = info.get("contour")
            if bbox is None:
                # No bbox; keep nucleus
                continue

            # Artifact decision: contour-based fraction of pixels above threshold
            artifact_mean = compute_bbox_mean(artifact_map, bbox)
            if contour is not None:
                artifact_fraction = compute_fraction_above_threshold_in_contour(
                    artifact_map,
                    contour,
                    args.artifact_threshold,
                )
                artifact_hit = artifact_fraction > args.artifact_fraction_threshold
            else:
                # Fallback: use bbox-mean if contour is missing
                artifact_fraction = 0.0
                artifact_hit = artifact_mean > args.artifact_threshold

            # RBC decision: contour-based fraction of pixels above threshold
            if rbc_map is not None and contour is not None:
                rbc_fraction = compute_fraction_above_threshold_in_contour(
                    rbc_map,
                    contour,
                    args.rbc_threshold,
                )
                rbc_hit = rbc_fraction > args.rbc_fraction_threshold
            else:
                rbc_fraction = 0.0
                rbc_hit = False

            # Remove nucleus if it overlaps high-probability Artifact or RBC
            if artifact_hit or rbc_hit:
                removed_ids.append(nid)
                if artifact_hit:
                    removed_by_artifact.append(nid)
                if rbc_hit:
                    removed_by_rbc.append(nid)
                if artifact_hit and rbc_hit:
                    removed_by_both.append(nid)

        kept_nuclei = total_nuclei - len(removed_ids)
        print(f"  nuclei removed total: {len(removed_ids)}")
        print(f"    removed by artifacts: {len(removed_by_artifact)}")
        print(f"    removed by rbc      : {len(removed_by_rbc)}")
        if rbc_map is not None:
            print(f"    removed by both     : {len(removed_by_both)}")
        print(f"  nuclei kept: {kept_nuclei}")

        # Build cleaned base 'nuc' dict with REINDEXED nucleus IDs:
        # remaining nuclei are renumbered sequentially from 1..K while
        # preserving their original (numeric) ordering.
        kept_ids_in_order = [nid for nid in all_ids if nid not in removed_ids]
        id_remap: Dict[str, str] = {
            old_id: str(new_idx)
            for new_idx, old_id in enumerate(kept_ids_in_order, start=1)
        }

        cleaned_nuc: Dict[str, dict] = {}
        for old_id, new_id in id_remap.items():
            cleaned_nuc[new_id] = nuc_dict[old_id]

        base_data_cleaned = dict(base_data)
        base_data_cleaned["nuc"] = cleaned_nuc

        # Build cleaned typeprob dict:
        #   - drop removed IDs
        #   - apply the same ID remapping so that keys stay aligned with base JSON.
        if isinstance(typeprob_data, dict):
            typeprob_cleaned: Dict[str, object] = {}
            for old_id, new_id in id_remap.items():
                if old_id in typeprob_data:
                    typeprob_cleaned[new_id] = typeprob_data[old_id]
        else:
            print("  [WARN] typeprob JSON is not a dict; leaving it unchanged.")
            typeprob_cleaned = typeprob_data

        # ------------------------------------------------------------------
        # Reclassification using Macrophage and Lymphocyte masks + typeprob
        # ------------------------------------------------------------------
        MACRO_CLASS = 3
        LYMPH_CLASS = 4

        if isinstance(typeprob_cleaned, dict) and (macro_map is not None or lymph_map is not None):
            for nid, nuc_info in cleaned_nuc.items():
                # Ensure we have probabilities for this nucleus
                probs = typeprob_cleaned.get(nid)
                if not isinstance(probs, (list, tuple)) or len(probs) < max(MACRO_CLASS, LYMPH_CLASS) + 1:
                    continue

                # Current nucleus type (any class 0-6 can be reclassified)
                current_type = nuc_info.get("type")

                contour = nuc_info.get("contour")
                if contour is None:
                    continue

                macro_flag = False
                lymph_flag = False

                if macro_map is not None:
                    macro_fraction = compute_fraction_above_threshold_in_contour(
                        macro_map,
                        contour,
                        args.macro_threshold,
                    )
                    macro_flag = macro_fraction > args.macro_fraction_threshold

                if lymph_map is not None:
                    lymph_fraction = compute_fraction_above_threshold_in_contour(
                        lymph_map,
                        contour,
                        args.lymph_threshold,
                    )
                    lymph_flag = lymph_fraction > args.lymph_fraction_threshold

                if not (macro_flag or lymph_flag):
                    continue

                # Build candidates excluding current type and find top-K
                candidates = [(i, float(p)) for i, p in enumerate(probs) if i != current_type]
                candidates.sort(key=lambda t: t[1], reverse=True)
                k = max(1, int(getattr(args, "reclass_top_k", 3)))
                topk = [cls for cls, p in candidates[:k]]

                in_top_macro = MACRO_CLASS in topk
                in_top_lymph = LYMPH_CLASS in topk

                # Apply decision rules
                new_type = current_type
                if macro_flag and not lymph_flag:
                    if in_top_macro:
                        new_type = MACRO_CLASS
                elif lymph_flag and not macro_flag:
                    if in_top_lymph:
                        new_type = LYMPH_CLASS
                elif macro_flag and lymph_flag:
                    if in_top_macro and in_top_lymph:
                        # Both in top-K: pick the higher probability
                        if probs[LYMPH_CLASS] > probs[MACRO_CLASS]:
                            new_type = LYMPH_CLASS
                        else:
                            new_type = MACRO_CLASS
                    elif in_top_macro:
                        new_type = MACRO_CLASS
                    elif in_top_lymph:
                        new_type = LYMPH_CLASS

                nuc_info["type"] = new_type

        # Write outputs
        out_base_path = os.path.join(args.out_json_dir, os.path.basename(base_path))
        out_typeprob_name = os.path.basename(typeprob_path)
        out_typeprob_path = os.path.join(args.out_typeprob_dir, out_typeprob_name)

        with open(out_base_path, "w") as f:
            json.dump(base_data_cleaned, f, indent=2)

        with open(out_typeprob_path, "w") as f:
            json.dump(typeprob_cleaned, f, indent=2)

        print(f"  wrote cleaned base JSON    -> {out_base_path}")
        print(f"  wrote cleaned typeprob JSON -> {out_typeprob_path}")

        summary_rows.append(
            f"{stem},{total_nuclei},{len(removed_ids)},{kept_nuclei},{len(removed_by_artifact)},{len(removed_by_rbc)},{len(removed_by_both)}"
        )

    # Write summary CSV in out-json-dir
    summary_path = os.path.join(args.out_json_dir, "filter_summary.csv")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_rows))

    print()
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()

