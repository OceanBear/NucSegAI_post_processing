#!/usr/bin/env python
"""
Per-tile regional features from QuPath *_cl.png maps on a fixed grid (e.g. 3×3, 4×4).

For each cell: palette fractions for all 6 CL classes (same legend as analyze_qupath_cl_maps).
Also writes contrast features: |mean(first column of cells) − mean(last column)| and
the same for first vs last row (left–right and top–bottom imbalance), per class;
plus strip contrasts |mean(left third of cols) − mean(right third)| and
|mean(top third of rows) − mean(bottom third)|.

Patchiness across grid cells (tumor, ignore, necrosis): max−min, sample std, and IQR of the
per-cell palette fractions (heterogeneity when imbalance is not aligned with LR/TB axes).

Cell index c{k} is row-major: k = row * grid_cols + col (c0 = top-left).

Default: only tiles listed in tile_categories_88_tiles.json (tumour_scar skipped → 87 tiles),
with columns labeled_tile_id, true_group, plus features. Writes a second CSV with mean/std
per true_group for every numeric feature (group-level summary).

Use --scan-all-cl-pngs to process every *_cl.png under --tiles-root (no labels / summary).

Uses the same RGB palette matching as analyze_qupath_cl_maps.py.

Requirements: pillow, numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Please install Pillow: pip install pillow") from e

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_qupath_cl_maps import CLASS_SPECS, find_cl_pngs
from analyze_labeled_tile_groups import resolve_cl_png

# Short names for CSV columns (stable order matches CLASS_SPECS)
CLASS_SHORT = ["tumor", "stroma", "immune_cells", "necrosis", "other", "ignore"]

# class_name -> index in CLASS_SHORT / fracs[..., j]
PATCHINESS_CLASS_INDICES: List[Tuple[str, int]] = [
    ("tumor", 0),
    ("ignore", 5),
    ("necrosis", 3),
]

GROUP_ORDER = ["bg", "margin", "tumour_inv", "tumour_lep"]


def load_labeled_tile_jobs(json_path: Path) -> List[Tuple[str, str]]:
    """Return (true_group, labeled_tile_id) for all tiles except tumour_scar."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    jobs: List[Tuple[str, str]] = []
    for group in GROUP_ORDER:
        if group not in data or not isinstance(data[group], list):
            continue
        for tid in data[group]:
            if isinstance(tid, str):
                jobs.append((group, tid))

    def sort_key(item: Tuple[str, str]) -> Tuple[int, str]:
        g, tid = item
        return (GROUP_ORDER.index(g) if g in GROUP_ORDER else 99, tid)

    jobs.sort(key=sort_key)
    return jobs


def feature_fieldnames(grid_rows: int, grid_cols: int) -> List[str]:
    n_cells = grid_rows * grid_cols
    names: List[str] = [
        "image_height",
        "image_width",
        "grid_rows",
        "grid_cols",
        "unknown_pixels_total",
    ]
    for idx in range(n_cells):
        for cname in CLASS_SHORT:
            names.append(f"c{idx}_{cname}")
    for cname in CLASS_SHORT:
        names.append(f"contrast_lr_{cname}")
        names.append(f"contrast_tb_{cname}")
        names.append(f"contrast_strip_lr_{cname}")
        names.append(f"contrast_strip_tb_{cname}")
    for cname, _j in PATCHINESS_CLASS_INDICES:
        names.append(f"cellspread_{cname}_maxmin")
        names.append(f"cellspread_{cname}_std")
        names.append(f"cellspread_{cname}_iqr")
    return names


def cell_patchiness_features(fracs: np.ndarray) -> Dict[str, str]:
    """
    Per class (tumor, ignore, necrosis): across all grid cells, max−min, std (ddof=1), IQR.
    fracs shape (grid_rows, grid_cols, 6).
    """
    flat = fracs.reshape(-1, fracs.shape[2])
    out: Dict[str, str] = {}
    n = flat.shape[0]
    for cname, j in PATCHINESS_CLASS_INDICES:
        vals = flat[:, j].astype(np.float64, copy=False)
        rng = float(np.max(vals) - np.min(vals)) if n else 0.0
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        if n:
            q75, q25 = np.percentile(vals, [75.0, 25.0])
            iqr = float(q75 - q25)
        else:
            iqr = 0.0
        out[f"cellspread_{cname}_maxmin"] = f"{rng:.10f}"
        out[f"cellspread_{cname}_std"] = f"{std:.10f}"
        out[f"cellspread_{cname}_iqr"] = f"{iqr:.10f}"
    return out


def compute_feature_row(
    arr: np.ndarray, gr: int, gc: int
) -> Tuple[Dict[str, str], np.ndarray, int]:
    """Return CSV value dict for feature columns only, fracs array, total_unknown."""
    fracs, _cell_u, total_unk = extract_grid_features(arr, gr, gc)
    row: Dict[str, str] = {
        "image_height": str(arr.shape[0]),
        "image_width": str(arr.shape[1]),
        "grid_rows": str(gr),
        "grid_cols": str(gc),
        "unknown_pixels_total": str(total_unk),
    }
    flat_idx = 0
    for ri in range(gr):
        for ci in range(gc):
            for j, cname in enumerate(CLASS_SHORT):
                row[f"c{flat_idx}_{cname}"] = f"{float(fracs[ri, ci, j]):.10f}"
            flat_idx += 1
    for j, cname in enumerate(CLASS_SHORT):
        row[f"contrast_lr_{cname}"] = f"{contrast_first_vs_last_col(fracs, j):.10f}"
        row[f"contrast_tb_{cname}"] = f"{contrast_first_vs_last_row(fracs, j):.10f}"
        row[f"contrast_strip_lr_{cname}"] = (
            f"{contrast_left_third_vs_right_third(fracs, j):.10f}"
        )
        row[f"contrast_strip_tb_{cname}"] = (
            f"{contrast_top_third_vs_bottom_third(fracs, j):.10f}"
        )
    row.update(cell_patchiness_features(fracs))
    return row, fracs, total_unk


def write_group_summary_csv(
    out_path: Path,
    rows: List[Dict[str, Any]],
    feature_cols: List[str],
) -> None:
    """One row per true_group: n_tiles, mean_*, std_* for each numeric feature."""
    by_g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g = r.get("true_group")
        if g:
            by_g[str(g)].append(r)

    summary_fields: List[str] = ["true_group", "n_tiles"]
    for col in feature_cols:
        summary_fields.append(f"mean_{col}")
        summary_fields.append(f"std_{col}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for g in GROUP_ORDER:
            if g not in by_g:
                continue
            grp_rows = by_g[g]
            n = len(grp_rows)
            out_row: Dict[str, str] = {"true_group": g, "n_tiles": str(n)}
            for col in feature_cols:
                vals = []
                for r in grp_rows:
                    v = r.get(col)
                    if v is None or v == "":
                        continue
                    try:
                        vals.append(float(v))
                    except ValueError:
                        continue
                if not vals:
                    out_row[f"mean_{col}"] = ""
                    out_row[f"std_{col}"] = ""
                else:
                    a = np.array(vals, dtype=np.float64)
                    out_row[f"mean_{col}"] = f"{float(np.mean(a)):.10f}"
                    out_row[f"std_{col}"] = (
                        f"{float(np.std(a, ddof=1)):.10f}" if n > 1 else "0.0"
                    )
            w.writerow(out_row)

    print(f"Wrote group summary {out_path}")


def axis_edges(length: int, n_parts: int) -> List[Tuple[int, int]]:
    """Partition [0, length) into n_parts contiguous half-open [start, end) intervals."""
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    if length < 1:
        raise ValueError("length must be >= 1")
    # Round boundaries so tiny images still get non-empty strips when possible.
    b = [round(i * length / n_parts) for i in range(n_parts + 1)]
    b[0] = 0
    b[-1] = length
    for i in range(1, len(b)):
        if b[i] < b[i - 1]:
            b[i] = b[i - 1]
    edges: List[Tuple[int, int]] = []
    for i in range(n_parts):
        s, e = b[i], b[i + 1]
        if e <= s:
            e = min(s + 1, length)
        edges.append((s, min(e, length)))
    return edges


def count_palette_in_flat_rgb(flat: np.ndarray) -> Tuple[np.ndarray, int]:
    """flat shape (N, 3) uint8 RGB. Returns counts[6], unknown."""
    counts = np.zeros(6, dtype=np.int64)
    if flat.size == 0:
        return counts, 0
    flat = flat.astype(np.uint8, copy=False)
    for cls_id, _name, rgb in CLASS_SPECS:
        m = (
            (flat[:, 0] == rgb[0])
            & (flat[:, 1] == rgb[1])
            & (flat[:, 2] == rgb[2])
        )
        counts[cls_id] = int(m.sum())
    total_assigned = int(counts.sum())
    unknown = int(flat.shape[0] - total_assigned)
    return counts, unknown


def cell_counts_and_fracs(
    arr: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Crop arr[y0:y1, x0:x1], return counts[6], unknown, fracs[6] over palette pixels in cell."""
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        z = np.zeros(6, dtype=np.float64)
        return np.zeros(6, dtype=np.int64), 0, z
    flat = patch.reshape(-1, 3)
    counts, unknown = count_palette_in_flat_rgb(flat)
    pal = float(np.sum(counts))
    if pal <= 0:
        return counts, unknown, np.zeros(6, dtype=np.float64)
    fracs = counts.astype(np.float64) / pal
    return counts, unknown, fracs


def tile_id_from_cl_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_cl"):
        return stem[: -len("_cl")]
    return stem


def extract_grid_features(
    arr: np.ndarray, grid_rows: int, grid_cols: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    arr: H×W×3 RGB uint8.

    Returns:
      fracs: (grid_rows, grid_cols, 6) palette fractions per cell
      cell_unknown: (grid_rows, grid_cols) unknown pixel count per cell
      total_unknown: whole-image unknown (sum over cells may differ if overlaps none — same as sum cell unknown)
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {arr.shape}")
    h, w = arr.shape[0], arr.shape[1]
    row_edges = axis_edges(h, grid_rows)
    col_edges = axis_edges(w, grid_cols)
    fracs = np.zeros((grid_rows, grid_cols, 6), dtype=np.float64)
    cell_unknown = np.zeros((grid_rows, grid_cols), dtype=np.int64)
    full_flat = arr.reshape(-1, 3)
    _, total_unknown = count_palette_in_flat_rgb(full_flat)

    for ri, (y0, y1) in enumerate(row_edges):
        for ci, (x0, x1) in enumerate(col_edges):
            counts, unk, f = cell_counts_and_fracs(arr, y0, y1, x0, x1)
            fracs[ri, ci] = f
            cell_unknown[ri, ci] = unk
    return fracs, cell_unknown, total_unknown


def contrast_first_vs_last_col(fracs: np.ndarray, class_idx: int) -> float:
    """abs(mean(frac[:,0,c]) - mean(frac[:,-1,c]))."""
    if fracs.shape[1] < 2:
        return 0.0
    a = float(np.mean(fracs[:, 0, class_idx]))
    b = float(np.mean(fracs[:, -1, class_idx]))
    return abs(a - b)


def contrast_first_vs_last_row(fracs: np.ndarray, class_idx: int) -> float:
    """abs(mean(frac[0,:,c]) - mean(frac[-1,:,c]))."""
    if fracs.shape[0] < 2:
        return 0.0
    a = float(np.mean(fracs[0, :, class_idx]))
    b = float(np.mean(fracs[-1, :, class_idx]))
    return abs(a - b)


def contrast_left_third_vs_right_third(fracs: np.ndarray, class_idx: int) -> float:
    """abs(mean(left third of cols) - mean(right third of cols)) per class."""
    n = fracs.shape[1]
    if n < 2:
        return 0.0
    k = max(1, n // 3)
    left = float(np.mean(fracs[:, :k, class_idx]))
    right = float(np.mean(fracs[:, n - k :, class_idx]))
    return abs(left - right)


def contrast_top_third_vs_bottom_third(fracs: np.ndarray, class_idx: int) -> float:
    """abs(mean(top third of rows) - mean(bottom third of rows)) per class."""
    m = fracs.shape[0]
    if m < 2:
        return 0.0
    k = max(1, m // 3)
    top = float(np.mean(fracs[:k, :, class_idx]))
    bot = float(np.mean(fracs[m - k :, :, class_idx]))
    return abs(top - bot)


def grid_metrics_for_rules(fracs: np.ndarray) -> Dict[str, float]:
    """
    Scalars for classify_tiles_from_cl_rules.py (3×3 or any grid).
    CLASS_SPECS order: tumor=0, necrosis=3, ignore=5.
    """
    j_tumor, j_necrosis, j_ignore = 0, 3, 5
    flat = fracs.reshape(-1, fracs.shape[2])

    def maxmin(j: int) -> float:
        v = flat[:, j]
        if v.size == 0:
            return 0.0
        return float(np.max(v) - np.min(v))

    def iqr(j: int) -> float:
        v = flat[:, j]
        if v.size == 0:
            return 0.0
        q75, q25 = np.percentile(v, [75.0, 25.0])
        return float(q75 - q25)

    return {
        "contrast_lr_ignore": contrast_first_vs_last_col(fracs, j_ignore),
        "contrast_lr_tumor": contrast_first_vs_last_col(fracs, j_tumor),
        "contrast_lr_necrosis": contrast_first_vs_last_col(fracs, j_necrosis),
        "contrast_tb_ignore": contrast_first_vs_last_row(fracs, j_ignore),
        "contrast_tb_tumor": contrast_first_vs_last_row(fracs, j_tumor),
        "contrast_tb_necrosis": contrast_first_vs_last_row(fracs, j_necrosis),
        "contrast_strip_lr_ignore": contrast_left_third_vs_right_third(
            fracs, j_ignore
        ),
        "contrast_strip_lr_tumor": contrast_left_third_vs_right_third(
            fracs, j_tumor
        ),
        "contrast_strip_lr_necrosis": contrast_left_third_vs_right_third(
            fracs, j_necrosis
        ),
        "contrast_strip_tb_ignore": contrast_top_third_vs_bottom_third(
            fracs, j_ignore
        ),
        "contrast_strip_tb_tumor": contrast_top_third_vs_bottom_third(
            fracs, j_tumor
        ),
        "contrast_strip_tb_necrosis": contrast_top_third_vs_bottom_third(
            fracs, j_necrosis
        ),
        "cellspread_tumor_maxmin": maxmin(j_tumor),
        "cellspread_ignore_maxmin": maxmin(j_ignore),
        "cellspread_necrosis_maxmin": maxmin(j_necrosis),
        "cellspread_ignore_iqr": iqr(j_ignore),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract per-cell CL palette fractions on a grid plus LR/TB contrasts."
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=Path(
            "/mnt/c/Apps/QuPath-v0.6.0-Windows/projects/JN_HandE_QuPath/tiles_manual"
        ),
        help="tiles_manual root (JN_TS_* subfolders with *_cl.png).",
    )
    parser.add_argument(
        "--labeled-json",
        type=Path,
        default=here / "tile_categories_88_tiles.json",
        help=(
            "Reference tile lists with true_group labels. "
            "Used unless --scan-all-cl-pngs is set (tumour_scar skipped → 87 tiles)."
        ),
    )
    parser.add_argument(
        "--scan-all-cl-pngs",
        action="store_true",
        help="Process every *_cl.png under --tiles-root (ignore --labeled-json; no group summary).",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=3,
        help="Number of row bands (e.g. 3 for 3×3 when cols=3).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=3,
        help="Number of column bands (e.g. 4 for a 4×4 grid when rows=4).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=here / "cl_grid_features_labeled.csv",
        help="Per-tile output CSV path.",
    )
    parser.add_argument(
        "--group-summary-csv",
        type=Path,
        default=None,
        help=(
            "Mean/std of each numeric feature per true_group. "
            "Default: <out-csv stem>_by_group_summary.csv (labeled mode only)."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N files (0 = off).",
    )
    args = parser.parse_args()

    if args.grid_rows < 1 or args.grid_cols < 1:
        raise SystemExit("--grid-rows and --grid-cols must be >= 1")

    root: Path = args.tiles_root
    if not root.is_dir():
        raise SystemExit(f"Tiles root does not exist: {root}")

    gr, gc = args.grid_rows, args.grid_cols
    feat_names = feature_fieldnames(gr, gc)

    labeled_mode = not args.scan_all_cl_pngs
    if labeled_mode:
        if not args.labeled_json.is_file():
            raise SystemExit(
                f"Labeled JSON not found: {args.labeled_json}\n"
                "Use --scan-all-cl-pngs to process all *_cl.png files instead."
            )
        jobs = load_labeled_tile_jobs(args.labeled_json)
        if not jobs:
            raise SystemExit(f"No labeled tiles in {args.labeled_json}")
        fieldnames = ["labeled_tile_id", "true_group", "tile_id", "cl_png"] + feat_names
    else:
        jobs = None
        fieldnames = ["tile_id", "cl_png"] + feat_names

    out_path: Path = args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collected: List[Dict[str, Any]] = []
    n_ok = 0
    n_skip = 0

    def process_one_path(p: Path, labeled_id: str = "", true_group: str = "") -> None:
        nonlocal n_ok, n_skip
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
            feat_part, _fracs, _tu = compute_feature_row(arr, gr, gc)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
            n_skip += 1
            return

        row: Dict[str, Any] = {
            "tile_id": tile_id_from_cl_path(p),
            "cl_png": str(p),
        }
        row.update(feat_part)
        if labeled_mode:
            row["labeled_tile_id"] = labeled_id
            row["true_group"] = true_group
        collected.append(row)
        n_ok += 1

    if labeled_mode:
        assert jobs is not None
        for i, (true_group, labeled_id) in enumerate(jobs, start=1):
            p = resolve_cl_png(root, labeled_id)
            if p is None:
                print(f"[WARN] missing CL map for {labeled_id}")
                n_skip += 1
                continue
            process_one_path(p, labeled_id=labeled_id, true_group=true_group)
            if args.progress_every and i % args.progress_every == 0:
                print(f"  processed {i} / {len(jobs)} ...")
    else:
        cl_paths = find_cl_pngs(root)
        if not cl_paths:
            raise SystemExit(f"No *_cl.png under {root}")
        for i, p in enumerate(cl_paths, start=1):
            process_one_path(p)
            if args.progress_every and i % args.progress_every == 0:
                print(f"  processed {i} / {len(cl_paths)} ...")

    if labeled_mode:
        collected.sort(
            key=lambda r: (
                GROUP_ORDER.index(r["true_group"])
                if r["true_group"] in GROUP_ORDER
                else 99,
                str(r["labeled_tile_id"]),
            )
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in collected:
            writer.writerow(row)

    print(f"Wrote {out_path} ({n_ok} tiles, grid {gr}×{gc})")
    if n_skip:
        print(f"Skipped / missing: {n_skip}")

    if labeled_mode and n_ok:
        summary_path = args.group_summary_csv
        if summary_path is None:
            summary_path = out_path.with_name(
                f"{out_path.stem}_by_group_summary{out_path.suffix}"
            )
        write_group_summary_csv(summary_path, collected, feat_names)


if __name__ == "__main__":
    main()
