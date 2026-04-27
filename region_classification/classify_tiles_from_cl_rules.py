#!/usr/bin/env python
"""
Rule-based tile categories from QuPath *_cl.png maps.

Uses a hierarchical rule tree:
  1) tumour_lep (strict): global ignore + tumor band (same idea as before)
  2) bg: global high ignore + low tumor + low tumor patchiness across grid cells
  3) margin: directional contrast (max of LR/TB/strip) and/or tumor patchiness;
     necrosis-high alone only margin if a weak directional cue is present
  4) tumour_inv: uniform ignore field + low max directional ignore contrast
  5) margin (fallback)

Grid features match extract_cl_grid_features.py (default 3×3). Use --legacy-global-only
to restore the older whole-tile-only three-step rules.

Input: all *_cl.png under --tiles-root (find_cl_pngs), or only tiles in --labeled-json
(tumour_scar omitted; same list as extract_cl_grid_features).

Requirements: pillow, numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Please install Pillow: pip install pillow") from e

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_labeled_tile_groups import resolve_cl_png
from analyze_qupath_cl_maps import find_cl_pngs
from extract_cl_grid_features import (
    count_palette_in_flat_rgb,
    extract_grid_features,
    grid_metrics_for_rules,
    load_labeled_tile_jobs,
)


def tile_id_from_cl_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_cl"):
        return stem[: -len("_cl")]
    return stem


def directional_ignore_max(gm: Dict[str, float]) -> float:
    """Max edge/strip imbalance for ignore: LR, TB, left-third vs right-third, top vs bottom."""
    return max(
        gm["contrast_lr_ignore"],
        gm["contrast_tb_ignore"],
        gm["contrast_strip_lr_ignore"],
        gm["contrast_strip_tb_ignore"],
    )


def directional_tumor_max(gm: Dict[str, float]) -> float:
    """Same as directional_ignore_max for tumor class."""
    return max(
        gm["contrast_lr_tumor"],
        gm["contrast_tb_tumor"],
        gm["contrast_strip_lr_tumor"],
        gm["contrast_strip_tb_tumor"],
    )


def global_palette_fractions(arr: np.ndarray) -> tuple[float, float]:
    """(frac_ignore, frac_tumor) over all palette pixels in RGB uint8 HxWx3."""
    flat = arr.reshape(-1, 3)
    counts, _unk = count_palette_in_flat_rgb(flat)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0, 0.0
    return float(counts[5]) / total, float(counts[0]) / total


def classify_tile_global_only(
    frac_ignore: float,
    frac_tumor: float,
    *,
    lep_ignore_gt: float,
    lep_tumor_min: float,
    lep_tumor_max: float,
    bg_ignore_gte: float,
    bg_tumor_lte: float,
    inv_margin_ignore_split: float,
) -> str:
    if frac_ignore > lep_ignore_gt and lep_tumor_min <= frac_tumor <= lep_tumor_max:
        return "tumour_lep"
    if frac_ignore >= bg_ignore_gte and frac_tumor <= bg_tumor_lte:
        return "bg"
    if frac_ignore <= inv_margin_ignore_split:
        return "tumour_inv"
    return "margin"


def classify_tile_with_grid(
    frac_ignore: float,
    frac_tumor: float,
    gm: Dict[str, float],
    *,
    lep_ignore_gt: float,
    lep_tumor_min: float,
    lep_tumor_max: float,
    bg_ignore_gte: float,
    bg_tumor_lte: float,
    bg_max_tumor_spread: float,
    margin_lr_ignore_gte: float,
    margin_nec_spread_gte: float,
    margin_tumor_spread_gte: float,
    margin_lr_tumor_gte: float,
    margin_nec_aux_dir_ignore_gte: float,
    margin_nec_aux_dir_tumor_gte: float,
    inv_max_ignore_spread: float,
    inv_max_lr_ignore: float,
    inv_margin_ignore_split: float,
) -> str:
    fi, ft = frac_ignore, frac_tumor
    dir_ig = directional_ignore_max(gm)
    dir_tu = directional_tumor_max(gm)

    if fi > lep_ignore_gt and lep_tumor_min <= ft <= lep_tumor_max:
        return "tumour_lep"

    if (
        fi >= bg_ignore_gte
        and ft <= bg_tumor_lte
        and gm["cellspread_tumor_maxmin"] < bg_max_tumor_spread
    ):
        return "bg"

    nec_hot = gm["cellspread_necrosis_maxmin"] >= margin_nec_spread_gte
    dir_hot = dir_ig >= margin_lr_ignore_gte
    tumor_hot = (
        gm["cellspread_tumor_maxmin"] >= margin_tumor_spread_gte
        and dir_tu >= margin_lr_tumor_gte
    )
    nec_margin = nec_hot and (
        dir_ig >= margin_nec_aux_dir_ignore_gte
        or dir_tu >= margin_nec_aux_dir_tumor_gte
    )

    if dir_hot or tumor_hot or nec_margin:
        return "margin"

    if (
        gm["cellspread_ignore_maxmin"] <= inv_max_ignore_spread
        and dir_ig <= inv_max_lr_ignore
    ):
        return "tumour_inv"

    if fi <= inv_margin_ignore_split:
        return "tumour_inv"
    return "margin"


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Classify tiles from CL maps using compositional + grid rules."
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=Path(
            "/mnt/c/Apps/QuPath-v0.6.0-Windows/projects/JN_HandE_QuPath/tiles_manual"
        ),
        help="tiles_manual root (JN_TS_* subfolders; only *_cl.png).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=here / "tile_categories_cl_rules.json",
        help="Output path for category JSON.",
    )
    parser.add_argument(
        "--labeled-json",
        type=Path,
        default=None,
        help=(
            "If set, only classify tiles listed here (tumour_scar skipped → 87 tiles). "
            "Uses resolve_cl_png under --tiles-root."
        ),
    )
    parser.add_argument(
        "--legacy-global-only",
        action="store_true",
        help="Use only whole-tile ignore/tumor + inv/margin split (no grid).",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=3,
        help="Grid rows for spatial rules (ignored with --legacy-global-only).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=3,
        help="Grid cols for spatial rules (ignored with --legacy-global-only).",
    )
    parser.add_argument(
        "--tile-size-mm2",
        type=float,
        default=2.0,
        help="Metadata only.",
    )
    parser.add_argument(
        "--lep-ignore-gt",
        type=float,
        default=0.23,
        help="Strict lep: frac_ignore > this.",
    )
    parser.add_argument(
        "--lep-tumor-min",
        type=float,
        default=0.06,
        help="Strict lep: min global tumor fraction.",
    )
    parser.add_argument(
        "--lep-tumor-max",
        type=float,
        default=0.12,
        help="Strict lep: max global tumor fraction.",
    )
    parser.add_argument(
        "--bg-ignore-gte",
        type=float,
        default=0.30,
        help="bg: global frac_ignore >= this.",
    )
    parser.add_argument(
        "--bg-tumor-lte",
        type=float,
        default=0.07,
        help="bg: global frac_tumor <= this.",
    )
    parser.add_argument(
        "--bg-max-tumor-spread",
        type=float,
        default=0.055,
        help="bg: cellspread_tumor_maxmin must be < this (patchiness gate).",
    )
    parser.add_argument(
        "--margin-lr-ignore-gte",
        type=float,
        default=0.10,
        help=(
            "margin: max(LR,TB,strip LR,strip TB) ignore contrast >= this triggers margin."
        ),
    )
    parser.add_argument(
        "--margin-nec-spread-gte",
        type=float,
        default=0.32,
        help=(
            "necrosis patchiness threshold; alone does not force margin—see "
            "--margin-nec-aux-dir-*."
        ),
    )
    parser.add_argument(
        "--margin-tumor-spread-gte",
        type=float,
        default=0.10,
        help="margin: with margin_lr_tumor_gte, tumor patchiness triggers margin.",
    )
    parser.add_argument(
        "--margin-lr-tumor-gte",
        type=float,
        default=0.085,
        help=(
            "margin: paired with margin_tumor_spread_gte; uses max directional tumor contrast."
        ),
    )
    parser.add_argument(
        "--margin-nec-aux-dir-ignore-gte",
        type=float,
        default=0.06,
        help=(
            "If necrosis spread is high, still require at least this max directional "
            "ignore contrast (or --margin-nec-aux-dir-tumor-gte for tumor) for margin."
        ),
    )
    parser.add_argument(
        "--margin-nec-aux-dir-tumor-gte",
        type=float,
        default=0.04,
        help="Companion to --margin-nec-aux-dir-ignore-gte for necrosis-only margin path.",
    )
    parser.add_argument(
        "--inv-max-ignore-spread",
        type=float,
        default=0.16,
        help="tumour_inv: cellspread_ignore_maxmin <= this (uniform ignore across cells).",
    )
    parser.add_argument(
        "--inv-max-lr-ignore",
        type=float,
        default=0.065,
        help=(
            "tumour_inv: max(LR,TB,strip) ignore contrast <= this (paired with ignore spread)."
        ),
    )
    parser.add_argument(
        "--inv-margin-ignore-split",
        type=float,
        default=0.25,
        help="Final tie-break: tumour_inv if global ignore <= this else margin.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N files (0 = off).",
    )
    args = parser.parse_args()

    if args.grid_rows < 1 or args.grid_cols < 1:
        raise SystemExit("--grid-rows and --grid-cols must be >= 1")

    root: Path = args.tiles_root
    if not root.is_dir():
        raise SystemExit(f"Tiles root does not exist: {root}")

    if args.labeled_json is not None:
        lj = args.labeled_json
        if not lj.is_file():
            raise SystemExit(f"Labeled JSON not found: {lj}")
        jobs = load_labeled_tile_jobs(lj)
        cl_paths: List[Path] = []
        missing: List[str] = []
        for _group, labeled_id in jobs:
            p = resolve_cl_png(root, labeled_id)
            if p is None:
                missing.append(labeled_id)
                continue
            cl_paths.append(p)
        if missing:
            print(f"[WARN] {len(missing)} labeled tile(s) missing CL map under {root}")
            for m in missing[:15]:
                print(f"    {m}")
            if len(missing) > 15:
                print(f"    ... and {len(missing) - 15} more")
        if not cl_paths:
            raise SystemExit("No CL paths resolved from labeled JSON")
    else:
        cl_paths = find_cl_pngs(root)
        if not cl_paths:
            raise SystemExit(f"No *_cl.png found under {root}")

    groups: Dict[str, List[str]] = {
        "bg": [],
        "margin": [],
        "tumour_inv": [],
        "tumour_lep": [],
    }

    thresholds: Dict[str, Any] = {
        "mode": "legacy_global_only" if args.legacy_global_only else "grid_rules",
        "labeled_json": str(args.labeled_json) if args.labeled_json else None,
        "restrict_to_labeled_list": args.labeled_json is not None,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "lep_ignore_gt": args.lep_ignore_gt,
        "lep_tumor_min": args.lep_tumor_min,
        "lep_tumor_max": args.lep_tumor_max,
        "bg_ignore_gte": args.bg_ignore_gte,
        "bg_tumor_lte": args.bg_tumor_lte,
        "inv_margin_ignore_split": args.inv_margin_ignore_split,
    }
    if not args.legacy_global_only:
        thresholds.update(
            {
                "rule_order": [
                    "1 tumour_lep: global ignore>t and tumor in [min,max]",
                    "2 bg: global bg gates AND cellspread_tumor_maxmin < bg_max_tumor_spread",
                    "3 margin: dir_ignore OR tumor patch+dir_tumor OR (necrosis AND aux dir)",
                    "4 tumour_inv: low ignore spread AND low max directional ignore contrast",
                    "5 tumour_inv if global ignore <= split else margin",
                ],
                "bg_max_tumor_spread": args.bg_max_tumor_spread,
                "margin_lr_ignore_gte": args.margin_lr_ignore_gte,
                "margin_nec_spread_gte": args.margin_nec_spread_gte,
                "margin_tumor_spread_gte": args.margin_tumor_spread_gte,
                "margin_lr_tumor_gte": args.margin_lr_tumor_gte,
                "margin_nec_aux_dir_ignore_gte": args.margin_nec_aux_dir_ignore_gte,
                "margin_nec_aux_dir_tumor_gte": args.margin_nec_aux_dir_tumor_gte,
                "inv_max_ignore_spread": args.inv_max_ignore_spread,
                "inv_max_lr_ignore": args.inv_max_lr_ignore,
            }
        )
    else:
        thresholds["rule_order"] = [
            "1 tumour_lep (global)",
            "2 bg (global)",
            "3 tumour_inv if ignore <= split else margin",
        ]

    n_ok = 0
    for i, p in enumerate(cl_paths, start=1):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
            fi, ft = global_palette_fractions(arr)
            if args.legacy_global_only:
                label = classify_tile_global_only(
                    fi,
                    ft,
                    lep_ignore_gt=args.lep_ignore_gt,
                    lep_tumor_min=args.lep_tumor_min,
                    lep_tumor_max=args.lep_tumor_max,
                    bg_ignore_gte=args.bg_ignore_gte,
                    bg_tumor_lte=args.bg_tumor_lte,
                    inv_margin_ignore_split=args.inv_margin_ignore_split,
                )
            else:
                fracs, _, _ = extract_grid_features(
                    arr, args.grid_rows, args.grid_cols
                )
                gm = grid_metrics_for_rules(fracs)
                label = classify_tile_with_grid(
                    fi,
                    ft,
                    gm,
                    lep_ignore_gt=args.lep_ignore_gt,
                    lep_tumor_min=args.lep_tumor_min,
                    lep_tumor_max=args.lep_tumor_max,
                    bg_ignore_gte=args.bg_ignore_gte,
                    bg_tumor_lte=args.bg_tumor_lte,
                    bg_max_tumor_spread=args.bg_max_tumor_spread,
                    margin_lr_ignore_gte=args.margin_lr_ignore_gte,
                    margin_nec_spread_gte=args.margin_nec_spread_gte,
                    margin_tumor_spread_gte=args.margin_tumor_spread_gte,
                    margin_lr_tumor_gte=args.margin_lr_tumor_gte,
                    margin_nec_aux_dir_ignore_gte=args.margin_nec_aux_dir_ignore_gte,
                    margin_nec_aux_dir_tumor_gte=args.margin_nec_aux_dir_tumor_gte,
                    inv_max_ignore_spread=args.inv_max_ignore_spread,
                    inv_max_lr_ignore=args.inv_max_lr_ignore,
                    inv_margin_ignore_split=args.inv_margin_ignore_split,
                )
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
            continue

        tid = tile_id_from_cl_path(p)
        groups[label].append(tid)
        n_ok += 1

        if args.progress_every and i % args.progress_every == 0:
            print(f"  processed {i} / {len(cl_paths)} ...")

    for k in groups:
        groups[k].sort()

    n_total = sum(len(v) for v in groups.values())
    payload: Dict[str, Any] = {
        "metadata": {
            "tile_size_mm2": args.tile_size_mm2,
            "description": (
                "Tile categories from CL rules (metadata.thresholds); no tumour_scar. "
                "Tile ids are *_cl.png stems without _cl."
            ),
            "total_tiles": n_total,
            "groups": {g: len(groups[g]) for g in groups},
            "cl_files_seen": len(cl_paths),
            "cl_files_ok": n_ok,
            "labeled_tiles_missing_cl": len(missing)
            if args.labeled_json is not None
            else None,
            "thresholds": thresholds,
        },
        **groups,
    }

    out = args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {out} ({n_ok} tiles)")


if __name__ == "__main__":
    main()
