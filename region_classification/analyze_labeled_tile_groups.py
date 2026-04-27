#!/usr/bin/env python
"""
Per-tile QuPath CL map features for labeled tiles (first 13 WSIs), aggregated by group.

Reuses CLASS_SPECS and count_pixels_in_image from analyze_qupath_cl_maps.py.

Expected tile layout:
  <tiles-root>/<JN_TS_XXX>/<tile_id>_cl.png

If that file is missing, falls back to the first match:
  <tiles-root>/<JN_TS_XXX>/*_tile_<x>_<y>_cl.png

Reads group lists from tile_categories_88_tiles.json (keys: bg, margin, tumour_inv,
tumour_lep, tumour_scar, ...).

Outputs (default: region_classification/):
  - labeled_tiles_per_tile_features.csv
  - labeled_tiles_group_summary.csv
  - labeled_tiles_group_boxplots.png

Requirements: pillow, numpy, matplotlib (same as analyze_qupath_cl_maps.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e

# Same directory as this script — import sibling module when run from repo root or here.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_qupath_cl_maps import CLASS_SPECS, count_pixels_in_image

# JN_TS_012_tumour_inv_tile_10912_14661 -> slide, category_token, x, y
_TILE_ID_RE = re.compile(
    r"^(JN_TS_\d+)_([a-z_]+)_tile_(\d+)_(\d+)$",
    re.IGNORECASE,
)


def parse_tile_id(tile_id: str) -> Tuple[str, str, int, int]:
    m = _TILE_ID_RE.match(tile_id.strip())
    if not m:
        raise ValueError(f"Tile id does not match expected pattern: {tile_id!r}")
    slide, category, xs, ys = m.groups()
    return slide, category.lower(), int(xs), int(ys)


def resolve_cl_png(tiles_root: Path, tile_id: str) -> Optional[Path]:
    slide, _cat, x, y = parse_tile_id(tile_id)
    wsi_dir = tiles_root / slide
    if not wsi_dir.is_dir():
        return None
    direct = wsi_dir / f"{tile_id}_cl.png"
    if direct.is_file():
        return direct
    pattern = f"*_tile_{x}_{y}_cl.png"
    matches = sorted(wsi_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer filename that contains full tile_id stem if possible
        for p in matches:
            if tile_id in p.stem:
                return p
        return matches[0]
    return None


def distribution_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) over the 6 palette classes using counts / sum(counts)."""
    total = float(np.sum(counts))
    if total <= 0:
        return float("nan")
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / total
        h -= p * math.log(p)
    return h


def load_labels_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_group_tiles(data: Dict[str, Any], skip_groups: frozenset[str]) -> List[Tuple[str, str]]:
    """Return list of (group_name, tile_id)."""
    rows: List[Tuple[str, str]] = []
    for key, value in data.items():
        if key == "metadata" or not isinstance(value, list):
            continue
        if key in skip_groups:
            continue
        for tile_id in value:
            if isinstance(tile_id, str):
                rows.append((key, tile_id))
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Per-tile CL features for labeled tiles, summarized by group."
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=Path(
            "/mnt/c/Apps/QuPath-v0.6.0-Windows/projects/JN_HandE_QuPath/tiles_manual"
        ),
        help="Root folder containing JN_TS_* subfolders with *_cl.png files.",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=here / "tile_categories_88_tiles.json",
        help="JSON with group keys and lists of tile ids.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here,
        help="Directory for CSV and figures.",
    )
    parser.add_argument(
        "--skip-groups",
        type=str,
        default="",
        help="Comma-separated group names to omit (e.g. tumour_scar).",
    )
    args = parser.parse_args()

    tiles_root: Path = args.tiles_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    skip = frozenset(
        g.strip() for g in args.skip_groups.split(",") if g.strip()
    )
    if not args.labels_json.is_file():
        raise SystemExit(f"Labels JSON not found: {args.labels_json}")
    if not tiles_root.is_dir():
        raise SystemExit(f"Tiles root does not exist: {tiles_root}")

    data = load_labels_json(args.labels_json)
    group_tiles = iter_group_tiles(data, skip)
    if not group_tiles:
        raise SystemExit("No tiles found in JSON (check skip-groups and structure).")

    class_names = [spec[1] for spec in CLASS_SPECS]
    per_tile_rows: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str]] = []

    for group, tile_id in group_tiles:
        cl_path = resolve_cl_png(tiles_root, tile_id)
        if cl_path is None:
            missing.append((group, tile_id))
            continue
        try:
            counts, unknown = count_pixels_in_image(cl_path)
        except Exception as e:
            print(f"[WARN] skip {cl_path}: {e}")
            missing.append((group, tile_id))
            continue

        total_px = int(np.sum(counts) + unknown)
        palette_px = int(np.sum(counts))
        denom_all = total_px if total_px > 0 else 1
        denom_pal = palette_px if palette_px > 0 else 1

        fracs_all = [int(counts[i]) / denom_all for i in range(6)]
        fracs_palette = [int(counts[i]) / denom_pal for i in range(6)]
        ent = distribution_entropy(counts)

        row: Dict[str, Any] = {
            "group": group,
            "tile_id": tile_id,
            "cl_png": str(cl_path),
            "total_pixels": total_px,
            "palette_pixels": palette_px,
            "unknown_pixels": int(unknown),
            "entropy_nats_palette": ent,
        }
        for i, name in enumerate(class_names):
            safe = name.lower().replace(" ", "_")
            row[f"count_{safe}"] = int(counts[i])
            row[f"frac_of_all_{safe}"] = fracs_all[i]
            row[f"frac_of_palette_{safe}"] = fracs_palette[i]
        per_tile_rows.append(row)

    if missing:
        print(f"[WARN] Missing or failed CL maps for {len(missing)} tiles:")
        for g, tid in missing[:20]:
            print(f"    {g}: {tid}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")

    if not per_tile_rows:
        raise SystemExit("No tiles were successfully processed.")

    # --- per-tile CSV ---
    per_tile_path = out_dir / "labeled_tiles_per_tile_features.csv"
    fieldnames = list(per_tile_rows[0].keys())
    with open(per_tile_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_tile_rows)
    print(f"Wrote {per_tile_path} ({len(per_tile_rows)} tiles)")

    # --- group summary ---
    groups = sorted({r["group"] for r in per_tile_rows})
    summary_path = out_dir / "labeled_tiles_group_summary.csv"
    frac_palette_keys = [f"frac_of_palette_{n.lower().replace(' ', '_')}" for n in class_names]

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        wo = csv.writer(f)
        header = (
            ["group", "n_tiles"]
            + [f"mean_{k}" for k in frac_palette_keys]
            + [f"std_{k}" for k in frac_palette_keys]
            + ["mean_entropy_nats", "std_entropy_nats", "mean_unknown_frac", "std_unknown_frac"]
        )
        wo.writerow(header)
        for g in groups:
            sub = [r for r in per_tile_rows if r["group"] == g]
            n = len(sub)
            means = []
            stds = []
            for k in frac_palette_keys:
                vals = np.array([r[k] for r in sub], dtype=np.float64)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if n > 1 else 0.0)
            ent_vals = np.array([r["entropy_nats_palette"] for r in sub], dtype=np.float64)
            unk_fracs = np.array(
                [r["unknown_pixels"] / max(r["total_pixels"], 1) for r in sub],
                dtype=np.float64,
            )
            wo.writerow(
                [g, n]
                + [f"{m:.10f}" for m in means]
                + [f"{s:.10f}" for s in stds]
                + [
                    f"{float(np.mean(ent_vals)):.10f}",
                    f"{float(np.std(ent_vals, ddof=1)) if n > 1 else 0.0:.10f}",
                    f"{float(np.mean(unk_fracs)):.10f}",
                    f"{float(np.std(unk_fracs, ddof=1)) if n > 1 else 0.0:.10f}",
                ]
            )
    print(f"Wrote {summary_path}")

    # --- boxplots: palette fractions + entropy by group ---
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes_flat = axes.ravel()
    for ax, k, title in zip(
        axes_flat[:6],
        frac_palette_keys,
        [f"Frac of palette\n({n})" for n in class_names],
    ):
        data_box = [[r[k] for r in per_tile_rows if r["group"] == g] for g in groups]
        ax.boxplot(data_box, labels=groups, showmeans=True)
        ax.set_ylabel("fraction")
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=35)
    ent_ax = axes_flat[6]
    ent_ax.boxplot(
        [[r["entropy_nats_palette"] for r in per_tile_rows if r["group"] == g] for g in groups],
        labels=groups,
        showmeans=True,
    )
    ent_ax.set_ylabel("entropy (nats)")
    ent_ax.set_title("Palette distribution entropy", fontsize=9)
    ent_ax.tick_params(axis="x", rotation=35)
    unk_ax = axes_flat[7]
    unk_ax.boxplot(
        [
            [r["unknown_pixels"] / max(r["total_pixels"], 1) for r in per_tile_rows if r["group"] == g]
            for g in groups
        ],
        labels=groups,
        showmeans=True,
    )
    unk_ax.set_ylabel("fraction of all pixels")
    unk_ax.set_title("Unknown / non-palette", fontsize=9)
    unk_ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Labeled tiles: CL class fractions and entropy by group", fontsize=12)
    fig.tight_layout()
    box_path = out_dir / "labeled_tiles_group_boxplots.png"
    fig.savefig(box_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {box_path}")


if __name__ == "__main__":
    main()
