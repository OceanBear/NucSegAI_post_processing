#!/usr/bin/env python
"""
Quantify pixel-class distribution across QuPath pixel classification maps (*_cl.png).

Expected layout:
  <tiles_manual_root>/<JN_TS_XXX>/*_cl.png

Each PNG is RGB (7921x7921); pixels use one of 6 palette colors (exact match).

Classes (QuPath legend):
  0: Tumor       (#c80000)
  1: Stroma      (#96c896)
  2: Immune      (#a05aa0)
  3: Necrosis    (#323232)
  4: Other       (#ffc800)
  5: Ignore      (#b4b4b4)

Outputs (default under this script's directory):
  - region_stats.csv
  - region_stats_pie.png

Requirements: pillow, numpy, matplotlib
  pip install pillow numpy matplotlib
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Please install Pillow: pip install pillow") from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e

import matplotlib.patheffects as pe

# (class_id, display_name, (R, G, B))
CLASS_SPECS: List[Tuple[int, str, Tuple[int, int, int]]] = [
    (0, "Tumor", (0xC8, 0x00, 0x00)),
    (1, "Stroma", (0x96, 0xC8, 0x96)),
    (2, "Immune cells", (0xA0, 0x5A, 0xA0)),
    (3, "Necrosis", (0x32, 0x32, 0x32)),
    (4, "Other", (0xFF, 0xC8, 0x00)),
    (5, "Ignore", (0xB4, 0xB4, 0xB4)),
]

RGB_TO_CLASS: Dict[Tuple[int, int, int], int] = {spec[2]: spec[0] for spec in CLASS_SPECS}


def find_cl_pngs(root: Path) -> List[Path]:
    paths: List[Path] = []
    for wsi_dir in sorted(root.iterdir()):
        if not wsi_dir.is_dir():
            continue
        paths.extend(sorted(wsi_dir.glob("*_cl.png")))
    return paths


def count_pixels_in_image(path: Path) -> Tuple[np.ndarray, int]:
    """
    Return per-class counts for this image (length 6) and unknown pixel count.
    """
    counts = np.zeros(6, dtype=np.int64)
    unknown = 0

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}: {path}")

    flat = arr.reshape(-1, 3).astype(np.uint8)
    # Pack as single uint32 key for vectorized lookup via Python dict is slow;
    # exact color match per class in loop is fine for 6 classes × N pixels in C.
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


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Aggregate pixel counts from QuPath *_cl.png maps."
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
        "--out-dir",
        type=Path,
        default=here,
        help="Directory for CSV and pie chart (default: region_classification/).",
    )
    args = parser.parse_args()

    root: Path = args.tiles_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise SystemExit(f"Tiles root does not exist or is not a directory: {root}")

    cl_files = find_cl_pngs(root)
    if not cl_files:
        raise SystemExit(f"No *_cl.png files found under {root}")

    total_counts = np.zeros(6, dtype=np.int64)
    total_unknown = 0
    n_files = 0

    for p in cl_files:
        try:
            c, unk = count_pixels_in_image(p)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
            continue
        total_counts += c
        total_unknown += unk
        n_files += 1
        if n_files % 50 == 0:
            print(f"  processed {n_files} files ...")

    grand_total = int(total_counts.sum() + total_unknown)
    print(f"Files processed: {n_files} / {len(cl_files)}")
    print(f"Total pixels (all images): {grand_total}")
    print(f"Unknown / non-palette pixels: {total_unknown}")

    # CSV
    csv_path = out_dir / "region_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class_id",
                "class_name",
                "hex_color",
                "pixel_count",
                "fraction_of_all_pixels",
                "fraction_of_labeled_pixels",
            ]
        )
        labeled = int(total_counts.sum())
        denom_label = labeled if labeled > 0 else 1
        denom_all = grand_total if grand_total > 0 else 1
        for cls_id, name, rgb in CLASS_SPECS:
            cnt = int(total_counts[cls_id])
            hex_c = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            w.writerow(
                [
                    cls_id,
                    name,
                    hex_c,
                    cnt,
                    f"{cnt / denom_all:.10f}",
                    f"{cnt / denom_label:.10f}",
                ]
            )
        w.writerow(
            [
                "unknown",
                "non_palette",
                "",
                total_unknown,
                f"{total_unknown / denom_all:.10f}",
                "",
            ]
        )

    print(f"Wrote {csv_path}")

    # Pie chart (6 classes only; optionally add unknown if > 0)
    labels = [spec[1] for spec in CLASS_SPECS]
    sizes = [int(total_counts[spec[0]]) for spec in CLASS_SPECS]
    colors_hex = [f"#{spec[2][0]:02x}{spec[2][1]:02x}{spec[2][2]:02x}" for spec in CLASS_SPECS]

    if total_unknown > 0:
        labels = labels + ["Unknown / other RGB"]
        sizes = sizes + [int(total_unknown)]
        colors_hex = colors_hex + ["#cccccc"]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors_hex,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
    )
    for t in autotexts:
        t.set_fontsize(9)
        # Add white outline so percentage text stays readable over slices.
        t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])
    ax.set_title(
        f"Region class pixel share ({n_files} classification maps)\n"
        f"Total pixels: {grand_total:,}"
    )
    fig.tight_layout()
    pie_path = out_dir / "region_stats_pie.png"
    fig.savefig(pie_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {pie_path}")


if __name__ == "__main__":
    main()
