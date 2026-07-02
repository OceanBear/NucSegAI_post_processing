#!/usr/bin/env python3
"""
Filter CSV centroid rows using ilastik probability masks.

For each CSV row with centroid (x, y), sample the centroid and neighbors in a
disk of configurable radius, then apply a 2-step rule:
  1) "hit" pixels are sampled pixels with probability > pixel-threshold
  2) remove row only if hit_fraction > fraction-threshold
where hit_fraction = (# hit pixels) / (# valid sampled pixels).

Expected CSV format:
  - Columns: class, x, y (at least 3 columns)
  - Optional header is preserved.

Expected H5 filename pattern:
  <stem>_Probabilities.h5
where <stem> matches <stem>.csv in --csv-dir.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import h5py  # type: ignore
import numpy as np  # type: ignore


def build_csv_map(csv_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in os.listdir(csv_dir):
        if name.lower().endswith(".csv"):
            stem = os.path.splitext(name)[0]
            out[stem] = os.path.join(csv_dir, name)
    return out


def build_h5_map(h5_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in os.listdir(h5_dir):
        low = name.lower()
        if not (low.endswith(".h5") or low.endswith(".hdf5")):
            continue
        stem = re.sub(r"_Probabilities\.h5$", "", name)
        stem = re.sub(r"\.hdf5$", "", stem, flags=re.IGNORECASE)
        out[stem] = os.path.join(h5_dir, name)
    return out


def load_prob_map(
    h5_path: str,
    *,
    dataset_path: str,
    channel: int,
    h5_layout: str,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if dataset_path not in f:
            raise KeyError(f"dataset {dataset_path!r} not found in {h5_path}")
        ds = f[dataset_path][...]

    arr = np.asarray(ds)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D array, got shape {arr.shape} in {h5_path}")
    if channel < 0 or channel >= arr.shape[2]:
        raise ValueError(
            f"channel {channel} out of range for shape {arr.shape} in {h5_path}"
        )

    # Keep coordinates in (Y, X, C) for downstream [y, x] indexing.
    if h5_layout == "yxc":
        arr_yx = arr
    elif h5_layout == "xyc":
        arr_yx = np.transpose(arr, (1, 0, 2))
    else:
        raise ValueError(f"unknown h5_layout: {h5_layout}")
    prob = arr_yx[:, :, channel].astype(np.float32)
    return prob


def row_looks_like_header(row: Sequence[str]) -> bool:
    if len(row) < 3:
        return False
    a = row[0].strip().lower()
    b = row[1].strip().lower()
    return a in {"class", "label", "gt"} and b in {"x", "col", "cx", "x_px"}


def disk_offsets(radius: float) -> List[Tuple[int, int]]:
    if radius <= 0:
        return [(0, 0)]
    r = int(radius)
    r2 = radius * radius
    out: List[Tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                out.append((dx, dy))
    out.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[1], p[0]))
    return out


def sample_prob(prob_map: np.ndarray, x: float, y: float) -> Optional[float]:
    h, w = prob_map.shape
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return None
    return float(prob_map[yi, xi])


def should_remove(
    prob_map: np.ndarray,
    x: float,
    y: float,
    *,
    pixel_threshold: float,
    fraction_threshold: float,
    offsets: Sequence[Tuple[int, int]],
) -> bool:
    valid = 0
    hit = 0
    for dx, dy in offsets:
        p = sample_prob(prob_map, x + dx, y + dy)
        if p is None:
            continue
        valid += 1
        if p > pixel_threshold:
            hit += 1
    if valid == 0:
        return False
    return (hit / valid) > fraction_threshold


def process_pair(
    csv_path: str,
    h5_path: str,
    out_csv_path: str,
    *,
    dataset_path: str,
    channel: int,
    h5_layout: str,
    pixel_threshold: float,
    fraction_threshold: float,
    radius: float,
    verbose: bool = False,
) -> Tuple[int, int]:
    prob_map = load_prob_map(
        h5_path,
        dataset_path=dataset_path,
        channel=channel,
        h5_layout=h5_layout,
    )
    offsets = disk_offsets(radius)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    has_header = bool(rows) and row_looks_like_header(rows[0])
    kept_rows: List[List[str]] = []
    if has_header:
        kept_rows.append(list(rows[0]))

    n_total = 0
    n_removed = 0

    for row in (rows[1:] if has_header else rows):
        if not row or all(not c.strip() for c in row):
            continue

        cells = list(row)
        while len(cells) < 3:
            cells.append("")

        try:
            x = float(cells[1])
            y = float(cells[2])
        except ValueError:
            # Keep malformed/non-numeric rows unchanged.
            kept_rows.append(cells)
            continue

        n_total += 1
        if should_remove(
            prob_map,
            x,
            y,
            pixel_threshold=pixel_threshold,
            fraction_threshold=fraction_threshold,
            offsets=offsets,
        ):
            n_removed += 1
            continue
        kept_rows.append(cells)

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(kept_rows)

    if verbose:
        print(
            f"{os.path.basename(csv_path)}: total={n_total} "
            f"removed={n_removed} kept={n_total - n_removed}"
        )
    return n_total, n_removed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", required=True, help="Directory with input CSV files.")
    p.add_argument(
        "--h5-dir",
        required=True,
        help="Directory with ilastik probability maps (*_Probabilities.h5).",
    )
    p.add_argument("--out-dir", required=True, help="Directory for filtered CSV files.")
    p.add_argument(
        "--dataset-path",
        default="/exported_data",
        help="Dataset path inside H5 (default: /exported_data).",
    )
    p.add_argument(
        "--h5-layout",
        choices=("yxc", "xyc"),
        default="yxc",
        help=(
            "Axis layout of H5 data. Use yxc for (Y, X, C) [default], "
            "xyc for (X, Y, C)."
        ),
    )
    p.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Mask channel index in H5 dataset (default: 0).",
    )
    p.add_argument(
        "--pixel-threshold",
        type=float,
        default=0.5,
        help="Per-pixel probability cutoff used to count hit pixels (default: 0.5).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="DEPRECATED alias for --pixel-threshold.",
    )
    p.add_argument(
        "--fraction-threshold",
        type=float,
        default=0.5,
        help=(
            "Remove row only if fraction of sampled pixels above pixel-threshold "
            "is greater than this value (default: 0.5)."
        ),
    )
    p.add_argument(
        "--radius",
        type=float,
        default=4.0,
        help="Neighbor search radius in pixels around centroid (default: 4).",
    )
    p.add_argument(
        "--only-stem",
        action="append",
        default=[],
        help="Process only this stem (basename without .csv); may be repeated.",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.threshold is not None:
        args.pixel_threshold = float(args.threshold)

    csv_map = build_csv_map(args.csv_dir)
    h5_map = build_h5_map(args.h5_dir)
    stems = sorted(set(csv_map) & set(h5_map))

    if args.only_stem:
        allow = set(args.only_stem)
        stems = [s for s in stems if s in allow]
        missing = allow.difference(stems)
        for s in sorted(missing):
            print(f"warning: stem {s!r} not available in both CSV/H5")

    if not stems:
        print("error: no overlapping CSV/H5 stems found")
        return 2

    total_rows = 0
    total_removed = 0
    removed_rows_by_tile: List[Tuple[str, int, int, int]] = []
    for stem in stems:
        csv_path = csv_map[stem]
        h5_path = h5_map[stem]
        out_csv_path = os.path.join(args.out_dir, f"{stem}.csv")
        try:
            n_total, n_removed = process_pair(
                csv_path,
                h5_path,
                out_csv_path,
                dataset_path=args.dataset_path,
                channel=args.channel,
                h5_layout=args.h5_layout,
                pixel_threshold=args.pixel_threshold,
                fraction_threshold=args.fraction_threshold,
                radius=args.radius,
                verbose=args.verbose,
            )
        except Exception as exc:
            print(f"error processing {stem}: {exc}")
            return 1
        total_rows += n_total
        total_removed += n_removed
        if n_removed > 0:
            removed_rows_by_tile.append((stem, n_removed, n_total - n_removed, n_total))

    summary_csv_path = os.path.join(args.out_dir, "removed_nuclei_summary.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["stem", "removed_nuclei", "kept_nuclei", "total_nuclei"])
        for stem, n_removed, n_kept, n_total in removed_rows_by_tile:
            w.writerow([stem, n_removed, n_kept, n_total])

    print(
        "summary: "
        f"tiles={len(stems)} rows={total_rows} removed={total_removed} "
        f"kept={total_rows - total_removed}"
    )
    print(f"removed-per-tile CSV: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
