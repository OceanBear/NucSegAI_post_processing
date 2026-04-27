#!/usr/bin/env python
"""
Compare GT cell labels vs DL cell labels CSVs per tile.

Both inputs are CSVs with first 3 columns as:
  A: class (t/f/l/o), B: x, C: y

For each GT row, this script looks for DL nuclei within --max-distance pixels.
Matching is GT-centric and radius-based (not exact coordinate matching):
  - if at least one DL nucleus of the same class is within radius -> correct
  - else if only different-class DL nuclei are within radius -> mismatched
  - else -> unmatched

Output CSV columns:
  A: gt_class
  B: x
  C: y
  D: dl_class
  E: match_status ("correct", "mismatched", "unmatched")

Usage:
  python compare_gt_dl_celllabels.py \
    --gt-csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915" \
    --dl-csv-dir "/mnt/j/HandE/results/latticea_test_data/dl_celllabels_mpp025_from04915" \
    --out-dir "/mnt/j/HandE/results/latticea_test_data/matched_gt_vs_dl" \
    --max-distance 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


VALID_CLASSES = {"t", "f", "l", "o"}


@dataclass
class CsvPoint:
    cls: str
    x_raw: str
    y_raw: str
    x: float
    y: float
    used: bool = False


def row_looks_like_header(row: Sequence[str]) -> bool:
    if len(row) < 3:
        return False
    a = row[0].strip().lower()
    b = row[1].strip().lower()
    return a in {"class", "label", "gt", "dl"} and b in {"x", "cx", "x_px", "col"}


def parse_csv_points(path: str, verbose: bool = False) -> List[CsvPoint]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if rows and row_looks_like_header(rows[0]):
        rows = rows[1:]

    points: List[CsvPoint] = []
    for row in rows:
        if not row or all(not c.strip() for c in row):
            continue
        cells = list(row)
        while len(cells) < 3:
            cells.append("")
        cls = cells[0].strip().lower()
        if cls not in VALID_CLASSES and verbose:
            print(f"warning: unexpected class {cls!r} in {path}", file=sys.stderr)
        try:
            x = float(cells[1])
            y = float(cells[2])
        except ValueError:
            if verbose:
                print(f"skip non-numeric row in {path}: {row}", file=sys.stderr)
            continue
        points.append(CsvPoint(cls=cls, x_raw=cells[1], y_raw=cells[2], x=x, y=y))
    return points


def pick_within_radius(dl_points: List[CsvPoint], gt_cls: str, x: float, y: float, max_distance: float) -> Optional[int]:
    """Pick nearest same-class candidate first; otherwise nearest different-class."""
    if max_distance <= 0:
        return None

    max_d2 = max_distance * max_distance
    same_best_i: Optional[int] = None
    same_best_d2 = max_d2
    any_best_i: Optional[int] = None
    any_best_d2 = max_d2

    for i, p in enumerate(dl_points):
        if p.used:
            continue
        dx = p.x - x
        dy = p.y - y
        d2 = dx * dx + dy * dy
        if d2 > max_d2:
            continue

        if any_best_i is None or d2 < any_best_d2:
            any_best_i = i
            any_best_d2 = d2

        if p.cls == gt_cls and (same_best_i is None or d2 < same_best_d2):
            same_best_i = i
            same_best_d2 = d2

    if same_best_i is not None:
        return same_best_i
    return any_best_i


def match_status(gt_cls: str, dl_cls: Optional[str]) -> str:
    if dl_cls is None:
        return "unmatched"
    return "correct" if gt_cls == dl_cls else "mismatched"


def iter_common_stems(gt_dir: str, dl_dir: str) -> List[str]:
    gt_stems = {os.path.splitext(n)[0] for n in os.listdir(gt_dir) if n.lower().endswith(".csv")}
    dl_stems = {os.path.splitext(n)[0] for n in os.listdir(dl_dir) if n.lower().endswith(".csv")}
    return sorted(gt_stems & dl_stems)


def process_pair(gt_csv_path: str, dl_csv_path: str, out_csv_path: str, max_distance: float, verbose: bool = False) -> Tuple[int, int, int, int]:
    gt_points = parse_csv_points(gt_csv_path, verbose=verbose)
    dl_points = parse_csv_points(dl_csv_path, verbose=verbose)

    out_rows: List[List[str]] = [["gt_class", "x", "y", "dl_class", "match_status"]]

    n_correct = 0
    n_mismatched = 0
    n_unmatched = 0

    for gt in gt_points:
        i = pick_within_radius(dl_points, gt.cls, gt.x, gt.y, max_distance=max_distance)

        dl_cls: Optional[str] = None
        if i is not None:
            dl_points[i].used = True
            dl_cls = dl_points[i].cls

        status = match_status(gt.cls, dl_cls)
        if status == "correct":
            n_correct += 1
        elif status == "mismatched":
            n_mismatched += 1
        else:
            n_unmatched += 1

        out_rows.append([gt.cls, gt.x_raw, gt.y_raw, dl_cls or "", status])

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(out_rows)

    total = n_correct + n_mismatched + n_unmatched
    return total, n_correct, n_mismatched, n_unmatched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-csv-dir", required=True, help="Directory of GT CSV files")
    ap.add_argument("--dl-csv-dir", required=True, help="Directory of DL CSV files")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-tile matched CSV")
    ap.add_argument(
        "--max-distance",
        type=float,
        default=5.0,
        help="Matching radius in pixels around each GT centroid (default: 5)",
    )
    ap.add_argument(
        "--only-stem",
        action="append",
        default=[],
        help="Process only this tile stem (repeatable)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    stems = iter_common_stems(args.gt_csv_dir, args.dl_csv_dir)
    if args.only_stem:
        keep = set(args.only_stem)
        stems = [s for s in stems if s in keep]
        missing = keep.difference(stems)
        for s in sorted(missing):
            print(f"warning: stem {s!r} not in both directories", file=sys.stderr)

    if not stems:
        print("error: no overlapping GT/DL csv stems found", file=sys.stderr)
        return 2

    totals = [0, 0, 0, 0]
    for stem in stems:
        gt_csv = os.path.join(args.gt_csv_dir, f"{stem}.csv")
        dl_csv = os.path.join(args.dl_csv_dir, f"{stem}.csv")
        out_csv = os.path.join(args.out_dir, f"{stem}.csv")
        try:
            cnt = process_pair(
                gt_csv,
                dl_csv,
                out_csv,
                max_distance=args.max_distance,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"error processing {stem}: {e}", file=sys.stderr)
            return 1

        for i in range(4):
            totals[i] += cnt[i]
        if args.verbose:
            print(
                f"{stem}: rows={cnt[0]} correct={cnt[1]} mismatched={cnt[2]} "
                f"unmatched={cnt[3]} -> {out_csv}"
            )

    print(
        "summary: "
        f"tiles={len(stems)} rows={totals[0]} correct={totals[1]} "
        f"mismatched={totals[2]} unmatched={totals[3]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

