#!/usr/bin/env python3
"""
Recall-only evaluation of CellViT++ Lizard predictions on partially labeled lattice-a tiles.

Ground truth: one CSV per tile with columns (class, x, y). Class letters:
  t = Tumor, l = Lymphocyte, f = Fibroblast, o = Other

Predictions: CellViT++ JSON per tile (default glob "*_cells.json") with Lizard types 0-5,
mapped to the same 4-class scheme as evaluate_lizard_to_old4.py.

A GT nucleus is a true positive when:
  1. Its (x, y) lies inside a predicted instance bounding box (centroid first; optional neighbor disk).
  2. The mapped prediction class equals the GT class.

Prediction bounding boxes in *_cells.json use (y, x) order:
  bbox = [[y_min, x_min], [y_max, x_max]]
GT CSV coordinates are (x, y).

Per-class recall = typed_TP / all_GT_nuclei_of_that_class.

Example:
  python evaluate_lizard_latticea_recall.py \\
      --pred-dir /mnt/j/HandE/results/latticea_test_data/cellvitpp_lizard/json \\
      --gt-csv-dir /mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915 \\
      --pred-glob "*_cells.json" \\
      --output-dir ./latticea_recall_results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate_lizard_to_old4 import (
    DEFAULT_PRED_STRIP_SUFFIXES,
    LIZARD_CLASS_NAMES,
    LIZARD_TO_OLD4,
    OLD4_CLASS_NAMES,
    build_file_map,
    extract_type_from_cell,
    iter_raw_cell_dicts,
    warn,
)

GT_LETTER_TO_OLD4: Dict[str, int] = {
    "t": 1,
    "l": 2,
    "f": 3,
    "o": 0,
}

GT_LETTER_NAMES: Dict[str, str] = {
    "t": "Tumor",
    "l": "Lymphocyte",
    "f": "Fibroblast",
    "o": "Other",
}

OLD4_TO_GT_LETTER: Dict[int, str] = {v: k for k, v in GT_LETTER_TO_OLD4.items()}


@dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True)
class PredNucleus:
    instance_id: str
    lizard_type: int
    mapped_old4: int
    bbox: BBox


def disk_offsets(radius: float) -> List[Tuple[float, float]]:
    if radius <= 0:
        return [(0.0, 0.0)]
    r = int(radius)
    r2 = radius * radius
    out: List[Tuple[float, float]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r2:
                out.append((float(dx), float(dy)))
    out.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[1], p[0]))
    return out


def disk_offsets_without_center(radius: float) -> List[Tuple[float, float]]:
    return [(dx, dy) for dx, dy in disk_offsets(radius) if dx != 0.0 or dy != 0.0]


def parse_bbox_yx(
    raw_bbox: object,
    *,
    source: str,
    instance_id: str,
) -> Optional[BBox]:
    """Parse bbox stored as [[y_min, x_min], [y_max, x_max]]."""
    if (
        not isinstance(raw_bbox, Sequence)
        or len(raw_bbox) != 2
        or not isinstance(raw_bbox[0], Sequence)
        or not isinstance(raw_bbox[1], Sequence)
        or len(raw_bbox[0]) != 2
        or len(raw_bbox[1]) != 2
    ):
        warn(f"skip invalid bbox in {source} instance {instance_id}: {raw_bbox!r}")
        return None

    try:
        y_min, x_min = float(raw_bbox[0][0]), float(raw_bbox[0][1])
        y_max, x_max = float(raw_bbox[1][0]), float(raw_bbox[1][1])
    except (TypeError, ValueError):
        warn(f"skip non-numeric bbox in {source} instance {instance_id}")
        return None

    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    if y_max < y_min or x_max < x_min:
        warn(f"skip degenerate bbox in {source} instance {instance_id}")
        return None

    return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def lizard_class_label(lizard_type: int) -> str:
    return LIZARD_CLASS_NAMES.get(lizard_type, f"Unknown ({lizard_type})")


def mapped_class_label(old4_type: int) -> str:
    return OLD4_CLASS_NAMES.get(old4_type, f"Unknown ({old4_type})")


def parse_pred_nuclei(json_path: Path) -> List[PredNucleus]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    out: List[PredNucleus] = []
    source = str(json_path)
    source = str(json_path)
    for instance_id, cell in iter_raw_cell_dicts(payload):
        bbox = parse_bbox_yx(cell.get("bbox"), source=source, instance_id=instance_id)
        if bbox is None:
            continue

        lizard_type = extract_type_from_cell(cell, expect_old4=False)
        if lizard_type is None:
            warn(f"skip instance {instance_id} in {json_path.name}: missing or unknown type")
            continue

        mapped = LIZARD_TO_OLD4.get(lizard_type)
        if mapped is None:
            warn(f"skip instance {instance_id} in {json_path.name}: unmapped lizard type {lizard_type}")
            continue

        out.append(
            PredNucleus(
                instance_id=instance_id,
                lizard_type=lizard_type,
                mapped_old4=mapped,
                bbox=bbox,
            )
        )
    return out


def pick_containing_nucleus(
    x: float,
    y: float,
    nuclei: Sequence[PredNucleus],
    *,
    neighbor_offsets: Sequence[Tuple[float, float]] = (),
) -> Optional[PredNucleus]:
    hits = [n for n in nuclei if n.bbox.contains(x, y)]
    if hits:
        hits.sort(key=lambda n: (n.bbox.area, n.instance_id))
        return hits[0]

    if not neighbor_offsets:
        return None

    hits = []
    for n in nuclei:
        if any(n.bbox.contains(x + dx, y + dy) for dx, dy in neighbor_offsets):
            hits.append(n)
    if not hits:
        return None
    hits.sort(key=lambda n: (n.bbox.area, n.instance_id))
    return hits[0]


def row_looks_like_gt_header(row: Sequence[str]) -> bool:
    if len(row) < 3:
        return False
    a, b = row[0].strip().lower(), row[1].strip().lower()
    return a in {"class", "label", "gt"} and b in {"x", "col", "cx", "x_px"}


def gt_letter_to_old4(gt_class: str) -> Optional[int]:
    key = gt_class.strip().lower()
    return GT_LETTER_TO_OLD4.get(key)


def match_status(gt_old4: int, pred: Optional[PredNucleus]) -> str:
    if pred is None:
        return "unmatched"
    if pred.mapped_old4 == gt_old4:
        return "correct"
    return "mismatched"


def process_pair(
    json_path: Path,
    gt_csv_path: Path,
    out_csv_path: Path,
    *,
    containment_radius: float,
    verbose: bool,
) -> Tuple[int, int, int, int, Dict[str, Tuple[int, int, int, int]]]:
    nuclei = parse_pred_nuclei(json_path)
    neighbor_offsets = disk_offsets_without_center(containment_radius)

    with gt_csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    data_rows = rows[1:] if rows and row_looks_like_gt_header(rows[0]) else rows

    out_rows: List[List[str]] = [[
        "class",
        "x",
        "y",
        "lizard_class",
        "mapped_pred_class",
        "match_status",
        "pred_instance_id",
    ]]

    n_correct = n_mismatched = n_unmatched = 0
    per_class: Dict[str, List[int]] = {}

    for row in data_rows:
        if not row or all(not c.strip() for c in row):
            continue
        cells = list(row)
        while len(cells) < 3:
            cells.append("")

        gt_class = cells[0]
        gt_key = gt_class.strip().lower()
        gt_old4 = gt_letter_to_old4(gt_class)
        if gt_old4 is None:
            if verbose:
                warn(f"skip unknown GT class {gt_class!r} in {gt_csv_path.name}")
            continue

        if gt_key not in per_class:
            per_class[gt_key] = [0, 0, 0, 0]

        try:
            x = float(cells[1])
            y = float(cells[2])
        except ValueError:
            if verbose:
                warn(f"skip non-numeric row in {gt_csv_path.name}: {row}")
            continue

        pred = pick_containing_nucleus(x, y, nuclei, neighbor_offsets=neighbor_offsets)
        status = match_status(gt_old4, pred)
        per_class[gt_key][0] += 1
        if status == "correct":
            n_correct += 1
            per_class[gt_key][1] += 1
        elif status == "mismatched":
            n_mismatched += 1
            per_class[gt_key][2] += 1
        else:
            n_unmatched += 1
            per_class[gt_key][3] += 1

        lizard_label = lizard_class_label(pred.lizard_type) if pred is not None else ""
        mapped_label = mapped_class_label(pred.mapped_old4) if pred is not None else ""
        pred_id = pred.instance_id if pred is not None else ""
        out_rows.append([cells[0], cells[1], cells[2], lizard_label, mapped_label, status, pred_id])

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(out_rows)

    n_data = n_correct + n_mismatched + n_unmatched
    per_class_out = {k: (v[0], v[1], v[2], v[3]) for k, v in per_class.items()}
    return n_data, n_correct, n_mismatched, n_unmatched, per_class_out


def recall(n_tp: int, n_gt: int) -> float:
    return float(n_tp / n_gt) if n_gt else 0.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recall evaluation for CellViT++ Lizard on lattice-a GT dots.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with CellViT++ prediction JSON files.")
    parser.add_argument("--gt-csv-dir", type=Path, required=True, help="Directory with GT CSV files per tile.")
    parser.add_argument(
        "--pred-glob",
        default="*_cells.json",
        help='Glob for prediction files (default: "*_cells.json").',
    )
    parser.add_argument(
        "--pred-strip-suffixes",
        default=",".join(DEFAULT_PRED_STRIP_SUFFIXES),
        help="Comma-separated filename stem suffixes to strip when matching prediction files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./latticea_recall_results"), help="Output directory.")
    parser.add_argument(
        "--containment-radius",
        type=float,
        default=0.0,
        help="Optional neighbor-disk radius in pixels if centroid misses all bounding boxes (default: 0).",
    )
    parser.add_argument("--only-stem", action="append", default=[], help="Process only this tile stem; repeatable.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.pred_dir.is_dir():
        print(f"error: prediction directory does not exist: {args.pred_dir}", file=sys.stderr)
        return 1
    if not args.gt_csv_dir.is_dir():
        print(f"error: GT CSV directory does not exist: {args.gt_csv_dir}", file=sys.stderr)
        return 1

    pred_strip = tuple(s for s in args.pred_strip_suffixes.split(",") if s)
    pred_map = build_file_map(args.pred_dir, args.pred_glob, pred_strip)

    gt_map: Dict[str, Path] = {}
    for path in sorted(args.gt_csv_dir.glob("*.csv")):
        gt_map[path.stem] = path

    common = sorted(set(pred_map) & set(gt_map))
    if args.only_stem:
        allow = set(args.only_stem)
        common = [s for s in common if s in allow]
        for stem in sorted(allow - set(common)):
            warn(f"stem {stem!r} not found in both directories")

    pred_only = sorted(set(pred_map) - set(gt_map))
    gt_only = sorted(set(gt_map) - set(pred_map))
    for stem in gt_only:
        warn(f"missing prediction for GT stem {stem!r}")
    for stem in pred_only:
        warn(f"missing GT CSV for prediction stem {stem!r}")

    if not common:
        print("error: no overlapping prediction/GT CSV stems found", file=sys.stderr)
        return 1

    totals = [0, 0, 0, 0]
    totals_by_class: Dict[str, List[int]] = {}
    per_file_rows: List[Dict[str, object]] = []

    for stem in common:
        counts = process_pair(
            pred_map[stem],
            gt_map[stem],
            args.output_dir / f"{stem}.csv",
            containment_radius=args.containment_radius,
            verbose=args.verbose,
        )
        n_data, n_correct, n_mismatched, n_unmatched, per_cls = counts
        totals[0] += n_data
        totals[1] += n_correct
        totals[2] += n_mismatched
        totals[3] += n_unmatched

        for cls, t in per_cls.items():
            if cls not in totals_by_class:
                totals_by_class[cls] = [0, 0, 0, 0]
            for i in range(4):
                totals_by_class[cls][i] += t[i]

        per_file_rows.append({
            "file": stem,
            "n_gt": n_data,
            "typed_tp": n_correct,
            "mismatched": n_mismatched,
            "unmatched": n_unmatched,
            "recall": recall(n_correct, n_data),
        })
        if args.verbose:
            print(
                f"{stem}: gt={n_data} tp={n_correct} mismatched={n_mismatched} "
                f"unmatched={n_unmatched} recall={recall(n_correct, n_data):.4f}"
            )

    class_order = ["t", "l", "f", "o"]
    extra = sorted(c for c in totals_by_class if c not in class_order)
    ordered_classes = [c for c in class_order if c in totals_by_class] + extra

    per_class_metrics: Dict[str, Dict[str, object]] = {}
    for cls in ordered_classes:
        t = totals_by_class[cls]
        per_class_metrics[cls] = {
            "name": GT_LETTER_NAMES.get(cls, cls),
            "support_gt": t[0],
            "typed_tp": t[1],
            "mismatched": t[2],
            "unmatched": t[3],
            "recall": recall(t[1], t[0]),
        }

    summary = {
        "n_files": len(common),
        "n_gt_nuclei": totals[0],
        "typed_tp": totals[1],
        "mismatched": totals[2],
        "unmatched": totals[3],
        "overall_recall": recall(totals[1], totals[0]),
        "lizard_to_old4_mapping": {str(k): v for k, v in LIZARD_TO_OLD4.items()},
        "gt_letter_to_old4": GT_LETTER_TO_OLD4,
        "per_class_metrics": per_class_metrics,
        "missing_prediction_stems": gt_only,
        "missing_gt_stems": pred_only,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = args.output_dir / "summary_recall.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    summary_csv = args.output_dir / "recall_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["class", "name", "support_gt", "typed_tp", "mismatched", "unmatched", "recall"])
        for cls in ordered_classes:
            m = per_class_metrics[cls]
            writer.writerow([
                cls,
                m["name"],
                m["support_gt"],
                m["typed_tp"],
                m["mismatched"],
                m["unmatched"],
                f"{m['recall']:.6f}",
            ])
        writer.writerow([
            "TOTAL",
            "All",
            totals[0],
            totals[1],
            totals[2],
            totals[3],
            f"{summary['overall_recall']:.6f}",
        ])

    per_file_csv = args.output_dir / "per_file_recall.csv"
    with per_file_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "n_gt", "typed_tp", "mismatched", "unmatched", "recall"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in per_file_rows:
            writer.writerow({**row, "recall": f"{row['recall']:.6f}"})

    print(
        f"summary: tiles={len(common)} gt={totals[0]} typed_tp={totals[1]} "
        f"overall_recall={summary['overall_recall']:.4f}"
    )
    print(f"wrote {summary_json}")
    print(f"wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
