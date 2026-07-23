#!/usr/bin/env python
"""
Match ground-truth CSV nuclei to HoVer-Net JSON contours and label each GT row.

For each GT row (class, x, y), first test whether the centroid itself lies in a
predicted contour (point-in-polygon). If not, test neighbor pixels in a disk
centered at (x, y) with configurable radius in pixels
(--containment-radius, default 4). If several contours contain candidate points,
the hit with the smallest polygon area is used (common tie-break for overlaps).

Output columns:
  - class
  - x
  - y
  - hovernet_class
  - match_status ("correct", "mismatched", or "unmatched")
  - nuc_id
  - type_prob

Status semantics:
  - correct: centroid or neighbor pixel inside chosen contour and mapped
             pred class matches GT
  - mismatched: neighborhood hit exists but mapped pred class differs
  - unmatched: no neighborhood pixel inside any contour

Strict matching (default) pred type (int) -> GT letter mapping:
  1,2 -> t; 6 -> f; 4 -> l; 0,3,5 -> o

Loose "tumor/lymphocyte/other" matching:
  - 1,2 must match t
  - 4 must match l
  - 0,3,5,6 can match either o or f

JSON layout: expects either {"nuc": {"1": {...}, ...}, ...} or a flat dict of
id -> nucleus objects with "contour" and "type".

Usage:
    python verify_hovernet_gt_json_matching.py \\
        --json-dir "/mnt/j/HandE/results/latticea_test_data/pred_scn/json" \\
        --gt-csv-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915" \\
        --out-dir "/mnt/j/HandE/results/latticea_test_data/gt_celllabels_mpp025_from04915_matched"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


PRED_TYPE_TO_GT: Dict[int, str] = {
    0: "o",
    1: "t",
    2: "t",
    3: "o",
    4: "l",
    5: "o",
    6: "f",
}

# Loose matching: accepted GT letters by pred type.
PRED_TYPE_TO_GT_ALLOWED_LOOSE: Dict[int, Tuple[str, ...]] = {
    0: ("o", "f"),
    1: ("t",),
    2: ("t",),
    3: ("o", "f"),
    4: ("l",),
    5: ("o", "f"),
    6: ("o", "f"),
}

# HoVer-Net type index -> human-readable class (column D).
PRED_TYPE_LABEL: Dict[int, str] = {
    0: "Undefined",
    1: "Epithelium (PD-L1 low and Ki67 low)",
    2: "Epithelium (PD-L1 hi or Ki67 hi)",
    3: "Macrophage",
    4: "Lymphocyte",
    5: "Vascular",
    6: "Fibroblast/Stroma",
}


def hovernet_class_label(pred_type: int) -> str:
    if pred_type in PRED_TYPE_LABEL:
        return PRED_TYPE_LABEL[pred_type]
    return f"Unknown type ({pred_type})"


def pnpoly(x: float, y: float, verts: Sequence[Sequence[float]]) -> bool:
    """Point-in-polygon (ray casting), WRF pnpoly-style; skips horizontal edges."""
    n = len(verts)
    if n < 3:
        return False
    c = False
    j = n - 1
    for i in range(n):
        xi, yi = float(verts[i][0]), float(verts[i][1])
        xj, yj = float(verts[j][0]), float(verts[j][1])
        if yj == yi:
            j = i
            continue
        if (yi > y) != (yj > y):
            xinters = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xinters:
                c = not c
        j = i
    return c


def polygon_area(verts: Sequence[Sequence[float]]) -> float:
    a = 0.0
    n = len(verts)
    if n < 3:
        return float("inf")
    for i in range(n):
        x1, y1 = float(verts[i][0]), float(verts[i][1])
        x2, y2 = float(verts[(i + 1) % n][0]), float(verts[(i + 1) % n][1])
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def disk_offsets(radius: float) -> List[Tuple[float, float]]:
    """
    Integer pixel offsets within a disk of given radius, ordered by distance.
    The first offset is always (0, 0).
    """
    if radius <= 0:
        return [(0.0, 0.0)]
    r = int(radius)
    r2 = radius * radius
    out: List[Tuple[float, float]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                out.append((float(dx), float(dy)))
    out.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[1], p[0]))
    return out


def disk_offsets_without_center(radius: float) -> List[Tuple[float, float]]:
    """Neighbor offsets only (excludes (0,0)), ordered by distance."""
    return [(dx, dy) for dx, dy in disk_offsets(radius) if dx != 0.0 or dy != 0.0]


def load_nuclei_dict(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    root = payload
    nuc = root.get("nuc")
    if isinstance(nuc, dict) and nuc:
        return nuc
    flat = {
        k: v
        for k, v in root.items()
        if isinstance(v, dict) and "contour" in v and "type" in v
    }
    if not flat:
        raise ValueError('JSON missing "nuc" object and no nucleus entries found at top level')
    return flat


@dataclass(frozen=True)
class PredNucleus:
    nucleus_id: str
    contour: List[List[float]]
    pred_type: int
    mapped_gt: Optional[str]
    area: float
    type_prob: str


def parse_pred_nuclei(nuclei: Mapping[str, object]) -> List[PredNucleus]:
    out: List[PredNucleus] = []
    for nid, raw in nuclei.items():
        if not isinstance(raw, dict):
            continue
        contour = raw.get("contour")
        ptype = raw.get("type")
        if not isinstance(contour, list) or not contour:
            continue
        try:
            pred_type = int(ptype)
        except (TypeError, ValueError):
            continue
        raw_type_prob = raw.get("type_prob", "")
        if isinstance(raw_type_prob, (dict, list)):
            type_prob = json.dumps(raw_type_prob, separators=(",", ":"))
        else:
            type_prob = "" if raw_type_prob is None else str(raw_type_prob)
        area = polygon_area(contour)
        out.append(
            PredNucleus(
                nucleus_id=str(nid),
                contour=contour,  # type: ignore[arg-type]
                pred_type=pred_type,
                mapped_gt=PRED_TYPE_TO_GT.get(pred_type),
                area=area,
                type_prob=type_prob,
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
    # Fast path: centroid-only check first.
    hits: List[PredNucleus] = [n for n in nuclei if pnpoly(x, y, n.contour)]
    if hits:
        hits.sort(key=lambda n: (n.area, n.nucleus_id))
        return hits[0]

    # Fallback: only if centroid misses, probe neighborhood pixels.
    if not neighbor_offsets:
        return None
    hits = []
    for n in nuclei:
        if any(pnpoly(x + dx, y + dy, n.contour) for dx, dy in neighbor_offsets):
            hits.append(n)
    if not hits:
        return None
    hits.sort(key=lambda n: (n.area, n.nucleus_id))
    return hits[0]


def row_looks_like_gt_header(row: Sequence[str]) -> bool:
    if len(row) < 3:
        return False
    a, b = row[0].strip().lower(), row[1].strip().lower()
    if a in {"class", "label", "gt"} and b in {"x", "col", "cx", "x_px"}:
        return True
    return False


def match_status_for_gt(
    gt_class: str,
    pred: Optional[PredNucleus],
    *,
    match_strategy: str = "strict",
) -> str:
    if pred is None:
        return "unmatched"
    gt = gt_class.strip().lower()
    if match_strategy == "strict":
        mapped = pred.mapped_gt
        if mapped is None or mapped != gt:
            return "mismatched"
        return "correct"
    if match_strategy == "loose_tlo":
        allowed = PRED_TYPE_TO_GT_ALLOWED_LOOSE.get(pred.pred_type)
        if allowed is None or gt not in allowed:
            return "mismatched"
        return "correct"
    raise ValueError(f"unknown match strategy: {match_strategy}")


def iter_stems(json_dir: str, gt_dir: str) -> List[str]:
    json_stems = {
        os.path.splitext(name)[0]
        for name in os.listdir(json_dir)
        if name.lower().endswith(".json")
    }
    csv_stems = {
        os.path.splitext(name)[0]
        for name in os.listdir(gt_dir)
        if name.lower().endswith(".csv")
    }
    return sorted(json_stems & csv_stems)


def process_pair(
    json_path: str,
    gt_csv_path: str,
    out_csv_path: str,
    *,
    match_strategy: str = "strict",
    containment_radius: float = 4.0,
    verbose: bool = False,
) -> Tuple[int, int, int, int, Dict[str, Tuple[int, int, int, int]]]:
    """
    Returns counts: (n_rows, n_correct, n_mismatched, n_unmatched, per_gt_class_counts)

    per_gt_class_counts maps gt letter -> (total, correct, mismatched, unmatched)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    nuclei_raw = load_nuclei_dict(payload)
    nuclei = parse_pred_nuclei(nuclei_raw)
    neighbor_offsets = disk_offsets_without_center(containment_radius)

    with open(gt_csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    has_header = bool(rows) and row_looks_like_gt_header(rows[0])
    data_rows = rows
    if has_header:
        data_rows = rows[1:]

    out_rows: List[List[str]] = [[
        "class",
        "x",
        "y",
        "hovernet_class",
        "match_status",
        "nuc_id",
        "type_prob",
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
        if gt_key not in per_class:
            per_class[gt_key] = [0, 0, 0, 0]  # total, correct, mismatched, unmatched
        try:
            x = float(cells[1])
            y = float(cells[2])
        except ValueError:
            if verbose:
                print(f"skip non-numeric row in {gt_csv_path}: {row}", file=sys.stderr)
            continue

        pred = pick_containing_nucleus(
            x, y, nuclei, neighbor_offsets=neighbor_offsets
        )
        status = match_status_for_gt(gt_class, pred, match_strategy=match_strategy)
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

        nuc_label = hovernet_class_label(pred.pred_type) if pred is not None else ""
        nuc_id = pred.nucleus_id if pred is not None else ""
        type_prob = pred.type_prob if pred is not None else ""
        out_cells = [cells[0], cells[1], cells[2], nuc_label, status, nuc_id, type_prob]
        out_rows.append(out_cells)

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(out_rows)

    n_data = n_correct + n_mismatched + n_unmatched
    per_class_out: Dict[str, Tuple[int, int, int, int]] = {
        k: (v[0], v[1], v[2], v[3]) for k, v in per_class.items()
    }
    return n_data, n_correct, n_mismatched, n_unmatched, per_class_out


def _safe_accuracy(n_correct: int, n_total: int) -> float:
    if n_total <= 0:
        return 0.0
    return float(n_correct) / float(n_total)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json-dir", required=True, help="Directory with HoVer-Net JSON per tile")
    p.add_argument("--gt-csv-dir", required=True, help="Directory with GT CSV per tile")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write augmented CSVs (same basename as GT)",
    )
    p.add_argument(
        "--only-stem",
        action="append",
        default=[],
        help="Process only this stem (basename without .csv); may be repeated",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--match-strategy",
        choices=("strict", "loose_tlo"),
        default="strict",
        help=(
            "Matching rule to use: strict (default) keeps original mapping; "
            "loose_tlo uses tumor/lymphocyte/other logic where o/f are merged."
        ),
    )
    p.add_argument(
        "--containment-radius",
        type=float,
        default=4.0,
        help=(
            "Radius in pixels around each GT centroid used for contour containment "
            "matching (default: 4)."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    stems = iter_stems(args.json_dir, args.gt_csv_dir)
    if args.only_stem:
        allow = set(args.only_stem)
        stems = [s for s in stems if s in allow]
        missing = allow.difference(stems)
        for s in sorted(missing):
            print(f"warning: stem {s!r} not in both directories", file=sys.stderr)

    if not stems:
        print("error: no overlapping JSON/CSV stems found", file=sys.stderr)
        return 2

    totals = [0, 0, 0, 0]
    totals_by_class: Dict[str, List[int]] = {}
    for stem in stems:
        json_path = os.path.join(args.json_dir, f"{stem}.json")
        gt_path = os.path.join(args.gt_csv_dir, f"{stem}.csv")
        out_path = os.path.join(args.out_dir, f"{stem}.csv")
        try:
            counts = process_pair(
                json_path,
                gt_path,
                out_path,
                match_strategy=args.match_strategy,
                containment_radius=args.containment_radius,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"error processing {stem}: {e}", file=sys.stderr)
            return 1
        for i in range(4):
            totals[i] += counts[i]
        per_cls = counts[4]
        for cls, t in per_cls.items():
            if cls not in totals_by_class:
                totals_by_class[cls] = [0, 0, 0, 0]
            totals_by_class[cls][0] += t[0]
            totals_by_class[cls][1] += t[1]
            totals_by_class[cls][2] += t[2]
            totals_by_class[cls][3] += t[3]
        if args.verbose:
            print(
                f"{stem}: rows={counts[0]} correct={counts[1]} "
                f"mismatched={counts[2]} unmatched={counts[3]} -> {out_path}"
            )

    # Write per-class + overall accuracy summary CSV.
    summary_csv_path = os.path.join(args.out_dir, "match_accuracy_summary.csv")
    os.makedirs(os.path.dirname(summary_csv_path) or ".", exist_ok=True)
    class_order = ["t", "l", "f", "o"]
    extra_classes = sorted(c for c in totals_by_class.keys() if c and c not in set(class_order))
    ordered_classes = [c for c in class_order if c in totals_by_class] + extra_classes
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["class", "total", "correct", "mismatched", "unmatched", "accuracy"])
        for cls in ordered_classes:
            t = totals_by_class.get(cls, [0, 0, 0, 0])
            acc = _safe_accuracy(t[1], t[0])
            w.writerow([cls, t[0], t[1], t[2], t[3], f"{acc:.6f}"])
        overall_acc = _safe_accuracy(totals[1], totals[0])
        w.writerow(["TOTAL", totals[0], totals[1], totals[2], totals[3], f"{overall_acc:.6f}"])

    print(
        "summary: "
        f"tiles={len(stems)} rows={totals[0]} correct={totals[1]} "
        f"mismatched={totals[2]} unmatched={totals[3]}"
    )
    print(f"accuracy summary written to: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
