#!/usr/bin/env python3
"""
Per-instance evaluation of CellViT++/SAM-H Lizard predictions against old 4-class GT.

Class mapping (Lizard 6-class -> old 4-class):
  0 Neutrophil          -> 0 Others
  1 Epithelial          -> 1 Tumor
  2 Lymphocyte          -> 2 Lymphocyte
  3 Plasma              -> 0 Others
  4 Eosinophil          -> 0 Others
  5 Connective tissue   -> 3 Fibroblast/Stroma

Old 4-class GT labels:
  0 Others, 1 Tumor, 2 Lymphocyte, 3 Fibroblast/Stroma

Matching:
  For each image/tile pair, GT and prediction cell contours are converted to polygons.
  Pairwise IoU is computed and one-to-one assignment uses the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment). Pairs with IoU >= --iou-threshold are
  detection true positives. Unmatched GT instances are detection false negatives;
  unmatched predictions are detection false positives.

Typed metrics:
  Among detection-matched pairs, typed TP counts pairs where GT type equals the
  mapped prediction type. Typed FP = unmatched predictions + matched pairs with
  wrong type. Typed FN = unmatched GT + matched pairs with wrong type.
  Classification accuracy among detected instances = typed_TP / detection_TP.

Example (GT and predictions may live in the same directory with different globs):
  python evaluate_lizard_to_old4.py \\
      --gt-dir /mnt/j/HandE/new_training_set/pred_4types/json \\
      --pred-dir /mnt/j/HandE/new_training_set/cellvitpp_lizard_vahadane_scn \\
      --gt-glob "*.json" \\
      --pred-glob "*_cells.json" \\
      --iou-threshold 0.3 \\
      --output-dir ./eval_results

If both GT and predictions are under the same folder:
  python evaluate_lizard_to_old4.py \\
      --gt-dir /mnt/j/HandE/new_training_set/cellvitpp_lizard_vahadane_scn \\
      --pred-dir /mnt/j/HandE/new_training_set/cellvitpp_lizard_vahadane_scn \\
      --gt-glob "<pattern for GT json files>" \\
      --pred-glob "*_cells.json" \\
      --iou-threshold 0.3 \\
      --output-dir ./eval_results

Requires: numpy, scipy, shapely (see requirements.txt in this folder).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# Editable class definitions and mapping
# ---------------------------------------------------------------------------

OLD4_CLASS_NAMES: Dict[int, str] = {
    0: "Others",
    1: "Tumor",
    2: "Lymphocyte",
    3: "Fibroblast/Stroma",
}

LIZARD_CLASS_NAMES: Dict[int, str] = {
    0: "Neutrophil",
    1: "Epithelial",
    2: "Lymphocyte",
    3: "Plasma",
    4: "Eosinophil",
    5: "Connective tissue",
}

# Lizard 6-class index -> old 4-class index
LIZARD_TO_OLD4: Dict[int, int] = {
    0: 0,  # Neutrophil -> Others
    1: 1,  # Epithelial -> Tumor
    2: 2,  # Lymphocyte -> Lymphocyte
    3: 2,  # Plasma -> Lymphocyte
    4: 0,  # Eosinophil -> Others
    5: 3,  # Connective tissue -> Fibroblast/Stroma
}

# GeoJSON / string label -> old 4-class (for GT files that store names)
NAME_TO_OLD4: Dict[str, int] = {
    "others": 0,
    "other": 0,
    "undefined": 0,
    "neutrophil": 0,
    "plasma": 0,
    "eosinophil": 0,
    "tumor": 1,
    "epithelial": 1,
    "epithelium": 1,
    "lymphocyte": 2,
    "fibroblast/stroma": 3,
    "fibroblast": 3,
    "stroma": 3,
    "connective tissue": 3,
}

# GeoJSON / string label -> Lizard 6-class
NAME_TO_LIZARD: Dict[str, int] = {v.lower(): k for k, v in LIZARD_CLASS_NAMES.items()}

DEFAULT_PRED_STRIP_SUFFIXES = ("_cells", "_cell_detection", "_typeprob")
DEFAULT_GT_STRIP_SUFFIXES: Tuple[str, ...] = ()


@dataclass
class CellInstance:
  """One cell/nucleus instance extracted from a JSON file."""

  instance_id: str
  contour: List[List[float]]
  type_id: int
  source_file: str
  polygon: Optional[BaseGeometry] = field(default=None, repr=False)


@dataclass
class MatchedPair:
  file_stem: str
  gt_instance_id: str
  pred_instance_id: str
  iou: float
  gt_type: int
  pred_type_original: int
  pred_type_mapped: int

  @property
  def type_match(self) -> bool:
    return self.gt_type == self.pred_type_mapped


@dataclass
class CountBundle:
  tp: int = 0
  fp: int = 0
  fn: int = 0

  def precision(self) -> float:
    denom = self.tp + self.fp
    return float(self.tp / denom) if denom else 0.0

  def recall(self) -> float:
    denom = self.tp + self.fn
    return float(self.tp / denom) if denom else 0.0

  def f1(self) -> float:
    p, r = self.precision(), self.recall()
    return float(2 * p * r / (p + r)) if (p + r) else 0.0

  def as_dict(self, prefix: str) -> Dict[str, float]:
    return {
      f"{prefix}_tp": self.tp,
      f"{prefix}_fp": self.fp,
      f"{prefix}_fn": self.fn,
      f"{prefix}_precision": self.precision(),
      f"{prefix}_recall": self.recall(),
      f"{prefix}_f1": self.f1(),
    }


def warn(msg: str) -> None:
  print(f"warning: {msg}", file=sys.stderr)


def normalize_label_name(name: str) -> str:
  return name.strip().lower().replace("_", " ")


def parse_type_value(raw: Any, *, expect_old4: bool) -> Optional[int]:
  """Parse integer or string type labels into a class index."""
  if raw is None:
    return None
  if isinstance(raw, bool):
    return None
  if isinstance(raw, int):
    if expect_old4 and raw in OLD4_CLASS_NAMES:
      return raw
    if not expect_old4 and raw in LIZARD_CLASS_NAMES:
      return raw
    # Allow numeric strings stored as int but out of declared range only if in range 0-5/0-3
    if expect_old4 and 0 <= raw <= 3:
      return raw
    if not expect_old4 and 0 <= raw <= 5:
      return raw
    return None
  if isinstance(raw, float) and raw.is_integer():
    return parse_type_value(int(raw), expect_old4=expect_old4)
  if isinstance(raw, str):
    key = normalize_label_name(raw)
    if expect_old4:
      if key.isdigit():
        return parse_type_value(int(key), expect_old4=True)
      return NAME_TO_OLD4.get(key)
    if key.isdigit():
      return parse_type_value(int(key), expect_old4=False)
    return NAME_TO_LIZARD.get(key)
  return None


def contour_from_geojson_geometry(geometry: Mapping[str, Any]) -> List[List[float]]:
  """Flatten GeoJSON Polygon / MultiPolygon to a single exterior ring when possible."""
  gtype = geometry.get("type")
  coords = geometry.get("coordinates")
  if not coords:
    return []

  if gtype == "Polygon":
    if not coords or not coords[0]:
      return []
    return [[float(x), float(y)] for x, y in coords[0]]

  if gtype == "MultiPolygon":
    # Use the largest polygon by vertex count as a simple heuristic.
    best: List[List[float]] = []
    for poly in coords:
      if poly and poly[0]:
        ring = [[float(x), float(y)] for x, y in poly[0]]
        if len(ring) > len(best):
          best = ring
    return best

  return []


def iter_raw_cell_dicts(payload: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
  """
  Yield (instance_id, cell_dict) from supported JSON layouts without assuming
  a single schema.
  """
  if isinstance(payload, list):
  # GeoJSON FeatureCollection as list, or bare feature list
    for i, item in enumerate(payload):
      if not isinstance(item, dict):
        continue
      if item.get("type") == "Feature" and isinstance(item.get("geometry"), dict):
        props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        geom = item["geometry"]
        contour = contour_from_geojson_geometry(geom)
        cell: Dict[str, Any] = {"contour": contour, "properties": props}
        if "classification" in props:
          cell["classification"] = props["classification"]
        feat_id = item.get("id")
        yield str(feat_id if feat_id is not None else i), cell
      elif "contour" in item or "type" in item:
        yield str(item.get("id", i)), item
    return

  if not isinstance(payload, dict):
    return

  if isinstance(payload.get("features"), list):
    yield from iter_raw_cell_dicts(payload["features"])
    return

  if isinstance(payload.get("cells"), list):
    for i, cell in enumerate(payload["cells"]):
      if isinstance(cell, dict):
        yield str(cell.get("id", i)), cell
    return

  nuc = payload.get("nuc")
  if isinstance(nuc, dict) and nuc:
    for nid, cell in nuc.items():
      if isinstance(cell, dict):
        yield str(nid), cell
    return

  # Flat dict of id -> nucleus objects
  for key, cell in payload.items():
    if key in {"mag", "wsi_metadata", "type_map", "metadata", "properties"}:
      continue
    if isinstance(cell, dict) and "contour" in cell:
      yield str(key), cell


def extract_type_from_cell(cell: Mapping[str, Any], *, expect_old4: bool) -> Optional[int]:
  if "type" in cell:
    parsed = parse_type_value(cell.get("type"), expect_old4=expect_old4)
    if parsed is not None:
      return parsed

  classification = cell.get("classification")
  if isinstance(classification, dict) and "name" in classification:
    name = str(classification["name"])
    if expect_old4:
      return NAME_TO_OLD4.get(normalize_label_name(name))
    return NAME_TO_LIZARD.get(normalize_label_name(name))

  props = cell.get("properties")
  if isinstance(props, dict):
    classification = props.get("classification")
    if isinstance(classification, dict) and "name" in classification:
      name = str(classification["name"])
      if expect_old4:
        return NAME_TO_OLD4.get(normalize_label_name(name))
      return NAME_TO_LIZARD.get(normalize_label_name(name))

  return None


def contour_to_polygon(contour: Sequence[Sequence[float]], source: str, instance_id: str) -> Optional[BaseGeometry]:
  if len(contour) < 3:
    warn(f"skip degenerate contour (<3 points) in {source} instance {instance_id}")
    return None

  try:
    coords = [(float(x), float(y)) for x, y in contour]
  except (TypeError, ValueError):
    warn(f"skip non-numeric contour in {source} instance {instance_id}")
    return None

  if coords[0] != coords[-1]:
    coords = coords + [coords[0]]

  poly = Polygon(coords)
  if poly.is_empty or poly.area <= 0:
    warn(f"skip empty/zero-area polygon in {source} instance {instance_id}")
    return None

  if not poly.is_valid:
    fixed = make_valid(poly)
    if fixed.is_empty:
      fixed = poly.buffer(0)
    if fixed.is_empty:
      warn(f"skip unfixable invalid polygon in {source} instance {instance_id}")
      return None
    poly = fixed

  if isinstance(poly, MultiPolygon):
    poly = max(poly.geoms, key=lambda g: g.area, default=None)
    if poly is None or poly.is_empty or poly.area <= 0:
      warn(f"skip empty MultiPolygon in {source} instance {instance_id}")
      return None

  if not isinstance(poly, Polygon) or poly.area <= 0:
    warn(f"skip non-polygon geometry in {source} instance {instance_id}")
    return None

  return poly


def extract_instances(path: Path, *, expect_old4: bool) -> List[CellInstance]:
  with path.open("r", encoding="utf-8") as f:
    payload = json.load(f)

  instances: List[CellInstance] = []
  source = str(path)
  for instance_id, cell in iter_raw_cell_dicts(payload):
    contour = cell.get("contour")
    if not isinstance(contour, list) or len(contour) < 3:
      continue

    type_id = extract_type_from_cell(cell, expect_old4=expect_old4)
    if type_id is None:
      warn(f"skip instance {instance_id} in {path.name}: missing or unknown type")
      continue

    polygon = contour_to_polygon(contour, source, instance_id)
    if polygon is None:
      continue

    instances.append(
      CellInstance(
        instance_id=instance_id,
        contour=[[float(x), float(y)] for x, y in contour],
        type_id=type_id,
        source_file=source,
        polygon=polygon,
      )
    )
  return instances


def strip_suffixes(stem: str, suffixes: Sequence[str]) -> str:
  for suffix in sorted(suffixes, key=len, reverse=True):
    if suffix and stem.endswith(suffix):
      return stem[: -len(suffix)]
  return stem


def build_file_map(
  directory: Path,
  glob_pattern: str,
  suffixes_to_strip: Sequence[str],
) -> Dict[str, Path]:
  mapping: Dict[str, Path] = {}
  for path in sorted(directory.glob(glob_pattern)):
    if not path.is_file():
      continue
    stem = strip_suffixes(path.stem, suffixes_to_strip)
    if stem in mapping:
      warn(f"duplicate basename {stem!r}; keeping {mapping[stem].name}, ignoring {path.name}")
      continue
    mapping[stem] = path
  return mapping


def polygon_iou(a: BaseGeometry, b: BaseGeometry) -> float:
  inter = a.intersection(b).area
  if inter <= 0:
    return 0.0
  union = a.union(b).area
  if union <= 0:
    return 0.0
  return float(inter / union)


def match_instances(
  gt_instances: Sequence[CellInstance],
  pred_instances: Sequence[CellInstance],
  iou_threshold: float,
) -> Tuple[List[MatchedPair], set[int], set[int]]:
  n_gt = len(gt_instances)
  n_pred = len(pred_instances)
  if n_gt == 0 or n_pred == 0:
    return [], set(range(n_gt)), set(range(n_pred))

  iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
  for i, gt in enumerate(gt_instances):
    gt_poly = gt.polygon
    assert gt_poly is not None
    for j, pred in enumerate(pred_instances):
      pred_poly = pred.polygon
      assert pred_poly is not None
      iou_matrix[i, j] = polygon_iou(gt_poly, pred_poly)

  # Maximize IoU via minimum cost assignment.
  cost = 1.0 - iou_matrix
  row_ind, col_ind = linear_sum_assignment(cost)

  pairs: List[MatchedPair] = []
  matched_gt: set[int] = set()
  matched_pred: set[int] = set()

  for i, j in zip(row_ind, col_ind):
    iou = float(iou_matrix[i, j])
    if iou < iou_threshold:
      continue
    gt = gt_instances[i]
    pred = pred_instances[j]
    mapped = LIZARD_TO_OLD4.get(pred.type_id, -1)
    pairs.append(
      MatchedPair(
        file_stem="",
        gt_instance_id=gt.instance_id,
        pred_instance_id=pred.instance_id,
        iou=iou,
        gt_type=gt.type_id,
        pred_type_original=pred.type_id,
        pred_type_mapped=mapped,
      )
    )
    matched_gt.add(i)
    matched_pred.add(j)

  unmatched_gt = set(range(n_gt)) - matched_gt
  unmatched_pred = set(range(n_pred)) - matched_pred
  return pairs, unmatched_gt, unmatched_pred


def update_confusion(confusion: np.ndarray, gt_type: int, pred_mapped: int) -> None:
  if 0 <= gt_type < confusion.shape[0] and 0 <= pred_mapped < confusion.shape[1]:
    confusion[gt_type, pred_mapped] += 1


def per_class_metrics_from_confusion(confusion: np.ndarray) -> Dict[str, Dict[str, float]]:
  out: Dict[str, Dict[str, float]] = {}
  n_classes = confusion.shape[0]
  for c in range(n_classes):
    tp = int(confusion[c, c])
    fp = int(confusion[:, c].sum() - tp)
    fn = int(confusion[c, :].sum() - tp)
    bundle = CountBundle(tp=tp, fp=fp, fn=fn)
    name = OLD4_CLASS_NAMES.get(c, str(c))
    out[name] = {
      "tp": bundle.tp,
      "fp": bundle.fp,
      "fn": bundle.fn,
      "precision": bundle.precision(),
      "recall": bundle.recall(),
      "f1": bundle.f1(),
      "support_gt": int(confusion[c, :].sum()),
    }
  return out


def evaluate_pair(
  file_stem: str,
  gt_path: Path,
  pred_path: Path,
  iou_threshold: float,
) -> Tuple[Dict[str, Any], List[MatchedPair], np.ndarray]:
  gt_instances = extract_instances(gt_path, expect_old4=True)
  pred_instances = extract_instances(pred_path, expect_old4=False)

  pairs, unmatched_gt, unmatched_pred = match_instances(gt_instances, pred_instances, iou_threshold)
  for pair in pairs:
    pair.file_stem = file_stem

  detection_tp = len(pairs)
  detection_fp = len(unmatched_pred)
  detection_fn = len(unmatched_gt)
  typed_tp = sum(1 for p in pairs if p.type_match)
  type_errors = detection_tp - typed_tp
  typed_fp = detection_fp + type_errors
  typed_fn = detection_fn + type_errors

  confusion = np.zeros((4, 4), dtype=np.int64)
  for pair in pairs:
    if 0 <= pair.pred_type_mapped < 4:
      update_confusion(confusion, pair.gt_type, pair.pred_type_mapped)

  file_metrics = {
    "file": file_stem,
    "gt_file": gt_path.name,
    "pred_file": pred_path.name,
    "n_gt": len(gt_instances),
    "n_pred": len(pred_instances),
    **CountBundle(tp=detection_tp, fp=detection_fp, fn=detection_fn).as_dict("detection"),
    **CountBundle(tp=typed_tp, fp=typed_fp, fn=typed_fn).as_dict("typed"),
    "classification_accuracy_detected": (
      float(typed_tp / detection_tp) if detection_tp else 0.0
    ),
  }
  return file_metrics, pairs, confusion


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = argparse.ArgumentParser(
    description="Evaluate CellViT++ Lizard predictions against old 4-class GT using contour IoU."
  )
  parser.add_argument("--gt-dir", type=Path, required=True, help="Directory with GT JSON files.")
  parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with prediction JSON files.")
  parser.add_argument(
    "--gt-glob",
    default="*.json",
    help='Glob pattern for GT files within --gt-dir (default: "*.json").',
  )
  parser.add_argument(
    "--pred-glob",
    default="*_cells.json",
    help='Glob pattern for prediction files within --pred-dir (default: "*_cells.json").',
  )
  parser.add_argument(
    "--gt-strip-suffixes",
    default=",".join(DEFAULT_GT_STRIP_SUFFIXES),
    help="Comma-separated filename stem suffixes to strip when matching GT files.",
  )
  parser.add_argument(
    "--pred-strip-suffixes",
    default=",".join(DEFAULT_PRED_STRIP_SUFFIXES),
    help="Comma-separated filename stem suffixes to strip when matching prediction files.",
  )
  parser.add_argument("--iou-threshold", type=float, default=0.3, help="Minimum IoU for a detection match.")
  parser.add_argument("--output-dir", type=Path, default=Path("./eval_results"), help="Output directory.")
  parser.add_argument("--verbose", action="store_true", help="Print per-file progress.")
  args = parser.parse_args(argv)

  if not args.gt_dir.is_dir():
    print(f"error: GT directory does not exist: {args.gt_dir}", file=sys.stderr)
    return 1
  if not args.pred_dir.is_dir():
    print(f"error: prediction directory does not exist: {args.pred_dir}", file=sys.stderr)
    return 1

  gt_strip = tuple(s for s in args.gt_strip_suffixes.split(",") if s)
  pred_strip = tuple(s for s in args.pred_strip_suffixes.split(",") if s)

  gt_map = build_file_map(args.gt_dir, args.gt_glob, gt_strip)
  pred_map = build_file_map(args.pred_dir, args.pred_glob, pred_strip)

  if not gt_map:
    warn(f"no GT files matched {args.gt_glob!r} under {args.gt_dir}")
  if not pred_map:
    warn(f"no prediction files matched {args.pred_glob!r} under {args.pred_dir}")

  common = sorted(set(gt_map) & set(pred_map))
  gt_only = sorted(set(gt_map) - set(pred_map))
  pred_only = sorted(set(pred_map) - set(gt_map))

  for stem in gt_only:
    warn(f"missing prediction for GT basename {stem!r} ({gt_map[stem].name})")
  for stem in pred_only:
    warn(f"missing GT for prediction basename {stem!r} ({pred_map[stem].name})")

  if not common:
    print("error: no matched GT/prediction file pairs by basename", file=sys.stderr)
    return 1

  all_pairs: List[MatchedPair] = []
  per_file_rows: List[Dict[str, Any]] = []
  confusion_total = np.zeros((4, 4), dtype=np.int64)

  totals = {
    "n_gt": 0,
    "n_pred": 0,
    "detection": CountBundle(),
    "typed": CountBundle(),
  }

  for stem in common:
    if args.verbose:
      print(f"processing {stem}")
    file_metrics, pairs, confusion = evaluate_pair(stem, gt_map[stem], pred_map[stem], args.iou_threshold)
    per_file_rows.append(file_metrics)
    all_pairs.extend(pairs)
    confusion_total += confusion

    totals["n_gt"] += file_metrics["n_gt"]
    totals["n_pred"] += file_metrics["n_pred"]
    totals["detection"].tp += file_metrics["detection_tp"]
    totals["detection"].fp += file_metrics["detection_fp"]
    totals["detection"].fn += file_metrics["detection_fn"]
    totals["typed"].tp += file_metrics["typed_tp"]
    totals["typed"].fp += file_metrics["typed_fp"]
    totals["typed"].fn += file_metrics["typed_fn"]

  detection = totals["detection"]
  typed = totals["typed"]
  summary = {
    "n_files": len(common),
    "n_gt_instances": totals["n_gt"],
    "n_pred_instances": totals["n_pred"],
    "iou_threshold": args.iou_threshold,
    "lizard_to_old4_mapping": {str(k): v for k, v in LIZARD_TO_OLD4.items()},
    "old4_class_names": {str(k): v for k, v in OLD4_CLASS_NAMES.items()},
    **detection.as_dict("detection"),
    **typed.as_dict("typed"),
    "classification_accuracy_detected": (
      float(typed.tp / detection.tp) if detection.tp else 0.0
    ),
    "per_class_metrics": per_class_metrics_from_confusion(confusion_total),
    "missing_prediction_basenames": gt_only,
    "missing_gt_basenames": pred_only,
  }

  out_dir = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  per_file_fields = [
    "file",
    "gt_file",
    "pred_file",
    "n_gt",
    "n_pred",
    "detection_tp",
    "detection_fp",
    "detection_fn",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "typed_tp",
    "typed_fp",
    "typed_fn",
    "typed_precision",
    "typed_recall",
    "typed_f1",
    "classification_accuracy_detected",
  ]
  save_csv(out_dir / "per_file_metrics.csv", per_file_rows, per_file_fields)

  matched_rows = [
    {
      "file": p.file_stem,
      "gt_instance_id": p.gt_instance_id,
      "pred_instance_id": p.pred_instance_id,
      "iou": f"{p.iou:.6f}",
      "gt_type": p.gt_type,
      "pred_type_original_6class": p.pred_type_original,
      "pred_type_mapped_4class": p.pred_type_mapped,
      "type_match": int(p.type_match),
    }
    for p in all_pairs
  ]
  save_csv(
    out_dir / "matched_instances.csv",
    matched_rows,
    [
      "file",
      "gt_instance_id",
      "pred_instance_id",
      "iou",
      "gt_type",
      "pred_type_original_6class",
      "pred_type_mapped_4class",
      "type_match",
    ],
  )

  confusion_rows: List[Dict[str, Any]] = []
  header = ["gt_class"] + [OLD4_CLASS_NAMES[i] for i in range(4)]
  for i in range(4):
    row = {"gt_class": OLD4_CLASS_NAMES[i]}
    for j in range(4):
      row[OLD4_CLASS_NAMES[j]] = int(confusion_total[i, j])
    confusion_rows.append(row)
  save_csv(out_dir / "confusion_matrix.csv", confusion_rows, header)

  print(json.dumps(summary, indent=2))
  print(f"\nWrote results to {out_dir.resolve()}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
