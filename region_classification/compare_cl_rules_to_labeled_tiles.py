#!/usr/bin/env python
"""
Compare rule-based CL classification JSON to the hand-labeled 88-tile reference.

Labeled ids look like: JN_TS_001_bg_tile_19702_7361
Predicted ids look like: JN_TS_001_tile_19702_7361

Matching is by (slide, tile_x, tile_y) only. tumour_scar is skipped (not in rules output).

Prints a summary to stdout; optional --out-json for a machine-readable report.

Requirements: stdlib only
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LABELED_TILE_RE = re.compile(
    r"^(JN_TS_\d+)_(bg|margin|tumour_inv|tumour_lep|tumour_scar)_tile_(\d+)_(\d+)$",
    re.IGNORECASE,
)
PRED_TILE_RE = re.compile(
    r"^(JN_TS_\d+)_tile_(\d+)_(\d+)$",
    re.IGNORECASE,
)

CLASSES = ["bg", "margin", "tumour_inv", "tumour_lep"]


def parse_labeled_id(tile_id: str) -> Optional[Tuple[str, int, int, str]]:
    m = LABELED_TILE_RE.match(tile_id.strip())
    if not m:
        return None
    slide, category, xs, ys = m.groups()
    return slide.upper(), int(xs), int(ys), category.lower()


def parse_pred_id(tile_id: str) -> Optional[Tuple[str, int, int]]:
    m = PRED_TILE_RE.match(tile_id.strip())
    if not m:
        return None
    slide, xs, ys = m.groups()
    return slide.upper(), int(xs), int(ys)


def tile_key(slide: str, x: int, y: int) -> str:
    return f"{slide}|{x}|{y}"


def load_labeled_tiles(path: Path) -> List[Tuple[str, str, str]]:
    """Return list of (tile_key, original_id, true_label). Skips tumour_scar."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows: List[Tuple[str, str, str]] = []
    for group in CLASSES + ["tumour_scar"]:
        if group not in data or not isinstance(data[group], list):
            continue
        for tid in data[group]:
            if not isinstance(tid, str):
                continue
            parsed = parse_labeled_id(tid)
            if parsed is None:
                continue
            slide, x, y, cat = parsed
            if cat == "tumour_scar":
                continue
            if cat not in CLASSES:
                continue
            rows.append((tile_key(slide, x, y), tid, cat))
    return rows


def load_predictions(path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Return (tile_key -> predicted label, metadata dict or {})."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    pred: Dict[str, str] = {}
    for group in CLASSES:
        if group not in data or not isinstance(data[group], list):
            continue
        for tid in data[group]:
            if not isinstance(tid, str):
                continue
            parsed = parse_pred_id(tid)
            if parsed is None:
                continue
            slide, x, y = parsed
            k = tile_key(slide, x, y)
            pred[k] = group
    return pred, meta


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare tile_categories_cl_rules.json to tile_categories_88_tiles.json."
    )
    parser.add_argument(
        "--labeled-json",
        type=Path,
        default=here / "tile_categories_88_tiles.json",
        help="Reference JSON with hand labels (88 tiles).",
    )
    parser.add_argument(
        "--predicted-json",
        type=Path,
        default=here / "tile_categories_cl_rules.json",
        help="Rule-based classifier output JSON.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write detailed report (mismatches, matrix, metrics) to this path.",
    )
    args = parser.parse_args()

    if not args.labeled_json.is_file():
        raise SystemExit(f"Missing labeled JSON: {args.labeled_json}")
    if not args.predicted_json.is_file():
        raise SystemExit(f"Missing predicted JSON: {args.predicted_json}")

    labeled = load_labeled_tiles(args.labeled_json)
    pred_map, pred_meta = load_predictions(args.predicted_json)

    cm: Dict[Tuple[str, str], int] = defaultdict(int)
    per_tile: List[Dict[str, Any]] = []
    missing: List[str] = []
    correct = 0

    for key, orig_id, true_l in labeled:
        pred_l = pred_map.get(key)
        if pred_l is None:
            missing.append(orig_id)
            per_tile.append(
                {
                    "labeled_tile_id": orig_id,
                    "true_label": true_l,
                    "predicted_label": None,
                    "match": False,
                    "reason": "not_in_predicted_json",
                }
            )
            continue
        ok = pred_l == true_l
        if ok:
            correct += 1
        cm[(true_l, pred_l)] += 1
        per_tile.append(
            {
                "labeled_tile_id": orig_id,
                "true_label": true_l,
                "predicted_label": pred_l,
                "match": ok,
            }
        )

    n = len(labeled)
    n_eval = n - len(missing)
    acc = correct / n if n else 0.0

    # Per-class precision / recall / F1 (micro-style denominators)
    def prf(cls: str) -> Tuple[float, float, float]:
        tp = cm[(cls, cls)]
        fp = sum(cm[(t, p)] for t in CLASSES for p in CLASSES if p == cls and t != cls)
        fn = sum(cm[(t, p)] for t in CLASSES for p in CLASSES if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    # Print report
    print(f"Labeled tiles (excluding tumour_scar): {n}")
    print(f"Found in predicted JSON: {n_eval}  |  Missing CL / key: {len(missing)}")
    print(f"Accuracy (on all labeled): {correct}/{n} = {acc:.4f}")
    if missing:
        print(f"\nMissing from predictions ({len(missing)}):")
        for m in missing[:30]:
            print(f"  {m}")
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(14) + "".join(c.ljust(14) for c in CLASSES)
    print(header)
    for t in CLASSES:
        row = t.ljust(14)
        for p in CLASSES:
            row += str(cm[(t, p)]).ljust(14)
        print(row)

    print("\nPer-class (labeled support as denominator for recall):")
    for c in CLASSES:
        support = sum(cm[(c, p)] for p in CLASSES)
        prec, rec, f1 = prf(c)
        print(f"  {c:12s}  support={support:3d}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    mismatches = [r for r in per_tile if not r["match"]]
    print(f"\nMismatches: {len(mismatches)}")
    for r in mismatches[:40]:
        pred = r["predicted_label"]
        reason = r.get("reason", "")
        extra = f" ({reason})" if reason else ""
        print(
            f"  {r['labeled_tile_id']}: true={r['true_label']} pred={pred}{extra}"
        )
    if len(mismatches) > 40:
        print(f"  ... and {len(mismatches) - 40} more (see --out-json)")

    if args.out_json:
        report = {
            "labeled_json": str(args.labeled_json),
            "predicted_json": str(args.predicted_json),
            "predicted_metadata": pred_meta,
            "n_labeled_ex_scar": n,
            "n_found_in_predictions": n_eval,
            "n_missing_predictions": len(missing),
            "missing_labeled_tile_ids": missing,
            "accuracy": acc,
            "confusion_matrix": {f"{t}|{p}": cm[(t, p)] for t in CLASSES for p in CLASSES},
            "per_class": {
                c: {
                    "support": sum(cm[(c, p)] for p in CLASSES),
                    "precision": prf(c)[0],
                    "recall": prf(c)[1],
                    "f1": prf(c)[2],
                }
                for c in CLASSES
            },
            "per_tile": per_tile,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
