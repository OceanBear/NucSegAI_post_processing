#!/usr/bin/env python
"""
Run a saved logistic-regression tile model on feature CSV and write category JSON.

Model bundle is produced by analyze_tile_logreg.py --save-model and contains:
  - pipeline
  - feature_names
  - classes
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np

CLASSES = ["bg", "margin", "tumour_inv", "tumour_lep"]


def infer_grid_cell_indices(fieldnames: Sequence[str]) -> List[int]:
    out: List[int] = []
    for k in fieldnames:
        if k.startswith("c") and k.endswith("_ignore"):
            stem = k[1 : -len("_ignore")]
            if stem.isdigit():
                out.append(int(stem))
    return sorted(set(out))


def with_global_means(rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    """In-place append global_frac_ignore/tumor if absent and c* columns exist."""
    if "global_frac_ignore" in fieldnames and "global_frac_tumor" in fieldnames:
        return
    cells = infer_grid_cell_indices(fieldnames)
    if not cells:
        return
    for r in rows:
        ig: List[float] = []
        tu: List[float] = []
        for idx in cells:
            k_ig = f"c{idx}_ignore"
            k_tu = f"c{idx}_tumor"
            if k_ig in r:
                try:
                    ig.append(float(r[k_ig]))
                except ValueError:
                    pass
            if k_tu in r:
                try:
                    tu.append(float(r[k_tu]))
                except ValueError:
                    pass
        r["global_frac_ignore"] = f"{float(np.mean(ig)):.10f}" if ig else "0.0"
        r["global_frac_tumor"] = f"{float(np.mean(tu)):.10f}" if tu else "0.0"


def load_feature_matrix(
    csv_path: Path, feature_names: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    with_global_means(rows, fieldnames)

    X: List[List[float]] = []
    tile_ids: List[str] = []
    for r in rows:
        tid = (r.get("tile_id") or "").strip()
        if not tid:
            continue
        feats: List[float] = []
        ok = True
        for c in feature_names:
            try:
                feats.append(float(r.get(c, "")))
            except (TypeError, ValueError):
                ok = False
                break
        if not ok:
            continue
        X.append(feats)
        tile_ids.append(tid)

    if not X:
        raise SystemExit("No valid rows with all required features and tile_id found.")
    return np.asarray(X, dtype=np.float64), tile_ids


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Predict tile categories with saved logreg model and write JSON."
    )
    p.add_argument(
        "--model",
        type=Path,
        default=here / "tile_logreg_model.joblib",
        help="Saved model bundle from analyze_tile_logreg.py.",
    )
    p.add_argument(
        "--features-csv",
        type=Path,
        default=here / "cl_grid_features_labeled_3x3.csv",
        help="Feature CSV with tile_id and model feature columns.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=here / "tile_categories_logreg.json",
        help="Output category JSON.",
    )
    p.add_argument(
        "--tile-size-mm2",
        type=float,
        default=2.0,
        help="Metadata only.",
    )
    args = p.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model bundle not found: {args.model}")
    if not args.features_csv.is_file():
        raise SystemExit(f"Features CSV not found: {args.features_csv}")

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    feature_names: List[str] = list(bundle["feature_names"])
    classes: List[str] = list(bundle.get("classes", CLASSES))

    X, tile_ids = load_feature_matrix(args.features_csv, feature_names)
    pred = pipe.predict(X)

    groups: Dict[str, List[str]] = {c: [] for c in classes}
    for tid, lab in zip(tile_ids, pred):
        if lab not in groups:
            continue
        groups[lab].append(tid)
    for k in groups:
        groups[k].sort()

    payload: Dict[str, Any] = {
        "metadata": {
            "tile_size_mm2": args.tile_size_mm2,
            "description": "Tile categories from saved multinomial logistic regression model.",
            "total_tiles": int(sum(len(v) for v in groups.values())),
            "groups": {k: len(v) for k, v in groups.items()},
            "model_bundle": str(args.model),
            "features_csv": str(args.features_csv),
        },
        **groups,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.out_json} ({payload['metadata']['total_tiles']} tiles)")


if __name__ == "__main__":
    main()

