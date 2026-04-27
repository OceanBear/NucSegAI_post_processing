#!/usr/bin/env python
"""
Train and evaluate multinomial L2-logistic regression on labeled CL grid features CSV.

- Reads region_classification/cl_grid_features_labeled_3x3.csv (or --csv).
- Drops tumour_scar; maps true_group to bg | margin | tumour_inv | tumour_lep.
- Adds global_frac_ignore / global_frac_tumor (mean of c{i}_ignore / c{i}_tumor over cells).
- Uses StandardScaler + LogisticRegression(penalty='l2', multi_class='multinomial').
- Tunes C via GroupKFold by slide (JN_TS_XXX from labeled_tile_id).
- Prints CV score, confusion matrix (out-of-fold predictions with best C), top |coef| features.

Requirements: numpy, scikit-learn (install with ``pip install scikit-learn``, not the deprecated ``sklearn`` PyPI name).

Example:
  python region_classification/analyze_tile_logreg.py \\
    --csv region_classification/cl_grid_features_labeled_3x3.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        GroupKFold,
        cross_val_predict,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "Please install dependencies: pip install scikit-learn numpy joblib"
    ) from e

CLASSES = ["bg", "margin", "tumour_inv", "tumour_lep"]
CELL_RE = re.compile(r"^c(\d+)_(tumor|ignore)$")
SLIDE_RE = re.compile(r"^(JN_TS_\d+)_", re.IGNORECASE)


def parse_slide(labeled_id: str) -> str:
    m = SLIDE_RE.match(labeled_id.strip())
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def infer_feature_columns(fieldnames: Sequence[str]) -> List[str]:
    skip = {
        "labeled_tile_id",
        "true_group",
        "tile_id",
        "cl_png",
    }
    out: List[str] = []
    for name in fieldnames:
        if name in skip:
            continue
        if name.startswith("global_"):
            continue
        out.append(name)
    return out


def add_global_means(rows: List[Dict[str, str]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Append global_frac_ignore, global_frac_tumor from c* cells if present."""
    by_cell: Dict[int, Dict[str, float]] = {}
    for row in rows:
        for k, v in row.items():
            m = CELL_RE.match(k)
            if not m:
                continue
            idx, kind = int(m.group(1)), m.group(2)
            try:
                val = float(v)
            except ValueError:
                continue
            if idx not in by_cell:
                by_cell[idx] = {}
            by_cell[idx][kind] = val
        break
    if not by_cell:
        return [], rows

    extra_cols = ["global_frac_ignore", "global_frac_tumor"]
    out_rows: List[Dict[str, str]] = []
    for row in rows:
        ign: List[float] = []
        tum: List[float] = []
        for idx in sorted(by_cell.keys()):
            k_ig = f"c{idx}_ignore"
            k_tu = f"c{idx}_tumor"
            if k_ig in row:
                try:
                    ign.append(float(row[k_ig]))
                except ValueError:
                    pass
            if k_tu in row:
                try:
                    tum.append(float(row[k_tu]))
                except ValueError:
                    pass
        r = dict(row)
        r["global_frac_ignore"] = f"{float(np.mean(ign)):.10f}" if ign else "0.0"
        r["global_frac_tumor"] = f"{float(np.mean(tum)):.10f}" if tum else "0.0"
        out_rows.append(r)
    return extra_cols, out_rows


def load_xy(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    extra, rows = add_global_means(rows)
    feature_cols = infer_feature_columns(fieldnames)
    for c in extra:
        if c not in feature_cols:
            feature_cols.append(c)

    X_list: List[List[float]] = []
    y_list: List[str] = []
    groups_list: List[str] = []
    ids: List[str] = []
    tile_ids: List[str] = []

    for row in rows:
        tg = (row.get("true_group") or "").strip().lower()
        if tg == "tumour_scar":
            continue
        if tg not in CLASSES:
            continue
        feats: List[float] = []
        ok = True
        for col in feature_cols:
            v = row.get(col, "")
            try:
                feats.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if not ok:
            continue
        lid = row.get("labeled_tile_id") or ""
        X_list.append(feats)
        y_list.append(tg)
        groups_list.append(parse_slide(lid))
        ids.append(lid)
        tile_ids.append(row.get("tile_id") or "")

    if not X_list:
        raise SystemExit("No labeled rows with valid features.")

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list)
    groups = np.asarray(groups_list)
    return X, y, groups, feature_cols, ids, tile_ids


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Multinomial L2 logistic regression on labeled CL grid CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=here / "cl_grid_features_labeled_3x3.csv",
        help="Labeled features CSV.",
    )
    parser.add_argument(
        "--c-values",
        type=str,
        default="0.001,0.01,0.05,0.1,0.5,1,5,10,50,100",
        help="Comma-separated C grid for inverse regularization strength.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="GroupKFold n_splits (capped by number of slide groups).",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Use class_weight='balanced' or None.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="LogisticRegression max_iter.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed (solver).",
    )
    parser.add_argument(
        "--top-k-coef",
        type=int,
        default=8,
        help="Top |coefficient| features per class to print.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write metrics + best params as JSON.",
    )
    parser.add_argument(
        "--out-oof-csv",
        type=Path,
        default=here / "tile_logreg_oof_predictions.csv",
        help="Where to write out-of-fold per-tile predictions CSV.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=here / "tile_logreg_model.joblib",
        help="Path to save trained model bundle (pipeline + feature order).",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    X, y, groups, feature_names, labeled_ids, tile_ids = load_xy(args.csv)

    uniq_groups = np.unique(groups)
    n_groups = len(uniq_groups)
    n_splits = min(args.cv_splits, n_groups)
    if n_splits < 2:
        raise SystemExit(
            f"Need at least 2 slide groups for grouped CV; got {n_groups}."
        )

    C_list = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]
    cw = "balanced" if args.class_weight == "balanced" else None

    # Only pass kwargs supported by this sklearn build.
    lr_kw: Dict[str, Any] = {
        "solver": "lbfgs",
        "class_weight": cw,
        "max_iter": args.max_iter,
        "random_state": args.random_state,
    }
    lr_params = inspect.signature(LogisticRegression.__init__).parameters
    if "multi_class" in lr_params:
        lr_kw["multi_class"] = "multinomial"

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**lr_kw)),
        ]
    )
    param_grid = {"lr__C": C_list}

    gkf = GroupKFold(n_splits=n_splits)
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*'penalty' was deprecated.*",
            category=FutureWarning,
        )
        grid.fit(X, y, groups=groups)

    best_C = grid.best_params_["lr__C"]
    print(f"Best C (GroupKFold n_splits={n_splits}, groups=slides): {best_C}")
    print(f"Mean CV accuracy: {grid.best_score_:.4f}")

    best_pipe = grid.best_estimator_
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*'penalty' was deprecated.*",
            category=FutureWarning,
        )
        y_oof = cross_val_predict(
            best_pipe, X, y, cv=gkf, groups=groups, n_jobs=-1
        )
    acc_oof = accuracy_score(y, y_oof)
    print(f"Out-of-fold accuracy (same CV, fixed best C): {acc_oof:.4f}")

    cm = confusion_matrix(y, y_oof, labels=CLASSES)
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "".join(f"{name:14s}" for name in CLASSES)
    print("true\\pred    " + header)
    for i, ti in enumerate(CLASSES):
        row = "".join(f"{cm[i, j]:14d}" for j in range(len(CLASSES)))
        print(f"{ti:12s}  {row}")

    print("\nClassification report (OOF):")
    print(
        classification_report(
            y,
            y_oof,
            labels=CLASSES,
            target_names=CLASSES,
            digits=3,
            zero_division=0,
        )
    )

    # Coefficients on full data (refitted best pipeline from grid)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*'penalty' was deprecated.*",
            category=FutureWarning,
        )
        best_pipe.fit(X, y)
    lr: LogisticRegression = best_pipe.named_steps["lr"]
    coef = lr.coef_
    print(f"\nTop |coef| features per class (after scaling; k={args.top_k_coef}):")
    for ci, cname in enumerate(lr.classes_):
        w = coef[ci]
        order = np.argsort(np.abs(w))[::-1][: args.top_k_coef]
        print(f"  {cname}:")
        for j in order:
            print(f"    {feature_names[j]:40s}  {w[j]:+.4f}")

    if args.out_oof_csv:
        args.out_oof_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_oof_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "labeled_tile_id",
                    "tile_id",
                    "slide_id",
                    "true_group",
                    "pred_group_oof",
                    "is_correct",
                ],
            )
            w.writeheader()
            for lid, tid, g, t, p in zip(labeled_ids, tile_ids, groups, y, y_oof):
                w.writerow(
                    {
                        "labeled_tile_id": lid,
                        "tile_id": tid,
                        "slide_id": g,
                        "true_group": t,
                        "pred_group_oof": p,
                        "is_correct": int(t == p),
                    }
                )
        print(f"\nWrote OOF predictions CSV: {args.out_oof_csv}")

    if args.save_model:
        bundle = {
            "pipeline": best_pipe,
            "feature_names": feature_names,
            "classes": list(CLASSES),
            "best_C": best_C,
            "class_weight": args.class_weight,
        }
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.save_model)
        print(f"Wrote model bundle: {args.save_model}")

    if args.out_json:
        payload: Dict[str, Any] = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_slide_groups": n_groups,
            "cv_splits": n_splits,
            "best_C": best_C,
            "mean_cv_accuracy": float(grid.best_score_),
            "oof_accuracy": float(acc_oof),
            "class_weight": args.class_weight,
            "feature_names": feature_names,
            "confusion_matrix_oof": cm.tolist(),
            "class_order": list(CLASSES),
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
