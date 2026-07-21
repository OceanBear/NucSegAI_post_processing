#!/usr/bin/env python3
"""
Fix 4-class nucleus JSON files whose type indices are offset by +1.

Some converted JSONs use:
  o -> 1, t -> 2, l -> 3, f -> 4

The standard mapping (type_info_4class.json) is:
  o -> 0, t -> 1, l -> 2, f -> 3

This script subtracts 1 from each nucleus "type" value and writes corrected JSONs.
All other fields are copied unchanged.

Usage:
  python fix_4class_json_types.py \\
    --json-dir "/mnt/j/HandE/results/latticea_test_data/converted_4class_json" \\
    --out-json-dir "/mnt/j/HandE/results/latticea_test_data/converted_4class_json_fixed"
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, Mapping, MutableMapping, Tuple


ABNORMAL_TO_NORMAL: Dict[int, int] = {
    1: 0,  # Others
    2: 1,  # Tumor
    3: 2,  # Lymphocyte
    4: 3,  # Fibroblast/Stroma
}


def load_nuclei_dict(payload: object) -> MutableMapping[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    nuc = payload.get("nuc")
    if isinstance(nuc, dict):
        return nuc
    raise ValueError('JSON missing top-level "nuc" object')


def remap_types(
    nuclei: Mapping[str, object],
) -> Tuple[int, int, Counter]:
    """
    Returns (n_changed, n_skipped, unexpected_type_counts).
    """
    changed = 0
    skipped = 0
    unexpected: Counter = Counter()

    for info in nuclei.values():
        if not isinstance(info, dict):
            skipped += 1
            continue
        raw_type = info.get("type")
        try:
            old_type = int(raw_type)
        except (TypeError, ValueError):
            skipped += 1
            continue

        if old_type in ABNORMAL_TO_NORMAL:
            new_type = ABNORMAL_TO_NORMAL[old_type]
            if new_type != old_type:
                info["type"] = new_type
                changed += 1
            else:
                skipped += 1
        else:
            unexpected[old_type] += 1
            skipped += 1

    return changed, skipped, unexpected


def process_file(in_path: str, out_path: str) -> Tuple[int, int, Counter]:
    with open(in_path, "r") as f:
        data = json.load(f)

    nuclei = load_nuclei_dict(data)
    changed, skipped, unexpected = remap_types(nuclei)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    return changed, skipped, unexpected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap 4-class nucleus JSON types from 1-4 to 0-3."
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Input directory with abnormal JSON files.",
    )
    parser.add_argument(
        "--out-json-dir",
        required=True,
        help="Output directory for corrected JSON files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, process only the first N JSON files (for quick tests).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_json_dir, exist_ok=True)

    names = sorted(
        name for name in os.listdir(args.json_dir) if name.lower().endswith(".json")
    )
    if args.max_files and args.max_files > 0:
        names = names[: args.max_files]

    total_changed = 0
    total_skipped = 0
    total_unexpected: Counter = Counter()

    for name in names:
        in_path = os.path.join(args.json_dir, name)
        out_path = os.path.join(args.out_json_dir, name)
        changed, skipped, unexpected = process_file(in_path, out_path)
        total_changed += changed
        total_skipped += skipped
        total_unexpected.update(unexpected)
        print(f"{name}: changed={changed}, skipped={skipped}, unexpected={dict(unexpected)}")

    print()
    print(f"Processed {len(names)} file(s)")
    print(f"Total changed={total_changed}, skipped={total_skipped}")
    if total_unexpected:
        print(f"Unexpected type values (left unchanged): {dict(sorted(total_unexpected.items()))}")
    print(f"Wrote outputs to: {args.out_json_dir}")


if __name__ == "__main__":
    main()
