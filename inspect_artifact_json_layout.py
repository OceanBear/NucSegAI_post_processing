#!/usr/bin/env python
"""
Inspect the layout of artifact-filter JSONs and corresponding typeprob JSONs,
and how they align with ilastik probability HDF5 files.

This script does NOT modify any files. It only reports:
  - Which stems exist in base JSON, typeprob JSON, and H5 Probabilities.
  - Whether base JSON and typeprob JSON share the same nuclei IDs/count.

Usage (adjust paths as needed):

    python inspect_artifact_json_layout.py \\
        --json-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/pred_artifacts/json" \\
        --typeprob-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/pred_artifacts/typeprob" \\
        --h5-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_post_processing/artifacts/Probabilities" \\
        --max-files 10
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple


def build_base_json_map(json_dir: str) -> Dict[str, str]:
    """
    Map STEM -> base JSON path from json_dir.

    Expected filenames, e.g.:
        JN_TS_004_bg_tile_26418_6375.json

    STEM is the basename without extension.
    """
    mapping: Dict[str, str] = {}
    for name in os.listdir(json_dir):
        if not name.lower().endswith(".json"):
            continue
        stem = os.path.splitext(name)[0]
        mapping[stem] = os.path.join(json_dir, name)
    return mapping


def build_typeprob_json_map(typeprob_dir: str) -> Dict[str, str]:
    """
    Map STEM -> typeprob JSON path from typeprob_dir.

    Expected filenames, e.g.:
        JN_TS_004_bg_tile_26418_6375_typeprob.json

    STEM is the part before the '_typeprob' suffix.
    """
    mapping: Dict[str, str] = {}
    for name in os.listdir(typeprob_dir):
        if not name.lower().endswith(".json"):
            continue
        stem_full = os.path.splitext(name)[0]
        stem = re.sub(r"_typeprob$", "", stem_full)
        mapping[stem] = os.path.join(typeprob_dir, name)
    return mapping


def build_h5_map(h5_dir: str) -> Dict[str, str]:
    """
    Map STEM -> H5 path from h5_dir.

    Expected filenames, e.g.:
        JN_TS_004_bg_tile_26418_6375_Probabilities.h5

    STEM is the part before '_Probabilities.h5'.
    """
    mapping: Dict[str, str] = {}
    for name in os.listdir(h5_dir):
        if not (name.lower().endswith(".h5") or name.lower().endswith(".hdf5")):
            continue
        stem = re.sub(r"_Probabilities\.h5$", "", name)
        mapping[stem] = os.path.join(h5_dir, name)
    return mapping


def load_nuclei_ids(json_path: str) -> Tuple[int, List[str]]:
    """Load nuclei IDs (keys of 'nuc' dict) from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    nuc = data.get("nuc", {})
    if not isinstance(nuc, dict):
        return 0, []
    ids = list(nuc.keys())
    return len(ids), ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect base JSON, typeprob JSON, and Probabilities H5 layout."
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory with base nuclei JSON files.",
    )
    parser.add_argument(
        "--typeprob-dir",
        required=True,
        help="Directory with *_typeprob.json files.",
    )
    parser.add_argument(
        "--h5-dir",
        required=True,
        help="Directory with *_Probabilities.h5 files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Max number of stems in the triple-intersection to inspect in detail (0 = all).",
    )

    args = parser.parse_args()

    base_map = build_base_json_map(args.json_dir)
    typeprob_map = build_typeprob_json_map(args.typeprob_dir)
    h5_map = build_h5_map(args.h5_dir)

    base_stems = set(base_map.keys())
    typeprob_stems = set(typeprob_map.keys())
    h5_stems = set(h5_map.keys())

    print(f"Base JSON count       : {len(base_stems)} (dir={args.json_dir})")
    print(f"Typeprob JSON count   : {len(typeprob_stems)} (dir={args.typeprob_dir})")
    print(f"H5 Probabilities count: {len(h5_stems)} (dir={args.h5_dir})")
    print()

    base_only = sorted(base_stems - typeprob_stems)
    typeprob_only = sorted(typeprob_stems - base_stems)
    json_h5_common = sorted(base_stems & h5_stems)
    triple_common = sorted(base_stems & typeprob_stems & h5_stems)

    print(f"Base-only stems (no typeprob): {len(base_only)}")
    if base_only:
        for s in base_only[:10]:
            print(f"  - {s}")
        if len(base_only) > 10:
            print(f"  ... ({len(base_only) - 10} more)")
    print()

    print(f"Typeprob-only stems (no base): {len(typeprob_only)}")
    if typeprob_only:
        for s in typeprob_only[:10]:
            print(f"  - {s}")
        if len(typeprob_only) > 10:
            print(f"  ... ({len(typeprob_only) - 10} more)")
    print()

    print(f"Stems with both base JSON and H5: {len(json_h5_common)}")
    print(f"Stems with base JSON, typeprob JSON, and H5: {len(triple_common)}")
    print()

    # Inspect a subset of stems where all three exist
    stems_to_check = triple_common if args.max_files <= 0 else triple_common[: args.max_files]
    if not stems_to_check:
        print("No stems found that have base JSON, typeprob JSON, and H5 all present.")
        return

    print(f"Inspecting up to {len(stems_to_check)} stems in the triple-intersection:\n")

    for stem in stems_to_check:
        base_path = base_map[stem]
        typeprob_path = typeprob_map[stem]
        h5_path = h5_map[stem]

        print("=" * 80)
        print(f"STEM: {stem}")
        print(f"  base JSON   : {base_path}")
        print(f"  typeprob JSON: {typeprob_path}")
        print(f"  H5          : {h5_path}")

        try:
            base_count, base_ids = load_nuclei_ids(base_path)
        except Exception as e:
            print(f"  [ERROR] reading base JSON nuclei: {e}")
            continue

        try:
            typeprob_count, typeprob_ids = load_nuclei_ids(typeprob_path)
        except Exception as e:
            print(f"  [ERROR] reading typeprob JSON nuclei: {e}")
            continue

        print(f"  base nuclei count    : {base_count}")
        print(f"  typeprob nuclei count: {typeprob_count}")

        if base_count != typeprob_count:
            print("  [MISMATCH] different nuclei counts between base and typeprob.")

        # Quick ID set comparison (limited for brevity)
        base_id_set = set(base_ids)
        typeprob_id_set = set(typeprob_ids)
        if base_id_set == typeprob_id_set:
            print("  [OK] base and typeprob share the same nucleus ID set.")
        else:
            missing_in_typeprob = sorted(base_id_set - typeprob_id_set)
            missing_in_base = sorted(typeprob_id_set - base_id_set)
            print("  [MISMATCH] nucleus ID sets differ.")
            if missing_in_typeprob:
                print(
                    f"    IDs present in base but missing in typeprob (showing up to 10): "
                    f"{missing_in_typeprob[:10]}"
                )
            if missing_in_base:
                print(
                    f"    IDs present in typeprob but missing in base (showing up to 10): "
                    f"{missing_in_base[:10]}"
                )

        print()


if __name__ == "__main__":
    main()

