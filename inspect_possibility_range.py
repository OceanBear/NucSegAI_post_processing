import argparse
import csv
import glob
import os

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-pixel probability distribution for channel 0 of "
            "ilastik /exported_data (H,W,C) in a directory of HDF5 files."
        )
    )
    parser.add_argument(
        "--h5-dir",
        required=True,
        help="Directory containing *_Probabilities.h5 files (artifacts or RBC).",
    )
    args = parser.parse_args()

    # Requested bins and labels: 10-percentile ranges
    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0 + 1e-6,
    ]
    labels = [
        "0-0.1",
        "0.1-0.2",
        "0.2-0.3",
        "0.3-0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-0.9",
        "0.9-1.0",
    ]
    threshold = 0.5

    paths = sorted(glob.glob(os.path.join(args.h5_dir, "*.h5")))
    print("Dir:", args.h5_dir)
    print("Files:", len(paths))

    if not paths:
        return

    total_counts = np.zeros(len(labels), dtype=np.int64)
    total_counts_above_thresh = np.zeros(len(labels), dtype=np.int64)
    total_pixels = 0
    total_pixels_above_thresh = 0
    zeros = 0
    nonzeros = 0

    for p in paths:
        with h5py.File(p, "r") as f:
            if "/exported_data" not in f:
                print(f"{os.path.basename(p)}: /exported_data not found, skipping")
                continue
            arr = f["/exported_data"][...]  # (H, W, C)

        if arr.ndim != 3:
            print(f"{os.path.basename(p)}: unexpected shape {arr.shape}, skipping")
            continue

        ch0 = arr[..., 0].astype(np.float32)  # channel 0 = artifact or RBC
        flat = ch0.ravel()
        total_pixels += flat.size
        zeros += int((flat == 0).sum())
        nonzeros += int((flat != 0).sum())

        # Histogram over all pixels
        hist_all, _ = np.histogram(flat, bins=bins)
        total_counts += hist_all

        # Histogram restricted to pixels above the chosen threshold
        above_mask = flat > threshold
        if above_mask.any():
            flat_above = flat[above_mask]
            total_pixels_above_thresh += flat_above.size
            hist_above, _ = np.histogram(flat_above, bins=bins)
            total_counts_above_thresh += hist_above

    print("total pixels:", total_pixels)
    print("zeros:", zeros, "nonzeros:", nonzeros)

    if total_pixels_above_thresh == 0:
        print(f"No pixels above threshold {threshold}; percentages above threshold are 0.")

    print("\nBin summary:")
    print(
        "bin_label\tcount_all\tpct_all\tcount_above_0.5\tpct_within_above_0.5"
    )
    rows = []
    for lab, c_all, c_above in zip(
        labels, total_counts, total_counts_above_thresh
    ):
        pct_all = 100.0 * c_all / total_pixels if total_pixels > 0 else 0.0
        pct_above = (
            100.0 * c_above / total_pixels_above_thresh
            if total_pixels_above_thresh > 0
            else 0.0
        )
        print(
            f"{lab}\t{int(c_all)}\t{pct_all:.4f}\t{int(c_above)}\t{pct_above:.4f}"
        )
        rows.append(
            [
                lab,
                int(c_all),
                pct_all,
                int(c_above),
                pct_above,
            ]
        )

    # Save CSV summary in the HDF5 directory
    csv_path = os.path.join(args.h5_dir, "inspect_possibility_range_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bin_label",
                "count_all_pixels",
                "pct_all_pixels",
                f"count_pixels_above_{threshold}",
                f"pct_within_pixels_above_{threshold}",
            ]
        )
        writer.writerows(rows)

    # Also store a simple overall summary file
    overall_csv_path = os.path.join(
        args.h5_dir, "inspect_possibility_range_overall.csv"
    )
    with open(overall_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "total_pixels",
                "zeros",
                "nonzeros",
                f"total_pixels_above_{threshold}",
            ]
        )
        writer.writerow(
            [
                int(total_pixels),
                int(zeros),
                int(nonzeros),
                int(total_pixels_above_thresh),
            ]
        )

    print(f"\nSaved bin summary CSV to: {csv_path}")
    print(f"Saved overall summary CSV to: {overall_csv_path}")


if __name__ == "__main__":
    main()
