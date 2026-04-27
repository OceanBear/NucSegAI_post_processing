# Default: 87 labeled tiles from tile_categories_88_tiles.json + group summary CSV.
# Use --scan-all-cl-pngs to process every *_cl.png under tiles_manual.
# Produces region_classification/cl_grid_features_labeled_3x3.csv

python region_classification/extract_cl_grid_features.py \
  --tiles-root /mnt/c/Apps/QuPath-v0.6.0-Windows/projects/JN_HandE_QuPath/tiles_manual \
  --grid-rows 3 --grid-cols 3 \
  --out-csv region_classification/cl_grid_features_labeled_3x3.csv
