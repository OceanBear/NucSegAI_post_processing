# Classify only the 87 labeled tiles (tile_categories_88_tiles.json; tumour_scar excluded).
# Writes a separate JSON so full-corpus tile_categories_cl_rules.json is unchanged.
#
# Grid-rule thresholds (defaults match classify_tiles_from_cl_rules.py; listed here for tuning).
python region_classification/classify_tiles_from_cl_rules.py \
  --tiles-root "/mnt/c/Apps/QuPath-v0.6.0-Windows/projects/JN_HandE_QuPath/tiles_manual" \
  --labeled-json region_classification/tile_categories_88_tiles.json \
  --out-json region_classification/tile_categories_cl_rules_87_tiles.json \
  --grid-rows 3 \
  --grid-cols 3 \
  --tile-size-mm2 2.0 \
  --lep-ignore-gt 0.23 \
  --lep-tumor-min 0.06 \
  --lep-tumor-max 0.12 \
  --bg-ignore-gte 0.30 \
  --bg-tumor-lte 0.07 \
  --bg-max-tumor-spread 0.055 \
  --margin-lr-ignore-gte 0.10 \
  --margin-nec-spread-gte 0.32 \
  --margin-tumor-spread-gte 0.10 \
  --margin-lr-tumor-gte 0.085 \
  --margin-nec-aux-dir-ignore-gte 0.06 \
  --margin-nec-aux-dir-tumor-gte 0.04 \
  --inv-max-ignore-spread 0.16 \
  --inv-max-lr-ignore 0.065 \
  --inv-margin-ignore-split 0.25 \
  --progress-every 50
