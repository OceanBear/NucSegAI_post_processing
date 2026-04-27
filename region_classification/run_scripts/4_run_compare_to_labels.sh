# Compare to labels
python region_classification/compare_cl_rules_to_labeled_tiles.py \
  --labeled-json region_classification/tile_categories_88_tiles.json \
  --predicted-json region_classification/tile_categories_logreg_87_tiles.json