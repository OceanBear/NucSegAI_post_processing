# Produces region_classification/tile_categories_logreg_87_tiles.json
python region_classification/predict_tile_logreg_to_json.py \
  --model region_classification/tile_logreg_model.joblib \
  --features-csv region_classification/cl_grid_features_all_3x3.csv \
  --out-json region_classification/tile_categories_logreg_all_tiles.json