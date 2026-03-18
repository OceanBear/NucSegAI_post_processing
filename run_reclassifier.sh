python reclassify_macro_lymph_from_masks.py \
  --json-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_wsi/json_filtered" \
  --typeprob-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_wsi/typeprob_filtered" \
  --macro-h5-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_post_processing/Macrophage/Probabilities" \
  --lymph-h5-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_post_processing/Lymphocytes/Probabilities" \
  --out-json-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_wsi/json_reclass" \
  --out-typeprob-dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_wsi/typeprob_reclass" \
  --macro-threshold 0.3 --macro-fraction-threshold 0.3 \
  --lymph-threshold 0.3 --lymph-fraction-threshold 0.3 \
  --reclass-top-k 5