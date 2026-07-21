# nucsegai_post_processing

Post-processing, inspection, and evaluation utilities for NucSegAI / QuPath / CellViT nuclei pipelines.

## Environment and dependencies

Recommended conda environment: `nucsegai_post_processing` (Python **3.11**).

```bash
conda activate nucsegai_post_processing
pip install -r requirements.txt
```

Core libraries (see `requirements.txt`):


| Library                  | Role                                        |
| ------------------------ | ------------------------------------------- |
| `numpy`                  | Arrays / numerical work across all packages |
| `h5py`                   | Read ilastik `*_Probabilities.h5` masks     |
| `pillow`                 | Load QuPath `*_cl.png` maps                 |
| `matplotlib`             | Region stats / labeled-tile plots           |
| `scikit-learn`, `joblib` | Tile logistic regression train / predict    |
| `scipy`, `shapely`       | CellViT instance IoU matching               |


`cellvit_inspection/requirements.txt` lists the smaller subset needed for CellViT evaluation alone.

---



## What is ilastik?

[ilastik](https://www.ilastik.org/) is an interactive learning and segmentation toolkit: you brush labels on images and it predicts class probabilities without requiring ML expertise. In this project we train ilastik pixel classifiers to produce **probability masks** (`*_Probabilities.h5`) that highlight unwanted regions(For example: **red blood cells** and **dark artifacts**). The scripts under `ilastik_inspection` and `ilastik_classifier` then use those masks to inspect thresholds and **filter matching nuclei out of NucSegAI segmentation JSONs** (and synced typeprob files). The same style of masks can also drive nucleus **reclassification** (Tumor / Lymphocyte / Fibroblast).

**4-class type IDs** (`type_info_4class.json`):


| ID  | Class             |
| --- | ----------------- |
| 0   | Others            |
| 1   | Tumor             |
| 2   | Lymphocyte        |
| 3   | Fibroblast/Stroma |


---



## `ilastik_inspection`

Exploratory tools for understanding ilastik `*_Probabilities.h5` maps and how they align with NucSegAI JSON coordinates. **Read-only** — none of these scripts modify JSONs.


| Script                            | Purpose                                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `inspect_prob_maps_and_json.py`   | Find H5 datasets, report shape/layout (channels-first vs last), and check spatial size against JSON coordinates. |
| `inspect_possibility_range.py`    | Histogram of per-pixel probabilities (channel 0) across a directory of H5 files (decile bins).                   |
| `inspect_artifact_json_layout.py` | Report which stems exist in base JSON / typeprob / H5, and whether nucleus IDs match between base and typeprob.  |
| `debug_check_rbc_axis.py`         | Sample JSON centroids under `(y,x,c)` vs `(x,y,c)` indexing to confirm H5 axis convention.                       |


Shell wrappers (`run_inspect_prob_*.sh`) point at Artifacts / Macrophage / Lymphocytes / RBC probability folders.

**Typical use:** run these before filtering or reclassification to confirm H5 layout and thresholds.

---



## `ilastik_classifier`

Nucleus-level cleanup and reclassification driven by ilastik probability masks.

### Filter (Artifact / RBC removal)

Implemented in `latticea_test_data/filter_nuclei_with_ilastik_mask.py`; invoked from here via `run_filter.sh`.

- Removes nuclei whose contour (or bbox) fraction above a probability threshold exceeds a fraction threshold.
- Deletes the same nucleus IDs from both base JSON (`nuc`) and typeprob JSON.
- Writes cleaned JSONs plus `filter_summary.csv`.

```bash
bash ilastik_classifier/run_filter.sh
```



### Reclassify (4-class)

`reclassify_from_masks.py` updates nucleus `type` using Tumor / Lymphocyte / Fibroblast ilastik masks plus typeprob vectors (length ≥ 4). It does **not** remove nuclei.

**Logic (per nucleus):**

1. For each class mask, compute the fraction of contour pixels with `Prob > class-threshold`.
2. Flag the class if that fraction exceeds `class-fraction-threshold`.
3. If no flags → keep original type.
4. Else take top-K typeprob classes (excluding current type); candidates = flagged ∩ top-K.
5. Reclassify to the candidate with highest mask fraction (ties: typeprob, then class id).
6. If already Tumor and only the tumor mask is flagged → keep Tumor.

```bash
# Lattice-a example
bash ilastik_classifier/run_reclassifier_latticea.sh

# SOW1885 example
bash ilastik_classifier/run_reclassifier.sh
```

Required dirs: `--json-dir`, `--typeprob-dir`, `--tumor-h5-dir`, `--lymph-h5-dir`, `--fibro-h5-dir`, `--out-json-dir`, `--out-typeprob-dir`.

Output: reclassified base JSONs, copied typeprob JSONs, and `reclassify_summary.csv`.

**Coordinate note:** H5 `/exported_data` is `(Y, X, C)`; JSON contours are `[x, y]`.

---



## `region_classification`

Tile-level region typing from QuPath pixel-classification maps (`*_cl.png`). Categories: `bg`, `margin`, `tumour_inv`, `tumour_lep` (labeled `tumour_scar` tiles are usually skipped).

QuPath CL palette (from `analyze_qupath_cl_maps.py`): Tumor, Stroma, Immune, Necrosis, Other, Ignore.

### Scripts


| Script                                 | Purpose                                                                                                   |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `analyze_qupath_cl_maps.py`            | Whole-tile class pixel counts → `region_stats.csv` (+ pie chart).                                         |
| `analyze_labeled_tile_groups.py`       | Per-tile CL features for the labeled 88-tile set, aggregated by group.                                    |
| `extract_cl_grid_features.py`          | Grid features (e.g. 3×3 cell fractions, LR/TB contrast, patchiness) → CSV.                                |
| `classify_tiles_from_cl_rules.py`      | Hierarchical rule tree over grid features → category JSON.                                                |
| `analyze_tile_logreg.py`               | Train multinomial L2 logistic regression (slide-grouped CV) on labeled grid CSV; optional `--save-model`. |
| `predict_tile_logreg_to_json.py`       | Apply a saved logreg bundle to a feature CSV → category JSON.                                             |
| `compare_cl_rules_to_labeled_tiles.py` | Compare predicted categories to `tile_categories_88_tiles.json`.                                          |




### Suggested pipeline (`run_scripts/`)

```text
1_run_extract_cl_grid_features.sh      # labeled tiles → cl_grid_features_*.csv
2_run_analyze_tile_logreg.sh           # train / OOF metrics
3_run_predict_tile_logreg_to_json.sh   # predict categories
4_run_compare_to_labels.sh             # score vs hand labels
```

Rule-based alternative: `run_classify_tiles_from_cl_rules.sh` (or `_88.sh` for the labeled subset). Use `1_run_extract_cl_grid_features_all.sh` / `3_run_predict_tile_logreg_to_json_all.sh` to process all CL tiles under a QuPath project.

---



## Other packages



### `latticea_test_data`

Lattice-a dataset helpers: match GT CSV labels to NucSegAI JSON contours (`verify_nucsegai_gt_json_matching.py`), compare GT vs DL cell-label CSVs, filter nuclei/CSV rows with artifact–RBC ilastik masks, fix offset 4-class type indices (`fix_4class_json_types.py`), and wrappers for CellViT recall evals on this set.

### `cellvit_inspection`

Evaluate CellViT++ / Lizard predictions against 4-class ground truth: instance IoU matching with Hungarian assignment (`evaluate_lizard_to_old4.py`), and centroid-in-bbox recall on partially labeled lattice-a tiles (`evaluate_lizard_latticea_recall.py`). Lizard 6-class types are mapped to the old 4-class scheme before scoring.