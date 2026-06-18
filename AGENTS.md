# Repository Guide for Future Codex Sessions

This repo is a Stanford Stats 305C Wi-Fi localization/modeling project. It is notebook-heavy, with reusable helpers in Python modules under `code/` and raw/processed data stored as CSV/NumPy artifacts.

## Environment

- Use the `stats305c` virtualenv.
- Python binary: `/Users/kevinluo/.virtualenvs/stats305c/bin/python`.
- Many notebooks assume they are run from their own directory and add both `os.getcwd()` and `os.getcwd()/..` to `sys.path`. When running scripts from the repo root, add `code/` to `PYTHONPATH` or `sys.path` if needed.
- `rg` may not be installed in this workspace; use `find`/Python inspection if `rg` is unavailable.

## Top-Level Layout

- `wifi_logger.py`: macOS Wi-Fi collection script. Writes to top-level `data/wifi_samples.csv`, `data/wifi_raw.jsonl`, and `data/wifi_raw/`. It uses `wdutil` and optional `wifi-unredactor`; do not run casually because it is side-effectful and tied to local hardware.
- `data/`: raw and cleaned survey data. Important files include `wifi_samples.csv`, `phone_location_log.csv`, `phone_location_log_corrected_edits.csv`, `df_clean.csv`, `data.csv`, and Sequoia-related CSVs.
- `code/`: notebooks, modeling helpers, derived arrays, CV splits, and plotting/modeling modules.
- `writeups/`: LaTeX milestone writeups and report images.
- `plots/`: generated plot artifacts.

## Core Python Modules

- `code/wifiplotting.py`: shared plotting and map utilities. Owns `OSMPlotContext`, OSM tile/building helpers, Wi-Fi heatmaps, AP Voronoi plots, and geo-CV visualization/error plotting helpers. Prefer adding reusable visualization code here instead of duplicating notebook cells.
- `code/modeling.py`: data splitting and aggregation utilities, including geographic train/test splits, geographic K-fold splits, leave-one-building-out splits, latitude/longitude binning, and binned RSSI summaries.
- `code/gp.py`: JAX Gaussian process implementation with Gibbs sampling and posterior prediction. It is computationally heavy; be careful with imports/execution inside notebooks.
- `code/correct_phone_locations.py`: interactive Matplotlib UI for correcting phone GPS tracks and attaching Wi-Fi metadata.
- `code/gridsearch/gibbs_gs.py`: batch/HPC-style Gibbs grid-search script that expects to run from `code/gridsearch/` and writes `results/job_<idx>`.
- `code/milestone4-slindermodel/helpers.py`: helper module for the separate slider/AP model notebooks.

## Data Artifacts Under `code/`

- `code/data/`: primary processed NumPy modeling arrays:
  - `X_train.npy`, `y_train.npy`, `obs_count.npy`, `obs_sse.npy`
  - `X_test.npy`, `coord_train.npy`, `coord_test.npy`, `world_train.npy`, `world_test.npy`
  - `ap_categories.npy`, `osm_context.pkl`
  - `geo_*` train/validation arrays for geographic validation.
- `code/cv/rand/` and `code/cv/geo/`: precomputed 5-fold CV splits. Each fold has `X_train_i`, `X_test_i`, `y_train_i`, `y_test_i`, `obs_count_*`, `obs_sse_*`, `coord_*`, `world_*`, `train_idx_i`, and `test_idx_i`.
- `code/gridsearch/`: grid-search-specific data/results, including local copies of train/test arrays, parameter grids, and CSV summaries.

## Notebook Organization

- `code/data cleaning.ipynb`: builds cleaned data from top-level raw survey data.
- `code/construct_training_data.ipynb`: constructs processed modeling arrays and CV split artifacts in `code/data/` and `code/cv/`.
- `code/gridsearch/GridSearch Setup.ipynb`: KNN and GP grid-search setup/evaluation, including KNN CV and spatial visualization.
- `code/gridsearch/Result Analysis.ipynb`: analysis of grid-search result CSVs.
- `code/milestone2/` and `code/milestone3/`: earlier EDA, baseline, and GP modeling notebooks.
- `code/milestone4/`: later modeling notebooks:
  - `gibbs sampler.ipynb`: JAX GP/Gibbs workflow, CV MSE, and geo-fold visualizations.
  - `full modeling.ipynb` and `gpytorch.ipynb`: GPyTorch experiments.
  - `prior_mean_eda.ipynb`, `aggregation.ipynb`, `accesspoints.ipynb`, `variational.ipynb`: supporting EDA/modeling work.
- `code/milestone4-slindermodel/`: separate PyTorch slider/access-point model experiments.

## Execution Caveats

- Do not run cells in `code/milestone4/gibbs sampler.ipynb` on this machine unless the user explicitly says to. The user has said the Gibbs code cannot be run here.
- Avoid full notebook execution unless requested. Many notebooks are exploratory and may be slow, produce large outputs, or overwrite generated artifacts.
- Be careful with commands that write generated data:
  - `code/construct_training_data.ipynb` can rewrite processed arrays and CV splits.
  - grid-search/Gibbs scripts can rewrite results.
  - `wifi_logger.py` appends/modifies top-level collection logs.
- For validation of helper/module edits, prefer static checks such as `py_compile`, import checks, JSON parsing of notebooks, and targeted non-mutating inspection.

## Editing Conventions

- Put reusable logic in Python modules under `code/`; keep notebooks focused on orchestration, experiments, and visualization calls.
- For shared plotting, prefer `code/wifiplotting.py`.
- For data splitting/aggregation helpers, prefer `code/modeling.py`.
- Preserve existing notebook-relative path conventions. Most notebook paths use `../data`, `../cv`, or local files depending on notebook directory.
- When editing notebooks programmatically, use JSON parsing and clear stale outputs for changed cells unless the user asks to preserve outputs.
- Do not revert unrelated dirty notebook changes; this repo commonly has active notebook edits.
