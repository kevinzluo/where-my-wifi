import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import geographic_kfold_split


ARRAY_NAMES = (
    "X",
    "y",
    "obs_count",
    "obs_sse",
    "coord",
    "world",
    "idx",
)


def load_training_artifacts(data_dir="data"):
    data_dir = Path(data_dir)
    return {
        "X": np.load(data_dir / "X_train.npy"),
        "y": np.load(data_dir / "y_train.npy"),
        "obs_count": np.load(data_dir / "obs_count.npy"),
        "obs_sse": np.load(data_dir / "obs_sse.npy"),
        "coord": np.load(data_dir / "coord_train.npy"),
        "world": np.load(data_dir / "world_train.npy"),
        "ap_categories": np.load(data_dir / "ap_categories.npy", allow_pickle=True),
    }


def load_modeling_dataframe(raw_csv="../data/sequoia_sets.csv"):
    sequoia = pd.read_csv(raw_csv)
    return sequoia[sequoia["rssi_set"].notna()].copy().reset_index()


def _save_bundle(out_dir, prefix, artifacts, idx):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(idx, dtype=int)
    np.save(out_dir / f"{prefix}_idx.npy", idx)
    np.save(out_dir / f"{prefix}_X.npy", artifacts["X"][idx])
    np.save(out_dir / f"{prefix}_y.npy", artifacts["y"][idx])
    np.save(out_dir / f"{prefix}_obs_count.npy", artifacts["obs_count"][idx])
    np.save(out_dir / f"{prefix}_obs_sse.npy", artifacts["obs_sse"][idx])
    np.save(out_dir / f"{prefix}_coord.npy", artifacts["coord"][idx])
    np.save(out_dir / f"{prefix}_world.npy", artifacts["world"][idx])


def _save_fold(out_dir, split_idx, artifacts, train_idx, test_idx):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    np.save(out_dir / f"train_idx_{split_idx}.npy", train_idx)
    np.save(out_dir / f"test_idx_{split_idx}.npy", test_idx)

    for name in ARRAY_NAMES[:-1]:
        source = artifacts[name]
        np.save(out_dir / f"{name}_train_{split_idx}.npy", source[train_idx])
        np.save(out_dir / f"{name}_test_{split_idx}.npy", source[test_idx])


def write_nested_geo_splits(
    *,
    out_root="eval_splits/geo_outer0",
    data_dir="data",
    raw_csv="../data/sequoia_sets.csv",
    outer_cv_dir="cv/geo",
    outer_fold=0,
    random_state=305,
):
    """
    Write nested geographic split artifacts without modifying existing code/cv folds.

    The outer split is copied from an existing geographic CV fold. Inner folds are
    generated only from the outer training rows and keep global row indices.
    """
    out_root = Path(out_root)
    artifacts = load_training_artifacts(data_dir)
    df = load_modeling_dataframe(raw_csv)
    outer_cv_dir = Path(outer_cv_dir)

    outer_train_idx = np.load(outer_cv_dir / f"train_idx_{outer_fold}.npy")
    outer_holdout_idx = np.load(outer_cv_dir / f"test_idx_{outer_fold}.npy")

    if len(set(outer_train_idx) & set(outer_holdout_idx)):
        raise ValueError("Outer train and holdout indices overlap.")
    if len(df) != artifacts["X"].shape[0]:
        raise ValueError("Modeling dataframe and X_train row counts do not match.")

    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "ap_categories.npy", artifacts["ap_categories"])
    _save_bundle(out_root, "outer_train", artifacts, outer_train_idx)
    _save_bundle(out_root, "outer_holdout", artifacts, outer_holdout_idx)

    outer_train_df = df.iloc[outer_train_idx].copy()
    for k in (3, 5):
        fold_dir = out_root / f"inner_geo{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        folds = geographic_kfold_split(
            outer_train_df,
            K=k,
            random_state=random_state,
        )
        np.save(fold_dir / "ap_categories.npy", artifacts["ap_categories"])
        for split_idx, (train_df, test_df) in enumerate(folds):
            _save_fold(
                fold_dir,
                split_idx,
                artifacts,
                train_df.index.to_numpy(dtype=int),
                test_df.index.to_numpy(dtype=int),
            )

    manifest = {
        "outer_source": str(outer_cv_dir),
        "outer_fold": int(outer_fold),
        "random_state": int(random_state),
        "n_total": int(artifacts["X"].shape[0]),
        "n_outer_train": int(len(outer_train_idx)),
        "n_outer_holdout": int(len(outer_holdout_idx)),
        "inner_folds": {"geo3": 3, "geo5": 5},
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def make_gp_stage1_grid():
    ls_xy_vals = [0.025, 0.04, 0.06]
    ls_z_vals = [0.25, 0.5]
    os_xyz_vals = [10, 25, 50]
    time_configs = [(0, 50), (5, 50), (5, 5)]
    ap_setups = [
        {"ap_form": "none", "ls_ap": 1.0, "os_ap": 0.0},
        {"ap_form": "mult", "ls_ap": 0.5, "os_ap": 0.0},
        {"ap_form": "mult", "ls_ap": 1.0, "os_ap": 0.0},
        {"ap_form": "add", "ls_ap": 0.25, "os_ap": 0.25},
        {"ap_form": "add", "ls_ap": 0.25, "os_ap": 0.5},
        {"ap_form": "add", "ls_ap": 0.5, "os_ap": 0.25},
        {"ap_form": "add", "ls_ap": 0.5, "os_ap": 0.5},
    ]

    rows = []
    for param_id, (ls_xy, ls_z, os_xyz, (os_t, ls_t), ap) in enumerate(
        product(ls_xy_vals, ls_z_vals, os_xyz_vals, time_configs, ap_setups)
    ):
        rows.append(
            {
                "param_id": param_id,
                "ls_xy": ls_xy,
                "ls_z": ls_z,
                "os_xyz": os_xyz,
                "ls_t": ls_t,
                "os_t": os_t,
                **ap,
            }
        )

    grid = pd.DataFrame(rows)
    if grid.shape != (378, 9):
        raise RuntimeError(f"Unexpected GP stage-1 grid shape: {grid.shape}.")
    if grid.drop(columns=["param_id"]).duplicated().any():
        raise RuntimeError("GP stage-1 grid contains duplicate parameter rows.")
    return grid


def write_gp_stage1_grid(path="gridsearch/gp_geo_stage1_grid.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_gp_stage1_grid()
    grid.to_csv(path, index=False)
    return grid


def result_path(result_dir, param_id, fold_idx):
    return Path(result_dir) / f"param_{int(param_id):04d}_fold_{int(fold_idx)}.csv"


def read_result_row(path):
    path = Path(path)
    if not path.exists():
        return None

    try:
        result = pd.read_csv(path)
    except Exception:
        return None
    if len(result) != 1:
        return None
    return result.iloc[0].to_dict()


def completed_gp_results(result_dir):
    result_dir = Path(result_dir)
    rows = []
    if not result_dir.exists():
        return pd.DataFrame()

    for path in sorted(result_dir.glob("param_*_fold_*.csv")):
        row = read_result_row(path)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_cv_results(results):
    if len(results) == 0:
        return pd.DataFrame()
    results = results.copy()
    if "total_elapsed" not in results.columns:
        results["total_elapsed"] = results["fit_elapsed"] + results["predict_elapsed"]
    grouped = (
        results.groupby("param_id", as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            n_folds=("fold_idx", "nunique"),
            mean_fit_elapsed=("fit_elapsed", "mean"),
            mean_predict_elapsed=("predict_elapsed", "mean"),
            mean_total_elapsed=("total_elapsed", "mean"),
        )
        .sort_values(["mean_mse", "std_mse"], na_position="last")
    )
    return grouped


def load_fold_arrays(split_dir, fold_idx):
    split_dir = Path(split_dir)
    return {
        "X_train": np.load(split_dir / f"X_train_{fold_idx}.npy"),
        "y_train": np.load(split_dir / f"y_train_{fold_idx}.npy"),
        "obs_count_train": np.load(split_dir / f"obs_count_train_{fold_idx}.npy"),
        "obs_sse_train": np.load(split_dir / f"obs_sse_train_{fold_idx}.npy"),
        "X_test": np.load(split_dir / f"X_test_{fold_idx}.npy"),
        "y_test": np.load(split_dir / f"y_test_{fold_idx}.npy"),
    }


def make_geo_cv_index_pairs(split_dir, n_splits):
    split_dir = Path(split_dir)
    return [
        (
            np.load(split_dir / f"train_idx_{fold_idx}.npy"),
            np.load(split_dir / f"test_idx_{fold_idx}.npy"),
        )
        for fold_idx in range(n_splits)
    ]


def write_result_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    pd.DataFrame([row]).to_csv(tmp_path, index=False)
    tmp_path.replace(path)
