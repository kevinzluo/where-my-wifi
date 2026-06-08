import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from eval_workflow import (
    load_fold_arrays,
    load_result_index,
    materialize_matching_result,
    result_path,
    result_index_key,
    write_result_row,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run one or more restartable GP CV fits.")
    parser.add_argument("--grid-path", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--param-id", type=int)
    parser.add_argument("--fold-idx", type=int)
    parser.add_argument("--jobs-json")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--samples", type=int, default=111)
    parser.add_argument("--burnin", type=int, default=10)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--calibration-iters", type=int, default=30)
    parser.add_argument("--jitter-factor", type=float, default=10.0)
    parser.add_argument("--predict-method", default="sequential")
    parser.add_argument("--key-seed", type=int, default=305)
    args = parser.parse_args()
    if args.jobs_json is None and (args.param_id is None or args.fold_idx is None):
        parser.error("Provide either --jobs-json or both --param-id and --fold-idx.")
    return args


def _jobs_from_args(args):
    if args.jobs_json is not None:
        jobs = json.loads(args.jobs_json)
        return [(int(job["param_id"]), int(job["fold_idx"])) for job in jobs]
    return [(int(args.param_id), int(args.fold_idx))]


def _run_one_fit(args, grid, fold_cache, result_index, param_id, fold_idx):
    import jax.numpy as jnp
    import jax.random as jr

    from gp import GaussianProcess
    from gp_kernels import make_indoor_outdoor_mean, make_wifi_kernel

    param_matches = grid.loc[grid["param_id"] == param_id]
    if len(param_matches) != 1:
        raise ValueError(f"Expected exactly one row for param_id={param_id}.")
    param_row = param_matches.iloc[0]

    out_path = result_path(args.result_dir, param_id, fold_idx)
    completed_row = materialize_matching_result(
        args.result_dir,
        param_row,
        fold_idx,
        result_index,
    )
    if completed_row is not None:
        print(f"Already complete for current config: {out_path}", flush=True)
        return completed_row

    total_start = time.time()
    if fold_idx not in fold_cache:
        fold_cache[fold_idx] = load_fold_arrays(args.split_dir, fold_idx)
    arrays = fold_cache[fold_idx]
    X_train = jnp.asarray(arrays["X_train"])
    y_train = jnp.asarray(arrays["y_train"])
    obs_count_train = jnp.asarray(arrays["obs_count_train"])
    obs_sse_train = jnp.asarray(arrays["obs_sse_train"])
    X_test = jnp.asarray(arrays["X_test"])
    y_test = jnp.asarray(arrays["y_test"])

    m = make_indoor_outdoor_mean(X_train, y_train)
    K = make_wifi_kernel(
        ap_form=param_row["ap_form"],
        ls_xy=param_row["ls_xy"],
        ls_z=param_row["ls_z"],
        os_xyz=param_row["os_xyz"],
        ls_t=param_row["ls_t"],
        os_t=param_row["os_t"],
        ls_ap=param_row["ls_ap"],
        os_ap=param_row["os_ap"],
    )

    gp = GaussianProcess(m, K)
    gp.fit(X_train, y_train, obs_count_train, obs_sse=obs_sse_train)

    start = time.time()
    chain = gp.gibbs(
        key=jr.PRNGKey(args.key_seed + 1000 * param_id + fold_idx),
        chains=args.chains,
        samples=args.samples,
        calibration_iters=args.calibration_iters,
        jitter_factor=args.jitter_factor,
    )
    fit_elapsed = time.time() - start

    cov_chains = chain[1][:, args.burnin::args.thin, :]
    start = time.time()
    pred_means, pred_vars = gp.predict(X_test, cov_chains, method=args.predict_method)
    predict_elapsed = time.time() - start

    y_hat = pred_means.mean(axis=0)
    mse = float(jnp.mean((y_test - y_hat) ** 2))

    row = {
        "param_id": int(param_id),
        "fold_idx": int(fold_idx),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "mse": mse,
        "fit_elapsed": float(fit_elapsed),
        "predict_elapsed": float(predict_elapsed),
        "total_elapsed": float(time.time() - total_start),
        "jitter": float(gp.jitter),
        **{
            col: param_row[col]
            for col in ["ap_form", "ls_xy", "ls_z", "os_xyz", "ls_t", "os_t", "ls_ap", "os_ap"]
        },
    }
    write_result_row(out_path, row)
    result_index[result_index_key(row)] = row
    print(
        f"Wrote {out_path} mse={row['mse']:.4f} "
        f"total={row['total_elapsed']:.1f}s",
        flush=True,
    )
    del gp, chain, cov_chains, pred_means, pred_vars, y_hat
    del X_train, y_train, obs_count_train, obs_sse_train, X_test, y_test
    gc.collect()
    return row


def main():
    args = parse_args()
    grid = pd.read_csv(args.grid_path)
    jobs = _jobs_from_args(args)
    fold_cache = {}
    result_index = load_result_index(args.result_dir)
    worker_start = time.time()
    print(f"Worker starting {len(jobs)} fit(s)", flush=True)

    for offset, (param_id, fold_idx) in enumerate(jobs, start=1):
        print(
            f"Worker fit {offset}/{len(jobs)}: param {param_id} fold {fold_idx}",
            flush=True,
        )
        _run_one_fit(args, grid, fold_cache, result_index, param_id, fold_idx)

    # The process will exit after each chunk, but clear Python/JAX state as much
    # as possible for single-process smoke tests and CPU runs.
    fold_cache.clear()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass
    gc.collect()
    print(f"Worker finished in {time.time() - worker_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
