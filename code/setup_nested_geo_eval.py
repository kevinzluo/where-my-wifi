from eval_workflow import write_gp_stage1_grid, write_nested_geo_splits
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent


def main():
    manifest = write_nested_geo_splits(
        out_root=CODE_DIR / "eval_splits" / "geo_outer0",
        data_dir=CODE_DIR / "data",
        raw_csv=REPO_DIR / "data" / "sequoia_sets.csv",
        outer_cv_dir=CODE_DIR / "cv" / "geo",
        outer_fold=0,
        random_state=305,
    )
    grid = write_gp_stage1_grid(CODE_DIR / "gridsearch" / "gp_geo_stage1_grid.csv")

    print("Wrote eval_splits/geo_outer0")
    print(manifest)
    print(f"Wrote gridsearch/gp_geo_stage1_grid.csv with {len(grid)} rows")


if __name__ == "__main__":
    main()
