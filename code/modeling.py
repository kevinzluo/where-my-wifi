import numpy as np
import pandas as pd


def _bin_edges(values, bins, bounds):
    if bounds is None:
        bounds = [values.min(), values.max()]
    if len(bounds) != 2:
        raise ValueError("bounds must contain exactly two values: [min, max].")

    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("bounds must be finite.")
    if lower == upper:
        pad = max(abs(lower) * 1e-12, 1e-12)
        lower -= pad
        upper += pad
    if lower > upper:
        raise ValueError("bounds must be finite and increasing.")

    if np.isscalar(bins):
        n_bins = int(bins)
        if n_bins <= 0:
            raise ValueError("bins must be a positive integer.")
        return np.linspace(lower, upper, n_bins + 1)

    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bin edges must be a one-dimensional array with at least two values.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin edges must be strictly increasing.")
    return edges


def _geographic_grid_assignments(
    df,
    lat_col,
    lon_col,
    lat_bins,
    lon_bins,
    lat_bounds,
    lon_bounds,
):
    missing_columns = [col for col in [lat_col, lon_col] if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing column(s): {', '.join(missing_columns)}")

    coords = df[[lat_col, lon_col]].astype(float)
    valid_coord = np.isfinite(coords[lat_col]) & np.isfinite(coords[lon_col])
    if not valid_coord.any():
        raise ValueError("No rows have finite latitude and longitude values.")

    lat_edges = _bin_edges(coords.loc[valid_coord, lat_col], lat_bins, lat_bounds)
    lon_edges = _bin_edges(coords.loc[valid_coord, lon_col], lon_bins, lon_bounds)

    lat_grid = np.digitize(coords[lat_col], lat_edges, right=False) - 1
    lon_grid = np.digitize(coords[lon_col], lon_edges, right=False) - 1

    # Include points exactly on the upper edge in the final cell.
    lat_grid = np.where(coords[lat_col] == lat_edges[-1], len(lat_edges) - 2, lat_grid)
    lon_grid = np.where(coords[lon_col] == lon_edges[-1], len(lon_edges) - 2, lon_grid)

    in_grid = (
        valid_coord
        & (lat_grid >= 0)
        & (lat_grid < len(lat_edges) - 1)
        & (lon_grid >= 0)
        & (lon_grid < len(lon_edges) - 1)
    )
    if not in_grid.any():
        raise ValueError("No rows fall inside the latitude/longitude grid.")

    row_cells = list(zip(lat_grid, lon_grid))
    occupied_cells = np.unique(np.column_stack([lat_grid[in_grid], lon_grid[in_grid]]), axis=0)
    return np.asarray(in_grid), row_cells, occupied_cells


def _cell_mask(in_grid, row_cells, selected_cells):
    selected_cells = {tuple(cell) for cell in selected_cells}
    return in_grid & np.array([cell in selected_cells for cell in row_cells])


def _valid_lonlat(df, lat_col, lon_col):
    missing_columns = [col for col in [lat_col, lon_col] if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing column(s): {', '.join(missing_columns)}")

    coords = df[[lat_col, lon_col]].astype(float)
    valid_coord = np.isfinite(coords[lat_col]) & np.isfinite(coords[lon_col])
    return coords, np.asarray(valid_coord)


def _footprint_contains_points(footprint, points):
    from matplotlib.path import Path as MplPath

    depth = np.zeros(points.shape[0], dtype=int)
    for ring in footprint["positive_rings"]:
        if len(ring) >= 3:
            depth += MplPath(ring).contains_points(points)
    for ring in footprint["negative_rings"]:
        if len(ring) >= 3:
            depth -= MplPath(ring).contains_points(points)
    return depth > 0


def _building_label(footprint, index):
    name = footprint.get("name")
    building_type = footprint.get("type", "building")
    building_id = footprint.get("id")
    if name:
        return str(name)
    if building_id is not None:
        return f"{building_type}/{building_id}"
    return f"building_{index}"


def geographic_train_test_split(
    df,
    lat_col="latitude",
    lon_col="longitude",
    lat_bins=20,
    lon_bins=20,
    train_frac=0.7,
    lat_bounds=None,
    lon_bounds=None,
    random_state=None,
):
    """
    Split rows by occupied latitude/longitude grid cells.

    The grid is built from lat_bins x lon_bins cells. Empty cells are ignored,
    then train_frac of the occupied cells are assigned to train. All rows in a
    selected cell go to the same split.
    """
    if not 0 <= train_frac <= 1:
        raise ValueError("train_frac must be between 0 and 1.")
    in_grid, row_cells, occupied_cells = _geographic_grid_assignments(
        df,
        lat_col,
        lon_col,
        lat_bins,
        lon_bins,
        lat_bounds,
        lon_bounds,
    )

    rng = np.random.default_rng(random_state)
    n_occupied = len(occupied_cells)
    n_train = int(round(train_frac * n_occupied))
    if 0 < train_frac < 1 and n_occupied > 1:
        n_train = int(np.clip(n_train, 1, n_occupied - 1))

    train_cell_idx = rng.choice(n_occupied, size=n_train, replace=False)
    train_mask = _cell_mask(in_grid, row_cells, occupied_cells[train_cell_idx])
    test_mask = in_grid & ~train_mask

    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def geographic_kfold_split(
    df,
    K=5,
    lat_col="latitude",
    lon_col="longitude",
    lat_bins=20,
    lon_bins=20,
    lat_bounds=None,
    lon_bounds=None,
    random_state=None,
):
    """
    Create K cross-validation folds from occupied latitude/longitude grid cells.

    Occupied cells are randomly split into K groups. Each returned fold is a
    (train_df, test_df) pair where one group of cells is held out for test and
    all remaining cells are used for train.
    """
    if int(K) != K or K < 2:
        raise ValueError("K must be an integer greater than or equal to 2.")
    K = int(K)

    in_grid, row_cells, occupied_cells = _geographic_grid_assignments(
        df,
        lat_col,
        lon_col,
        lat_bins,
        lon_bins,
        lat_bounds,
        lon_bounds,
    )
    if K > len(occupied_cells):
        raise ValueError("K cannot be larger than the number of occupied grid cells.")

    rng = np.random.default_rng(random_state)
    shuffled_cells = occupied_cells[rng.permutation(len(occupied_cells))]
    heldout_cell_groups = np.array_split(shuffled_cells, K)

    folds = []
    for test_cells in heldout_cell_groups:
        test_mask = _cell_mask(in_grid, row_cells, test_cells)
        train_mask = in_grid & ~test_mask
        folds.append((df.loc[train_mask].copy(), df.loc[test_mask].copy()))
    return folds


def leave_one_building_out_split(
    df,
    lat_col="latitude",
    lon_col="longitude",
    context=None,
    building_footprints=None,
    min_test_rows=1,
):
    """
    Create leave-one-building-out datasets from OSM building footprints.

    For each building footprint, rows whose lat/lon fall inside that building
    are held out for test. All finite-coordinate rows outside that building are
    used for train. Buildings with fewer than min_test_rows observations are
    skipped.

    Returns a list of dictionaries with keys:
        building: metadata for the held-out building
        train: training DataFrame
        test: held-out DataFrame
        mask: boolean mask over df marking held-out rows
    """
    if min_test_rows < 0:
        raise ValueError("min_test_rows must be nonnegative.")

    coords, valid_coord = _valid_lonlat(df, lat_col, lon_col)
    if not valid_coord.any():
        raise ValueError("No rows have finite latitude and longitude values.")

    if building_footprints is None:
        from wifiplotting import OSMPlotContext

        if context is None:
            context = OSMPlotContext.from_dataframe(df, lon_col=lon_col, lat_col=lat_col)
        if not context.load_buildings():
            raise RuntimeError(
                "OSM building footprints are unavailable; cannot build leave-one-building-out splits."
            ) from context.building_error
        building_footprints = context.building_footprints

    if context is None:
        raise ValueError(
            "context is required when precomputed building_footprints are provided."
        )

    if not building_footprints:
        raise ValueError("No building footprints were provided or loaded.")

    x, y = context.to_world(
        coords.loc[valid_coord, lon_col].to_numpy(),
        coords.loc[valid_coord, lat_col].to_numpy(),
    )
    valid_points = np.column_stack([x, y])
    valid_indices = np.flatnonzero(valid_coord)

    splits = []
    for index, footprint in enumerate(building_footprints):
        valid_mask = _footprint_contains_points(footprint, valid_points)
        mask = np.zeros(len(df), dtype=bool)
        mask[valid_indices] = valid_mask

        n_test = int(mask.sum())
        if n_test < min_test_rows:
            continue

        train_mask = valid_coord & ~mask
        building = {
            "index": index,
            "id": footprint.get("id"),
            "type": footprint.get("type"),
            "name": footprint.get("name"),
            "label": _building_label(footprint, index),
            "n_test": n_test,
            "n_train": int(train_mask.sum()),
        }
        splits.append({
            "building": building,
            "train": df.loc[train_mask].copy(),
            "test": df.loc[mask].copy(),
            "mask": pd.Series(mask, index=df.index, name="held_out_building"),
        })

    return splits
