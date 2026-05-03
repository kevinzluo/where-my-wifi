#!/usr/bin/env python3
"""
Interactive correction UI for the phone_location_log.csv GPS track.

Run from the where-my-wifi directory:

    workon stats305c
    python code/correct_phone_locations.py

Controls:
    click            preview a correction for the current point
    shift-click      commit the clicked correction and advance
    enter/return     commit the preview and advance
    right/space/n    approve with no new correction edit and advance
    left/b           go back one point
    g                jump to an observation row by number
    s                save
    q                save and quit
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd

from wifiplotting import OSMPlotContext, TILE_SIZE, lonlat_to_world

RAW_COLUMNS = [
    "timestamp",
    "sample_index",
    "latitude",
    "longitude",
    "altitude_m",
]
CORRECTION_COLUMNS = ["latitude_correction_edit", "longitude_correction_edit"]
BASE_OUTPUT_COLUMNS = RAW_COLUMNS + CORRECTION_COLUMNS
WIFI_METADATA_COLUMNS = [
    "wifi_environment",
    "wifi_timestamp_utc",
    "wifi_timestamp_delta_seconds",
    "wifi_measurement_set_id",
    "wifi_building",
    "wifi_floor",
    "wifi_row",
]
OUTPUT_COLUMNS = BASE_OUTPUT_COLUMNS + WIFI_METADATA_COLUMNS


@dataclass(frozen=True)
class CorrectionDelta:
    latitude: float
    longitude: float


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_raw_locations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        na_values=["null", "none", "nan"],
        keep_default_na=True,
    )
    if df.empty:
        raise ValueError(f"No rows found in {path}.")

    for column in ["sample_index", "latitude", "longitude", "altitude_m"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_coords = df[["latitude", "longitude"]].isna().any(axis=1)
    if missing_coords.any():
        first_bad = int(missing_coords.idxmax())
        raise ValueError(
            f"Row {first_bad} in {path} is missing a latitude or longitude."
        )
    return df


def raw_prefix_matches(raw_df: pd.DataFrame, corrected_df: pd.DataFrame) -> bool:
    if len(corrected_df) > len(raw_df):
        return False
    raw_prefix = raw_df.iloc[: len(corrected_df)]
    if corrected_df["timestamp"].astype(str).tolist() != raw_prefix["timestamp"].astype(str).tolist():
        return False
    for column in ["sample_index", "latitude", "longitude", "altitude_m"]:
        raw_values = pd.to_numeric(raw_prefix[column], errors="coerce").to_numpy()
        corrected_values = pd.to_numeric(corrected_df[column], errors="coerce").to_numpy()
        if not np.allclose(raw_values, corrected_values, equal_nan=True):
            return False
    return True


def initialize_corrected_locations(raw_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        corrected_df = pd.read_csv(output_path, keep_default_na=False)
        missing = [column for column in BASE_OUTPUT_COLUMNS if column not in corrected_df.columns]
        if missing:
            raise ValueError(
                f"{output_path} is missing expected columns: {', '.join(missing)}"
            )
        corrected_df = corrected_df[BASE_OUTPUT_COLUMNS].copy()
        if not raw_prefix_matches(raw_df, corrected_df):
            backup_path = backup_existing_output(output_path)
            backup_message = f" Backed it up to {backup_path}." if backup_path else ""
            raise ValueError(
                f"{output_path} does not match the raw log prefix.{backup_message}"
            )
        if len(corrected_df) < len(raw_df):
            new_rows = raw_df.iloc[len(corrected_df) :].copy()
            new_rows["latitude_correction_edit"] = np.nan
            new_rows["longitude_correction_edit"] = np.nan
            corrected_df = pd.concat(
                [corrected_df, new_rows[BASE_OUTPUT_COLUMNS]],
                ignore_index=True,
            )
        for column in CORRECTION_COLUMNS:
            corrected_df[column] = pd.to_numeric(corrected_df[column], errors="coerce")
        return corrected_df

    corrected_df = raw_df.copy()
    corrected_df["latitude_correction_edit"] = np.nan
    corrected_df["longitude_correction_edit"] = np.nan
    return corrected_df[BASE_OUTPUT_COLUMNS]


def read_wifi_samples(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def parse_timestamp_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.replace("", pd.NA),
        errors="coerce",
        format="mixed",
        utc=True,
    )


def empty_wifi_metadata(index: pd.Index) -> pd.DataFrame:
    metadata = pd.DataFrame(index=index)
    metadata["wifi_environment"] = "null"
    metadata["wifi_timestamp_utc"] = pd.NA
    metadata["wifi_timestamp_delta_seconds"] = np.nan
    metadata["wifi_measurement_set_id"] = pd.NA
    metadata["wifi_building"] = pd.NA
    metadata["wifi_floor"] = pd.NA
    metadata["wifi_row"] = pd.NA
    return metadata[WIFI_METADATA_COLUMNS]


def build_wifi_metadata(
    raw_df: pd.DataFrame,
    wifi_df: pd.DataFrame,
    *,
    tolerance_seconds: float = 30.0,
) -> pd.DataFrame:
    metadata = empty_wifi_metadata(raw_df.index)
    if wifi_df.empty:
        return metadata

    required = {"sample_timestamp_utc", "measurement_set_timestamp_utc", "environment"}
    missing = sorted(required - set(wifi_df.columns))
    if missing:
        raise ValueError(f"Wi-Fi samples are missing expected columns: {', '.join(missing)}")

    phone = pd.DataFrame(index=raw_df.index)
    phone["phone_row"] = raw_df.index
    phone["phone_timestamp_utc"] = parse_timestamp_utc(raw_df["timestamp"])

    wifi = wifi_df.reset_index(names="wifi_row").copy()
    sample_timestamp = parse_timestamp_utc(wifi["sample_timestamp_utc"])
    measurement_timestamp = parse_timestamp_utc(wifi["measurement_set_timestamp_utc"])
    wifi["wifi_lookup_timestamp_utc"] = sample_timestamp.combine_first(measurement_timestamp)

    valid_phone = phone["phone_timestamp_utc"].notna()
    valid_wifi = wifi["wifi_lookup_timestamp_utc"].notna()
    if not valid_phone.any() or not valid_wifi.any():
        return metadata

    optional_columns = [
        column
        for column in ["measurement_set_id", "environment", "building", "floor"]
        if column in wifi.columns
    ]
    matched = pd.merge_asof(
        phone.loc[valid_phone].sort_values("phone_timestamp_utc", kind="stable"),
        wifi.loc[valid_wifi, ["wifi_row", "wifi_lookup_timestamp_utc"] + optional_columns].sort_values(
            "wifi_lookup_timestamp_utc",
            kind="stable",
        ),
        left_on="phone_timestamp_utc",
        right_on="wifi_lookup_timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    ).set_index("phone_row")

    matched_mask = matched["wifi_row"].notna()
    if not matched_mask.any():
        return metadata

    matched_rows = matched.loc[matched_mask]
    metadata.loc[matched_rows.index, "wifi_environment"] = (
        matched_rows["environment"].replace("", pd.NA).fillna("null").astype(str)
    )
    metadata.loc[matched_rows.index, "wifi_timestamp_utc"] = (
        matched_rows["wifi_lookup_timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    )
    metadata.loc[matched_rows.index, "wifi_timestamp_delta_seconds"] = (
        matched_rows["wifi_lookup_timestamp_utc"] - matched_rows["phone_timestamp_utc"]
    ).dt.total_seconds()
    if "measurement_set_id" in matched_rows.columns:
        metadata.loc[matched_rows.index, "wifi_measurement_set_id"] = (
            matched_rows["measurement_set_id"].replace("", pd.NA)
        )
    if "building" in matched_rows.columns:
        metadata.loc[matched_rows.index, "wifi_building"] = (
            matched_rows["building"].replace("", pd.NA)
        )
    if "floor" in matched_rows.columns:
        metadata.loc[matched_rows.index, "wifi_floor"] = (
            matched_rows["floor"].replace("", pd.NA)
        )
    metadata.loc[matched_rows.index, "wifi_row"] = matched_rows["wifi_row"].astype("Int64")
    return metadata[WIFI_METADATA_COLUMNS]


def attach_wifi_metadata(
    corrected_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    wifi_df: pd.DataFrame,
    *,
    tolerance_seconds: float = 30.0,
) -> pd.DataFrame:
    corrected_df = corrected_df[BASE_OUTPUT_COLUMNS].copy()
    metadata = build_wifi_metadata(
        raw_df,
        wifi_df,
        tolerance_seconds=tolerance_seconds,
    )
    return pd.concat([corrected_df.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)[
        OUTPUT_COLUMNS
    ]


def save_corrected_locations(corrected_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_output_columns(corrected_df).to_csv(output_path, index=False, na_rep="")


def backup_existing_output(output_path: Path) -> Optional[Path]:
    if not output_path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = output_path.with_name(f"{output_path.name}.bak.{timestamp}")
    suffix = 1
    while backup_path.exists():
        backup_path = output_path.with_name(f"{output_path.name}.bak.{timestamp}.{suffix}")
        suffix += 1
    shutil.copy2(output_path, backup_path)
    return backup_path


def ensure_output_columns(corrected_df: pd.DataFrame) -> pd.DataFrame:
    output_df = corrected_df.copy()
    missing_base = [column for column in BASE_OUTPUT_COLUMNS if column not in output_df.columns]
    if missing_base:
        raise ValueError(f"Corrected data is missing columns: {', '.join(missing_base)}")
    if "wifi_environment" not in output_df.columns:
        output_df["wifi_environment"] = "null"
    output_df["wifi_environment"] = output_df["wifi_environment"].replace("", pd.NA).fillna("null")
    for column in WIFI_METADATA_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = pd.NA
    return output_df[OUTPUT_COLUMNS]


def first_unprocessed_index(corrected_df: pd.DataFrame) -> int:
    processed = correction_mask(corrected_df)
    if processed.all():
        return max(0, len(corrected_df) - 1)
    return int((~processed).idxmax())


def correction_mask(corrected_df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(corrected_df["latitude_correction_edit"], errors="coerce").notna()
        & pd.to_numeric(corrected_df["longitude_correction_edit"], errors="coerce").notna()
    )


def world_to_lonlat(x: float, y: float, zoom: int) -> Tuple[float, float]:
    scale = TILE_SIZE * (2 ** zoom)
    lon = x / scale * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / scale))))
    return lon, lat


def approximate_meters(delta: CorrectionDelta, latitude: float) -> Tuple[float, float]:
    lat_m = delta.latitude * 111_320.0
    lon_m = delta.longitude * 111_320.0 * math.cos(math.radians(latitude))
    return lat_m, lon_m


def padded_bounds_meters(
    lons: Sequence[float],
    lats: Sequence[float],
    *,
    pad_meters: float = 90.0,
    pad_fraction: float = 0.30,
) -> Tuple[float, float, float, float]:
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    center_lat = (lat_min + lat_max) / 2.0
    lat_pad_degrees = pad_meters / 111_320.0
    lon_scale = 111_320.0 * max(0.2, math.cos(math.radians(center_lat)))
    lon_pad_degrees = pad_meters / lon_scale
    lat_pad = max(lat_pad_degrees, (lat_max - lat_min) * pad_fraction)
    lon_pad = max(lon_pad_degrees, (lon_max - lon_min) * pad_fraction)
    return lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad


class PhoneLocationCorrectionUI:
    def __init__(
        self,
        raw_df: pd.DataFrame,
        corrected_df: pd.DataFrame,
        output_path: Path,
        *,
        history_window: int = 50,
        future_window: int = 25,
        start_index: Optional[int] = None,
        pad_meters: float = 90.0,
        max_tiles: int = 48,
        min_zoom: int = 15,
        max_zoom: int = 19,
    ) -> None:
        self.raw_df = raw_df.reset_index(drop=True)
        self.corrected_df = corrected_df.reset_index(drop=True)
        self.output_path = output_path
        self.history_window = history_window
        self.future_window = future_window
        self.index = first_unprocessed_index(corrected_df) if start_index is None else start_index
        self.index = min(max(0, self.index), len(self.raw_df) - 1)
        self.pad_meters = pad_meters
        self.max_tiles = max_tiles
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

        self.preview_delta: Optional[CorrectionDelta] = None
        self.basemap_loaded = False
        self.context: Optional[OSMPlotContext] = None
        self.context_cache = {}
        self.updating_slider = False

        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.subplots_adjust(bottom=0.16)
        slider_ax = self.fig.add_axes([0.14, 0.055, 0.72, 0.035])
        self.row_slider = Slider(
            slider_ax,
            "Row",
            1,
            len(self.raw_df),
            valinit=self.index + 1,
            valstep=1,
        )
        self.row_slider.on_changed(self.on_slider_changed)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.reset_preview()

    def run(self) -> None:
        self.redraw()
        plt.show()

    def row_edit(self, index: int) -> Optional[CorrectionDelta]:
        row = self.corrected_df.loc[index]
        lat = pd.to_numeric(row["latitude_correction_edit"], errors="coerce")
        lon = pd.to_numeric(row["longitude_correction_edit"], errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            return None
        return CorrectionDelta(float(lat), float(lon))

    def cumulative_delta_before(self, index: int) -> CorrectionDelta:
        if index <= 0:
            return CorrectionDelta(0.0, 0.0)
        edits = self.corrected_df.loc[: index - 1, CORRECTION_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        return CorrectionDelta(
            latitude=float(edits["latitude_correction_edit"].fillna(0.0).sum()),
            longitude=float(edits["longitude_correction_edit"].fillna(0.0).sum()),
        )

    def cumulative_delta_at(self, index: int) -> CorrectionDelta:
        before = self.cumulative_delta_before(index)
        edit = self.row_edit(index)
        if edit is None:
            return before
        return CorrectionDelta(
            latitude=before.latitude + edit.latitude,
            longitude=before.longitude + edit.longitude,
        )

    def reset_preview(self) -> None:
        self.preview_delta = self.cumulative_delta_at(self.index)

    def corrected_lonlat(self, index: int, delta: Optional[CorrectionDelta] = None) -> Tuple[float, float]:
        if delta is None:
            delta = self.cumulative_delta_at(index)
        row = self.raw_df.loc[index]
        lat = float(row["latitude"]) + delta.latitude
        lon = float(row["longitude"]) + delta.longitude
        return lon, lat

    def current_corrected_lonlat(self) -> Tuple[float, float]:
        return self.corrected_lonlat(self.index, self.preview_delta)

    def displayed_lonlat(self, index: int) -> Tuple[float, float]:
        if index == self.index:
            return self.current_corrected_lonlat()
        if index > self.index:
            return self.corrected_lonlat(index, self.preview_cumulative_delta_at(index))
        return self.corrected_lonlat(index)

    def preview_cumulative_delta_at(self, index: int) -> CorrectionDelta:
        preview = self.preview_delta or self.cumulative_delta_at(self.index)
        if index <= self.index:
            return preview
        edits = self.corrected_df.loc[self.index + 1 : index, CORRECTION_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        return CorrectionDelta(
            latitude=preview.latitude + float(edits["latitude_correction_edit"].fillna(0.0).sum()),
            longitude=preview.longitude + float(edits["longitude_correction_edit"].fillna(0.0).sum()),
        )

    def recent_committed_indices(self) -> Sequence[int]:
        processed = correction_mask(self.corrected_df)
        candidates = [
            idx
            for idx in range(max(0, self.index - self.history_window), self.index)
            if bool(processed.iloc[idx])
        ]
        return candidates[-self.history_window :]

    def future_indices(self) -> Sequence[int]:
        end = min(len(self.raw_df), self.index + 1 + self.future_window)
        return list(range(self.index + 1, end))

    def current_bounds(self) -> Tuple[float, float, float, float]:
        lons = []
        lats = []

        for idx in self.recent_committed_indices():
            lon, lat = self.corrected_lonlat(idx)
            lons.append(lon)
            lats.append(lat)

        lon, lat = self.current_corrected_lonlat()
        lons.append(lon)
        lats.append(lat)

        for idx in self.future_indices():
            lon, lat = self.displayed_lonlat(idx)
            lons.append(lon)
            lats.append(lat)

        return padded_bounds_meters(lons, lats, pad_meters=self.pad_meters)

    def context_cache_key(self, context: OSMPlotContext) -> Tuple[int, int, int, int, int]:
        left, top = lonlat_to_world(context.bounds[0], context.bounds[3], context.zoom)
        right, bottom = lonlat_to_world(context.bounds[2], context.bounds[1], context.zoom)
        return (
            context.zoom,
            math.floor(left / TILE_SIZE),
            math.floor(right / TILE_SIZE),
            math.floor(top / TILE_SIZE),
            math.floor(bottom / TILE_SIZE),
        )

    def make_context(self) -> OSMPlotContext:
        context = OSMPlotContext(
            self.current_bounds(),
            max_tiles=self.max_tiles,
            min_zoom=self.min_zoom,
            max_zoom=self.max_zoom,
        )
        key = self.context_cache_key(context)
        cached = self.context_cache.get(key)
        if cached is not None:
            context.image, context.extent = cached
            context.basemap_loaded = True
        else:
            context.load_basemap()
            if context.basemap_loaded:
                self.context_cache[key] = (context.image, context.extent)
        return context

    def redraw(self) -> None:
        self.ax.clear()
        self.context = self.make_context()
        if self.context.basemap_loaded:
            self.ax.imshow(self.context.image, extent=self.context.extent, origin="upper")
            self.context.apply_world_axis(self.ax)
            self.basemap_loaded = True
        else:
            self.basemap_loaded = False
            left, bottom, right, top = self.current_bounds()
            self.ax.set_xlim(left, right)
            self.ax.set_ylim(bottom, top)
            self.ax.set_xlabel("Longitude")
            self.ax.set_ylabel("Latitude")
            self.ax.grid(alpha=0.25)

        self.draw_history()
        self.draw_future_points()
        self.draw_current_point()
        self.draw_status()
        self.sync_slider()
        self.fig.canvas.draw_idle()

    def project_lonlats(self, lons: Sequence[float], lats: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        lons_array = np.array(lons)
        lats_array = np.array(lats)
        if self.basemap_loaded and self.context is not None:
            return self.context.to_world(lons_array, lats_array)
        return lons_array, lats_array

    def draw_history(self) -> None:
        indices = list(self.recent_committed_indices())
        if not indices:
            return
        lons, lats = zip(*(self.corrected_lonlat(idx) for idx in indices))
        xs, ys = self.project_lonlats(lons, lats)
        colors = plt.get_cmap("Greys")(np.linspace(0.25, 0.85, len(indices)))
        self.ax.scatter(
            xs,
            ys,
            c=colors,
            s=40,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
            label=f"past {len(indices)}",
        )

    def draw_future_points(self) -> None:
        indices = list(self.future_indices())
        if not indices:
            return
        lons, lats = zip(*(self.displayed_lonlat(idx) for idx in indices))
        xs, ys = self.project_lonlats(lons, lats)
        colors = self.future_colors(indices)
        sizes = self.future_sizes(len(indices))
        edgecolors = ["black"] + ["white"] * (len(indices) - 1)
        linewidths = [1.8] + [0.45] * (len(indices) - 1)
        self.ax.scatter(
            xs,
            ys,
            c=colors,
            marker="o",
            s=sizes,
            edgecolors=edgecolors,
            linewidths=linewidths,
            zorder=8,
            label=f"next {len(indices)}",
        )

    def future_sizes(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.array([])
        if count == 1:
            return np.array([120])
        if count == 2:
            return np.array([120, 82])
        return np.concatenate([[120, 82], np.linspace(58, 24, count - 2)])

    def future_colors(self, indices: Sequence[int]) -> np.ndarray:
        count = len(indices)
        intensities = self.sequence_intensities(count)
        alphas = self.sequence_alphas(count)
        return np.array(
            [
                self.environment_color(self.row_environment(index), intensity, alpha)
                for index, intensity, alpha in zip(indices, intensities, alphas)
            ]
        )

    def sequence_alphas(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.array([])
        if count == 1:
            return np.array([1.0])
        if count == 2:
            return np.array([1.0, 0.70])
        return np.concatenate([[1.0, 0.70], np.linspace(0.52, 0.18, count - 2)])

    def sequence_intensities(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.array([])
        if count == 1:
            return np.array([1.0])
        if count == 2:
            return np.array([1.0, 0.45])
        return np.concatenate([[1.0, 0.45], np.linspace(0.34, 0.10, count - 2)])

    def row_environment(self, index: int) -> str:
        if "wifi_environment" not in self.corrected_df.columns:
            return "null"
        environment = self.corrected_df.loc[index, "wifi_environment"]
        if pd.isna(environment) or str(environment).strip() == "":
            return "null"
        return str(environment).strip().lower()

    def environment_color(self, environment: str, intensity: float, alpha: float) -> Tuple[float, float, float, float]:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        if environment == "indoor":
            light = np.array([1.0, 0.78, 0.78])
            strong = np.array([0.86, 0.02, 0.02])
        elif environment == "outdoor":
            light = np.array([0.72, 1.0, 0.72])
            strong = np.array([0.00, 0.55, 0.00])
        else:
            light = np.array([0.86, 0.74, 1.0])
            strong = np.array([0.42, 0.02, 0.88])
        color = light * (1.0 - intensity) + strong * intensity
        return float(color[0]), float(color[1]), float(color[2]), float(alpha)

    def draw_current_point(self) -> None:
        row = self.raw_df.loc[self.index]
        raw_lon = float(row["longitude"])
        raw_lat = float(row["latitude"])
        corrected_lon, corrected_lat = self.current_corrected_lonlat()

        if self.basemap_loaded and self.context is not None:
            raw_x, raw_y = self.context.to_world(raw_lon, raw_lat)
            corrected_x, corrected_y = self.context.to_world(corrected_lon, corrected_lat)
        else:
            raw_x, raw_y = raw_lon, raw_lat
            corrected_x, corrected_y = corrected_lon, corrected_lat

        raw_visible = self.point_in_current_view(raw_x, raw_y)
        if raw_visible:
            self.ax.scatter(
                [raw_x],
                [raw_y],
                marker="x",
                c="black",
                s=80,
                linewidths=1.8,
                alpha=0.65,
                zorder=12,
                label="raw current",
            )
        self.ax.scatter(
            [corrected_x],
            [corrected_y],
            marker="*",
            c="#d62728",
            s=330,
            edgecolors="white",
            linewidths=1.4,
            zorder=30,
            label="corrected current",
        )
        if raw_visible:
            self.ax.plot([raw_x, corrected_x], [raw_y, corrected_y], color="#d62728", alpha=0.50, zorder=11)
        self.ax.legend(loc="upper right", framealpha=0.9)

    def point_in_current_view(self, x: float, y: float) -> bool:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)

    def current_wifi_status(self) -> Tuple[str, str, str, str]:
        row = self.corrected_df.loc[self.index]
        environment = row.get("wifi_environment", "null")
        if pd.isna(environment) or str(environment).strip() == "":
            environment = "null"
        building = row.get("wifi_building", "null")
        if pd.isna(building) or str(building).strip() == "":
            building = "null"
        floor = row.get("wifi_floor", "null")
        if pd.isna(floor) or str(floor).strip() == "":
            floor = "null"
        delta = pd.to_numeric(row.get("wifi_timestamp_delta_seconds", np.nan), errors="coerce")
        delta_text = "null" if pd.isna(delta) else f"{float(delta):+.2f}s"
        return str(environment), str(building), str(floor), delta_text

    def draw_status(self) -> None:
        row = self.raw_df.loc[self.index]
        processed = int(correction_mask(self.corrected_df).sum())
        cumulative_delta = self.preview_delta or CorrectionDelta(0.0, 0.0)
        edit_delta = self.preview_edit_for_current()
        lat_m, lon_m = approximate_meters(cumulative_delta, float(row["latitude"]))
        edit_lat_m, edit_lon_m = approximate_meters(edit_delta, float(row["latitude"]))
        saved = self.row_edit(self.index) is not None
        status = "saved" if saved and edit_delta == self.row_edit(self.index) else "preview"
        if self.row_edit(self.index) is None and edit_delta == CorrectionDelta(0.0, 0.0):
            status = "carried"
        time_of_day = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(time_of_day):
            time_text = str(row["timestamp"])
        else:
            time_text = time_of_day.strftime("%Y-%m-%d %H:%M:%S %z")
        wifi_environment, wifi_building, wifi_floor, wifi_delta_text = self.current_wifi_status()

        self.ax.set_title(
            f"Row {self.index + 1}/{len(self.raw_df)} | sample {row['sample_index']} | "
            f"time {time_text} | env {wifi_environment} | building {wifi_building} | "
            f"floor {wifi_floor} | wifi delta {wifi_delta_text}\n"
            f"{status} cumulative: "
            f"dlat {cumulative_delta.latitude:.8f} ({lat_m:+.1f} m), "
            f"dlon {cumulative_delta.longitude:.8f} ({lon_m:+.1f} m) | "
            f"edit: dlat {edit_delta.latitude:.8f} ({edit_lat_m:+.1f} m), "
            f"dlon {edit_delta.longitude:.8f} ({edit_lon_m:+.1f} m)"
        )
        controls = (
            "Click preview | Shift-click commit | Enter commit | "
            "Right/Space/n no new edit | Left/b back | g jump | s save | q save+quit"
        )
        self.ax.text(
            0.01,
            0.01,
            f"{processed}/{len(self.raw_df)} rows corrected\n"
            "Past gray; future indoor red, outdoor green, null purple; +1 is largest/outlined\n"
            f"{controls}",
            transform=self.ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
            zorder=10,
        )

    def event_lonlat(self, event) -> Optional[Tuple[float, float]]:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        if self.basemap_loaded and self.context is not None:
            return world_to_lonlat(float(event.xdata), float(event.ydata), self.context.zoom)
        return float(event.xdata), float(event.ydata)

    def delta_from_lonlat(self, lon: float, lat: float) -> CorrectionDelta:
        row = self.raw_df.loc[self.index]
        return CorrectionDelta(
            latitude=lat - float(row["latitude"]),
            longitude=lon - float(row["longitude"]),
        )

    def on_click(self, event) -> None:
        lonlat = self.event_lonlat(event)
        if lonlat is None:
            return
        lon, lat = lonlat
        self.preview_delta = self.delta_from_lonlat(lon, lat)
        if event.key and "shift" in str(event.key).lower():
            self.commit_current(self.preview_edit_for_current())
            self.advance()
        else:
            self.redraw()

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in {"enter", "return"}:
            self.commit_current(self.preview_edit_for_current())
            self.advance()
        elif key in {"right", " ", "space", "n"}:
            self.commit_current(self.row_edit(self.index) or CorrectionDelta(0.0, 0.0))
            self.advance()
        elif key in {"left", "b"}:
            self.go_back()
        elif key == "g":
            self.prompt_for_jump()
        elif key == "s":
            self.save()
            print(f"Saved {self.output_path}")
        elif key == "q":
            self.save()
            print(f"Saved {self.output_path}")
            plt.close(self.fig)

    def preview_edit_for_current(self) -> CorrectionDelta:
        cumulative_before = self.cumulative_delta_before(self.index)
        preview = self.preview_delta or cumulative_before
        return CorrectionDelta(
            latitude=preview.latitude - cumulative_before.latitude,
            longitude=preview.longitude - cumulative_before.longitude,
        )

    def commit_current(self, edit: CorrectionDelta) -> None:
        self.corrected_df.loc[self.index, "latitude_correction_edit"] = edit.latitude
        self.corrected_df.loc[self.index, "longitude_correction_edit"] = edit.longitude
        self.preview_delta = self.cumulative_delta_at(self.index)
        self.save()

    def advance(self) -> None:
        if self.index < len(self.raw_df) - 1:
            self.index += 1
        self.reset_preview()
        self.redraw()

    def go_back(self) -> None:
        if self.index > 0:
            self.index -= 1
        self.reset_preview()
        self.redraw()

    def jump_to_row(self, row_number: int) -> None:
        self.index = min(max(0, row_number - 1), len(self.raw_df) - 1)
        self.reset_preview()
        self.redraw()

    def prompt_for_jump(self) -> None:
        try:
            raw_value = input(f"Jump to row number 1-{len(self.raw_df)}: ").strip()
            if not raw_value:
                return
            self.jump_to_row(int(raw_value))
        except Exception as exc:
            print(f"Could not jump: {exc}")

    def on_slider_changed(self, value: float) -> None:
        if self.updating_slider:
            return
        self.jump_to_row(int(round(value)))

    def sync_slider(self) -> None:
        self.updating_slider = True
        try:
            self.row_slider.set_val(self.index + 1)
        finally:
            self.updating_slider = False

    def save(self) -> None:
        save_corrected_locations(self.corrected_df, self.output_path)


def parse_args() -> argparse.Namespace:
    root = project_root_from_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data" / "phone_location_log.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "phone_location_log_corrected_edits.csv",
    )
    parser.add_argument(
        "--wifi-samples",
        type=Path,
        default=root / "data" / "wifi_samples.csv",
        help="Wi-Fi samples CSV used to annotate indoor/outdoor status.",
    )
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--future", type=int, default=25)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument(
        "--pad-meters",
        type=float,
        default=30.0,
        help="Pad around the displayed correction window, in meters.",
    )
    parser.add_argument("--max-tiles", type=int, default=48)
    parser.add_argument("--min-zoom", type=int, default=15)
    parser.add_argument("--max-zoom", type=int, default=19)
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create/update the corrected CSV and exit without opening the UI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = read_raw_locations(args.input)
    corrected_df = initialize_corrected_locations(raw_df, args.output)
    wifi_df = read_wifi_samples(args.wifi_samples)
    corrected_df = attach_wifi_metadata(corrected_df, raw_df, wifi_df)
    if args.init_only:
        save_corrected_locations(corrected_df, args.output)
        print(f"Wrote {args.output} with {len(corrected_df)} rows.")
        return

    ui = PhoneLocationCorrectionUI(
        raw_df,
        corrected_df,
        args.output,
        history_window=args.history,
        future_window=args.future,
        start_index=args.start_index,
        pad_meters=args.pad_meters,
        max_tiles=args.max_tiles,
        min_zoom=args.min_zoom,
        max_zoom=args.max_zoom,
    )
    ui.run()


if __name__ == "__main__":
    main()
