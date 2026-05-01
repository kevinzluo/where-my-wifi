import json
import math
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

TILE_SIZE = 256
USER_AGENT = "stats305c-eda/1.0"

TL_CORNER = [37.430582, -122.173904]
BR_CORNER = [37.42705, -122.169413]


def first_non_null(series, fallback="unknown"):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else fallback


def lonlat_to_world(lon, lat, zoom):
    scale = TILE_SIZE * (2 ** zoom)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = (lon + 180.0) / 360.0 * scale
    lat = np.clip(lat, -85.05112878, 85.05112878)
    lat_rad = np.radians(lat)
    y = (1.0 - np.arcsinh(np.tan(lat_rad)) / math.pi) / 2.0 * scale
    x, y = np.broadcast_arrays(x, y)
    if x.ndim == 0 and y.ndim == 0:
        return float(x), float(y)
    return x, y


def padded_bounds(lons, lats, pad_fraction=0.18, min_pad=0.0007):
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    lon_pad = max(min_pad, (lon_max - lon_min) * pad_fraction)
    lat_pad = max(min_pad, (lat_max - lat_min) * pad_fraction)
    return lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad


def choose_zoom(lons, lats, max_tiles=20, min_zoom=14, max_zoom=19):
    bounds = padded_bounds(lons, lats)
    return choose_zoom_for_bounds(bounds, max_tiles=max_tiles, min_zoom=min_zoom, max_zoom=max_zoom)


def choose_zoom_for_bounds(bounds, max_tiles=20, min_zoom=14, max_zoom=19):
    for zoom in range(max_zoom, min_zoom - 1, -1):
        left, top = lonlat_to_world(bounds[0], bounds[3], zoom)
        right, bottom = lonlat_to_world(bounds[2], bounds[1], zoom)
        x0 = math.floor(left / TILE_SIZE)
        x1 = math.floor(right / TILE_SIZE)
        y0 = math.floor(top / TILE_SIZE)
        y1 = math.floor(bottom / TILE_SIZE)
        tile_count = (x1 - x0 + 1) * (y1 - y0 + 1)
        if tile_count <= max_tiles:
            return zoom
    return min_zoom


def fetch_tile(x, y, zoom, timeout=5):
    max_index = 2 ** zoom - 1
    if x < 0 or y < 0 or x > max_index or y > max_index:
        raise ValueError("Tile outside valid range.")
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return mpimg.imread(BytesIO(response.read()), format="png")


def fetch_tile_mosaic(bounds, zoom, timeout=5):
    left, top = lonlat_to_world(bounds[0], bounds[3], zoom)
    right, bottom = lonlat_to_world(bounds[2], bounds[1], zoom)
    x0 = math.floor(left / TILE_SIZE)
    x1 = math.floor(right / TILE_SIZE)
    y0 = math.floor(top / TILE_SIZE)
    y1 = math.floor(bottom / TILE_SIZE)

    rows = []
    for y in range(y0, y1 + 1):
        row_tiles = [
            fetch_tile(x, y, zoom, timeout=timeout)
            for x in range(x0, x1 + 1)
        ]
        rows.append(np.concatenate(row_tiles, axis=1))

    image = np.concatenate(rows, axis=0)
    extent = (x0 * TILE_SIZE, (x1 + 1) * TILE_SIZE,
              (y1 + 1) * TILE_SIZE, y0 * TILE_SIZE)
    return image, extent


def geometry_to_world_ring(geometry, zoom):
    if len(geometry) < 3:
        return []
    lons = [node["lon"] for node in geometry]
    lats = [node["lat"] for node in geometry]
    return list(zip(*lonlat_to_world(lons, lats, zoom)))


def ring_area(ring):
    if len(ring) < 3:
        return 0.0
    xs = np.asarray([point[0] for point in ring])
    ys = np.asarray([point[1] for point in ring])
    return 0.5 * np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)


def orient_ring(ring, clockwise=False):
    ring = list(ring)
    area = ring_area(ring)
    if (clockwise and area > 0) or (not clockwise and area < 0):
        ring.reverse()
    return ring


def footprint_to_path(footprint):
    vertices = []
    codes = []
    for ring, clockwise in (
        [(ring, False) for ring in footprint["positive_rings"]]
        + [(ring, True) for ring in footprint["negative_rings"]]
    ):
        ring = orient_ring(ring, clockwise=clockwise)
        if len(ring) < 3:
            continue
        vertices.extend(ring)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1))
        vertices.append(ring[0])
        codes.append(MplPath.CLOSEPOLY)
    if not vertices:
        return None
    return MplPath(vertices, codes)


def fetch_building_footprints(bounds, zoom, timeout=10, pad=None):
    south, west, north, east = bounds[1], bounds[0], bounds[3], bounds[2]
    query = (
        "[out:json][timeout:20];"
        f'nwr["building"]({south},{west},{north},{east});'
        "out body geom;"
    )
    payload = urlencode({"data": query}).encode("utf-8")
    request = Request(
        "https://overpass-api.de/api/interpreter",
        data=payload,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Overpass building query failed with HTTP {exc.code}: {body[:500]}"
        ) from exc

    footprints = []
    for element in data.get("elements", []):
        positive_rings = []
        negative_rings = []
        if element.get("type") == "relation":
            for member in element.get("members", []):
                role = member.get("role") or "outer"
                if role == "part":
                    continue
                ring = geometry_to_world_ring(
                    member.get("geometry") or [], zoom)
                if not ring:
                    continue
                if role == "inner":
                    negative_rings.append(ring)
                else:
                    positive_rings.append(ring)
        else:
            ring = geometry_to_world_ring(element.get("geometry") or [], zoom)
            if ring:
                positive_rings.append(ring)

        if positive_rings:
            tags = element.get("tags") or {}
            footprints.append({
                "id": element.get("id"),
                "type": element.get("type"),
                "name": tags.get("name"),
                "positive_rings": positive_rings,
                "negative_rings": negative_rings,
            })
    return footprints


def fetch_building_polygons(bounds, zoom, timeout=10, pad=None):
    footprints = fetch_building_footprints(
        bounds,
        zoom,
        timeout=timeout,
        pad=pad,
    )
    return [
        ring
        for footprint in footprints
        for ring in footprint["positive_rings"]
    ]


class OSMPlotContext:
    def __init__(
        self,
        bounds,
        zoom=None,
        *,
        tile_timeout=5,
        building_timeout=10,
        max_tiles=20,
        min_zoom=14,
        max_zoom=19,
    ):
        self.bounds = tuple(bounds)
        if zoom is None:
            zoom = choose_zoom_for_bounds(
                self.bounds,
                max_tiles=max_tiles,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
            )
        self.zoom = zoom
        self.tile_timeout = tile_timeout
        self.building_timeout = building_timeout

        self.image = None
        self.extent = None
        self.basemap_loaded = False
        self.basemap_error = None

        self.building_footprints = None
        self.building_polygons = None
        self.buildings_loaded = False
        self.building_error = None

    @classmethod
    def from_bounds(
        cls,
        init_lons,
        init_lats,
        zoom=None,
        *,
        pad_fraction=0.18,
        min_pad=0.0007,
        **kwargs,
    ):

        bounds = padded_bounds(init_lons, init_lats,
                               pad_fraction=pad_fraction, min_pad=min_pad)
        if zoom is None:
            zoom_kwargs = {
                key: kwargs[key]
                for key in ("max_tiles", "min_zoom", "max_zoom")
                if key in kwargs
            }
            zoom = choose_zoom_for_bounds(bounds, **zoom_kwargs)
        return cls(bounds, zoom=zoom, **kwargs)

    @classmethod
    def from_dataframe(
        cls,
        df,
        lon_col="longitude",
        lat_col="latitude",
        zoom=None,
        *,
        pad_fraction=0.18,
        min_pad=0.0007,
        **kwargs,
    ):
        coords = df[[lon_col, lat_col]].dropna()
        if coords.empty:
            raise ValueError(
                "Cannot build OSMPlotContext from an empty coordinate set.")

        lons = coords[lon_col].to_numpy()
        lats = coords[lat_col].to_numpy()
        bounds = padded_bounds(
            lons, lats, pad_fraction=pad_fraction, min_pad=min_pad)
        if zoom is None:
            zoom_kwargs = {
                key: kwargs[key]
                for key in ("max_tiles", "min_zoom", "max_zoom")
                if key in kwargs
            }
            zoom = choose_zoom_for_bounds(bounds, **zoom_kwargs)
        return cls(bounds, zoom=zoom, **kwargs)

    def to_world(self, lon, lat):
        return lonlat_to_world(lon, lat, self.zoom)

    def load_basemap(self):
        if self.image is not None or self.basemap_error is not None:
            return self.basemap_loaded
        try:
            self.image, self.extent = fetch_tile_mosaic(
                self.bounds,
                self.zoom,
                timeout=self.tile_timeout,
            )
            self.basemap_loaded = True
        except Exception as exc:
            self.basemap_error = exc
            self.basemap_loaded = False
        return self.basemap_loaded

    def load_buildings(self):
        if self.building_footprints is not None or self.building_error is not None:
            return self.buildings_loaded
        try:
            self.building_footprints = fetch_building_footprints(
                self.bounds,
                self.zoom,
                timeout=self.building_timeout,
            )
            self.building_polygons = [
                ring
                for footprint in self.building_footprints
                for ring in footprint["positive_rings"]
            ]
            self.buildings_loaded = True
        except Exception as exc:
            self.building_error = exc
            self.building_footprints = None
            self.building_polygons = None
            self.buildings_loaded = False
        return self.buildings_loaded

    def _metadata(self):
        return {
            "bounds": self.bounds,
            "zoom": self.zoom,
            "basemap_loaded": self.basemap_loaded,
            "buildings_loaded": self.buildings_loaded,
            "building_count": len(self.building_footprints or []),
            "basemap_error": self.basemap_error,
            "building_error": self.building_error,
        }

    def generate_base_axis(
        self,
        ax=None,
        *,
        figsize=(12, 9),
        draw_buildings=True,
        building_style=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        self.load_basemap()
        if self.basemap_loaded:
            ax.imshow(self.image, extent=self.extent, origin="upper")

            if draw_buildings:
                self.load_buildings()
                if self.building_footprints:
                    style = {
                        "facecolor": (1, 1, 1, 0.08),
                        "edgecolor": (0, 0, 0, 0.55),
                        "linewidth": 0.8,
                    }
                    if building_style:
                        style.update(building_style)
                    for footprint in self.building_footprints:
                        path = footprint_to_path(footprint)
                        if path is not None:
                            ax.add_patch(PathPatch(path, **style))

            self.apply_world_axis(ax)

        return fig, ax, self._metadata()

    def apply_world_axis(self, ax):
        left, top = self.to_world(self.bounds[0], self.bounds[3])
        right, bottom = self.to_world(self.bounds[2], self.bounds[1])
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def contains_building(self, lon, lat):
        if not self.load_buildings():
            raise RuntimeError(
                "OSM building polygons are unavailable; cannot determine containment."
            ) from self.building_error

        x, y = self.to_world(lon, lat)
        x_arr, y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        is_scalar = x_arr.ndim == 0 and y_arr.ndim == 0
        points = np.column_stack([x_arr.ravel(), y_arr.ravel()])

        contained = np.zeros(points.shape[0], dtype=bool)
        for footprint in self.building_footprints:
            depth = np.zeros(points.shape[0], dtype=int)
            for ring in footprint["positive_rings"]:
                if len(ring) >= 3:
                    depth += MplPath(ring).contains_points(points)
            for ring in footprint["negative_rings"]:
                if len(ring) >= 3:
                    depth -= MplPath(ring).contains_points(points)
            contained |= depth > 0

        if is_scalar:
            return bool(contained[0])
        return contained.reshape(x_arr.shape)


def aggregate_wifi_points(wifi_df, value=None, new_data=None):
    wifi_df = wifi_df.copy()
    if new_data is not None:
        wifi_df['new_data'] = new_data
        value = 'new_data'

    not_nan = wifi_df[wifi_df[value].notna()]
    is_nan = wifi_df[wifi_df[value].isna()]

    grouped = (
        not_nan.groupby(["latitude", "longitude"], as_index=False)
        .agg(
            mean_value=(value, "mean"),
            sample_count=(value, "size"),
            # building=("building", first_non_null),
            # label=("label", first_non_null),
        )
        .sort_values(["mean_value", "sample_count"], ignore_index=True)
    )

    na_grouped = (
        is_nan.groupby(["latitude", "longitude"], as_index=False)
        .agg(
            mean_value=(value, "mean"),
            sample_count=(value, "size"),
        )
        .sort_values(["mean_value", "sample_count"], ignore_index=True)
    )
    return grouped, na_grouped


def plot_agg_wifi_heatmap(
    wifi_df,
    value=None,
    new_data=None,
    zoom=None,
    cmap="RdYlGn",
    invert_cmap=False,
    plotname=None,
    show_na=False,
    context=None,
):
    '''
    Plot WiFi data on heatmap
    wifi_df : used to define map locations
    value : column name to be plotted, default = rssi
    new_data : new data to be plotted, same ordering as wifi_df, overrides `value`
    context : optional OSMPlotContext to reuse OSM basemap/building data

    Coordinate priority: phone GPS, then mac CoreLocation, then legacy CoreLocation. Zero-placeholder wdutil rows use NaN
    '''
    if new_data is not None:
        if plotname is None:
            plotname = "UNKNOWN_VALUE_NAME"
    else:
        if value is None:
            value = "rssi_sample"
            plotname = "RSSI"
        elif plotname is None:
            plotname = value.capitalize()

    points, points_na = aggregate_wifi_points(wifi_df, value, new_data)

    if points_na.shape[0] == 0:
        show_na = False

    lons = points["longitude"].tolist()
    lats = points["latitude"].tolist()
    mean_value = points["mean_value"].to_numpy()
    bounds = padded_bounds(lons, lats)
    if zoom is None:
        zoom = choose_zoom(lons, lats)

    if context is None:
        context = OSMPlotContext(bounds, zoom=zoom)
    fig, ax, metadata = context.generate_base_axis(figsize=(12, 9))
    basemap_loaded = metadata["basemap_loaded"]

    if not basemap_loaded:
        print(
            f"OSM tiles unavailable ({metadata['basemap_error']}). "
            "Falling back to a plain lat/lon heatmap."
        )
    elif metadata["building_error"] is not None:
        print(
            f"OSM building footprints unavailable ({metadata['building_error']}).")

    if basemap_loaded:
        xs, ys = context.to_world(
            points["longitude"].to_numpy(),
            points["latitude"].to_numpy(),
        )
        if show_na:
            xs_na, ys_na = context.to_world(
                points_na["longitude"].to_numpy(),
                points_na["latitude"].to_numpy(),
            )
    else:
        xs = points["longitude"].to_numpy()
        ys = points["latitude"].to_numpy()
        if show_na:
            xs_na = points_na["longitude"].to_numpy()
            ys_na = points_na["latitude"].to_numpy()

    if invert_cmap:
        cmap += '_r'
    sizes = 70 + 14 * points["sample_count"].to_numpy()
    scatter = ax.scatter(
        xs,
        ys,
        c=mean_value,
        s=sizes,
        cmap=cmap,
        alpha=0.92,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.88)
    colorbar.set_label(f"Mean {value}")

    if show_na:
        sizes_na = 70 + 14 * points_na["sample_count"].to_numpy()
        scatter = ax.scatter(
            xs_na,
            ys_na,
            c='black',
            s=sizes_na,
            alpha=show_na,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

    if basemap_loaded:
        context.apply_world_axis(ax)
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(lats))))

    ax.set_title(f"Spatial Heatmap of {plotname}")
    ax.text(
        0.01,
        0.01,
        f"{'Smaller' if invert_cmap else 'Larger'} values are better. Marker size = sample count.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    # fig.tight_layout()
    return fig, ax, points, points_na
