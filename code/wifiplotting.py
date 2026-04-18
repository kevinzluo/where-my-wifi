import json
import math
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.collections import PolyCollection

TILE_SIZE = 256
USER_AGENT = "stats305c-eda/1.0"

def first_non_null(series, fallback="unknown"):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else fallback


def lonlat_to_world(lon, lat, zoom):
    scale = TILE_SIZE * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    return x, y


def padded_bounds(lons, lats, pad_fraction=0.18, min_pad=0.0007):
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    lon_pad = max(min_pad, (lon_max - lon_min) * pad_fraction)
    lat_pad = max(min_pad, (lat_max - lat_min) * pad_fraction)
    return lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad


def choose_zoom(lons, lats, max_tiles=20, min_zoom=14, max_zoom=19):
    bounds = padded_bounds(lons, lats)
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


def fetch_tile_mosaic(bounds, zoom):
    left, top = lonlat_to_world(bounds[0], bounds[3], zoom)
    right, bottom = lonlat_to_world(bounds[2], bounds[1], zoom)
    x0 = math.floor(left / TILE_SIZE)
    x1 = math.floor(right / TILE_SIZE)
    y0 = math.floor(top / TILE_SIZE)
    y1 = math.floor(bottom / TILE_SIZE)

    rows = []
    for y in range(y0, y1 + 1):
        row_tiles = [fetch_tile(x, y, zoom) for x in range(x0, x1 + 1)]
        rows.append(np.concatenate(row_tiles, axis=1))

    image = np.concatenate(rows, axis=0)
    extent = (x0 * TILE_SIZE, (x1 + 1) * TILE_SIZE, (y1 + 1) * TILE_SIZE, y0 * TILE_SIZE)
    return image, extent


def fetch_building_polygons(bounds, zoom, timeout=10):
    south, west, north, east = bounds[1], bounds[0], bounds[3], bounds[2]
    query = (
        "[out:json][timeout:20];"
        f"way[\\\"building\\\"]({south},{west},{north},{east});"
        "out geom;"
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
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    polygons = []
    for element in data.get("elements", []):
        geometry = element.get("geometry") or []
        if len(geometry) < 3:
            continue
        polygons.append(
            [lonlat_to_world(node["lon"], node["lat"], zoom) for node in geometry]
        )
    return polygons


def aggregate_rssi_points(wifi_df, value):
    subdf = wifi_df.dropna(
        subset=["selected_latitude", "selected_longitude", value]
    ).copy()
    if subdf.empty:
        raise ValueError(f"No rows with selected coordinates and usable {value} values were found.")

    subdf["label"] = subdf["waypoint_id"].fillna(subdf["building"]).fillna("unknown")
    grouped = (
        subdf.groupby(["selected_latitude", "selected_longitude"], as_index=False)
        .agg(
            mean_value=(value, "mean"),
            sample_count=(value, "size"),
            building=("building", first_non_null),
            label=("label", first_non_null),
        )
        .sort_values(["mean_value", "sample_count"], ignore_index=True)
    )
    return grouped


def plot_rssi_heatmap(wifi_df, value="wdutil_rssi_effective_dbm", zoom=None):
    points = aggregate_rssi_points(wifi_df, value)

    lons = points["selected_longitude"].tolist()
    lats = points["selected_latitude"].tolist()
    mean_value = points["mean_value"].to_numpy()
    bounds = padded_bounds(lons, lats)
    zoom = zoom or choose_zoom(lons, lats)

    fig, ax = plt.subplots(figsize=(12, 9))
    basemap_loaded = False

    try:
        image, extent = fetch_tile_mosaic(bounds, zoom)
        ax.imshow(image, extent=extent, origin="upper")
        basemap_loaded = True
    except Exception as exc:
        print(f"OSM tiles unavailable ({exc}). Falling back to a plain lat/lon heatmap.")

    if basemap_loaded:
        try:
            polygons = fetch_building_polygons(bounds, zoom)
        except Exception as exc:
            print(f"OSM building footprints unavailable ({exc}).")
            polygons = []
        if polygons:
            ax.add_collection(
                PolyCollection(
                    polygons,
                    facecolor=(1, 1, 1, 0.08),
                    edgecolor=(0, 0, 0, 0.55),
                    linewidth=0.8,
                )
            )

    if basemap_loaded:
        xs, ys = zip(
            *(lonlat_to_world(lon, lat, zoom) for lon, lat in zip(points["selected_longitude"], points["selected_latitude"]))
        )
    else:
        xs = points["selected_longitude"].to_numpy()
        ys = points["selected_latitude"].to_numpy()

    sizes = 70 + 14 * points["sample_count"].to_numpy()
    scatter = ax.scatter(
        xs,
        ys,
        c=mean_value,
        s=sizes,
        cmap="RdYlGn",
        alpha=0.92,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.88)
    colorbar.set_label(f"Mean {value}")

    if basemap_loaded:
        left, top = lonlat_to_world(bounds[0], bounds[3], zoom)
        right, bottom = lonlat_to_world(bounds[2], bounds[1], zoom)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(lats))))

    ax.set_title(f"Spatial heatmap of {value} using phone-first coordinates")
    ax.text(
        0.01,
        0.01,
        f"Coordinate priority: phone GPS, then mac CoreLocation, then legacy CoreLocation. Zero-placeholder wdutil rows use NaN. Marker size = sample count.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    return fig, ax, points