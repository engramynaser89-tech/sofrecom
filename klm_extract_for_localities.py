import os
import re
import math
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D

ROOT_DIR = Path("/Users/ibenferhqt/Downloads/localities")
MAPS_DIR = ROOT_DIR / "maps"
MAPS_DIR.mkdir(exist_ok=True)

# --- Helpers -------------------------------------------------------------

LOCALITY_ID_RE = re.compile(r"^(\d{5})")

# Find last floating-point number in a string (e.g. "15325-LTE_UE_RSRP\n-87.25")
FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")

def extract_locality_id(filename: str):
    """Return first 5 digits from the start of the filename (before extension)."""
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    m = LOCALITY_ID_RE.match(name)
    return m.group(1) if m else None

def strip_namespaces(root):
    """Remove namespaces from tags so we can use simple names like 'Placemark'."""
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return root

def extract_rsrp_from_description(text: str):
    """
    Parse RSRP value from a <description> like:
      '15325-LTE_UE_RSRP\\n-87.25'
    We'll take the last float found in the text.
    """
    if not text:
        return None
    matches = FLOAT_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None

def parse_coordinates(coord_text: str):
    """
    Parse 'lon,lat[,alt]' from <coordinates> tag.
    Example: '42.795433,18.394373' -> (42.795433, 18.394373)
    """
    if not coord_text:
        return None, None
    coord_text = coord_text.strip()
    # Sometimes KML might put extra spaces; take first token
    first_token = coord_text.split()[0]
    parts = first_token.split(",")
    if len(parts) < 2:
        return None, None
    try:
        lon = float(parts[0])
        lat = float(parts[1])
        return lon, lat
    except ValueError:
        return None, None

def parse_kml_from_bytes(kml_bytes: bytes, file_path: Path):
    """
    Parse KML from raw bytes and return a list of point dicts:
      {"lon": float, "lat": float, "rsrp": float, "file_path": str}
    """
    root = ET.fromstring(kml_bytes)
    root = strip_namespaces(root)

    points = []
    for placemark in root.findall(".//Placemark"):
        desc_el = placemark.find("description")
        rsrp = extract_rsrp_from_description(desc_el.text if desc_el is not None else None)

        point_el = placemark.find(".//Point")
        coords_el = point_el.find("coordinates") if point_el is not None else None
        lon, lat = parse_coordinates(coords_el.text if coords_el is not None else None)

        if (
            rsrp is not None
            and not math.isnan(rsrp)
            and lon is not None
            and lat is not None
        ):
            points.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "rsrp": rsrp,
                    "file_path": str(file_path),
                }
            )
    return points

def parse_kml_file(path: Path):
    """Parse a .kml file from disk and return list of point dicts."""
    with path.open("rb") as f:
        kml_bytes = f.read()
    return parse_kml_from_bytes(kml_bytes, path)

def parse_kmz_file(path: Path):
    """
    Parse a .kmz (zip) file. We'll read the first .kml inside the archive.
    Return list of point dicts.
    """
    points = []
    with zipfile.ZipFile(path, "r") as zf:
        # Find first .kml inside (common: 'doc.kml')
        kml_names = [name for name in zf.namelist() if name.lower().endswith(".kml")]
        if not kml_names:
            return points
        kml_bytes = zf.read(kml_names[0])
        points = parse_kml_from_bytes(kml_bytes, path)
    return points

# --- Main processing: per-file stats + per-locality aggregation ----------

results = []  # per-file summary
locality_points = {}  # locality_id -> list of {lon, lat, rsrp, file_path}

for root, dirs, files in os.walk(ROOT_DIR):
    for fname in files:
        lower = fname.lower()
        if not (lower.endswith(".kml") or lower.endswith(".kmz")):
            continue

        fpath = Path(root) / fname
        locality_id = extract_locality_id(fname)

        if locality_id is None:
            print(f"Skipping {fpath} (no 5-digit locality id at start of filename)")
            continue

        # Extract points depending on extension
        if lower.endswith(".kml"):
            points = parse_kml_file(fpath)
        else:
            points = parse_kmz_file(fpath)

        total_points = len(points)
        rsrps = [p["rsrp"] for p in points]

        if total_points == 0:
            pct_below_100 = float("nan")
            num_below_100 = 0
        else:
            num_below_100 = sum(1 for v in rsrps if v < -100.0)
            pct_below_100 = 100.0 * num_below_100 / total_points

        results.append(
            {
                "locality_id": locality_id,
                "file_path": str(fpath),
                "num_points": total_points,
                "num_rsrp_lt_-100": num_below_100,
                "pct_rsrp_lt_-100": pct_below_100,
            }
        )

        # Aggregate per locality for mapping later
        locality_points.setdefault(locality_id, []).extend(points)

        print(
            f"{locality_id} | {fpath.name} | "
            f"points={total_points}, RSRP<-100: {num_below_100} "
            f"({pct_below_100:.2f}%)"
        )

# --- Save summary to CSV -------------------------------------------------

df = pd.DataFrame(results)
out_csv = ROOT_DIR / "localities_rsrp_summary.csv"
df.to_csv(out_csv, index=False)
print(f"\nSummary saved to: {out_csv}")

# --- Per-locality maps with OSM background ------------------------------

def rsrp_color(rsrp):
    """
    Drive-test style colors:
      red   : rsrp < -110
      orange: -110 <= rsrp < -100
      yellow: -100 <= rsrp < -90
      green : rsrp >= -90
    """
    if rsrp < -110:
        return "red"
    elif rsrp < -100:
        return "orange"
    elif rsrp < -90:
        return "yellow"
    else:
        return "green"

for locality_id, pts in locality_points.items():
    if not pts:
        continue

    print(f"Creating map for locality {locality_id} with {len(pts)} points...")

    pts_df = pd.DataFrame(pts)
    pts_df["color"] = pts_df["rsrp"].apply(rsrp_color)

    # Compute percentages per RSRP bin (based on colors)
    total_pts = len(pts_df)
    counts = {
        "red":    (pts_df["color"] == "red").sum(),
        "orange": (pts_df["color"] == "orange").sum(),
        "yellow": (pts_df["color"] == "yellow").sum(),
        "green":  (pts_df["color"] == "green").sum(),
    }
    percentages = {
        c: (100.0 * counts[c] / total_pts) if total_pts > 0 else 0.0
        for c in counts
    }

    gdf = gpd.GeoDataFrame(
        pts_df,
        geometry=gpd.points_from_xy(pts_df["lon"], pts_df["lat"]),
        crs="EPSG:4326",
    )
    gdf_3857 = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot actual points (no labels here; legend is manual to always show all bins)
    for color in ["red", "orange", "yellow", "green"]:
        subset = gdf_3857[gdf_3857["color"] == color]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            markersize=8,
            color=color,
            alpha=0.8,
        )

    # Add OSM basemap (EPSG:3857 / Web Mercator)
    ctx.add_basemap(ax, crs=gdf_3857.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_axis_off()
    ax.set_title(f"Locality {locality_id} — LTE UE RSRP drive test", fontsize=12)

    # Build a full legend with all 4 bins + percentages, even if some bins are empty
    legend_defs = [
        ("red",    "RSRP < -110 dBm"),
        ("orange", "-110 ≤ RSRP < -100 dBm"),
        ("yellow", "-100 ≤ RSRP < -90 dBm"),
        ("green",  "RSRP ≥ -90 dBm"),
    ]

    handles = []
    for color, base_label in legend_defs:
        pct = percentages[color]
        label = f"{base_label} ({pct:.1f}%)"
        handle = Line2D(
            [], [], marker="o", linestyle="None",
            markersize=6, color=color, label=label
        )
        handles.append(handle)

    ax.legend(handles=handles, loc="lower left", title="RSRP bins")

    plt.tight_layout()
    out_jpg = MAPS_DIR / f"{locality_id}_rsrp_map.jpg"
    plt.savefig(out_jpg, dpi=200)
    plt.close(fig)

    print(f"Saved map: {out_jpg}")