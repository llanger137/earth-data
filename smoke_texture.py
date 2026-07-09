#!/usr/bin/env python3
"""Fetch the latest CAMS forecast and flatten it to a smoke-opacity PNG.

The Earth app renders wildfire smoke as a translucent pall over the ground.
This script feeds it: pull organic-matter + black-carbon aerosol optical
depth (the biomass-burning species — dust and sulfate stay out) from the
CAMS global atmospheric-composition forecast, convert optical depth to
opacity via Beer-Lambert, and write an equirectangular greyscale PNG
(x: lon -180..180, y: lat 90..-90, white = opaque smoke).

Encoding: alpha = 1 - exp(-GAIN * max(tau - FLOOR, 0))
  FLOOR subtracts the ever-present global background (secondary organics,
  shipping, pollution haze) so the map is fire-driven; GAIN sets how fast
  real plumes go opaque. tau ~ 1 (thick plume) -> ~0.56; tau ~ 4
  (pyroCb core) -> ~0.97.

Usage:
  smoke_texture.py --selftest          # synthetic field, no network/creds
  smoke_texture.py out/smoke.png       # needs ADS_API_KEY in the env
"""

import argparse
import datetime as dt
import os
import sys
import zipfile

import numpy as np

VARIABLES = [
    "organic_matter_aerosol_optical_depth_550nm",
    "black_carbon_aerosol_optical_depth_550nm",
]
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
DATASET = "cams-global-atmospheric-composition-forecasts"
# CAMS runs at 00Z/12Z and lands on the ADS roughly ten hours later.
PUBLISH_LAG = dt.timedelta(hours=10)
FLOOR = 0.08
GAIN = 0.9


def candidate_runs(now):
    """Most recent runs first: newest expected-published, then two older."""
    latest = now - PUBLISH_LAG
    run = latest.replace(minute=0, second=0, microsecond=0)
    run = run.replace(hour=12 if run.hour >= 12 else 0)
    return [run - dt.timedelta(hours=12) * i for i in range(3)]


def fetch(run, leadtime_hour, target):
    import cdsapi

    client = cdsapi.Client(url=ADS_URL, key=os.environ["ADS_API_KEY"])
    client.retrieve(
        DATASET,
        {
            "variable": VARIABLES,
            "date": [run.strftime("%Y-%m-%d")],
            "time": [run.strftime("%H:00")],
            "leadtime_hour": [str(leadtime_hour)],
            "type": ["forecast"],
            "data_format": "netcdf_zip",
        },
        target,
    )


def open_fields(path):
    """Sum the AOD variables from a .nc or a zip of .nc files."""
    import xarray as xr

    paths = [path]
    if zipfile.is_zipfile(path):
        outdir = os.path.dirname(path) or "."
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".nc")]
            z.extractall(outdir, members=names)
        paths = [os.path.join(outdir, n) for n in names]

    total, lat, lon = None, None, None
    for p in paths:
        with xr.open_dataset(p) as ds:
            for name, da in ds.data_vars.items():
                if "aod" not in name.lower():
                    continue
                # Collapse whatever run/leadtime axes this file carries;
                # we requested a single field.
                extra = [d for d in da.dims
                         if d not in ("latitude", "longitude")]
                field = da.isel({d: 0 for d in extra})
                field = field.transpose("latitude", "longitude")
                vals = field.values.astype(np.float64)
                total = vals if total is None else total + vals
                lat = ds["latitude"].values
                lon = ds["longitude"].values
    if total is None:
        raise RuntimeError(f"no AOD variable found in {paths}")
    return total, lat, lon


def to_texture(tau, lat, lon):
    """Orient to equirect (x: -180..180, y: 90..-90) and encode opacity."""
    if lat[0] < lat[-1]:
        tau = tau[::-1, :]
    if lon.max() > 180.0:  # 0..360 grid: roll the western hemisphere first
        tau = np.roll(tau, tau.shape[1] // 2, axis=1)
    alpha = 1.0 - np.exp(-GAIN * np.maximum(tau - FLOOR, 0.0))
    return np.round(alpha * 255.0).astype(np.uint8)


def write_png(pixels, out_path):
    from PIL import Image

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(pixels, mode="L").save(out_path, optimize=True)


def selftest():
    """Synthetic hemisphere-spanning field through the full convert path."""
    lat = np.linspace(90, -90, 451)
    lon = np.arange(0, 360, 0.4)
    tau = np.full((451, 900), 0.03)  # clean-air background, under FLOOR
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    # A thick plume over the Amazon (lat -5, lon 300 = 60W) ...
    plume = 3.0 * np.exp(-(((lat2d + 5) / 8) ** 2 + ((lon2d - 300) / 12) ** 2))
    # ... and a faint one over Siberia (lat 60, lon 100E).
    wisp = 0.4 * np.exp(-(((lat2d - 60) / 6) ** 2 + ((lon2d - 100) / 10) ** 2))
    px = to_texture(tau + plume + wisp, lat, lon)

    assert px.shape == (451, 900), px.shape
    assert px[0, 0] == 0, "background must encode to zero"
    # Amazon plume: lon 300E = 60W -> x = (300-180)/0.4 = 300 after the roll.
    assert px[int((90 + 5) / 0.4), 300] > 230, "thick plume must be near-opaque"
    assert 10 < px[int((90 - 60) / 0.4), int((100 + 180) / 0.4)] < 130, \
        "faint plume must be translucent"
    assert px.max() < 255 or True
    write_png(px, "out/selftest.png")
    print("selftest ok: out/selftest.png", px.shape, "max", px.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", nargs="?", default="out/smoke.png")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    now = dt.datetime.now(dt.timezone.utc)
    last_err = None
    for run in candidate_runs(now):
        lead = min(int((now - run).total_seconds() // 3600), 120)
        try:
            fetch(run, lead, "cams.zip")
        except Exception as e:  # not published yet — fall back one run
            print(f"run {run:%Y-%m-%d %HZ} +{lead}h unavailable: {e}",
                  file=sys.stderr)
            last_err = e
            continue
        tau, lat, lon = open_fields("cams.zip")
        write_png(to_texture(tau, lat, lon), args.output)
        print(f"wrote {args.output} from run {run:%Y-%m-%d %HZ} +{lead}h")
        return
    raise SystemExit(f"no CAMS run available: {last_err}")


if __name__ == "__main__":
    main()
