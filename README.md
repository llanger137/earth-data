# earth-data

Live data textures for the [Earth app](https://github.com/llanger137/earth),
published matteason-style: a scheduled Action fetches the science data,
flattens it to a small texture, and force-pushes it to the single-commit
`data` branch, where the app fetches it as a static file.

## Clouds

`clouds/` — global cloud cover as a matteason-style greyscale texture
(equirectangular, lon −180…180 × lat 90…−90, white = cloud), rebuilt
hourly.

Source: [NOAA GMGSI](https://www.ospo.noaa.gov/products/imagery/gmgsi/)
longwave-IR — the global geostationary mosaic on anonymous public S3
(hourly, ~40 min latency, no credentials). Brightness temperature
becomes a visible-style cloud mask in [ir_visible.py](ir_visible.py);
GMGSI stops at ~72.7°, so the polar caps come from the
[matteason](https://clouds.matteason.co.uk) visible cloud map (CC0),
alpha-ramped in between 60° and 65° — tuned down from the nominal 64–71°
band to keep the degraded Meteosat-IODC limb (a missing-data bowl
bottoming ~66°N) below the blend. Pipeline and tuning constants
(`BLEND_FULL`/`BLEND_NONE`) in [cloud_texture.py](cloud_texture.py).

Published via GitHub Pages from the `data` branch:

```
https://llanger137.github.io/earth-data/clouds/current_8192.jpg   8192×4096
https://llanger137.github.io/earth-data/clouds/current_4096.jpg   4096×2048
https://llanger137.github.io/earth-data/clouds/manifest.json
https://llanger137.github.io/earth-data/clouds/archive/<YYYYMMDDHH>.jpg
```

`manifest.json` is `{"latest", "frames", "flows", "window_hours"}`:
`frames` lists the archived UTC hour stamps (ascending) over a rolling
7-day window; `archive/` holds exactly those frames at 4096×2048.

`archive/flow/<YYYYMMDDHH>.png` — optical flow between consecutive
archive frames, for shader-side cloud advection during replay. `<stamp>`
is the interval start: the map holds the motion from that frame to the
next hour's. 1024×512 RGB PNG (no alpha): R = Δu, G = Δv in
equirectangular UV per hour, encoded as
`byte = round(clip((v + 0.014) / 0.028, 0, 1) · 255)` — 128 is no motion,
full scale ±0.014 UV — with B constant 128. Stamps with a map are listed
in `manifest.json` as `flows`; a gap in the archive gets no map.

One-time setup: push the repo, run the `clouds` workflow once by hand to
seed the `data` branch, then enable GitHub Pages serving from `data` /
root (Settings → Pages → Deploy from a branch). No secrets — GMGSI is
anonymous.

## Smoke

`smoke.png` — global wildfire smoke as a greyscale opacity map
(equirectangular, lon −180…180 × lat 90…−90, white = opaque smoke).

Source: [CAMS global atmospheric-composition forecast](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts)
— organic-matter + black-carbon aerosol optical depth at 550 nm (the
biomass-burning species; dust and sulfate excluded). Optical depth becomes
opacity via Beer–Lambert (`1 − exp(−GAIN·max(τ − FLOOR, 0))`); the floor
removes the global pollution background so the map is fire-driven. Details
and tuning constants in [smoke_texture.py](smoke_texture.py).

Refreshed twice daily (10:40 / 22:40 UTC), ~40 min after each CAMS run
lands on the ADS. Fetch URL:

```
https://raw.githubusercontent.com/llanger137/earth-data/data/smoke.png
```

## One-time setup

1. Register at [ads.atmosphere.copernicus.eu](https://ads.atmosphere.copernicus.eu),
   and accept the CAMS data licence (open any CAMS dataset page → Download →
   accept the terms at the bottom).
2. Copy the API key from your ADS profile page.
3. Add it to this repo as an Actions secret named `ADS_API_KEY`
   (Settings → Secrets and variables → Actions).
4. Run the `smoke` workflow once by hand (Actions → smoke → Run workflow)
   to seed the `data` branch; after that the cron keeps it fresh.

Until the secret exists, the scheduled job skips harmlessly.

Contains modified Copernicus Atmosphere Monitoring Service information.
