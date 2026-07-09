# earth-data

Live data textures for the [Earth app](https://github.com/llanger137/earth),
published matteason-style: a scheduled Action fetches the science data,
flattens it to a small texture, and force-pushes it to the single-commit
`data` branch, where the app fetches it as a static file.

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
