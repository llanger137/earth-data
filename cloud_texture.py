#!/usr/bin/env python3
"""Fetch the latest GMGSI IR mosaics and build the cloud textures.

The Earth app drapes live clouds over the globe. This script feeds it
hourly: pull NOAA's GMGSI global geostationary IR composite (anonymous
public S3, ~40 min latency), map brightness temperature to a visible-style
cloud mask (ir_visible.py: white = cloud), upscale it into an 8192x4096
equirectangular canvas (x: lon -180..180, y: lat 90..-90), and fill the
polar caps — GMGSI stops at ~72.7 deg — from the matteason visible cloud
map (CC0, 3-hourly).  GMGSI missing-data fill (dqf != 0: the bowl above
the Meteosat-IODC disk edge, thin sector-seam strips) also falls through
to matteason instead of printing as saturated white.

Two GMGSI channels are read, not one.  Longwave (11 um) carries the
picture; shortwave (3.9 um) is read only to recover warm low cloud, whose
top is so close to the ground's own temperature that longwave cannot
separate the two — see sw_fusion.py for the physics, the validation, and
the places the rescue deliberately declines to fire.  Shortwave enters as
a shift in longwave counts applied before the tone curve, so the curve's
tuning survives untouched and the app needs no change at all.  If the
shortwave frame is missing or unreadable the run publishes longwave-only
output, exactly as it did before the channel was added.

Blend: pure GMGSI for |lat| <= BLEND_FULL, linear ramp to BLEND_NONE,
pure matteason above.  Above BLEND_NONE nothing observed anything, so the
caps are smoothed to a single brightness carried up from the last belt
that did (depole) rather than left to print matteason's fill as a
starburst.  Outputs land in <outdir>: current_8192.jpg,
current_4096.jpg, archive/<YYYYMMDDHH>.jpg (frame's own UTC hour, rolling
WINDOW_HOURS deep), archive/flow/<YYYYMMDDHH>.png (optical flow over the
hour that stamp starts, for shader-side advection), archive/aurora/
<YYYYMMDDHH>.png (the OVATION aurora nowcast baked to the app's texture,
same rolling window), and manifest.json
{"latest": stamp, "frames": [stamps ascending], "flows": [flow stamps],
"aurora": [aurora stamps], "window_hours": 168}.
Every frame in the lookback newer than the published latest is archived,
not just the newest, so a failed or cancelled run's hour backfills on
the next one instead of holing the window.

Usage:
  cloud_texture.py --selftest         # synthetic frames, no network
  cloud_texture.py [out/clouds]       # build from live GMGSI + matteason
"""

import argparse
import datetime as dt
import io
import json
import os
import re
import sys

import numpy as np

S3 = "https://noaa-gmgsi-pds.s3.amazonaws.com"
PRODUCT = "GMGSI_LW"
# The 3.9um companion, used only to rescue warm low cloud that LW alone
# renders as clear sky (sw_fusion.py). Same blend, same grid, same
# 10-minute window; published a few minutes after LW (15:36 vs 15:39 on
# 2026-07-25), comfortably before this workflow's :45 run. When it is
# missing or unreadable the run continues on LW alone, which is exactly
# the output this pipeline produced before the channel was added.
PRODUCT_SW = "GMGSI_SW"
MATTEASON_URL = "https://clouds.matteason.co.uk/images/8192x4096/clouds.jpg"
WIDTH, HEIGHT = 8192, 4096
BLEND_FULL = 60.0  # pure GMGSI up to this |lat| ...
BLEND_NONE = 65.0  # ... pure matteason above this; linear ramp between
# (60..65 rather than the GMGSI edge ~72.7: the Meteosat-IODC sector is
# saturated missing-data fill above its disk edge, a bowl bottoming ~66N
# with hard boundaries that show through any higher band.)
# Pole treatment (depole below). No geostationary satellite sees a pole,
# so matteason's cap is invented: a flat 232 above ~87 deg, and below that
# a belt whose longitudinal contrast the converging equirect grid funnels
# into a radial starburst. POLE_NONE stays clear of BLEND_NONE so the
# reference belt reads matteason only, never the GMGSI blend below it.
POLE_NONE = 72.0  # untouched up to this |lat| ...
POLE_FULL = 84.0  # ... fully smoothed above; smoothstep between
POLE_ARC_KM = 440.0  # longitudinal kernel measured on the ground, not in
# pixels, so it means the same thing on the 8192 canvas and the 4096 output
POLE_BELT = 6.0  # depth of the reference belt just below POLE_NONE
EARTH_CIRC_KM = 40030.0
WINDOW_HOURS = 168
JPEG_QUALITY = 85
MAX_LOOKBACK_HOURS = 6
# Optical flow between consecutive archive frames, for shader-side cloud
# advection during replay. FLOW_MAX_UV is the full-scale displacement in
# equirect UV units — must match FLOW_MAX_UV in the app's globe.frag,
# which decodes uv_disp = (texel * 2 - 1) * FLOW_MAX_UV.
FLOW_MAX_UV = 0.014  # sized above the ~0.011 UV in-band (|lat|<55) peak on
# real GMGSI pairs so fast fronts/cirrus don't clip; the faster polar tail
# does clip, but the shader dissolves flow to zero there anyway.
FLOW_W, FLOW_H = 1024, 512
FLOW_PAD = 64  # horizontal wrap pad (px) so dateline motion reads short
# Content sanity for the matteason frame: a decodable-but-garbage image
# (all black / all white / flat grey, or the wrong size) must fail the
# run and keep the last good publish — the poisoned hour's archive frame
# would otherwise serve corrupt for the full window, since archive/ is
# never rewritten. Bounds are deliberately loose; the live frame
# measured mean 169 / std 78 (2026-07-18).
MATTEASON_MEAN_MIN, MATTEASON_MEAN_MAX = 5.0, 250.0
MATTEASON_STD_MIN = 20.0
# Aurora nowcast: NOAA SWPC OVATION, a 1x1deg global grid of visible-aurora
# probability, archived beside the clouds so History can replay it. bake_aurora
# below MUST stay in lockstep with the app's AuroraService (aurora_service.dart)
# — same equirect convention, same Gaussian, same 16-bit hi/lo split — or a
# historical frame would decode unlike a live one.
AURORA_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
AURORA_W, AURORA_H = 360, 181
AURORA_BLUR_RADIUS = 4
AURORA_BLUR_SIGMA = 1.5


def find_frames(now):
    """Newest S3 key per hour over the lookback window, oldest first.

    Every hour is listed, not just the newest with data: a failed or
    cancelled run leaves its hour recoverable, and the next run archives
    all frames newer than the published latest instead of only the top —
    otherwise a single missed run is a permanent hole in frames[]."""
    import requests

    hour = now.replace(minute=0, second=0, microsecond=0)
    keys = []
    for back in range(MAX_LOOKBACK_HOURS, -1, -1):
        t = hour - dt.timedelta(hours=back)
        prefix = f"{PRODUCT}/{t:%Y/%m/%d/%H}/"
        r = requests.get(S3, params={"list-type": "2", "prefix": prefix},
                         timeout=60)
        r.raise_for_status()
        found = re.findall(r"<Key>([^<]+)</Key>", r.text)
        if found:
            keys.append(sorted(found)[-1])
    if not keys:
        raise SystemExit(f"no {PRODUCT} frame in the last "
                         f"{MAX_LOOKBACK_HOURS}h")
    return keys


def sw_lookup(lw_key):
    """(S3 prefix listing the SW hour, the LW frame's own `_s` stamp).

    Split out from the fetch so the string handling is testable without
    network: getting it wrong pairs two frames that only look related.
    """
    m = re.search(r"_s(\d+)", lw_key)
    if not m:
        return None, None
    head = lw_key.rsplit("/", 1)[0]
    if not head.startswith(f"{PRODUCT}/"):
        return None, None
    return f"{PRODUCT_SW}/{head[len(PRODUCT) + 1:]}/", m.group(1)


def find_sibling_sw(lw_key):
    """The GMGSI_SW key covering the same start time as `lw_key`, or None.

    Matched on the filename's own `_s<stamp>` token rather than by taking
    the newest key in the hour: the two products are produced by separate
    jobs, and pairing a 15:00 LW frame with a 15:10 SW frame would put a
    10-minute parallax into a per-pixel channel difference.
    """
    import requests

    prefix, stamp = sw_lookup(lw_key)
    if not prefix:
        return None
    r = requests.get(S3, params={"list-type": "2", "prefix": prefix},
                     timeout=60)
    r.raise_for_status()
    for k in sorted(re.findall(r"<Key>([^<]+)</Key>", r.text)):
        if f"_s{stamp}" in k:
            return k
    return None


def low_cloud_boost(lw_key, ir, lat, lon, stamp):
    """LW counts to add so warm low cloud stops reading as clear sky.

    Returns None -- meaning "publish LW-only, exactly as before this
    channel existed" -- for every failure: no sibling frame yet, a fetch
    error, a grid that does not match. A missing SW hour must cost that
    hour its low cloud, never the whole publish; archive/ is never
    rewritten, so a hard failure here would hole the replay window
    permanently.

    `ir`, `lat`, `lon` are already oriented; `stamp` is the frame's own
    UTC time, so a backfilled hour gets that hour's terminator and not
    the terminator of whenever the run happens to be catching up.
    """
    from sw_fusion import low_cloud_shift, solar_zenith_cos

    try:
        key = find_sibling_sw(lw_key)
        if not key:
            print(f"no {PRODUCT_SW} sibling for {lw_key}; LW-only this hour",
                  file=sys.stderr)
            return None
        sw, swbad, slat, slon, _ = read_gmgsi(fetch(f"{S3}/{key}"))
        if sw.shape != ir.shape:
            print(f"{PRODUCT_SW} grid {sw.shape} != {PRODUCT} {ir.shape}; "
                  f"LW-only this hour", file=sys.stderr)
            return None
        sw, _, _ = orient(sw, slat, slon)
        swbad, _, _ = orient(swbad, slat, slon)
        return low_cloud_shift(ir, sw, solar_zenith_cos(lat, lon, stamp),
                               lat, bad=swbad > 0.5)
    except Exception as e:
        print(f"{PRODUCT_SW} fusion skipped ({e}); LW-only this hour",
              file=sys.stderr)
        return None


def fetch(url):
    import requests

    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


def read_gmgsi(buf):
    """Stored IR counts, bad-pixel mask, 1-D axes, and the frame's own
    timestamp.  dqf != 0 marks missing-data fill (20: the bowl north of
    the Meteosat-IODC usable disk and thin sector-seam strips; -128: the
    top edge row) — `data` holds saturated fill there, not cloud."""
    import h5py

    with h5py.File(io.BytesIO(buf), "r") as f:
        ir = f["data"][0].astype(np.float32)  # 0-255 brightness temperature
        bad = (f["dqf"][0][:] != 0).astype(np.float32)
        lat = f["lat"][:, 0].astype(np.float64)  # constant along columns
        lon = f["lon"][0, :].astype(np.float64)  # constant along rows
        stamp = dt.datetime.fromtimestamp(int(f["time"][0]), dt.timezone.utc)
    return ir, bad, lat, lon, stamp


def orient(ir, lat, lon):
    """North-up rows, columns ascending from the -180 meridian."""
    if lat[0] < lat[-1]:
        ir, lat = ir[::-1], lat[::-1]
    lon = (lon + 180.0) % 360.0 - 180.0
    k = int(np.argmin(lon))  # westernmost column
    if k:
        ir, lon = np.roll(ir, -k, axis=1), np.roll(lon, -k)
    return ir, lat, lon


def composite(vis, bad, lat, lon, canvas):
    """Bilinear-upscale the GMGSI band into the canvas, ramping at the edge.

    Row/column mapping comes from the file's actual lat/lon axes; x wraps
    at the dateline.  Wherever `bad` (missing-data fill, dqf != 0) lands,
    the matteason canvas shows through instead — the sector-seam fill
    strips would otherwise print as bright lines.
    """
    h, w = canvas.shape
    lat_out = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    inside = (lat_out <= lat[0]) & (lat_out >= lat[-1])
    i0, i1 = np.nonzero(inside)[0][[0, -1]]

    n = lon.size
    lon_out = -180.0 + (np.arange(w) + 0.5) * (360.0 / w)
    edges = np.append(lon, lon[0] + 360.0)  # wrap segment across the seam
    c = np.interp(np.where(lon_out < lon[0], lon_out + 360.0, lon_out),
                  edges, np.arange(n + 1.0)) % n
    c0 = np.floor(c).astype(int)
    cf = (c - c0).astype(np.float32)
    c1 = (c0 + 1) % n

    r = np.interp(lat_out[i0:i1 + 1], lat[::-1],
                  np.arange(lat.size - 1.0, -1.0, -1.0))
    r0 = np.floor(r).astype(int)
    rf = (r - r0).astype(np.float32)[:, None]
    r1 = np.minimum(r0 + 1, lat.size - 1)

    def resample(src):
        cols = src.astype(np.float32)
        cols = cols[:, c0] * (1.0 - cf) + cols[:, c1] * cf
        return cols[r0] * (1.0 - rf) + cols[r1] * rf

    # Weight the resample by good coverage and renormalize: a masked
    # pixel holds saturated fill, and plain bilinear would bleed it into
    # the neighbouring output pixels as a dim rim along every masked
    # strip (alpha only partly suppresses them). Where coverage is zero
    # the numerator is zero too, and alpha hides the band entirely.
    good = 1.0 - np.clip(bad.astype(np.float32), 0.0, 1.0)
    w = resample(good)
    band = resample(vis * good) / np.maximum(w, 1e-6)
    alpha = np.clip((BLEND_NONE - np.abs(lat_out[i0:i1 + 1]))
                    / (BLEND_NONE - BLEND_FULL), 0.0, 1.0)
    alpha = alpha.astype(np.float32)[:, None] * np.clip(w, 0.0, 1.0)
    out = canvas.astype(np.float32)
    out[i0:i1 + 1] = out[i0:i1 + 1] * (1.0 - alpha) + band * alpha
    return np.round(out).clip(0, 255).astype(np.uint8)


def _smoothstep(x, edge0, edge1):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ring_mean(row, n, passes=2):
    """Circular moving average of `n` columns, wrapping at the dateline.
    Two box passes rather than one: a single box leaves visible corners
    where the kernel steps on and off a bright feature."""
    w = row.size
    for _ in range(passes):
        pad = n // 2
        ext = np.concatenate([row[-pad:], row, row[:n - pad]])
        cs = np.concatenate([[0.0], np.cumsum(ext, dtype=np.float64)])
        row = ((cs[n:n + w] - cs[:w]) / n).astype(np.float32)
    return row


def depole(px):
    """Stop the invented polar caps from printing as radial starbursts.

    Nothing in geostationary orbit can see a pole, so everything above
    ~72 deg here is matteason's fill rather than an observation, and it
    arrives with two separate tells.  Its longitudinal contrast is still
    near full strength at 84 deg (row std ~8 where the grid has already
    squeezed 360 deg of longitude into an arc a few hundred km wide), so
    the projection funnels it into a star with its point on the pole.
    And above ~87 deg it is a flat 232 — brighter than the 206 belt it
    sits inside, which renders the star on a glowing disc.

    So: average each row over a fixed arc on the ground (POLE_ARC_KM),
    which widens in columns as the grid converges, and hold the whole cap
    at the zonal mean of the last belt below it.  Both ramp in over
    POLE_NONE..POLE_FULL, so rows equatorward of POLE_NONE come back
    untouched and the cap has no rim.  The result is the honest rendering
    of a place nothing observed: one brightness, no invented structure,
    matching the latitude that did get looked at.
    """
    out = px.astype(np.float32).copy()
    h, w = out.shape
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    cosl = np.maximum(np.cos(np.radians(lat)), 1e-9)
    frac = _smoothstep(np.abs(lat), POLE_NONE, POLE_FULL)

    for i in np.nonzero(frac > 0.0)[0]:
        n = int(min(w, max(2, round(POLE_ARC_KM * w
                                    / (EARTH_CIRC_KM * cosl[i])))))
        out[i] += frac[i] * (_ring_mean(out[i], n) - out[i])

    # Measure the belt after the blur, not before: the reference has to be
    # the mean the smoothing actually converges to.
    zonal = out.mean(axis=1)
    for sign in (1.0, -1.0):
        sl = sign * lat
        ref = zonal[(sl > POLE_NONE - POLE_BELT) & (sl <= POLE_NONE)].mean()
        rows = np.nonzero(sl > POLE_NONE)[0]
        out[rows] += (frac[rows] * (ref - zonal[rows]))[:, None]
    return np.round(out).clip(0, 255).astype(np.uint8)


def write_archive_frame(px, stamp, outdir):
    """A backfilled hour: just its archive frame. current_* and the
    manifest belong to the newest frame's write_outputs, which runs last
    and sweeps every archived stamp into frames[]."""
    from PIL import Image

    os.makedirs(os.path.join(outdir, "archive"), exist_ok=True)
    half = Image.fromarray(px, mode="L") \
        .resize((WIDTH // 2, HEIGHT // 2), Image.LANCZOS)
    half.save(os.path.join(outdir, "archive",
                           f"{stamp.strftime('%Y%m%d%H')}.jpg"),
              quality=JPEG_QUALITY)


def flow_uv(prev_px, next_px):
    """Farneback optical flow prev -> next, in equirect UV units.

    Inputs are FLOW_W x FLOW_H greyscale arrays. Both get wrap-padded
    horizontally first so motion across the dateline reads as a short
    hop, not a full-width jump; the pad is cropped off afterwards.
    Pixel displacements become UV as (dx/FLOW_W, dy/FLOW_H) — row-down
    positive dy already is increasing uv.y, so no sign flip — and clip
    to +/-FLOW_MAX_UV, the shader's full-scale range."""
    import cv2

    def pad(a):
        return np.hstack([a[:, -FLOW_PAD:], a, a[:, :FLOW_PAD]])

    flow = cv2.calcOpticalFlowFarneback(
        pad(prev_px), pad(next_px), None, pyr_scale=0.5, levels=5,
        winsize=25, iterations=3, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    uv = flow[:, FLOW_PAD:FLOW_PAD + FLOW_W] \
        / np.array([FLOW_W, FLOW_H], np.float32)
    # A NaN would survive clip and cast to byte 0 = full-scale westward:
    # a garbage streak in the replay. Farneback shouldn't produce one from
    # finite uint8 input, but "shouldn't" is not a publish guarantee.
    return np.clip(np.nan_to_num(uv), -FLOW_MAX_UV, FLOW_MAX_UV)


def encode_flow(uv):
    """UV flow to bytes: R = dx, G = dy (128 = still), B = 128 constant."""
    px = np.full(uv.shape[:2] + (3,), 128, np.uint8)
    px[..., :2] = np.round(np.clip(
        (uv + FLOW_MAX_UV) / (2 * FLOW_MAX_UV), 0.0, 1.0) * 255.0)
    return px


def write_flow_frame(tag_a, tag_b, outdir):
    """archive/flow/<tag_a>.png: cloud motion over the hour tag_a starts.

    RGB with no alpha channel — the app's renderer premultiplies alpha
    on decode, which would corrupt the displacement bytes."""
    from PIL import Image

    def load(tag):
        return np.asarray(Image.open(
            os.path.join(outdir, "archive", f"{tag}.jpg"))
            .convert("L").resize((FLOW_W, FLOW_H), Image.LANCZOS))

    os.makedirs(os.path.join(outdir, "archive", "flow"), exist_ok=True)
    Image.fromarray(encode_flow(flow_uv(load(tag_a), load(tag_b))),
                    mode="RGB") \
        .save(os.path.join(outdir, "archive", "flow", f"{tag_a}.png"))


def update_flows(frames, stamp, outdir):
    """Fill missing flow maps, prune stale ones, return their stamps.

    Every adjacent pair of archived frames exactly 1 h apart gets a map
    named for the interval start; holes in the archive get none. Only
    missing maps are computed — one Farneback solve (~1-3 s) per run in
    the steady state, and the first deploy backfills the whole existing
    window as a one-time cost of a few minutes. A flow map leaves the
    window by the same rule as the frame it starts from."""
    flowdir = os.path.join(outdir, "archive", "flow")
    os.makedirs(flowdir, exist_ok=True)

    def when(tag):
        return dt.datetime.strptime(tag, "%Y%m%d%H") \
            .replace(tzinfo=dt.timezone.utc)

    for a, b in zip(frames, frames[1:]):
        if (when(b) - when(a) == dt.timedelta(hours=1)
                and not os.path.exists(os.path.join(flowdir, f"{a}.png"))):
            # One unreadable archive JPEG must cost that pair its flow map
            # (the app crossfades there), not fail the run — archive frames
            # are never rewritten, so a hard fail here would block every
            # future publish on a frame nothing will ever repair.
            try:
                write_flow_frame(a, b, outdir)
            except Exception as e:
                print(f"flow {a}->{b} skipped: {e}")
    flows = []
    for name in os.listdir(flowdir):
        m = re.fullmatch(r"(\d{10})\.png", name)
        if not m:
            continue
        if (stamp - when(m.group(1))) >= dt.timedelta(hours=WINDOW_HOURS):
            os.remove(os.path.join(flowdir, name))
        else:
            flows.append(m.group(1))
    return sorted(flows)


def bake_aurora(coordinates):
    """OVATION [lon, lat, prob] grid -> (AURORA_H, AURORA_W, 4) uint8, the
    same texture AuroraService bakes live (aurora_service.dart): equirect
    (Greenwich at u=0.5, north pole at row 0), the spurious equator seam
    dropped, a separable Gaussian (sigma 1.5, radius 4; longitude wraps,
    latitude clamps), encoded 16-bit with the high byte in R, low in G, B
    mirroring R, alpha opaque. Non-negative throughout, so round-half-up
    (floor(x + 0.5)) matches Dart's round-half-away-from-zero exactly."""
    arr = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    lon = arr[:, 0].astype(np.int64)
    lat = arr[:, 1].astype(np.int64)
    keep = np.abs(lat) > 2  # OVATION emits a false low-value equator seam
    col = (lon + 180) % AURORA_W  # u=0.5 <-> lon 0
    row = 90 - lat  #                v=0   <-> lat +90
    val = np.floor(arr[:, 2] / 100.0 * 255.0 + 0.5).clip(0, 255)
    grid = np.zeros((AURORA_H, AURORA_W), np.float64)
    grid[row[keep], col[keep]] = val[keep]

    off = np.arange(-AURORA_BLUR_RADIUS, AURORA_BLUR_RADIUS + 1)
    k = np.exp(-(off ** 2) / (2 * AURORA_BLUR_SIGMA ** 2))
    knorm = k.sum()
    tmp = np.zeros_like(grid)
    for t, w in zip(off, k):
        tmp += w * np.roll(grid, -int(t), axis=1)  # longitude is periodic
    tmp /= knorm
    rows = np.arange(AURORA_H)
    out = np.zeros_like(grid)
    for t, w in zip(off, k):
        out += w * tmp[np.clip(rows + int(t), 0, AURORA_H - 1), :]  # poles clamp
    out /= knorm

    v16 = np.floor(out * 257.0 + 0.5).clip(0, 65535).astype(np.uint32)
    rgba = np.empty((AURORA_H, AURORA_W, 4), np.uint8)
    rgba[..., 0] = (v16 >> 8).astype(np.uint8)
    rgba[..., 1] = (v16 & 0xFF).astype(np.uint8)
    rgba[..., 2] = rgba[..., 0]
    rgba[..., 3] = 255
    return rgba


def write_outputs(px, stamp, outdir, aurora_rgba=None):
    """current_8192/current_4096/archive frame, prune, rewrite manifest.
    When [aurora_rgba] is given it lands as this hour's aurora frame; the
    manifest's aurora list is rebuilt from whatever aurora frames survive
    the window regardless, so a skipped fetch just holds the last ones."""
    from PIL import Image

    os.makedirs(os.path.join(outdir, "archive"), exist_ok=True)
    full = Image.fromarray(px, mode="L")
    half = full.resize((WIDTH // 2, HEIGHT // 2), Image.LANCZOS)
    full.save(os.path.join(outdir, "current_8192.jpg"), quality=JPEG_QUALITY)
    half.save(os.path.join(outdir, "current_4096.jpg"), quality=JPEG_QUALITY)
    tag = stamp.strftime("%Y%m%d%H")
    half.save(os.path.join(outdir, "archive", f"{tag}.jpg"),
              quality=JPEG_QUALITY)

    frames = []
    for name in os.listdir(os.path.join(outdir, "archive")):
        m = re.fullmatch(r"(\d{10})\.jpg", name)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y%m%d%H") \
            .replace(tzinfo=dt.timezone.utc)
        if (stamp - t) >= dt.timedelta(hours=WINDOW_HOURS):
            os.remove(os.path.join(outdir, "archive", name))
        else:
            frames.append(m.group(1))
    flows = update_flows(sorted(frames), stamp, outdir)

    # Aurora: an independent layer on the same rolling window. It lands under
    # the newest cloud hour's tag (OVATION has no past, so backfilled hours
    # get none) and prunes by the same age rule.
    auroradir = os.path.join(outdir, "archive", "aurora")
    os.makedirs(auroradir, exist_ok=True)
    if aurora_rgba is not None:
        Image.fromarray(aurora_rgba, mode="RGBA").save(
            os.path.join(auroradir, f"{tag}.png"))
    aurora = []
    for name in os.listdir(auroradir):
        m = re.fullmatch(r"(\d{10})\.png", name)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y%m%d%H") \
            .replace(tzinfo=dt.timezone.utc)
        if (stamp - t) >= dt.timedelta(hours=WINDOW_HOURS):
            os.remove(os.path.join(auroradir, name))
        else:
            aurora.append(m.group(1))

    manifest = {"latest": tag, "frames": sorted(frames), "flows": flows,
                "aurora": sorted(aurora), "window_hours": WINDOW_HOURS}
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest


def set_output(updated):
    """Tell the surrounding Action whether there is anything to publish."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"updated={'true' if updated else 'false'}\n")


def selftest():
    """Synthetic GMGSI + matteason through the full composite path."""
    outdir = "out/clouds-selftest"
    lat = np.linspace(-72.7368, 72.7154, 3000)  # ascending: exercises flip
    lon = (np.arange(5000) * 0.072 + 180.0) % 360.0 - 180.0  # 0-start: roll
    vis = np.full((3000, 5000), 180.0, np.float32)
    # Bright blob at lat 30, lon 45 (in the synthetic pre-orient layout).
    rows = np.abs(lat - 30.0) < 2.0
    cols = np.abs(lon - 45.0) < 2.0
    vis[np.ix_(rows, cols)] = 255.0
    # Bad-pixel (dqf fill) blob at lat -30, lon -45: matteason must show.
    # Zeroed shell around it, saturated fill inside — the shape real dqf
    # regions take (255 fill against dark clear sky at the sector seams).
    vis[np.ix_(np.abs(lat + 30.0) < 4.0, np.abs(lon + 45.0) < 4.0)] = 0.0
    bad = np.zeros_like(vis)
    bad[np.ix_(np.abs(lat + 30.0) < 2.0, np.abs(lon + 45.0) < 2.0)] = 1.0
    vis[np.ix_(np.abs(lat + 30.0) < 2.0, np.abs(lon + 45.0) < 2.0)] = 255.0
    canvas = np.full((HEIGHT, WIDTH), 60, np.uint8)

    v, la, lo = orient(vis, lat, lon)
    assert la[0] > la[-1] and np.all(np.diff(lo) > 0), "orient failed"
    b, _, _ = orient(bad, lat, lon)
    px = composite(v, b, la, lo, canvas)
    assert px.shape == (HEIGHT, WIDTH), px.shape

    def row(deg):
        return int((90.0 - deg) / 180.0 * HEIGHT)

    assert px[row(0), 100] == 180, "equator must be pure GMGSI"
    assert px[row(72.5), 100] == 60, "above BLEND_NONE must be pure matteason"
    assert px[row(89), 100] == 60 and px[row(-89), 100] == 60, "poles"
    mid = px[row(-(BLEND_FULL + BLEND_NONE) / 2), 100]
    assert 100 < mid < 140, f"ramp midpoint should sit between layers: {mid}"
    assert px[row(30), int((45 + 180) / 360 * WIDTH)] == 255, "blob misplaced"
    assert px[row(-30), int((-45 + 180) / 360 * WIDTH)] == 60, \
        "bad pixels must fall through to matteason"
    # The strip flanking the blob sits in the zeroed shell: only canvas
    # (60) and vis (0) may blend there. Plain bilinear would bleed the
    # 255 fill into a brighter rim along the blob's edge.
    shell = px[row(-29):row(-31),
               int((-48.5 + 180) / 360 * WIDTH):
               int((-46.9 + 180) / 360 * WIDTH)]
    assert shell.max() <= 61, \
        f"masked fill bled into its neighbours: max {shell.max()}"

    # depole, on a caricature of what matteason actually delivers: a
    # uniform belt, wedges above 74 deg standing in for the starburst
    # (harsher than the real ~8-30 row std up there), and an invented
    # flat cap above 87 deg brighter than the belt it sits in. Run at a
    # different grid size than the canvas, since the kernel is measured
    # on the ground and must not care about resolution.
    ph, pw = 512, 1024
    plat = 90.0 - (np.arange(ph) + 0.5) * (180.0 / ph)
    star = np.full((ph, pw), 200, np.uint8)
    star[np.abs(plat) > 74.0] = np.where(
        (np.arange(pw) * 12 // pw) % 2, 255, 120)
    star[np.abs(plat) > 87.0] = 250
    fixed = depole(star)
    below = np.abs(plat) <= POLE_NONE
    assert np.array_equal(star[below], fixed[below]), \
        "depole reached below POLE_NONE"
    for sign in (1, -1):
        near = int(np.argmin(np.abs(plat - sign * 88.0)))
        at_full = int(np.argmin(np.abs(plat - sign * POLE_FULL)))
        assert fixed[near].std() < 1.0, \
            f"starburst survived at the pole: std {fixed[near].std():.1f}"
        assert fixed[at_full].std() < star[at_full].std() / 4.0, \
            f"POLE_FULL barely smoothed: std {fixed[at_full].std():.1f}"
        # The invented cap must be held at the belt below it (200), not
        # left to glow at its own fill value (250).
        assert abs(float(fixed[near].mean()) - 200.0) < 1.5, \
            f"cap not carried to the belt mean: {fixed[near].mean():.1f}"
    assert POLE_NONE - POLE_BELT > BLEND_NONE, \
        "reference belt must read matteason only, not the GMGSI blend"

    # Flow core: a dense field of small cloud-puff blobs rolled right by
    # ~2 deg of longitude must read back as positive dx of that size.
    # The texture must be fine-grained — this Farneback tuning tracks
    # detail near its winsize (25 px) and averages the flat background
    # into anything larger, under-reading isolated smooth shapes to near
    # zero. Real IR frames at 1024x512 are exactly this kind of field.
    rng = np.random.default_rng(0)
    field = np.zeros((FLOW_H, FLOW_W), np.float32)
    r = 12
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    puff = np.exp(-(yy ** 2 + xx ** 2) / (2 * 5.0 ** 2)) * 220.0
    for _ in range(800):
        cy = rng.integers(r, FLOW_H - r)
        cols = (np.arange(-r, r + 1) + rng.integers(0, FLOW_W)) % FLOW_W
        field[cy - r:cy + r + 1, cols] += puff
    prev = np.round(field).clip(0, 255).astype(np.uint8)
    shift = round(FLOW_W * 2.0 / 360.0)  # ~2 deg of longitude in px
    uv = flow_uv(prev, np.roll(prev, shift, axis=1))
    expect = shift / FLOW_W
    med = float(np.median(uv[..., 0][prev > 60]))
    assert 0.7 * expect < med < 1.3 * expect, \
        f"blob flow off: median dx {med:.5f}, expected ~{expect:.5f}"
    enc = encode_flow(uv)
    assert enc.shape == (FLOW_H, FLOW_W, 3) and enc.dtype == np.uint8
    assert np.all(enc[..., 2] == 128), "B channel must stay 128"
    dec = enc[..., :2] / 255.0 * (2 * FLOW_MAX_UV) - FLOW_MAX_UV
    assert np.abs(dec - uv).max() <= 2 * FLOW_MAX_UV / 255.0, \
        "encode/decode must round-trip within one quantization step"

    # A stale and a fresh archive frame: only the stale one gets pruned,
    # and its flow map goes with it. The fresh frame is real data (the
    # flow pass decodes it) one hour before the latest, so the adjacent
    # pair yields archive/flow/2026071823.png — identical frames, near-
    # still bytes.
    stamp = dt.datetime(2026, 7, 19, 0, tzinfo=dt.timezone.utc)
    os.makedirs(os.path.join(outdir, "archive", "flow"), exist_ok=True)
    open(os.path.join(outdir, "archive", "2026071100.jpg"), "wb").close()
    open(os.path.join(outdir, "archive", "flow", "2026071100.png"),
         "wb").close()
    write_archive_frame(px, stamp - dt.timedelta(hours=1), outdir)
    manifest = write_outputs(px, stamp, outdir)
    assert manifest == {"latest": "2026071900",
                        "frames": ["2026071823", "2026071900"],
                        "flows": ["2026071823"],
                        "aurora": [],
                        "window_hours": WINDOW_HOURS}, manifest
    assert not os.path.exists(os.path.join(outdir, "archive",
                                           "2026071100.jpg")), "prune failed"
    assert not os.path.exists(os.path.join(outdir, "archive", "flow",
                                           "2026071100.png")), \
        "flow prune failed"
    from PIL import Image
    fp = Image.open(os.path.join(outdir, "archive", "flow",
                                 "2026071823.png"))
    assert fp.size == (FLOW_W, FLOW_H) and fp.mode == "RGB", \
        (fp.size, fp.mode)
    assert np.abs(np.asarray(fp).astype(np.int16) - 128).max() <= 1, \
        "identical frames must encode as (near-)zero motion"

    # Aurora bake: a 9x9 full-probability patch (radius-4 blur, so its centre
    # sees only saturated taps) must peak at ~65535 in the 16-bit R/G split;
    # B mirrors R, alpha is opaque, an on-equator cell is dropped, and the
    # far hemisphere stays dark.
    coords = []
    for la in range(-90, 91):
        for lo in range(360):
            p = 100 if (196 <= lo <= 204 and 61 <= la <= 69) else 0
            if lo == 200 and la == 0:
                p = 100  # equator seam: bake must drop it
            coords.append([lo, la, p])
    aur = bake_aurora(coords)
    assert aur.shape == (AURORA_H, AURORA_W, 4) and aur.dtype == np.uint8
    assert np.array_equal(aur[..., 2], aur[..., 0]), "B must mirror R"
    assert np.all(aur[..., 3] == 255), "alpha opaque"
    ccol = (200 + 180) % AURORA_W
    peak = (int(aur[90 - 65, ccol, 0]) << 8) | int(aur[90 - 65, ccol, 1])
    assert peak >= 65000, f"saturated patch centre too dim: {peak}"
    assert aur[90 - 0, ccol, 0] == 0 and aur[90 - 0, ccol, 1] == 0, \
        "equator seam must be dropped"
    assert aur[90 - (-80), (0 + 180) % AURORA_W, 0] == 0, "far cell not dark"

    # Shortwave sibling lookup: same date path under the SW product, and
    # the LW frame's own start stamp carried across so the two frames are
    # paired by observation time rather than by "newest in the hour".
    lwk = ("GMGSI_LW/2026/07/25/15/GLOBCOMPLIR_v3r0_blend_s202607251500000"
           "_e202607251509599_c202607251536500.nc")
    pre, sstamp = sw_lookup(lwk)
    assert pre == "GMGSI_SW/2026/07/25/15/", pre
    assert sstamp == "202607251500000", sstamp
    # The real 15Z pair differs in creation time (c...1536500 vs c...1539153)
    # and in the product token (LIR vs SIR), so matching must key on _s only.
    swk = ("GMGSI_SW/2026/07/25/15/GLOBCOMPSIR_v3r0_blend_s202607251500000"
           "_e202607251509599_c202607251539153.nc")
    assert f"_s{sstamp}" in swk and swk.startswith(pre)
    other = swk.replace("_s202607251500000", "_s202607251510000")
    assert f"_s{sstamp}" not in other, "a 15:10 frame must not match 15:00"
    assert sw_lookup("nonsense.nc") == (None, None)

    # Aurora archive lands under the newest hour, prunes stale on the window.
    open(os.path.join(outdir, "archive", "aurora", "2026071100.png"),
         "wb").close()
    m2 = write_outputs(px, stamp, outdir, aur)
    assert m2["aurora"] == ["2026071900"], m2["aurora"]
    assert not os.path.exists(os.path.join(outdir, "archive", "aurora",
                                           "2026071100.png")), "aurora prune"
    ap = Image.open(os.path.join(outdir, "archive", "aurora", "2026071900.png"))
    assert ap.size == (AURORA_W, AURORA_H) and ap.mode == "RGBA", \
        (ap.size, ap.mode)
    print(f"selftest ok: {outdir}", px.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default="out/clouds")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stand-in", action="store_true",
                    help="plain inversion instead of ir_visible (testing)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.stand_in:
        # GMGSI stores display counts: high = cold = cloud. Crude stretch
        # (surface ~70 -> black) so test renders resemble the real mapping.
        def to_visible(ir):
            return np.round((ir - 70.0) * (255.0 / 130.0)) \
                .clip(0, 255).astype(np.uint8)
    else:
        from ir_visible import ir_to_visible as to_visible

    keys = find_frames(dt.datetime.now(dt.timezone.utc))
    latest = ""
    manifest_path = os.path.join(args.outdir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            latest = json.load(f).get("latest") or ""
    # Everything newer than the published latest, oldest first — one
    # frame in the steady state, more when a run failed or was cancelled
    # and left hours behind. Stamps compare chronologically as strings.
    new = [k for k in keys
           if re.search(r"_s(\d{10})", k).group(1) > latest]  # sYYYYMMDDHH
    if not new:
        print(f"already current at {latest}; nothing to publish")
        set_output(False)
        return

    from PIL import Image
    canvas = Image.open(io.BytesIO(fetch(MATTEASON_URL))).convert("L")
    if canvas.size != (WIDTH, HEIGHT):
        raise SystemExit(f"matteason frame is {canvas.size}, "
                         f"not {WIDTH}x{HEIGHT}")
    canvas = np.asarray(canvas)
    mean, std = float(canvas.mean()), float(canvas.std())
    if not (MATTEASON_MEAN_MIN <= mean <= MATTEASON_MEAN_MAX
            and std >= MATTEASON_STD_MIN):
        raise SystemExit(f"matteason frame fails content sanity "
                         f"(mean {mean:.1f}, std {std:.1f})")

    # The aurora nowcast is "now", not per-frame — fetch it once and archive
    # it under the newest hour. A hiccup here must not sink the cloud run, so
    # a failure just skips this hour's aurora and holds the last ones.
    aurora_rgba = None
    try:
        aurora_rgba = bake_aurora(json.loads(fetch(AURORA_URL))["coordinates"])
    except Exception as e:
        print(f"aurora fetch/bake failed, skipping this hour: {e}",
              file=sys.stderr)

    # Backfilled hours reuse the current matteason canvas for their caps
    # (it's 3-hourly anyway); only the newest frame becomes current_*.
    for key in new:
        ir, bad, lat, lon, stamp = read_gmgsi(fetch(f"{S3}/{key}"))
        # Orient before the curve now, because the shortwave rescue needs
        # latitude and the terminator per pixel, and both are only
        # meaningful once rows are north-up and columns start at -180.
        ir, olat, olon = orient(ir, lat, lon)
        bad, _, _ = orient(bad, lat, lon)
        # The shift goes in BEFORE the tone curve, not after it: "this
        # cloud is warmer than it looks" is a statement about the input
        # value, so shifting there inherits the curve's monotonicity and
        # leaves its tuned mid/shoulder section (the 2026-07-24
        # convective retune) untouched.
        boost = low_cloud_boost(key, ir, olat, olon, stamp)
        if boost is not None:
            ir = ir + boost
        vis = to_visible(ir)
        px = depole(composite(vis, bad, olat, olon, canvas))
        if key is new[-1]:
            manifest = write_outputs(px, stamp, args.outdir, aurora_rgba)
        else:
            write_archive_frame(px, stamp, args.outdir)
    set_output(True)
    print(f"wrote {args.outdir} at {manifest['latest']} "
          f"({len(manifest['frames'])} archived, {len(new)} new)")


if __name__ == "__main__":
    main()
