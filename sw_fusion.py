#!/usr/bin/env python3
"""Recover warm low cloud that longwave IR alone cannot see.

The problem this solves, concretely: on 2026-07-25 a stratus deck sat
over the Twin Cities all morning.  GMGSI LW read 81 there, the
ir_visible LUT mapped that to 12, the app shader's pow(cloud, 1.6) turned
12/255 into 0.75% opacity, and the globe showed clear sky over a city
that could not see the sun.  matteason's visible frame read 140 in the
same place.

Why LW cannot fix this alone: a summer stratus top is only a few kelvin
colder than the ground under it, so its 11um brightness temperature lands
in the same 70..105 band as clear warm land at night.  ir_visible.py's
docstring already names this ("an ambiguity no per-pixel mapping can
break") and resolves it conservatively -- dim -- because painting every
clear night desert white is the worse failure.

The 3.9um shortwave channel (GMGSI_SW, same blend, same grid, same
10-minute window as GMGSI_LW) breaks the tie, because at 3.9um a water
cloud does not behave like the ground:

  * by day it REFLECTS sunlight, so its apparent 3.9um temperature runs
    warm -- warmer than its own 11um temperature;
  * by night it emits with lower emissivity than the surface, so its
    3.9um temperature runs COLD instead.

The sign flips.  Measured on the 2026-07-25 15Z and 12Z frames, in
stored counts (both channels carry NOAA's 0-255 "brightness temperature"
scaling, and the night clear-sky ridge below confirms the two scales
agree to within a couple of counts):

  day   cloud (LW-SW) median +11, clear +8  -- separable after the ridge
        below is removed
  night cloud (LW-SW) median  -3, clear -1  -- opposite sign

A single global LW-SW rule would therefore have suppressed real cloud
every night.  This module keeps the two regimes apart and blends them by
solar zenith.

VALIDATION (labels from GMGSI_VIS, which shares the grid and timestamp
exactly -- matteason does not, being 3-hourly on another projection, and
its 3 h of drift alone dragged apparent separability down to near
chance).  Measured inside LW's blind toe, 70 <= LW <= 100:

  day, mu > 0.42   AUC 0.937 (LW alone) -> 0.981 (with SW)
                   at a fixed 10% false-alarm rate, cloud caught
                   87% -> 93%
  held out on the 12Z frame with constants fitted only on 15Z:
                   AUC 0.964 -> 0.996; 88.6% of VIS-cloud scores > 0.5
                   against 0.8% of VIS-clear
  night            AUC 0.805 -> 0.845, cloud caught 53% -> 66%
                   (a floor: the only night label available is the same
                   deck seen by VIS 3 h later, so it carries that much
                   advection error)

WHERE IT DELIBERATELY DOES NOTHING.  Three regions where the rule was
measured to fail or could not be checked at all, and the boost tapers to
zero rather than guess:

  * near the terminator (-0.05 < mu < 0.42): the solar term is too weak
    to difference.  Measured AUC 0.445 at mu 0.15-0.25 and 0.837 at
    0.25-0.42, both WORSE than LW's own 0.833/0.898 -- using SW there
    would actively hurt.
  * poleward of ~55 deg: GMGSI_VIS gave no usable clear/cloud label
    split at 15Z, so the rule is simply unvalidated there.  It fades out
    by 60 deg, which is also where cloud_texture.py starts handing over
    to matteason.
  * LW above ~100: not ambiguous any more -- that is real cold cloud,
    and the tuned mid/shoulder part of the ir_visible curve (including
    the 2026-07-24 convective retune) must not be disturbed.

The output is a SHIFT IN LW COUNTS, not a brightness.  Shifting the input
of the existing curve says exactly the right thing physically -- "this
pixel is cloud whose top is too warm to look like one" -- and it inherits
the curve's monotonicity and its tuned upper half for free.  The app and
the shader need no change at all.

Usage:
  sw_fusion.py --selftest
"""

import numpy as np

# Clear-sky ridge of (LW - SW) by day, in stored counts, fitted over
# VIS-clear pixels on the 15Z frame: the difference is not zero even in
# clear air, because bare ground reflects some 3.9um too and the two
# channels' 0-255 scalings are close but not identical.  A straight line
# fitted the ridge to within 2.9 counts across LW 64..105, and held on
# the 12Z frame unchanged, so it is kept parametric rather than baked as
# a per-frame table that would overfit one hour.
DAY_RIDGE = (-0.1811, 13.29)  # (LW-SW)_clear ~= 13.29 - 0.1811 * LW
# By night there is no solar term and the ridge is nearly flat: measured
# median (SW - LW) over clear pixels was 0..1 counts across the toe.
NIGHT_RIDGE = 0.5

# Score ramps: residual value that scores 0, and the value that scores 1.
# Day edges are the 88th percentile of the clear residual and the 65th of
# the cloud residual -- deliberately overlapping-tolerant, since the ramp
# is soft and a half-score only half-shifts.
DAY_RAMP = (8.0, 25.0)
# Night residuals are small AND integer-quantized (the whole cloud/clear
# separation lives in about 4 counts), so the ramp is correspondingly
# tight.  This is the weak half of the rule; NIGHT_SHIFT reflects that.
NIGHT_RAMP = (1.5, 5.0)

# How far a fully-confident pixel is allowed to move along the LW curve.
# Day: 45 counts takes the Twin Cities case (LW 81) to an effective 126,
# which the ir_visible curve renders ~112 -- close to matteason's 140 in
# the same place, and enough for the shader to draw it at ~26% opacity
# instead of 0.75%.  Night gets less because it earned less: AUC 0.845
# against the day rule's 0.981.
DAY_SHIFT = 45.0
NIGHT_SHIFT = 28.0

# Regime weights by cos(solar zenith).  The gap between NIGHT_MU and
# DAY_MU is the measured dead zone and is intentionally left unboosted.
NIGHT_MU = (-0.02, -0.12)  # full night rule at or below -0.12
DAY_MU = (0.30, 0.45)      # full day rule at or above 0.45

# Toe gate: full effect through the ambiguous band, gone before the
# curve's tuned mid-cloud section.
TOE_FULL, TOE_NONE = 100.0, 130.0
# Latitude gate: unvalidated poleward of 55, and matteason takes over at
# 60-65 anyway (cloud_texture.BLEND_FULL/BLEND_NONE).
LAT_FULL, LAT_NONE = 50.0, 60.0


def _smoothstep(x, edge0, edge1):
    """Hermite 0..1 ramp; works for edge0 > edge1 (a descending ramp)."""
    t = np.clip((np.asarray(x, np.float32) - edge0) / (edge1 - edge0),
                0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def solar_zenith_cos(lat, lon, when):
    """cos(solar zenith) on the outer grid of 1-D `lat` x `lon` degrees.

    NOAA's low-precision solar position: good to a fraction of a degree,
    which is far finer than the 0.15-wide mu ramps it feeds.  `when` is
    an aware UTC datetime -- the frame's own timestamp, not wall clock,
    so a backfilled hour gets that hour's terminator.
    """
    doy = when.timetuple().tm_yday
    frac = when.hour + when.minute / 60.0 + when.second / 3600.0
    g = np.radians((360.0 / 365.24) * (doy - 2))
    decl = np.radians(23.44) * -np.cos(np.radians(360.0 / 365.24) * (doy + 10))
    eot = -7.659 * np.sin(g) + 9.863 * np.sin(2 * g + np.radians(102.9))
    ha = np.radians(15.0 * (frac + eot / 60.0 - 12.0)
                    + np.asarray(lon, np.float64)[None, :])
    la = np.radians(np.asarray(lat, np.float64))[:, None]
    return (np.sin(la) * np.sin(decl)
            + np.cos(la) * np.cos(decl) * np.cos(ha)).astype(np.float32)


def low_cloud_shift(lw, sw, mu, lat, bad=None):
    """LW counts to ADD before the ir_visible curve, as float32.

    `lw` and `sw` are the two GMGSI frames as stored (same shape, same
    grid, oriented identically), `mu` is cos(solar zenith) broadcastable
    to that shape, `lat` is the per-row latitude in degrees.  `bad` is
    SW's dqf mask, and passing it is NOT optional in the pipeline: a
    dqf-flagged SW pixel holds saturated 255 fill, and 255 against a toe
    LW of ~81 is a residual of +174 in the NIGHT direction -- exactly the
    shape of a confident nocturnal stratus detection.  Left unmasked, the
    Meteosat-IODC fill bowl and the sector seams would paint themselves
    as solid low cloud every night.

    Returns >= 0 everywhere: this rule may only ever ADD cloud, never
    remove it, so a bad SW frame can dim nothing that LW already found.
    """
    lw = np.asarray(lw, np.float32)
    sw = np.asarray(sw, np.float32)
    mu = np.broadcast_to(np.asarray(mu, np.float32), lw.shape)

    day_resid = (lw - sw) - (DAY_RIDGE[1] + DAY_RIDGE[0] * lw)
    night_resid = (sw - lw) - NIGHT_RIDGE

    day = _smoothstep(day_resid, *DAY_RAMP) * _smoothstep(mu, *DAY_MU)
    night = _smoothstep(night_resid, *NIGHT_RAMP) * _smoothstep(mu, *NIGHT_MU)

    shift = day * DAY_SHIFT + night * NIGHT_SHIFT

    # Gates. The toe gate keeps the tuned mid/shoulder curve untouched;
    # the latitude gate stops at the edge of what VIS could validate.
    shift *= _smoothstep(lw, TOE_NONE, TOE_FULL)
    latg = _smoothstep(np.abs(np.asarray(lat, np.float32)), LAT_NONE, LAT_FULL)
    shift *= latg[:, None] if latg.ndim == 1 else latg

    # A fill pixel (-9999), a NaN, or anything SW's dqf flags must not be
    # promoted into cloud.
    good = np.isfinite(lw) & np.isfinite(sw) & (lw > 0.0)
    if bad is not None:
        good &= ~np.asarray(bad, bool)
    return np.where(good, shift, 0.0).astype(np.float32)


def selftest():
    import datetime as dt

    # --- ramp/gate mechanics -------------------------------------------
    lat = np.array([0.0, 0.0, 0.0, 0.0])
    hot = np.full((4, 1), 1.0, np.float32)   # high sun
    dark = np.full((4, 1), -1.0, np.float32)  # deep night

    # Day: a strongly reflecting low cloud (big LW-SW) must shift hard.
    lw = np.array([[81.0], [81.0], [81.0], [81.0]], np.float32)
    sw_cloud = lw - (np.polyval(DAY_RIDGE, lw) + 30.0)
    s = low_cloud_shift(lw, sw_cloud, hot, lat)
    assert np.all(s > 0.9 * DAY_SHIFT), s
    # Day: a pixel sitting exactly on the clear ridge must not move.
    sw_clear = lw - np.polyval(DAY_RIDGE, lw)
    assert np.all(low_cloud_shift(lw, sw_clear, hot, lat) == 0.0)

    # Night: the sign is the other way round. The same reflecting-looking
    # pixel must NOT be boosted at night, and the emissivity-deficit one
    # must be.
    assert np.all(low_cloud_shift(lw, sw_cloud, dark, lat) == 0.0), \
        "day-signed residual must not fire at night"
    sw_night = lw + NIGHT_RIDGE + 8.0
    sn = low_cloud_shift(lw, sw_night, dark, lat)
    assert np.all(sn > 0.9 * NIGHT_SHIFT), sn
    assert np.all(low_cloud_shift(lw, sw_night, hot, lat) == 0.0), \
        "night-signed residual must not fire by day"

    # Terminator dead zone: neither rule may fire where both were measured
    # to be worse than LW alone.
    for m in (-0.01, 0.0, 0.1, 0.2, 0.29):
        mm = np.full((4, 1), m, np.float32)
        assert np.all(low_cloud_shift(lw, sw_cloud, mm, lat) == 0.0), m
        assert np.all(low_cloud_shift(lw, sw_night, mm, lat) == 0.0), m

    # Toe gate: cold tops keep the tuned curve.
    cold = np.full((4, 1), 140.0, np.float32)
    assert np.all(low_cloud_shift(cold, cold - 40.0, hot, lat) == 0.0), \
        "LW above TOE_NONE must not be shifted"

    # Latitude gate.
    polar = np.array([70.0, 65.0, 61.0, -70.0])
    assert np.all(low_cloud_shift(lw, sw_cloud, hot, polar) == 0.0)
    mid = np.array([0.0, 20.0, 40.0, 49.0])
    assert np.all(low_cloud_shift(lw, sw_cloud, hot, mid) > 0.9 * DAY_SHIFT)

    # Never negative, never NaN-propagating, fill stays put.
    junk = np.array([[-9999.0], [np.nan], [0.0], [81.0]], np.float32)
    out = low_cloud_shift(junk, junk - 40.0, hot, lat)
    assert np.all(np.isfinite(out)) and np.all(out >= 0.0), out
    assert out[0, 0] == 0.0 and out[1, 0] == 0.0 and out[2, 0] == 0.0

    # SW's own dqf fill is the dangerous case: saturated 255 against a toe
    # LW reads as a textbook night-stratus residual. Unmasked it fires;
    # masked it must not.
    fill = np.full((4, 1), 255.0, np.float32)
    assert np.all(low_cloud_shift(lw, fill, dark, lat) > 0.0), \
        "this is the trap the bad mask exists to close"
    masked = low_cloud_shift(lw, fill, dark, lat,
                             bad=np.ones((4, 1), bool))
    assert np.all(masked == 0.0), masked

    # --- the case that started this ------------------------------------
    # Twin Cities, 2026-07-25 15Z: LW 81.1, SW 58.4 as measured.
    tc = low_cloud_shift(np.array([[81.1]], np.float32),
                         np.array([[58.4]], np.float32),
                         np.array([[0.52]], np.float32),
                         np.array([44.95]))
    assert tc[0, 0] > 40.0, f"Twin Cities stratus must shift hard: {tc}"
    from ir_visible import ir_to_visible
    before = int(ir_to_visible(np.array([81.1]))[0])
    after = int(ir_to_visible(np.array([81.1 + tc[0, 0]]))[0])
    assert before < 20 and after > 100, (before, after)

    # --- solar geometry sanity -----------------------------------------
    noon = dt.datetime(2026, 7, 25, 12, 0, tzinfo=dt.timezone.utc)
    mu = solar_zenith_cos(np.array([0.0]), np.array([0.0, 180.0]), noon)
    assert mu[0, 0] > 0.9 and mu[0, 1] < -0.9, mu
    jun = dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.timezone.utc)
    arctic = solar_zenith_cos(np.array([66.6]), np.array([0.0]), jun)
    assert arctic[0, 0] > 0, "arctic midsummer must be lit"

    print(f"selftest ok: Twin Cities 81.1 -> +{tc[0, 0]:.0f} counts, "
          f"render {before} -> {after}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    else:
        ap.error("nothing to do but --selftest")
