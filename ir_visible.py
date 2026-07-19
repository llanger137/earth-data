#!/usr/bin/env python3
"""Map GMGSI longwave-IR values to a visible-look cloud greyscale.

The Earth app's cloud layer is matteason's visible-style map (white puffy
clouds on black).  The 73N-73S band is being switched to NOAA GMGSI LW
(hourly, global, day and night), whose raw values look wrong on the globe:
the field is a washed-out grey where clear sky sits at mid levels and only
deep convection is bright.  ir_to_visible() is the pure per-pixel value
mapping that makes the IR frame sit next to matteason's look.

Value convention (verified empirically on the 2026-07-19 00Z and 02Z
frames, not assumed): the `data` variable stores a 0-255 scaled
brightness where HIGH = COLD = CLOUD, i.e. it is already white-is-cloud.
Measured as stored: clear desert (night Sahara, day Australia) 62-79,
clear tropical ocean 66-91, marine stratocumulus 79-101, mid cloud
~100-140, frontal/convective tops 150-220.  The value 255 appeared only
as missing-data fill (dqf == 20): the bowl north of the Meteosat-IODC
usable disk and thin strips on the sector seams at ~22E and ~94E.  It
maps to white here; masking it spatially is the pipeline's job.  (The
IODC limb also degrades below its flagged bowl — 35-85E poleward of
~60N is structureless smear, brightening falsely from ~66N — which is
why cloud_texture.py blends to matteason at 60-65 rather than the
contract's original 64-71.)

The curve (monotone cubic through CONTROL_POINTS, baked to a 256-entry
LUT):
  * black point 70 - the clear warm-surface bulk (ocean and night desert,
    values 62-92 with p75 = 92) crushes to black, so the Sahara and the
    trade-wind oceans read dark as in the reference;
  * slow toe 70..105 - marine stratus and warm low cloud share values
    79-95 with clear night land, an ambiguity no per-pixel mapping can
    break, so this zone renders dim-but-present (~10-60) rather than
    black or bright;
  * steep rise 105..190 - where matteason-white cloud lives (its solid
    cloud has IR p25 = 112, p50 = 140, p75 = 169), giving fronts and
    convection their white;
  * saturation >= 190 - cold tops go near-white; 255 stays 255.
Tuned against the matteason 8192x4096 frame of the same hour using
class-conditional stats (matteason<=10 vs >=220 pixel populations over
60S-60N) and side-by-side renders of both day and night hemispheres.

Documented residuals: cold night/winter high terrain (pre-dawn Tibet
85-119, winter-night Altiplano ~107-190 depending on hour) keeps
faint-to-mid grey false cloud -
matteason's own night imagery paints those regions bright too, so the
swap does not add a mismatch; stratocumulus decks (Peru, California) are
much dimmer than matteason's white sheets; matteason's desert-albedo
bleed (interior Australia rendered white by day) has no IR counterpart -
deserts go correctly black, which reads as a fix, not a regression.

The mapping is orientation-agnostic: input is the GMGSI `data` array as
stored (any shape, floats; -9999 fill and NaN map to black) and the
pipeline handles row order and longitude alignment before/after.

Usage:
  ir_visible.py --selftest                    # synthetic ramp, no network
  ir_visible.py gmgsi.nc matteason.jpg out/   # side-by-side re-tune aid
"""

import argparse
import os

import numpy as np

# (stored IR value, output brightness) - monotone increasing in both.
CONTROL_POINTS = (
    (0, 0),
    (70, 0),      # black point: clear warm surface ends here
    (88, 25),     # toe: stratus / clear-land ambiguity stays dim
    (105, 60),
    (135, 140),   # mid cloud picks up body
    (165, 218),
    (190, 247),   # cold tops saturate
    (255, 255),
)


def _monotone_cubic_lut(points):
    """Bake a Fritsch-Carlson monotone cubic through `points` into a
    256-entry uint8 LUT.  Monotone data in -> monotone curve out, no
    overshoot, smooth slopes (no piecewise-linear banding kinks)."""
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    h = np.diff(x)
    delta = np.diff(y) / h

    d = np.empty_like(x)
    d[0], d[-1] = delta[0], delta[-1]
    for i in range(1, len(x) - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:  # weighted harmonic mean keeps the interpolant monotone
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    v = np.arange(256, dtype=np.float64)
    seg = np.clip(np.searchsorted(x, v, side="right") - 1, 0, len(h) - 1)
    t = (v - x[seg]) / h[seg]
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    out = (h00 * y[seg] + h10 * h[seg] * d[seg]
           + h01 * y[seg + 1] + h11 * h[seg] * d[seg + 1])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


_LUT = _monotone_cubic_lut(CONTROL_POINTS)


def ir_to_visible(ir):
    """GMGSI LW values as stored -> visible-look greyscale (uint8, same
    shape).  White = cloud, black = clear.  Pure per-pixel mapping;
    NaN and negative fill values (-9999) come out black."""
    vals = np.asarray(ir, dtype=np.float32)
    vals = np.where(np.isnan(vals), 0.0, vals)
    idx = np.clip(np.rint(vals), 0, 255).astype(np.uint8)
    return _LUT[idx]


def selftest():
    """Synthetic ramp through the full mapping - no network, no files."""
    ramp = np.arange(256, dtype=np.float32)
    out = ir_to_visible(ramp)
    assert out.dtype == np.uint8 and out.shape == ramp.shape
    assert np.all(np.diff(out.astype(np.int32)) >= 0), "must be monotone"
    assert out[0] == 0 and out[70] == 0, "clear surface must be black"
    assert out[255] == 255, "coldest value must be white"
    assert out[190] >= 240, "cold tops must be near-white"
    assert 5 <= out[88] <= 45, "stratus zone must be dim but present"
    assert 100 <= out[135] <= 180, "mid cloud must keep structure"

    field = np.array([[[-9999.0, np.nan], [64.0, 140.0]],
                      [[80.0, 200.0], [255.0, 30.0]]], dtype=np.float32)
    px = ir_to_visible(field)
    assert px.shape == field.shape, "shape must survive any rank"
    assert px[0, 0, 0] == 0 and px[0, 0, 1] == 0, "fill/NaN must be black"
    assert px[1, 1, 0] == 255, "255 (incl. dqf fill) stays white"
    print("selftest ok: LUT",
          [int(_LUT[v]) for v in (0, 70, 88, 105, 135, 165, 190, 255)])


def _load_gmgsi(path):
    import h5py

    with h5py.File(path, "r") as f:
        return f["data"][0]


def compare(gmgsi_path, matteason_path, outdir):
    """Write raw/transformed/reference/side-by-side PNGs for re-tuning.

    The matteason frame is cropped to the GMGSI latitude band
    (72.7N..72.7S) and resized to the same grid so rows line up.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(outdir, exist_ok=True)

    ir = _load_gmgsi(gmgsi_path)
    tx = ir_to_visible(ir)
    h, w = tx.shape

    m = Image.open(matteason_path).convert("L")
    mh = m.size[1]
    r0 = round((90 - 72.7154) / 180 * mh)
    r1 = round((90 + 72.7368) / 180 * mh)
    ref = np.asarray(m.crop((0, r0, m.size[0], r1)).resize((w, h)))

    raw = np.clip(np.nan_to_num(np.asarray(ir, np.float32)),
                  0, 255).astype(np.uint8)
    for name, px in [("raw_ir", raw), ("transformed", tx),
                     ("matteason_ref", ref)]:
        Image.fromarray(px).save(os.path.join(outdir, name + ".png"))
    gap = np.full((16, w), 128, np.uint8)
    Image.fromarray(np.vstack([tx, gap, ref])).save(
        os.path.join(outdir, "side_by_side.png"))
    print(f"wrote raw_ir/transformed/matteason_ref/side_by_side to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gmgsi_nc", nargs="?", help="GMGSI_LW NetCDF path")
    ap.add_argument("matteason", nargs="?", help="matteason frame path")
    ap.add_argument("outdir", nargs="?", default="out/compare")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.gmgsi_nc and args.matteason):
        ap.error("need GMGSI .nc and matteason image paths (or --selftest)")
    compare(args.gmgsi_nc, args.matteason, args.outdir)


if __name__ == "__main__":
    main()
