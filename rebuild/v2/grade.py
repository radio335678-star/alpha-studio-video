#!/usr/bin/env python3
"""Luminous Indigo colour grade for the v2 promo (1280x720 backgrounds, per-frame).

Replaces render.py's crude `INDIGO_GRADE = [1.12, 0.93, 1.10]` multiply, which
laid a lavender cast over everything and left the cyan-blue dot fields off-brand.

Signal path (all vectorised float32, no per-pixel Python):

  1. float conversion; Rec.709 luma Y and chroma plane (x = R-Y, y = B-Y)
  2. *neutral* hue pull inside that opponent plane. Y is carried through
     untouched and only the (x, y) chroma is steered, so R == G == B stays
     exactly R == G == B: whites stay white, greys stay grey, there is no
     lavender cast and black never drifts purple. Chroma is steered toward a
     blend of two brand anchors -- indigo #6C4BFF (dominant) and cyan #2BD9FF
     (accent). Selection uses squared-chord distance on the unit chroma circle
     (2 - 2 u.v), which is monotone in the hue angle but needs no atan2, and the
     cyan claim is gated on chroma *purity*, so clipped / white-mixed cyan-blue
     of the dot fields falls back to indigo while genuinely on-brand cyan accents
     survive intact.
  3. S-curve on luma only (hue and saturation preserving)
  4. gamut-safe chroma rescale (scales chroma, never clips a channel, so no hue
     shift on the blown highlights) then a black lift: 0,0,0 -> exactly #07070C
     and 255,255,255 -> exactly 255
  5. saturation boost scaling with `energy`
  6. optional bloom (energy > 0.4 only): soft-knee highlight -> 8x box
     downsample -> gaussian -> upsample -> 9..12% back, indigo tinted
  7. monochrome film grain, sigma-equivalent 1.5/255, seeded per frame

Work is done in cache-resident row blocks: a naive full-frame version carries
~30 live 3.7 MB temporaries (110 MB working set, 2.3x slower here), while 48-row
blocks keep everything in L2/L3. Measured on one core at 1280x720: ~25 ms
without bloom, ~33 ms with bloom (see `benchmark`).

Drop-in for render.py (v2), replacing grade_bg():
    frame = grade.indigo_look(BG[bg](W, H, t=t_abs, energy=e, frame=f), t_abs, e)

Public API
----------
    indigo_look(frame_uint8, t, energy) -> uint8 ndarray   (the grade)
    grade_image(PIL.Image, t, energy)   -> PIL.Image
    punch_curve(energy=0.5, n=1024)     -> (n, 3) transfer LUT
    hue_targets()                       -> brand anchor geometry + tuning
    contrast_amount / sat_amount / bloom_amount / bloom_tint
    verify(shots=None, synth=None)      -> dict of QA numbers
    make_qa(out_dir=None)               -> writes _grade_*.jpg + _grade_sheet.jpg
    benchmark(n=15)                     -> timing dict

CLI
---
    python3 grade.py in1.jpg in2.jpg ... outdir/   # grade stills (QA)
    python3 grade.py --qa                            # regenerate QA stills/sheet
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

try:  # scipy: fast separable gaussian for the bloom (box fallback if absent)
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Brand
# --------------------------------------------------------------------------- #
INDIGO = (108, 75, 255)     # #6C4BFF primary
CYAN = (43, 217, 255)       # #2BD9FF secondary
BASE = (7, 7, 12)           # #07070C
WHITE = (255, 255, 255)

INDIGO_F = np.array(INDIGO, dtype=np.float32) / 255.0
CYAN_F = np.array(CYAN, dtype=np.float32) / 255.0
BASE_F = np.array(BASE, dtype=np.float32) / 255.0
LIFT_K = (1.0 - BASE_F).astype(np.float32)      # per-channel lift slope
BASE_S = tuple(float(v) for v in BASE_F)
LIFT_S = tuple(float(v) for v in LIFT_K)

LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722
# exact inverse of the (Y, x = R-Y, y = B-Y) decomposition:
#   R = Y + x,  B = Y + y,  G = Y + GR_GX*x + GR_GY*y
GR_GX = -0.29715
GR_GY = -0.10092

GRAIN_SIGMA = 1.5 / 255.0    # spec: film grain ~1.5/255
GRAIN_AMP = GRAIN_SIGMA * math.sqrt(3.0)     # uniform -> matching sigma
GRAIN_SEED_XOR = 0x9E3779B9
BLOOM_ENERGY_MIN = 0.4       # spec: bloom only above this energy
BLOOM_KNEE = 0.82
BLOOM_DS = 8                 # 1280x720 -> 160x90 for the blur
FPS = 30.0
BLOCK_ROWS = 48              # cache-resident working set per pass (measured best)

# Purity reference: the chroma magnitude an *undiluted* brand cyan can have at
# a given luma.  Above ~0.41 luma that is the gamut-limited cyan (line through
# #2BD9FF towards white); below it, a dim cyan accent is legitimately diluted,
# so the ceiling follows the cyan ray through the origin (M/Y = 0.86) instead
# and dim cyan accents survive instead of being read as washed teal.
#   m_max(Y) = min(PUR_A + PUR_B*Y, PUR_C*Y)
PUR_A, PUR_B, PUR_C = 0.50551, 0.15590, 1.40

F32 = np.float32
F0, F1, F3 = F32(0.0), F32(1.0), F32(3.0)

TUNING = dict(
    cyan_sigma_deg=5.5,      # half-width of the cyan claim (deg)
    purity_power=2.0,        # how harshly white-mixed cyan is rejected
    chroma_knee=0.030,       # chroma below this counts as neutral
    pull=1.00,
    gamut_floor=0.40,        # hot highlights keep chroma, clip a little
)

def hue_targets():
    """Geometry of the two brand hue anchors in the (x, y) opponent plane."""
    out = {}
    for name, c in (("indigo", INDIGO_F), ("cyan", CYAN_F)):
        y = float(c[0]) * LUMA_R + float(c[1]) * LUMA_G + float(c[2]) * LUMA_B
        x = float(c[0]) - y
        yy = float(c[2]) - y
        m = math.hypot(x, yy)
        out[name] = {
            "rgb": tuple(int(round(float(v) * 255)) for v in c),
            "luma": y, "chroma": m,
            "angle": math.degrees(math.atan2(yy, x)) % 360.0,
            "unit": (x / m, yy / m),
        }
    gap = abs(out["indigo"]["angle"] - out["cyan"]["angle"]) % 360.0
    out["gap_deg"] = min(gap, 360.0 - gap)
    out.update(TUNING)
    return out


def contrast_amount(energy: float) -> float:
    """S-curve mix amount; punchier through the drop."""
    return 0.20 + 0.18 * float(np.clip(energy, 0.0, 1.0))


def sat_amount(energy: float) -> float:
    """Saturation multiplier; punchier through the drop."""
    return 1.0 + 0.28 * float(np.clip(energy, 0.0, 1.0))


def bloom_amount(energy: float) -> float:
    """Bloom mix, 9% .. 12% as the drop builds."""
    return 0.09 + 0.03 * float(np.clip(energy, 0.0, 1.0))


def bloom_tint() -> np.ndarray:
    """Indigo bloom tint (55% indigo / 45% white), luma-normalised."""
    tint = 0.55 * INDIGO_F + 0.45
    return (tint / tint.mean()).astype(np.float32)


def _curve(x, a):
    """Gentle S-curve in place: mix identity with a smoothstep by amount `a`."""
    s = x * x * (F3 - x * F32(2.0))
    x += (s - x) * F32(a)
    return x


def _curve_lut(energy: float = 0.5, n: int = 1024) -> np.ndarray:
    """Sampled S-curve, straight from the expression the hot loop uses."""
    x = np.linspace(0.0, 1.0, int(n), dtype=np.float32)
    return _curve(x, contrast_amount(energy))


def punch_curve(energy: float = 0.5, n: int = 1024) -> np.ndarray:
    """(n, 3) transfer curve of the grade, in 0..1 output units.

    Row x is the response to a neutral input of luma x:
        out_c = base_c + curve(x) * (1 - base_c)
    so it maps 0 -> #07070C exactly, 1 -> 255 exactly, and is monotonic in
    between (nothing crushed, nothing clipped). Sampled from the same analytic
    expression the hot loop uses, so LUT and code cannot drift.
    """
    y = _curve_lut(energy, n)[:, None]
    return (BASE_F[None, :] + y * (1.0 - BASE_F[None, :])).astype(np.float32)


# --------------------------------------------------------------------------- #
# Small array helpers (bloom)
# --------------------------------------------------------------------------- #
def _box2(a):
    """Exact 2x2 box downsample of an (h, w) or (h, w, c) array."""
    return (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) * F32(0.25)


def _down(a, k=BLOOM_DS):
    """k x k box downsample by repeated 2x2 passes: no alias flicker."""
    while k > 1:
        a = _box2(a)
        k //= 2
    return a


def _blur(a, sigma=2.2):
    """Separable gaussian on a small image."""
    if _HAVE_SCIPY:
        if a.ndim == 3:
            return _gaussian_filter(a, sigma=(sigma, sigma, 0.0), mode="nearest")
        return _gaussian_filter(a, sigma=sigma, mode="nearest")
    # pragma: no cover - 3x3 box fallback
    b = np.pad(a, ((0, 0), (1, 1)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    acc = b[:, 0:a.shape[1]] + b[:, 1:a.shape[1] + 1] + b[:, 2:a.shape[1] + 2]
    return acc * F32(1.0 / 3.0)


def _grain(h, w, frame_index):
    """Monochrome film grain, sigma-equivalent 1.5/255, deterministic per frame.

    Drawn from a cheap uniform source (same seed -> same frame noise) and added
    equally to all three channels, so it can never introduce a colour cast.
    """
    seed = (int(frame_index) & 0xFFFFFFFF) ^ GRAIN_SEED_XOR
    g = np.random.default_rng(seed).integers(-32768, 32768, (h, w), dtype=np.int16)
    g = g.astype(np.float32)
    g *= F32(GRAIN_AMP / 32767.0)
    return g


# --------------------------------------------------------------------------- #
# The grade
# --------------------------------------------------------------------------- #
def _block_consts(energy, strength):
    """Per-frame scalar constants for the block kernel."""
    tg = hue_targets()
    (ix, iy), (cx, cy) = tg["indigo"]["unit"], tg["cyan"]["unit"]
    # squared-chord half width == (2 sin(sigma/2))^2, i.e. radius == 1 exactly
    # at the cyan anchor and 0 at cyan_sigma_deg away from it.
    w2 = (2.0 * math.sin(math.radians(tg["cyan_sigma_deg"]) * 0.5)) ** 2
    return dict(
        ix=F32(ix), iy=F32(iy), cx=F32(cx), cy=F32(cy),
        dd_x=F32(cx - ix), dd_y=F32(cy - iy),
        dot_c=F32(w2 - 2.0),
        inv_w=F32(1.0 / w2),
        purity_a=F32(PUR_A), purity_b=F32(PUR_B), purity_c=F32(PUR_C),
        purity_pow=int(tg["purity_power"]),
        knee=F32(1.0 / float(tg["chroma_knee"])) * F32(float(strength) * tg["pull"]),
        gamut_floor=F32(tg["gamut_floor"]),
        sat=F32(sat_amount(energy)), contrast=F32(contrast_amount(energy)),
        lift_k=LIFT_S, base=BASE_S,
        gx=F32(GR_GX), gy=F32(GR_GY),
        lr=F32(LUMA_R), lg=F32(LUMA_G), lb=F32(LUMA_B),
    )


def _block_kernel(c):
    """Build the closure that grades one cache-resident block in place."""
    ix, iy, cx, cy = c["ix"], c["iy"], c["cx"], c["cy"]
    dc, invw = c["dot_c"], c["inv_w"]
    ddx, ddy = c["dd_x"], c["dd_y"]
    pa, pb, pc, ppow = c["purity_a"], c["purity_b"], c["purity_c"], c["purity_pow"]
    knee, sat, con = c["knee"], c["sat"], c["contrast"]
    gamut_floor = c["gamut_floor"]
    gx, gy = c["gx"], c["gy"]
    lr, lg, lb = c["lr"], c["lg"], c["lb"]
    kr, kg, kb = c["lift_k"]
    br, bg, bb = c["base"]

    def kernel(r, g, b):
        # --- luma + chroma -----------------------------------------------------
        Y = r * lr
        Y += g * lg
        Y += b * lb
        x = r - Y
        y = b - Y
        M = np.sqrt(x * x + y * y)

        # --- neutral hue pull toward the brand anchors -------------------------
        # claim = how close the unit chroma direction sits to the cyan anchor,
        # as 2 u.cyan == 2 - |u - cyan|^2 (monotone in angle, no atan2 needed).
        s = x * cx
        s += y * cy
        s *= F32(2.0)
        s /= np.maximum(M, F32(1e-5))
        s += dc
        s *= invw                      # > 0 inside the cyan claim radius
        np.clip(s, F0, F1, out=s)
        p = s * s                      # quadratic falloff, no exp needed
        # purity: a clipped / white-mixed cyan-blue may not claim the accent
        pur = pb * Y
        pur += pa
        sh = Y * pc
        np.minimum(pur, sh, out=pur)
        np.maximum(pur, F32(1e-6), out=pur)
        pur = M / pur
        np.clip(pur, F0, F1, out=pur)
        for _ in range(ppow - 1):
            pur *= pur
        p *= pur

        # presence gate: neutrals are untouched -> greys stay truly neutral
        w = M * knee                   # knee already carries `pull`
        np.clip(w, F0, F1, out=w)
        w *= w

        # blended target direction, (1-p)*indigo + p*cyan = indigo + p*delta.
        # Left unnormalised: the blend is 0.83..1.0 long, which only trims
        # mid-way chroma a little.
        tx = p * ddx
        tx += ix
        ty = p * ddy
        ty += iy
        x += w * (M * tx - x)
        y += w * (M * ty - y)

        # --- S-curve on luma only (hue and saturation preserving) -------------
        Y = _curve(Y, con)

        # --- one chroma scale: gamut limit vs the requested saturation --------
        # `up` is how much chroma the pixel's luma can still afford, so boosting
        # saturates until the gamut says stop and never rotates a clipped hue.
        gv = x * gx
        gv += y * gy
        hi = np.maximum(np.maximum(x, gv), y)
        up = F1
        up -= Y
        up /= np.maximum(hi, F32(1e-5))
        np.clip(up, gamut_floor, sat, out=up)
        x *= up
        y *= up
        gv *= up                       # linear: no need to recompute from x, y

        # --- rebuild RGB + black lift: 0 -> #07070C, 1 -> 255 ------------------
        # out_c = base_c + (Y + chroma_c) * (1 - base_c); Y keeps the per
        # channel slope so 255,255,255 lands on 255 exactly.
        np.add(x, Y * kr, out=r)
        r += br
        np.add(y, Y * kb, out=b)
        b += bb
        np.add(gv, Y * kg, out=g)
        g += bg
        return r, g, b
    return kernel


def _highlight_src(r, g, b):
    """Soft-knee highlight energy feeding the bloom (a scalar field).

    The bloom is specified as indigo tinted, so it is driven by highlight
    *energy* (channel max) rather than colour: one plane to pool and blur
    instead of three, and the tint supplies the hue.
    """
    mx = np.maximum(r, g)
    mx = np.maximum(mx, b)
    w = mx - F32(BLOOM_KNEE)
    np.clip(w, F0, F32(1.0 - BLOOM_KNEE), out=w)
    w *= F32(1.0 / (1.0 - BLOOM_KNEE))
    w *= w
    w *= mx
    return w


def indigo_look(frame, t: float = 0.0, energy: float = 0.5, *,
                frame_index=None, bloom=None, grain: float = GRAIN_SIGMA,
                strength: float = 1.0, block: int = BLOCK_ROWS, ret_float: bool = False):
    """Grade one RGB uint8 frame -> uint8 frame of the same size.

    frame_index  grain seed; defaults to round(t * 30)
    bloom        None -> automatic (only when energy > BLOOM_ENERGY_MIN)
    grain        sigma in 0..1 units (0 disables)
    """
    f = np.asarray(frame)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("indigo_look expects an HxWx3 RGB frame, got %r" % (f.shape,))
    if f.dtype != np.uint8:
        f = np.clip(f, 0, 255).astype(np.uint8)

    energy = float(np.clip(energy, 0.0, 1.0))
    if frame_index is None:
        frame_index = int(round(float(t) * FPS))
    h, w = f.shape[:2]
    if bloom is None:
        bloom = energy > BLOOM_ENERGY_MIN
    bs = max(8, int(block) & ~7)                    # keep the 8-row bloom grid
    kernel = _block_kernel(_block_consts(energy, strength))

    scale = F32(1.0 / 255.0)
    R = np.multiply(f[:, :, 0], scale, dtype=np.float32)
    G = np.multiply(f[:, :, 1], scale, dtype=np.float32)
    B = np.multiply(f[:, :, 2], scale, dtype=np.float32)

    # ---- pass 1: grade blocks, collecting the highlight field ----------------
    hl = np.empty((h, w), np.float32) if bloom else None
    for y0 in range(0, h, bs):
        y1 = min(y0 + bs, h)
        r, g, b = kernel(R[y0:y1], G[y0:y1], B[y0:y1])
        R[y0:y1] = r
        G[y0:y1] = g
        B[y0:y1] = b
        if bloom:
            hl[y0:y1] = _highlight_src(r, g, b)

    # ---- pool, blur, scale and tint the highlight field -----------------------
    small = None
    if bloom:
        small = _down(hl)
        hl = None
        small = _blur(small, sigma=2.2)
        peak = float(small.max())
        if peak <= 1e-4:
            bloom = False
            small = None
        else:
            small *= F32(bloom_amount(energy) * 0.62 / peak)
            small = np.ascontiguousarray(small[:, :, None] * bloom_tint()[None, None, :])

    # ---- pass 2: add bloom, add grain, quantise ------------------------------
    gn = _grain(h, w, frame_index) if (grain and grain > 0.0) else None
    if gn is not None and grain != GRAIN_SIGMA:
        gn *= F32(grain / GRAIN_SIGMA)
    out = None if ret_float else np.empty((h, w, 3), np.uint8)
    for y0 in range(0, h, bs):
        y1 = min(y0 + bs, h)
        r, g, b = R[y0:y1], G[y0:y1], B[y0:y1]
        if bloom:
            s0 = y0 // BLOOM_DS
            up = np.repeat(np.repeat(small[s0:s0 + -(-(y1 - y0) // BLOOM_DS)],
                                    BLOOM_DS, axis=0), BLOOM_DS, axis=1)
            up = up[:y1 - y0, :w]
            r += up[:, :, 0]
            g += up[:, :, 1]
            b += up[:, :, 2]
        if gn is not None:
            n = gn[y0:y1]
            r += n
            g += n
            b += n
        if ret_float:
            continue
        r *= F32(255.0)
        r += F32(0.5)
        g *= F32(255.0)
        g += F32(0.5)
        b *= F32(255.0)
        b += F32(0.5)
        np.clip(r, F0, F32(255.0), out=r)
        np.clip(g, F0, F32(255.0), out=g)
        np.clip(b, F0, F32(255.0), out=b)
        o = out[y0:y1]
        o[:, :, 0] = r
        o[:, :, 1] = g
        o[:, :, 2] = b
    if ret_float:
        return np.clip(np.stack([R, G, B], axis=2), 0.0, 1.0)
    return out


def grade_image(img: Image.Image, t: float = 0.0, energy: float = 0.5, **kw) -> Image.Image:
    """PIL convenience wrapper."""
    return Image.fromarray(indigo_look(np.asarray(img.convert("RGB")), t, energy, **kw))


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
# (bg kind, t, energy, frame index, bright-threshold for the hue check)
QA_SHOTS = [
    ("tunnel", 3.0, 0.3, 90, 0.45),
    ("tunnel", 3.0, 0.9, 90, 0.45),
    ("rings", 8.0, 0.9, 240, 0.45),
    ("waves", 12.0, 0.6, 360, 0.45),
    ("solid", 5.0, 0.2, 150, 0.045),   # near-black plate: measure its core glow
    ("particles", 5.0, 0.5, 150, 0.25),
    ("grid", 7.0, 0.7, 210, 0.40),
]
W_, H_ = 1280, 720
MIN_BRIGHT = 200          # below this a hue statistic is not meaningful


def _raw(kind, t, energy, fr, w=W_, h=H_):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import engine as E
    return E.__dict__["bg_" + kind](w, h, t=t, energy=float(energy), frame=int(fr))


def _lum(a):
    return a[:, :, 0] * LUMA_R + a[:, :, 1] * LUMA_G + a[:, :, 2] * LUMA_B


def _hue_stats(rgb01, ythr=0.45, mthr=0.02):
    """Circular mean hue (deg, in the grade's own opponent plane) + coverage."""
    Y = _lum(rgb01)
    x = rgb01[:, :, 0] - Y
    y = rgb01[:, :, 2] - Y
    M = np.hypot(x, y)
    m = (Y > ythr) & (M > mthr)
    n = int(m.sum())
    if n < 20:
        return {"n": n, "hue_deg": float("nan"), "frac_cyan": 0.0, "mean_rgb": None}
    a = np.arctan2(y[m], x[m])
    ang = math.degrees(math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean()))) % 360.0
    cy = hue_targets()["cyan"]["angle"]
    da = np.abs((np.degrees(a) - cy + 180.0) % 360.0 - 180.0)
    sel = rgb01[m]
    return {"n": n, "hue_deg": ang, "frac_cyan": float((da < 10.0).mean()),
            "mean_rgb": [round(float(v * 255), 1) for v in sel.mean(axis=0)]}


def _patch(a01, y0, x0, s=64):
    return [float(v) for v in a01[y0:y0 + s, x0:x0 + s].reshape(-1, 3).mean(axis=0)]


def _synthetics():
    P = 256
    tile = lambda c: np.tile(np.array(c, np.uint8)[None, None, :], (P, P, 1))
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 48, axis=0)[:, :, None]
    return {
        "white": np.full((P, P, 3), 255, np.uint8),
        "black": np.zeros((P, P, 3), np.uint8),
        "grey_ramp": np.repeat(ramp, 3, axis=2),
        "grey128": np.full((P, P, 3), 128, np.uint8),
        "brand_cyan": tile(CYAN),
        "brand_indigo": tile(INDIGO),
    }


def make_qa(out_dir=None):
    """Render the QA stills, before/after pairs and contact sheets."""
    out_dir = out_dir or HERE
    names, shots = [], []
    for kind, t, e, fr, ythr in QA_SHOTS:
        raw = _raw(kind, t, e, fr)
        g = indigo_look(raw, t, e, frame_index=fr)
        tag = "%s_e%02d" % (kind, int(round(e * 100)))
        p = os.path.join(out_dir, "_grade_%s.jpg" % tag)
        Image.fromarray(g).save(p, quality=93)
        names.append(p)
        pb = os.path.join(out_dir, "_grade_%s_ba.jpg" % tag)
        Image.fromarray(np.concatenate([raw, g], axis=1)).resize((1280, 360), Image.LANCZOS).save(pb, quality=92)
        names.append(pb)
        shots.append({"kind": kind, "t": t, "energy": e, "frame": fr,
                      "ythr": ythr, "raw": raw, "graded": g})

    # contact sheet: before | after per shot, labelled
    cw, ch, pad, hdr = 620, 349, 14, 34
    sheet = Image.new("RGB", (2 * cw + pad * 3, len(shots) * (ch + hdr) + pad * (len(shots) + 1)),
                      (10, 10, 16))
    dr = ImageDraw.Draw(sheet)
    for i, s in enumerate(shots):
        y = pad + i * (ch + hdr)
        dr.text((pad, y + 10), "%s  t=%.1fs  energy=%.1f   BEFORE (engine.bg_*)" % (s["kind"], s["t"], s["energy"]),
                fill=(140, 140, 165))
        dr.text((pad + cw + pad, y + 10), "AFTER (indigo_look)", fill=(150, 200, 255))
        sheet.paste(Image.fromarray(s["raw"]).resize((cw, ch), Image.LANCZOS), (pad, y + hdr))
        sheet.paste(Image.fromarray(s["graded"]).resize((cw, ch), Image.LANCZOS), (pad * 2 + cw, y + hdr))
    sp = os.path.join(out_dir, "_grade_sheet.jpg")
    sheet.save(sp, quality=90)
    names.append(sp)

    # synthetics sheet
    syn = _synthetics()
    tiles = [("white", syn["white"]), ("black", syn["black"]), ("grey ramp", syn["grey_ramp"]),
             ("mid grey 128", syn["grey128"]), ("#2BD9FF", syn["brand_cyan"]),
             ("#6C4BFF", syn["brand_indigo"])]
    ss = Image.new("RGB", (len(tiles) * 176 + 16, 216), (10, 10, 16))
    ds = ImageDraw.Draw(ss)
    for i, (k, arr) in enumerate(tiles):
        gg = indigo_look(arr, 0.0, 0.5)
        ds.text((14 + i * 176, 10), k, fill=(200, 200, 220))
        ss.paste(Image.fromarray(gg).resize((164, 164), Image.NEAREST), (12 + i * 176, 34))
    spp = os.path.join(out_dir, "_grade_synth.jpg")
    ss.save(spp, quality=95)
    names.append(spp)
    return names, shots, syn


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def benchmark(n=15):
    """Wall clock for indigo_look on 1280x720, single core."""
    import time as _t
    raw = _raw("tunnel", 3.0, 0.9, 90)
    out = {"scipy": _HAVE_SCIPY, "size": "1280x720", "block_rows": BLOCK_ROWS}
    for e in (0.3, 0.9):
        indigo_look(raw, 3.0, e, frame_index=90)          # warm up
        ts = []
        for i in range(n):
            t0 = _t.perf_counter()
            indigo_look(raw, 3.0, e, frame_index=90 + i)
            ts.append((_t.perf_counter() - t0) * 1e3)
        ts.sort()
        out["e%.1f" % e] = {"median": round(ts[len(ts) // 2], 2),
                            "min": round(ts[0], 2), "max": round(ts[-1], 2)}
    ts = []
    for i in range(n):
        t0 = _t.perf_counter()
        indigo_look(raw, 3.0, 0.9, frame_index=90 + i, bloom=False)
        ts.append((_t.perf_counter() - t0) * 1e3)
    ts.sort()
    out["no_bloom"] = {"median": round(ts[len(ts) // 2], 2)}
    out["budget_ms"] = 40.0
    out["pass"] = out["e0.9"]["median"] < 40.0 and out["e0.3"]["median"] < 40.0
    return out


def verify(shots=None, synth=None, bench=True):
    """Run every check, return a report dict."""
    if shots is None or synth is None:
        _, shots, synth = make_qa()
    tg = hue_targets()
    ind_ang, cy_ang = tg["indigo"]["angle"], tg["cyan"]["angle"]
    rep = {"hue_targets": tg, "checks": {}, "shots": {}}
    G8 = lambda a, **k: (indigo_look(a, **k).astype(np.float32) / 255.0)

    # (a) white stays white (+-4) -- grain ON, averaged over a 64x64 patch
    gw = G8(synth["white"], t=0.0, energy=0.5)
    wp = _patch(gw, 64, 64, 64)
    rep["checks"]["a_white"] = {
        "patch_mean_255": [round(v * 255, 2) for v in wp],
        "delta": [round(v * 255 - 255, 2) for v in wp],
        "min_255": round(float(gw.min() * 255), 2),
        "pass": all(abs(v * 255 - 255) <= 4 for v in wp),
    }

    # (b) black lands on BASE #07070C (+-6)
    gb = G8(synth["black"], t=0.0, energy=0.5)
    bp = _patch(gb, 32, 32, 64)
    rep["checks"]["b_black"] = {
        "patch_mean_255": [round(v * 255, 2) for v in bp],
        "target": list(BASE),
        "delta": [round(v * 255 - t, 2) for v, t in zip(bp, BASE)],
        "pass": all(abs(v * 255 - t) <= 6 for v, t in zip(bp, BASE)),
    }

    # (d) mid-greys: the grade must not rotate neutral input at all.  A neutral
    #     pixel can only follow the analytic lift response
    #     out_c = base_c + curve(v)*(1 - base_c), so any hue rotation (a lavender
    #     cast) shows up as a deviation from it.  Grain/bloom are measured apart.
    gg = G8(synth["grey_ramp"], t=0.0, energy=0.5, grain=0.0, bloom=False)
    band = gg[:, 40:216, :]                                  # input levels 40..215
    lv = np.arange(40, 216, dtype=np.float32) / 255.0
    yl = np.interp(lv, np.linspace(0, 1, 1024), _curve_lut(0.5))
    want = BASE_F[None, :] + yl[:, None] * (1.0 - BASE_F[None, :])
    dev = np.abs(band - want[None, :, :]).max(axis=0)        # per level, 3 channels
    lvl = band.mean(axis=(0, 2))
    steps = np.diff(lvl)
    rg = float(np.abs(band[:, :, 0] - band[:, :, 1]).max() * 255)   # R vs G: 0 = no rotation
    blu = (band[:, :, 2] - np.maximum(band[:, :, 0], band[:, :, 1])).mean(axis=0)
    rep["checks"]["d_grey"] = {
        "levels_in": [40, 128, 215],
        "levels_out": [round(float(v * 255), 1) for v in (lvl[0], lvl[88], lvl[-1])],
        "max_dev_from_neutral_response_255": round(float(dev.max() * 255), 4),
        "R_minus_G_max_255": round(rg, 4),
        "blue_bias_at_black_40_128_215": [round(float(blu[i] * 255), 2) for i in (0, 88, 175)],
        "monotonic": bool((steps >= 0).all()),
        "min_step_255": round(float(steps.min() * 255), 4),
        "pass": bool(float(dev.max() * 255) <= 0.5 and rg <= 0.5 and bool((steps >= 0).all())),
    }

    # brand anchors survive the pull
    for key, arr, src in (("cyan", synth["brand_cyan"], CYAN), ("indigo", synth["brand_indigo"], INDIGO)):
        ga = G8(arr, t=0.0, energy=0.5, grain=0.0)
        st = _hue_stats(ga, ythr=0.05, mthr=0.02)
        want = cy_ang if key == "cyan" else ind_ang
        rep["checks"]["anchor_" + key] = {
            "src": list(src), "out_255": [round(v * 255, 1) for v in _patch(ga, 64, 64, 64)],
            "hue_deg": round(st["hue_deg"], 2),
            "hue_shift_deg": round(abs((st["hue_deg"] - want + 180) % 360 - 180), 2),
        }

    # (c) bright dot field mean hue within +-8 deg of #6C4BFF
    for s in shots:
        gi = s["graded"].astype(np.float32) / 255.0
        ri = s["raw"].astype(np.float32) / 255.0
        st = _hue_stats(gi, s["ythr"], 0.02)
        rw = _hue_stats(ri, s["ythr"], 0.02)
        key = "%s_e%02d" % (s["kind"], int(round(s["energy"] * 100)))
        if st["n"] < MIN_BRIGHT:
            rep["shots"][key] = {"n_bright": st["n"], "raw_hue": rw["hue_deg"],
                                 "graded_hue": st["hue_deg"], "delta_vs_indigo": None,
                                 "pass_c": None, "skipped": "too few bright px",
                                 "frac_near_cyan": st["frac_cyan"], "mean_bright_rgb": st["mean_rgb"]}
            continue
        dh = abs((st["hue_deg"] - ind_ang + 180.0) % 360.0 - 180.0)
        rep["shots"][key] = {"n_bright": st["n"], "raw_hue": round(rw["hue_deg"], 2),
                             "graded_hue": round(st["hue_deg"], 2),
                             "delta_vs_indigo": round(dh, 2), "pass_c": bool(dh <= 8.0),
                             "frac_near_cyan": round(st["frac_cyan"], 4),
                             "mean_bright_rgb": st["mean_rgb"]}
    graded = [v for v in rep["shots"].values() if v["pass_c"] is not None]
    rep["checks"]["c_bright_hue"] = {
        "indigo_angle_deg": round(ind_ang, 2), "cyan_angle_deg": round(cy_ang, 2),
        "tol_deg": 8.0, "n_tested": len(graded),
        "worst": round(max(v["delta_vs_indigo"] for v in graded), 2) if graded else None,
        "pass": bool(graded) and all(v["pass_c"] for v in graded),
    }
    rep["pass"] = (all(rep["checks"][k]["pass"] for k in ("a_white", "b_black", "d_grey"))
                   and rep["checks"]["c_bright_hue"]["pass"])
    if bench:
        rep["timing_ms"] = benchmark()
    return rep


def _print_report(rep):
    tg, c = rep["hue_targets"], rep["checks"]
    print("\nLuminous Indigo grade -- verification" + ("" if rep["pass"] else "   [FAIL]"))
    print("  anchors: indigo #%02X%02X%02X @ %.2f deg | cyan #%02X%02X%02X @ %.2f deg | gap %.1f deg"
          % (*tg["indigo"]["rgb"], tg["indigo"]["angle"], *tg["cyan"]["rgb"],
             tg["cyan"]["angle"], tg["gap_deg"]))
    print("  tuning: cyan claim sigma %.1f deg, purity^%d, chroma knee %.3f, pull %.2f"
          % (tg["cyan_sigma_deg"], tg["purity_power"], tg["chroma_knee"], tg["pull"]))

    a = c["a_white"]
    print("\n(a) white stays white, tol +-4/255")
    print("    255,255,255 -> %s  (delta %s)   PASS=%s" % (a["patch_mean_255"], a["delta"], a["pass"]))
    b = c["b_black"]
    print("(b) black lands on BASE #07070C, tol +-6/255")
    print("    0,0,0 -> %s  (delta vs %s)   PASS=%s" % (b["patch_mean_255"], b["target"], b["pass"]))
    d = c["d_grey"]
    print("(d) mid-greys stay neutral (no lavender cast)")
    print("    grey ramp in %s -> out %s   monotonic=%s (min step %.3f/255)" % (
        d["levels_in"], d["levels_out"], d["monotonic"], d["min_step_255"]))
    print("    max deviation from the analytic neutral lift: %.4f/255" % d["max_dev_from_neutral_response_255"])
    print("    max |R-G| over the ramp: %.4f/255 (0 = no hue rotation)" % d["R_minus_G_max_255"])
    print("    residual blue bias (brand base tint) at 40/128/215: %s/255" % d["blue_bias_at_black_40_128_215"])
    print("    PASS=%s" % d["pass"])
    print("    brand anchors: #2BD9FF -> %s (hue %.2f, shift %.2f deg) | #6C4BFF -> %s (hue %.2f, shift %.2f deg)" % (
        c["anchor_cyan"]["out_255"], c["anchor_cyan"]["hue_deg"], c["anchor_cyan"]["hue_shift_deg"],
        c["anchor_indigo"]["out_255"], c["anchor_indigo"]["hue_deg"], c["anchor_indigo"]["hue_shift_deg"]))
    h = c["c_bright_hue"]
    print("(c) bright dot field mean hue vs #6C4BFF (%.2f deg), tol +-%.0f deg" % (h["indigo_angle_deg"], h["tol_deg"]))
    for k, v in rep["shots"].items():
        if v["pass_c"] is None:
            print("    %-14s n=%6d  SKIPPED (%s)" % (k, v["n_bright"], v["skipped"]))
        else:
            print("    %-14s n=%6d  raw %6.2f -> graded %6.2f  delta %5.2f  PASS=%s  cyan-share %.3f  rgb %s" % (
                k, v["n_bright"], v["raw_hue"], v["graded_hue"], v["delta_vs_indigo"],
                v["pass_c"], v["frac_near_cyan"], v["mean_bright_rgb"]))
    print("    worst delta %.2f deg over %d tested shots   PASS=%s" % (h["worst"], h["n_tested"], h["pass"]))
    t = rep.get("timing_ms")
    if t:
        print("\ntiming -- %s, single core, scipy=%s, block_rows=%d" % (t["size"], t["scipy"], t["block_rows"]))
        for k in ("e0.3", "e0.9"):
            print("    energy %s : median %.2f ms  (min %.2f / max %.2f)" % (
                k[-3:], t[k]["median"], t[k]["min"], t[k]["max"]))
        print("    no bloom : median %.2f ms" % t["no_bloom"]["median"])
        print("    budget %.0f ms  PASS=%s" % (t["budget_ms"], t["pass"]))
    print("\noverall: %s\n" % ("PASS" if rep["pass"] else "FAIL"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def split_outdir(argv):
    """argparse cannot do `inputs... outdir`; peel a trailing directory off."""
    a = list(argv)
    outdir = None
    if len(a) > 1 and (a[-1].endswith(("/", os.sep)) or os.path.isdir(a[-1])):
        outdir = a.pop()
    return a, outdir


def grade_stills(inputs, outdir, energy=0.5, t=0.0, bloom=None, grain=GRAIN_SIGMA,
                 report=True):
    """Grade stills on disk. Returns (results, report) where results is a list
    of dicts with the input path, the output path and the graded array."""
    os.makedirs(outdir, exist_ok=True)
    tg = hue_targets()
    ind_ang = tg["indigo"]["angle"]
    results = []
    for p in inputs:
        im = Image.open(p).convert("RGB")
        arr = indigo_look(np.asarray(im), t, energy, bloom=bloom, grain=grain)
        o = os.path.join(outdir, os.path.splitext(os.path.basename(p))[0] + "_graded.jpg")
        Image.fromarray(arr).save(o, quality=93)
        item = {"in": p, "out": o, "frame": arr}
        if report:
            a = arr.astype(np.float32) / 255.0
            Y = _lum(a)
            dark = a[Y <= max(np.percentile(Y, 0.2), 1e-4)]
            bright = Y > max(0.45, float(np.percentile(Y, 99.5)))
            st = _hue_stats(a, ythr=float(np.percentile(Y, 99.5)) * 0.45 + 1e-9, mthr=0.02)
            item.update({
                "mean_rgb": [round(float(v * 255), 1) for v in a.reshape(-1, 3).mean(axis=0)],
                "black_pct": [round(float(v * 255), 1) for v in dark.mean(axis=0)],
                "black_vs_base": [round(float(v * 255 - b), 1) for v, b in zip(dark.mean(axis=0), BASE)],
                "p50_luma": round(float(np.percentile(Y, 50)), 4),
                "p999_luma": round(float(np.percentile(Y, 99.9)), 4),
                "bright_hue": round(st["hue_deg"], 2),
                "bright_hue_delta": round(abs((st["hue_deg"] - ind_ang + 180.0) % 360.0 - 180.0), 2),
                "bright_frac": round(float(bright.mean()), 3),
            })
        print("  %s -> %s" % (p, o))
        results.append(item)
    return results


def _print_still_report(results, energy):
    print("\ngraded stills @ energy %.2f" % energy)
    for r in results:
        if "mean_rgb" not in r:
            continue
        print("  %s" % os.path.basename(r["in"]))
        print("     mean %s   p50 luma %.3f  p99.9 luma %.3f  bright px %.1f%%" % (
            r["mean_rgb"], r["p50_luma"], r["p999_luma"], 100 * r["bright_frac"]))
        print("     black point %s (vs #%02X%02X%02X: %s)   bright-field hue %.2f deg (%.2f from #6C4BFF)" % (
            r["black_pct"], BASE[0], BASE[1], BASE[2], r["black_vs_base"],
            r["bright_hue"], r["bright_hue_delta"]))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args_list, outdir_arg = split_outdir(argv)
    ap = argparse.ArgumentParser(description="Luminous Indigo per-frame colour grade")
    ap.add_argument("inputs", nargs="*", help="input stills to grade")
    ap.add_argument("--qa", action="store_true", help="write QA stills + contact sheets")
    ap.add_argument("--verify", action="store_true", help="run the full numeric checks")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--energy", type=float, default=0.5)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--bloom", dest="bloom", action="store_true", default=None)
    ap.add_argument("--no-bloom", dest="bloom", action="store_false")
    ap.add_argument("--grain", type=float, default=GRAIN_SIGMA)
    ap.add_argument("--out", default=None, help="output / QA directory")
    args = ap.parse_args(args_list)

    shots = synth = None
    results = None
    ok = True
    if args.qa or (not args.inputs and not args.verify):
        names, shots, synth = make_qa(args.out)
        print("wrote:")
        for n in names:
            print("   ", n)
    if args.inputs:
        outdir = outdir_arg or args.out or os.path.join(HERE, "graded")
        results = grade_stills(args.inputs, outdir, args.energy, args.t,
                               args.bloom, args.grain, report=not args.no_verify)
    if not args.no_verify and shots is not None:
        rep = verify(shots, synth)
        _print_report(rep)
        with open(os.path.join(args.out or HERE, "_grade_report.json"), "w") as fh:
            json.dump(rep, fh, indent=2, default=float)
        ok = rep["pass"] and rep["timing_ms"]["pass"]
    elif results:
        _print_still_report(results, args.energy)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())