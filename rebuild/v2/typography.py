"""Refined text-colour / typography system for the v2 promo (self-contained).

Everything here draws *in place* on a PIL image or a uint8 ndarray and is
**beat-agnostic**: entrances are functions of a local-time float ``t``
(seconds since the element's own entrance), never of beat indices. The only
dependency is ``engine`` (fonts, springs, stagger, scrim helpers) — nothing in
``render.py`` / ``engine.py`` is modified.

Design goals vs. v1 (which was flat white + one indigo glow, washed lavender
subs colliding with bright dot fields):

1. **Gradient bodies, not flat fills.** Headline glyphs are painted on a temp
   RGBA layer, then the RGB is ramped top->bottom through numpy and re-masked
   by the glyph alpha, so the gradient is clipped to the ink exactly.
2. **Crisp rim light.** A 1-2px warm-white band derived from
   ``alpha - shift(alpha, k)`` (i.e. the top edge of every stroke) is
   composited *over* the gradient body. The body top is deliberately held
   *below* pure white so the rim reads as light rather than as nothing.
3. **Glow that never washes the edge.** Two-stage glow (wide indigo haze +
   tight cyan bleed) is drawn from the glyph alpha and composited *under* the
   core, so the anti-aliased glyph edge stays the sharpest thing on screen.
4. **Dark separation ring.** ``blur(alpha) - alpha`` tinted to BASE at ~0.3
   sits between glow and core. This is what keeps text readable over bright
   dot fields even with no scrim.
5. **Real hierarchy.** Inter Tight ~700 for words, Space Grotesk for numerals,
   IBM Plex Mono (SemiBold) for eyebrows. Tracking is size-aware: -2px at
   display sizes, +3px on subs, +8px on eyebrows.
6. **Adaptive contrast.** ``plate="auto"`` samples the luminance under the text
   box and picks the scrim strength needed to keep the fill legible.

Public API
----------
``headline`` ``sub_line`` ``stat_number`` ``eyebrow`` ``headline_block``
``plate`` / ``plate_all`` ``PALETTE``, the geometry helpers
(``headline_box`` ``sub_box`` ``stat_box`` ``eyebrow_box``
``headline_block_boxes``), and low-level pieces (``gradient_core``,
``rim_light``, ``two_stage_glow``, ``separation_ring``, ``measure``).

Ordering note: a scrim darkens whatever is already on the canvas. For a
multi-element block call ``plate_all(img, [*box_helpers], auto=True)`` *before*
drawing the text — a per-element ``plate=`` argument draws the scrim after
earlier elements and will dim them.

Run ``python3 typography.py`` to render the QA stills ``_typo_*.jpg``.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import engine as E  # noqa: E402  (local module, path set above)

Color = Tuple[int, int, int]
BBox = Tuple[int, int, int, int]

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------

INDIGO: Color = (108, 75, 255)      # #6C4BFF  primary
CYAN: Color = (43, 217, 255)        # #2BD9FF  secondary
BASE: Color = (7, 7, 12)            # #07070C  canvas / separation ring

# Headline body ramp. Deliberately short of pure white at the top so the rim
# highlight has somewhere to go, and indigo-tinted at the bottom so the glyph
# feels lit from above and seated in the palette.
HEAD_TOP: Color = (228, 224, 255)
HEAD_BOT: Color = (163, 152, 247)
RIM: Color = (255, 250, 242)        # warm near-white rim light

# Accent ramp (cyan numerals / accent words)
ACC_TOP: Color = (214, 247, 255)
ACC_BOT: Color = (43, 158, 235)
ACCENT: Color = CYAN
ACCENT_WARM: Color = (255, 240, 214)

# Secondary type
SUB_FILL: Color = (182, 174, 240)   # #B6AEF0 muted lavender
SUB_DIM: Color = (146, 139, 196)    # for tertiary/meta lines
EYEBROW_FILL: Color = CYAN

PALETTE: Dict[str, Color] = {
    "indigo": INDIGO, "cyan": CYAN, "base": BASE,
    "head_top": HEAD_TOP, "head_bot": HEAD_BOT, "rim": RIM,
    "acc_top": ACC_TOP, "acc_bot": ACC_BOT, "accent": ACCENT,
    "accent_warm": ACCENT_WARM, "sub": SUB_FILL, "sub_dim": SUB_DIM,
    "eyebrow": EYEBROW_FILL,
}

# Entrance defaults (match the v1 rhythm, exposed so they can be tuned).
DEF_OFFSET_Y = 60.0
DEF_DAMPING = 14.0
DEF_STIFFNESS = 170.0
DEF_MASS = 1.0
DEF_STAGGER_WINDOW = 0.55
DEF_STAGGER_DUR = 0.45
SUB_DELAY = 0.85                   # sub lines breathe after the headline


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def clamp01(x) -> float:
    return max(0.0, min(1.0, float(x)))


def _writeback(img, base: Image.Image, backing) -> None:
    """Mirror engine.draw_text's in-place semantics for PIL / ndarray targets."""
    if backing is not None:
        backing[:, :, :3] = np.asarray(base.convert("RGB"))
        return
    try:
        if getattr(img, "mode", None) in ("RGB", "RGBA"):
            img.paste(base.convert(img.mode))
    except Exception:
        pass


def _composite(img, layer: Image.Image, xy=None) -> Image.Image:
    """Alpha-composite ``layer`` (at integer ``xy`` or full-size) onto ``img``."""
    base, backing = E._as_rgba(img)
    if xy is None:
        base = Image.alpha_composite(base, layer)
    else:
        base.alpha_composite(layer, (int(round(xy[0])), int(round(xy[1]))))
    _writeback(img, base, backing)
    return base


# ----------------------------------------------------------------------------
# Fonts — engine.get_font under the hood, with real variable-weight axes
# ----------------------------------------------------------------------------

_VFONT: Dict[tuple, object] = {}


def _font(alias: str, size: int, weight: Optional[float] = None):
    """Resolve ``alias`` at ``size`` with an optional variable-font weight.

    ``E.get_font`` is used for path resolution + offline fallback, but its
    result is *shared and cached* by engine — mutating it would leak the
    weight into every other caller (including render.py). So when a weight is
    requested we re-open the same file as our own instance and set the axis.
    """
    size = int(size)
    key = (str(alias), size, weight)
    if key in _VFONT:
        return _VFONT[key]
    base = E.get_font(alias, size)
    font = base
    if weight is not None and hasattr(base, "set_variation_by_axes") and getattr(base, "path", None):
        try:
            axes = base.get_variation_axes()
            vals = []
            for ax in axes:
                nm = ax.get("name", b"")
                nm = nm.decode() if isinstance(nm, bytes) else str(nm)
                if nm.lower().startswith("weight"):
                    v = float(np.clip(weight, ax["minimum"], ax["maximum"]))
                else:
                    v = float(ax["default"])
                vals.append(v)
            font = ImageFont.truetype(base.path, size, layout_engine=base.layout_engine)
            font.set_variation_by_axes(vals)
        except Exception:
            font = base
    _VFONT[key] = font
    return font


def _mono_font(size: float):
    """IBM Plex Mono SemiBold when the cached file exists, else Regular.

    SemiBold is a *separate* file in the plexmono family (not an axis on the
    variable font), so this degrades cleanly when it was never downloaded.
    """
    try:
        return _font("mono-bold", int(size), None)
    except Exception:
        return _font("mono", int(size), None)


def auto_tracking(size: float, kind: str = "display") -> int:
    """Size-aware tracking in px (negative = tighter)."""
    size = float(size)
    if kind == "display":
        if size >= 95:
            return -2
        if size >= 60:
            return -1
        return 0
    if kind == "numeral":
        return -2 if size >= 90 else -1
    if kind == "sub":
        return 3
    if kind == "eyebrow":
        return 8
    return 0


# ----------------------------------------------------------------------------
# Measurement — mirrors engine.draw_text's per-char layout exactly
# ----------------------------------------------------------------------------

def _advances(text: str, font, spacing: float) -> np.ndarray:
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    out = []
    for ch in text:
        try:
            out.append(float(probe.textlength(ch, font=font)))
        except Exception:
            out.append(float(font.size) * 0.6)
    return np.asarray(out, dtype=np.float64)


def _line_box(text: str, font, spacing: float, factors=None):
    """(advances, total_width, ascent, descent) matching engine's char path.

    ``factors`` optionally scales individual advances (used to tighten the
    space a comma claims in a display numeral).
    """
    adv = _advances(text, font, spacing)
    if factors is not None:
        adv = adv * np.clip(np.asarray(factors, dtype=np.float64).ravel(), 0.05, 4.0)
    total = float(adv.sum()) + float(spacing) * max(len(text) - 1, 0)
    try:
        asc, desc = font.getmetrics()
    except Exception:
        asc, desc = int(font.size * 0.8), int(font.size * 0.2)
    return adv, total, float(asc), float(desc)


def measure(text: str, font, spacing: float = 0.0, factors=None) -> BBox:
    """Tight ink bbox relative to engine's layout box origin (top-left)."""
    if not text:
        return (0, 0, 0, 0)
    adv, total, asc, desc = _line_box(text, font, spacing, factors)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    bx0 = by0 = 10 ** 9
    bx1 = by1 = -(10 ** 9)
    cx = 0.0
    for i, ch in enumerate(text):
        if ch.strip():
            b = probe.textbbox((cx, asc), ch, font=font, anchor="ls")
            bx0 = min(bx0, b[0]); by0 = min(by0, b[1])
            bx1 = max(bx1, b[2]); by1 = max(by1, b[3])
        cx += adv[i] + spacing
    if bx1 < bx0:
        return (0, 0, int(math.ceil(total)), asc)
    return (int(math.floor(bx0)), int(math.floor(by0)),
            int(math.ceil(bx1)), int(math.ceil(by1)))


def _anchor_offsets(anchor: str, total_w: float, asc: float, desc: float):
    ax = {"l": 0.0, "m": 0.5, "r": 1.0}.get(anchor[0].lower(), 0.5)
    vkey = anchor[1].lower() if len(anchor) > 1 else "m"
    ay = {"a": 0.0, "t": 0.0, "m": 0.5, "s": 1.0, "b": 1.0, "d": 1.0}.get(vkey, 0.5)
    total_h = asc + desc
    return ax, ay, total_h


# ----------------------------------------------------------------------------
# Glyph raster -> alpha mask on a padded local canvas
# ----------------------------------------------------------------------------

def _paint_mask(size, text, font, spacing, char_alphas, alpha_scale=1.0,
                dx=0.0, dy=0.0, origin=(0.0, 0.0), factors=None):
    """Draw per-char alphas into an "L" canvas; returns the alpha array."""
    adv, total, asc, desc = _line_box(text, font, spacing, factors)
    w, h = size
    canvas = Image.new("L", size, 0)
    d = ImageDraw.Draw(canvas)
    n = len(text)
    cas = (np.ones(n) if char_alphas is None
           else np.clip(np.asarray(char_alphas, dtype=np.float64).ravel(), 0, 1))
    cx = origin[0] + dx
    by = origin[1] + dy + asc
    for i, ch in enumerate(text):
        a = float(alpha_scale) * float(cas[i]) if i < n else 0.0
        if a > 0.003 and ch.strip():
            d.text((cx, by), ch, font=font,
                   fill=int(255 * clamp01(a)), anchor="ls")
        cx += adv[i] + spacing
    return np.asarray(canvas, dtype=np.uint8)


def _mask_extent(alpha: np.ndarray):
    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def gradient_core(alpha: np.ndarray, top: Color, bottom: Color,
                  span: Optional[Tuple[float, float]] = None,
                  text_h: Optional[float] = None) -> Image.Image:
    """RGBA layer: glyph ``alpha`` masked with a vertical colour ramp.

    ``span`` is (y0, y1) in canvas px. Default: the tight ink extent, so the
    ramp always runs exactly from the top of the tallest glyph to the bottom of
    the deepest. Pass ``span=(0, text_h)`` (``align="font"``) to make every
    line of a multi-line block share one ramp.
    """
    h, w = alpha.shape
    if span is None:
        ext = _mask_extent(alpha)
        if ext is None:
            return Image.new("RGBA", (w, h), (0, 0, 0, 0))
        y0, y1 = float(ext[1]), float(ext[3])
    else:
        y0, y1 = float(span[0]), float(span[1])
    if y1 - y0 < 1e-6:
        y1 = y0 + 1.0
    t = np.clip((np.arange(h, dtype=np.float32) - y0) / (y1 - y0), 0.0, 1.0)
    c0 = np.asarray(top, dtype=np.float32)
    c1 = np.asarray(bottom, dtype=np.float32)
    col = c0[None, :] + (c1 - c0)[None, :] * t[:, None]        # (h, 3)
    # Broadcast the (h, 3) ramp straight into the RGBA buffer rather than
    # materialising an (h, w, 3) float copy first.
    arr = np.empty((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = col[:, 0, None]
    arr[:, :, 1] = col[:, 1, None]
    arr[:, :, 2] = col[:, 2, None]
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def rim_light(alpha: np.ndarray, color: Color = RIM, strength: float = 0.9,
              thickness: float = 2.0) -> np.ndarray:
    """Thin crisp highlight along the *top* edge of every stroke.

    ``alpha - shift_down(alpha, k)`` isolates exactly the band of pixels where
    a glyph starts and nothing is below it, i.e. the top rim of every stem and
    curve. Returns a uint8 alpha band ready for compositing.
    """
    k = max(1, int(round(thickness)))
    lagged = np.zeros_like(alpha)
    lagged[k:, :] = alpha[:-k, :]
    a = alpha.astype(np.float32) / 255.0
    l = lagged.astype(np.float32) / 255.0
    band = np.clip(a - l, 0.0, 1.0)
    # Bias toward solid glyph interior so AA fringes don't get a bright halo.
    band *= np.clip(a * 1.35, 0.0, 1.0)
    band *= float(np.clip(strength, 0.0, 1.0))
    return np.clip(band * 255.0, 0, 255).astype(np.uint8)


def _tinted(alpha: np.ndarray, color: Color) -> Image.Image:
    """Solid-colour RGBA layer from an alpha array (no dstack copy)."""
    h, w = alpha.shape
    arr = np.empty((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def separation_ring(alpha: np.ndarray, color: Color = BASE,
                    strength: float = 0.32, radius: float = 3.0
                    ) -> Optional[Image.Image]:
    """Soft dark halo hugging the glyph exterior — the anti-bright-dot trick."""
    if strength <= 0.002:
        return None
    img = Image.fromarray(alpha, "L")
    r = max(1.0, float(radius))
    halo = np.asarray(img.filter(ImageFilter.GaussianBlur(r * 1.6)), dtype=np.int16)
    ring = np.clip(halo - alpha.astype(np.int16), 0, 255).astype(np.uint8)
    ring = np.asarray(Image.fromarray(ring, "L")
                      .filter(ImageFilter.GaussianBlur(r * 0.45)), dtype=np.uint8)
    if int(ring.max()) < 4:
        return None
    peak = max(1, int(ring.max()))
    ring = Image.fromarray(ring, "L").point(
        [min(255, int(v * strength * 255.0 / peak)) for v in range(256)])
    return _tinted(np.asarray(ring, dtype=np.uint8), color)


def two_stage_glow(alpha: np.ndarray, size: float, color: Color = INDIGO,
                   wide_alpha: float = 0.24, tight_alpha: float = 0.30,
                   accent: Color = CYAN) -> Optional[Image.Image]:
    """Wide colour haze + tight bleed, both *under* the core so edges stay crisp.

    Built from the glyph alpha rather than from a filled text layer, which is
    why it never brightens the glyph body itself.
    """
    img = Image.fromarray(alpha, "L")
    layers = [(max(2.0, size * 0.17), wide_alpha, color),
              (max(1.0, size * 0.05), tight_alpha, accent)]
    acc: Optional[Image.Image] = None
    for radius, a, col in layers:
        if a <= 0.002:
            continue
        b = _blur_soft(img, radius)
        if a != 1.0:
            b = b.point([min(255, int(v * float(a))) for v in range(256)])
        layer = _tinted(np.asarray(b, dtype=np.uint8), col)
        acc = layer if acc is None else Image.alpha_composite(acc, layer)
    return acc


def _blur_soft(img: Image.Image, radius: float) -> Image.Image:
    """Gaussian blur, evaluated at reduced resolution for large radii.

    A soft glow carries no detail above a few pixels, so blurring a downsampled
    copy and scaling back up is visually identical but ~k^2 cheaper. Small radii
    (where resampling *would* be visible) are done at full resolution.
    """
    r = float(radius)
    k = int(min(6, max(1, r // 6.0)))
    if k <= 1:
        return img.filter(ImageFilter.GaussianBlur(r))
    small = img.resize((max(1, img.width // k), max(1, img.height // k)),
                       Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(r / k))
    return small.resize(img.size, Image.BILINEAR)


# ----------------------------------------------------------------------------
# Entrance
# ----------------------------------------------------------------------------

def entrance(t: float, delay: float = 0.0, offset_y: float = DEF_OFFSET_Y,
             damping: float = DEF_DAMPING, stiffness: float = DEF_STIFFNESS,
             mass: float = DEF_MASS) -> Tuple[float, float]:
    """Beat-agnostic entrance. Returns ``(alpha, dy)`` for local time ``t``."""
    lt = float(t) - float(delay)
    if lt <= 0:
        return 0.0, float(offset_y)
    s = float(E.spring(lt, damping=damping, stiffness=stiffness, mass=mass))
    a = clamp01(s)
    return a, float(offset_y) * (1.0 - a)


def _char_alphas(text: str, t: float, delay: float, window: float,
                 dur: float) -> np.ndarray:
    lt = float(t) - float(delay)
    if lt <= 0:
        return np.zeros(len(text))
    return E.stagger_alphas(len(text), clamp01(lt / max(dur, 1e-6)), window=window)


def _wipe_alphas(text: str, font, spacing: float, progress: float,
                 feather_px: float, factors=None) -> np.ndarray:
    """Soft gradient wipe across the string: per-char alpha from a soft mask."""
    adv, total, asc, desc = _line_box(text, font, spacing, factors)
    n = len(text)
    if n == 0:
        return np.zeros(0)
    lefts = np.zeros(n)
    cx = 0.0
    for i in range(n):
        lefts[i] = cx
        cx += adv[i] + spacing
    fe = max(4.0, float(feather_px))
    p = clamp01(progress)
    front = p * (total + fe)
    return np.clip((front - lefts) / fe, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Plate — soft dark rounded scrim for busy backgrounds
# ----------------------------------------------------------------------------

def _as_plate_kwargs(plate) -> Dict[str, object]:
    if plate is None or plate is False:
        return {"enabled": False}
    if plate is True:
        return {"enabled": True, "strength": 0.62}
    if isinstance(plate, str):
        if plate.lower() in ("auto", "adaptive"):
            return {"enabled": True, "auto": True}
        return {"enabled": True, "strength": 0.62}
    if isinstance(plate, (int, float)):
        return {"enabled": True, "strength": float(plate)}
    if isinstance(plate, dict):
        d = dict(plate)
        d["enabled"] = True
        return d
    if isinstance(plate, (tuple, list)):
        return {"enabled": True, "strength": float(plate[0])}
    return {"enabled": True, "strength": 0.62}


def _region_luma(img, bbox: BBox) -> Tuple[float, float]:
    """(mean, p95) luminance of the region — drives ``plate="auto"``."""
    base, _ = E._as_rgba(img)
    W, H = base.size
    x0 = int(max(0, min(W, bbox[0]))); y0 = int(max(0, min(H, bbox[1])))
    x1 = int(max(0, min(W, bbox[2]))); y1 = int(max(0, min(H, bbox[3])))
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    arr = np.asarray(base.convert("RGB"), dtype=np.float32)[y0:y1, x0:x1]
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return float(lum.mean()), float(np.percentile(lum, 95))


def plate(img, bbox: BBox, strength: Optional[float] = None, auto: bool = False,
          pad: Tuple[int, int] = (34, 26), radius: int = 26,
          feather: Optional[float] = None, color: Color = (4, 4, 9),
          target_p95: float = 96.0, floor: float = 0.22,
          ceil: float = 0.80, shape: str = "rrect") -> Dict[str, float]:
    """Soft dark scrim behind a text box. Accepts one bbox or a list of boxes.

    ``auto=True`` measures the background luminance under the region and scales
    the scrim so the 95th-percentile background brightness lands under
    ``target_p95`` — that is what kills the "sub vanishes into bright dots"
    problem without darkening already-clean scenes.

    ``feather`` defaults to a fraction of the region size (so a tall 3-line
    block gets a proportionally softer shoulder than a small eyebrow), and is
    applied twice to avoid the flat plateau / visible edge that a single
    normalized blur produces.

    NOTE: a scrim darkens whatever is already on the canvas. For multi-element
    blocks draw the scrim *first* (see ``plate_all`` + the ``*_box`` helpers)
    rather than relying on a per-element ``plate=`` argument, which would
    darken the element drawn before it.
    """
    boxes = [tuple(b) for b in (bbox if isinstance(bbox, (list, tuple)) and
                                bbox and isinstance(bbox[0], (list, tuple))
                                else [bbox])]
    boxes = [b for b in boxes if (b[2] - b[0]) >= 1 and (b[3] - b[1]) >= 1]
    if not boxes:
        return {"strength": 0.0, "p95": 0.0, "mean": 0.0, "box": None}
    bx0 = min(b[0] for b in boxes); by0 = min(b[1] for b in boxes)
    bx1 = max(b[2] for b in boxes); by1 = max(b[3] for b in boxes)
    base, backing = E._as_rgba(img)
    W, H = base.size
    px, py = int(pad[0]), int(pad[1])
    x0 = max(0, int(bx0) - px); y0 = max(0, int(by0) - py)
    x1 = min(W, int(bx1) + px); y1 = min(H, int(by1) + py)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return {"strength": 0.0, "p95": 0.0, "mean": 0.0, "box": None}
    mean_l, p95 = _region_luma(base, (x0, y0, x1, y1))
    s = float(strength) if strength is not None else 0.62
    if auto and strength is None:
        need = clamp01((p95 - target_p95) / max(255.0 - target_p95, 1e-6))
        s = float(np.clip(floor + (ceil - floor) * need, floor, ceil))
    s = float(np.clip(s, 0.0, 0.88))
    fsz = min(float(x1 - x0), float(y1 - y0))
    # Feather scales with the SHORT side: a wide, short scrim needs a much
    # larger feather than a square one or its straight edges stay visible
    # against a high-contrast dot field.
    f = float(feather) if feather is not None else float(np.clip(fsz * 0.50, 30.0, 130.0))
    m = Image.new("L", (x1 - x0, y1 - y0), 0)
    d = ImageDraw.Draw(m)
    if shape == "ellipse":
        d.ellipse([0, 0, (x1 - x0) - 1, (y1 - y0) - 1], fill=255)
    else:
        d.rounded_rectangle([0, 0, (x1 - x0) - 1, (y1 - y0) - 1],
                            radius=int(min(radius, max(1, fsz // 2))), fill=255)
    if f > 0.5:
        m = m.filter(ImageFilter.GaussianBlur(f))
        peak = int(np.asarray(m).max())
        if peak > 8:
            m = m.point(lambda v, pk=peak: min(255, int(v * 255.0 / pk)))
        m = m.filter(ImageFilter.GaussianBlur(f * 0.55))
    # Radial falloff on top of the blur: without this a rounded rect keeps a
    # visible straight top/bottom edge over a high-contrast dot field, because
    # blurring alone leaves the plateau boundary legible.
    mw, mh = (x1 - x0), (y1 - y0)
    ny_, nx_ = np.mgrid[0:mh, 0:mw].astype(np.float32)
    rx = (nx_ - (mw - 1) * 0.5) / max((mw - 1) * 0.5, 1.0)
    ry = (ny_ - (mh - 1) * 0.5) / max((mh - 1) * 0.5, 1.0)
    vign = np.clip(1.18 - 0.50 * (rx * rx + ry * ry), 0.0, 1.0)
    m = Image.fromarray(
        (np.asarray(m, dtype=np.float32) * vign).clip(0, 255).astype(np.uint8), "L")
    scrim = Image.new("RGBA", (x1 - x0, y1 - y0), tuple(color) + (0,))
    scrim.putalpha(m.point(lambda v, k=s: int(v * k)))
    base.alpha_composite(scrim, (x0, y0))
    _writeback(img, base, backing)
    return {"strength": s, "p95": p95, "mean": mean_l,
            "box": (x0, y0, x1, y1)}


def plate_all(img, boxes, **kw) -> Dict[str, float]:
    """Draw one merged scrim covering several text boxes. Call *before* the text."""
    return plate(img, list(boxes), **kw)


# Internal alias: the treatments take a ``plate=`` argument, which shadows the
# module-level function name inside their bodies.
_plate = plate


# ----------------------------------------------------------------------------
# Shared text geometry
#
# Everything (drawing *and* the ``*_box`` helpers used for plate layout) goes
# through ``_layout`` so a pre-computed plate can never disagree with where the
# glyphs actually land.
# ----------------------------------------------------------------------------

def _layout(text: str, size: float, alias: str, weight: Optional[float],
            tracking: Optional[int], anchor: str, kind: str,
            font=None, factors=None) -> Dict[str, object]:
    sp = float(auto_tracking(size, kind) if tracking is None else tracking)
    font = font if font is not None else _font(alias, int(size), weight)
    adv, total, asc, desc = _line_box(text, font, sp, factors)
    ax, ay, total_h = _anchor_offsets(anchor, total, asc, desc)
    ib = measure(text, font, sp, factors)
    # Padding must exceed 3 sigma of the widest glow blur (size * 0.17) or the
    # blurred glow gets sliced off at the canvas edge as a hard rectangle.
    pad = int(math.ceil(max(float(size) * 0.72, 56)))
    pad4 = pad + 4.0
    # The glyph box is always painted at a FIXED spot inside the padded canvas;
    # the anchor math is carried by `dest` at paste-back time. (Folding the
    # anchor into the paint origin instead pushes centred text to a negative
    # canvas x, where it gets silently clipped.)
    origin_ink = (-total * ax, -total_h * ay)
    return {"font": font, "sp": sp, "total": total, "asc": asc, "desc": desc,
            "ax": ax, "ay": ay, "total_h": total_h, "ib": ib, "pad": pad,
            "factors": factors, "origin_ink": origin_ink, "pad4": pad4,
            "origin": (pad4, pad4),
            "dest": (origin_ink[0] - pad4, origin_ink[1] - pad4),
            "canvas": (int(math.ceil(total)) + 2 * pad + 8,
                       int(math.ceil(total_h)) + 2 * pad + 8)}


def _rest_bbox(lay: Dict[str, object], xy, dy: float = 0.0) -> BBox:
    """Ink bbox for a resting (fully entered) position — what plates must cover.

    Uses the *unpadded* ink origin: the canvas padding exists only to hold the
    glow, and must never leak into geometry the caller lays out against.
    """
    ib, org = lay["ib"], lay["origin_ink"]
    bx0 = int(xy[0] + org[0]) + ib[0]
    by0 = int(round(xy[1] + dy + org[1])) + ib[1]
    return (bx0, by0, bx0 + (ib[2] - ib[0]), by0 + (ib[3] - ib[1]))


def headline_box(xy, text: str, size: int = 104, *, anchor: str = "lm",
                 weight: float = 700.0, tracking: Optional[int] = None) -> BBox:
    """Resting ink bbox of a ``headline`` call (no drawing)."""
    return _rest_bbox(_layout(str(text), float(size), "inter", weight, tracking,
                              anchor, "display"), xy)


def stat_box(xy, text: str = "1,000,000", size: int = 176, *,
             anchor: str = "mm", weight: float = 700.0,
             tracking: Optional[int] = None,
             tighten_commas: float = 0.62) -> BBox:
    """Resting ink bbox of a ``stat_number`` call (no drawing)."""
    text = str(text)
    factors = np.ones(len(text), dtype=np.float64)
    if tighten_commas and tighten_commas != 1.0:
        for i, c in enumerate(text):
            if c in ",.":
                factors[i] = float(tighten_commas)
    return _rest_bbox(_layout(text, float(size), "grotesk", weight, tracking,
                              anchor, "numeral", factors=factors), xy)


def sub_box(xy, text: str, size: int = 30, *, anchor: str = "lm",
            weight: float = 450.0, tracking: int = 3,
            rule: bool = False, rule_gap: int = 30) -> BBox:
    """Resting ink bbox of a ``sub_line`` call (rule area included when asked)."""
    lay = _layout(str(text), float(size), "inter", weight, tracking, anchor, "sub")
    bb = _rest_bbox(lay, xy)
    if rule:
        bb = (bb[0], bb[1] - int(rule_gap) - 2, bb[2], bb[3])
    return bb


def eyebrow_box(xy, text: str, size: int = 21, *, anchor: str = "lm",
                tracking: int = 8, upper: bool = True, rule: bool = False,
                rule_gap: int = 18) -> BBox:
    """Resting ink bbox of an ``eyebrow`` call (no drawing)."""
    t = str(text).upper() if upper else str(text)
    font = _mono_font(size)
    lay = _layout(t, float(size), "mono", None, tracking, anchor, "eyebrow",
                  font=font)
    bb = _rest_bbox(lay, xy)
    if rule:
        bb = (bb[0], bb[1] - int(rule_gap) - 2, bb[2], bb[3])
    return bb


# ----------------------------------------------------------------------------
# Headline
# ----------------------------------------------------------------------------

def headline(img, xy, text: str, size: int, t: float = 1.0, *,
             anchor: str = "lm",
             weight: float = 700.0,
             tracking: Optional[int] = None,
             delay: float = 0.0,
             offset_y: float = DEF_OFFSET_Y,
             damping: float = DEF_DAMPING,
             stiffness: float = DEF_STIFFNESS,
             mass: float = DEF_MASS,
             stagger_window: float = DEF_STAGGER_WINDOW,
             stagger_dur: float = DEF_STAGGER_DUR,
             reveal: str = "none",
             reveal_dur: float = 0.9,
             reveal_axis: Tuple[float, float] = (0.0, 1.0),
             accent: Union[None, str, Sequence[str], Dict[str, Color]] = None,
             accent_color: Color = ACCENT,
             accent_ramp: Tuple[Color, Color] = (ACC_TOP, ACC_BOT),
             top_color: Color = HEAD_TOP,
             bot_color: Color = HEAD_BOT,
             rim: Optional[Color] = RIM,
             rim_strength: float = 0.9,
             glow: Color = INDIGO,
             glow_accent: Color = CYAN,
             glow_wide: float = 0.24,
             glow_tight: float = 0.30,
             separation: float = 0.32,
             plate: Union[None, str, float, dict] = None,
             alpha: float = 1.0,
             grad_align: str = "ink",
             factors=None,
             blur_copies: int = 5) -> BBox:
    """Display headline. Returns the ink bbox in image coords.

    Entrance: ``offset_y`` rise, alpha from ``E.spring`` (d14/k170 by default),
    per-char stagger window 0.55s. ``reveal`` is one of ``"none"``,
    ``"wipe-left"`` (soft gradient mask over char alphas) or ``"blur"``
    (several decreasing-alpha/offset copies faking depth).
    """
    text = str(text)
    if not text:
        return (0, 0, 0, 0)
    size = float(size)
    L = _layout(text, size, "inter", weight, tracking, anchor, "display",
                factors=factors)
    font, sp = L["font"], L["sp"]
    total, asc, total_h, origin = L["total"], L["asc"], L["total_h"], L["origin"]
    cw, ch = L["canvas"]
    dest = L["dest"]

    a_glob, dy = entrance(t, delay, offset_y, damping, stiffness, mass)
    a_glob *= clamp01(alpha)
    if a_glob <= 0.004:
        return (0, 0, 0, 0)

    rev = (reveal or "none").lower()
    if rev == "wipe-left":
        cas = _wipe_alphas(text, font, sp, clamp01((t - delay) / max(reveal_dur, 1e-6)),
                           max(10.0, size * 0.30), factors)
    elif rev == "blur":
        cas = np.full(len(text), clamp01(a_glob * 1.15), dtype=np.float64)
    else:
        cas = _char_alphas(text, t, delay, stagger_window, stagger_dur)
    if float(cas.max(initial=0.0)) <= 0.004:
        return (0, 0, 0, 0)

    alpha_arr = _paint_mask((cw, ch), text, font, sp, cas, 1.0, 0.0, 0.0, origin, factors)
    if int(alpha_arr.max()) < 3:
        return (0, 0, 0, 0)

    # accent segmentation (exact same advances -> pixel-perfect alignment)
    acc_flags = np.zeros(len(text), dtype=np.float64)
    acc_cols: List[Optional[Color]] = [None] * len(text)
    if accent:
        spec: Dict[str, Color] = {}
        if isinstance(accent, dict):
            spec = dict(accent)
        else:
            keys = [accent] if isinstance(accent, str) else list(accent)
            for k in keys:
                spec[k] = accent_color
        lower = text.lower()
        for key, col in spec.items():
            k = str(key).lower()
            if not k:
                continue
            start = 0
            while True:
                j = lower.find(k, start)
                if j < 0:
                    break
                for i in range(j, min(j + len(k), len(text))):
                    acc_flags[i] = 1.0
                    acc_cols[i] = col if isinstance(col, (tuple, list)) else accent_color
                start = j + 1
    base_mask = _paint_mask((cw, ch), text, font, sp, cas, 1.0, 0.0, 0.0, origin, factors)
    if acc_flags.max() > 0:
        acc_mask = _paint_mask((cw, ch), text, font, sp,
                               cas * acc_flags, 1.0, 0.0, 0.0, origin, factors)
        base_mask = _paint_mask(
            (cw, ch), text, font, sp, cas * (1.0 - acc_flags), 1.0, 0.0, 0.0, origin,
            factors)
    else:
        acc_mask = None

    span = None
    if grad_align == "font":
        span = (origin[1], origin[1] + total_h)

    ox = int(round(xy[0] + dest[0]))
    oy = int(round(xy[1] + dy + dest[1]))

    # Everything that sits *under* the core is accumulated into a single layer
    # and pasted once: each _composite converts the whole 1280x720 frame to
    # RGBA and back, so batching them is the difference between 5 full-frame
    # round trips per headline and 1.
    under: Optional[Image.Image] = None

    def _under(layer: Optional[Image.Image]) -> None:
        nonlocal under
        if layer is None:
            return
        under = layer if under is None else Image.alpha_composite(under, layer)

    # 1) separation ring (dark halo) -> survives bright dot fields
    _under(separation_ring(alpha_arr, BASE, separation, max(1.5, size * 0.030)))
    # 2) two-stage glow, under the core
    _under(two_stage_glow(alpha_arr, size, glow, glow_wide, glow_tight, glow_accent))

    # 3) depth copies for reveal="blur" (decreasing alpha + offset)
    if rev == "blur":
        step = max(1.0, size * 0.045)
        dxv, dyv = reveal_axis
        nrm = math.hypot(dxv, dyv) or 1.0
        dxv, dyv = dxv / nrm, dyv / nrm
        for k in range(1, int(max(1, blur_copies)) + 1):
            f = k / (int(max(1, blur_copies)) + 1.0)
            m = _paint_mask((cw, ch), text, font, sp, cas, 0.30 * (1.0 - f),
                            dxv * step * k * a_glob, dyv * step * k * a_glob,
                            origin, factors)
            lay = gradient_core(m, top_color, bot_color, span, total_h)
            _under(lay.filter(ImageFilter.GaussianBlur(size * 0.012 * k)))
    if under is not None:
        _composite(img, under, (ox, oy))

    # 4) core: gradient body + crisp rim
    def _core(mask: np.ndarray, top: Color, bot: Color) -> Image.Image:
        lay = gradient_core(mask, top, bot, span, total_h)
        if rim is not None:
            band = rim_light(mask, rim, rim_strength, max(1.0, size * 0.022))
            if int(band.max()) > 3:
                lay = Image.alpha_composite(lay, _tinted(band, rim))
        return lay

    _composite(img, _core(base_mask, top_color, bot_color), (ox, oy))
    if acc_mask is not None and int(acc_mask.max()) > 3:
        uniform = all((c or accent_color) == accent_color for c in acc_cols)
        at, ab = (accent_ramp if uniform else (None, None))
        if at is None:  # heterogeneous accents: flat per-word tint
            lay = gradient_core(acc_mask, accent_ramp[0], accent_ramp[0], span, total_h)
        else:
            lay = gradient_core(acc_mask, at, ab, span, total_h)
        if rim is not None:
            band = rim_light(acc_mask, rim, rim_strength * 0.9, max(1.0, size * 0.022))
            if int(band.max()) > 3:
                lay = Image.alpha_composite(lay, _tinted(band, RIM))
        _composite(img, lay, (ox, oy))

    return _rest_bbox(L, xy, dy)


def headline_block(img, xy, lines: Sequence[str], sizes: Optional[Sequence[float]] = None,
                   line_height: float = 1.14, t: float = 1.0, line_delay: float = 0.15,
                   **kw) -> BBox:
    """Stack of headlines with cascaded entrances (v1's text_block, refined).

    ``xy`` is the anchor of the *first* line; each line is anchored at its own
    vertical middle and stepped down by ``size * line_height``. Returns the
    union ink bbox.

    For a block you also want to plate, call ``plate_all`` first with
    ``headline_block_boxes(...)`` so the scrim lands under every line.
    """
    lines = [str(l) for l in lines]
    if not lines:
        return (0, 0, 0, 0)
    if sizes is None:
        sizes = [{1: 132, 2: 108, 3: 84}[min(len(lines), 3)]] * len(lines)
    anchor = kw.pop("anchor", "lm")
    base_delay = float(kw.pop("delay", 0.0))
    boxes: List[BBox] = []
    y = float(xy[1])
    for i, ln in enumerate(lines):
        s = float(sizes[i])
        call = dict(kw)
        call["delay"] = base_delay + i * float(line_delay)
        bb = headline(img, (xy[0], y), ln, s, t, anchor=anchor, **call)
        if bb != (0, 0, 0, 0):
            boxes.append(bb)
        y = y + s * float(line_height)
    if not boxes:
        return (0, 0, 0, 0)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def headline_block_boxes(xy, lines: Sequence[str],
                         sizes: Optional[Sequence[float]] = None,
                         line_height: float = 1.14, anchor: str = "lm",
                         **kw) -> List[BBox]:
    """Resting ink bboxes of a ``headline_block`` — feed these to ``plate_all``.

    Draw the scrim first, then the block, otherwise the block's own lines will
    sit on top of an already-darkened canvas.
    """
    lines = [str(l) for l in lines]
    if not lines:
        return []
    if sizes is None:
        sizes = [{1: 132, 2: 108, 3: 84}[min(len(lines), 3)]] * len(lines)
    return [headline_box((xy[0], float(xy[1]) + i * float(sizes[i]) * line_height),
                         ln, sizes[i], anchor=anchor, **kw)
            for i, ln in enumerate(lines)]


# ----------------------------------------------------------------------------
# Sub-line
# ----------------------------------------------------------------------------

def sub_line(img, xy, text: str, size: int = 30, t: float = 1.0, *,
             anchor: str = "lm",
             weight: float = 450.0,
             tracking: int = 3,
             delay: float = SUB_DELAY,
             fill: Color = SUB_FILL,
             glow: Optional[Color] = INDIGO,
             glow_alpha: float = 0.26,
             glow_radius: Optional[int] = None,
             separation: float = 0.42,
             plate: Union[None, str, float, dict] = None,
             rule: bool = False,
             rule_gap: int = 30,
             rule_color: Color = INDIGO,
             rule_fade: float = 0.34,
             rule_height: int = 1,
             stagger_window: float = 0.5,
             stagger_dur: float = 0.5,
             offset_y: float = 26.0,
             accent: Union[None, str, Sequence[str], Dict[str, Color]] = None,
             accent_color: Color = ACCENT_WARM,
             alpha: float = 1.0) -> BBox:
    """Secondary line: muted lavender, +3 tracking, optional hairline rule.

    Drawn through ``E.draw_text`` (per-char path, so tracking + stagger are
    exact), preceded by a dark separation pass and an adaptive plate when the
    background under it is busy.
    """
    text = str(text)
    if not text:
        return (0, 0, 0, 0)
    size = float(size)
    font = _font("inter", int(size), weight)
    sp = float(tracking)
    a, dy = entrance(t, delay, offset_y, DEF_DAMPING, DEF_STIFFNESS, DEF_MASS)
    a *= clamp01(alpha)
    if a <= 0.004:
        return (0, 0, 0, 0)
    cas = _char_alphas(text, t, delay, stagger_window, stagger_dur)
    if float(cas.max(initial=0.0)) <= 0.004:
        return (0, 0, 0, 0)

    adv, total, asc, desc = _line_box(text, font, sp)
    ax, ay, total_h = _anchor_offsets(anchor, total, asc, desc)
    ib = measure(text, font, sp)
    lx = xy[0] - total * ax
    ty = xy[1] + dy - total_h * ay
    bx0 = int(lx) + ib[0]; by0 = int(round(ty)) + ib[1]
    bx1 = bx0 + (ib[2] - ib[0]); by1 = by0 + (ib[3] - ib[1])

    pk = _as_plate_kwargs(plate)
    if pk["enabled"]:
        extra = 0
        if rule:
            extra = int(rule_gap) + 6
        box = (bx0, by0 - extra, bx1, by1)
        _plate(img, box, strength=pk.get("strength"), auto=pk.get("auto", False),
              pad=(30, 22), radius=18, feather=30.0)

    if separation > 0.004:
        pad = int(math.ceil(size * 0.6)) + 8
        m = _paint_mask((int(total) + 2 * pad, int(total_h) + 2 * pad), text, font, sp,
                        cas, 1.0, 0.0, 0.0, (pad - total * ax, pad - total_h * ay))
        ring = separation_ring(m, (3, 3, 8), separation, max(1.0, size * 0.055))
        if ring is not None:
            _composite(img, ring, (int(round(lx)) - pad, int(round(ty)) - pad))

    # accent words inside the sub (warm-white / cyan)
    if accent:
        spec: Dict[str, Color] = (dict(accent) if isinstance(accent, dict)
                                  else {k: accent_color for k in
                                        ([accent] if isinstance(accent, str) else accent)})
        lower = text.lower()
        for key, col in spec.items():
            k = str(key).lower()
            start = 0
            while True:
                j = lower.find(k, start)
                if j < 0:
                    break
                c = col if isinstance(col, (tuple, list)) else accent_color
                for i in range(j, min(j + len(k), len(text))):
                    cx = lx + float(adv[:i].sum()) + sp * i
                    E.draw_text(img, (cx, xy[1] + dy + asc), text[i],
                                font=font, size=int(size), fill=c, anchor="ls",
                                alpha=a * float(cas[i]), spacing=0)
                start = j + 1

    gr = int(glow_radius) if glow_radius else max(3, int(size * 0.30))
    E.draw_text(img, (xy[0], xy[1] + dy), text, font=font, size=int(size),
                fill=fill, anchor=anchor, glow=glow, glow_radius=gr,
                glow_alpha=glow_alpha, alpha=a, spacing=sp, char_alphas=cas)

    if rule:
        _draw_rule(img, lx, ty - float(rule_gap), total, a, rule_color,
                   rule_fade, rule_height)
    return (bx0, by0, bx1, by1)


def _draw_rule(img, x: float, y: float, width: float, a: float,
               color: Color, fade: float, height: int = 1) -> BBox:
    """Hairline rule that wipes in with the entrance, fading at both ends."""
    w = max(2, int(round(width * clamp01(a))))
    h = max(1, int(height))
    if w < 3:
        return (0, 0, 0, 0)
    layer = Image.new("RGBA", (w, h), tuple(color) + (0,))
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = color[0]; rgb[:, :, 1] = color[1]; rgb[:, :, 2] = color[2]
    al = np.zeros((h, w), dtype=np.uint8)
    fx = np.arange(w, dtype=np.float32)
    fl = np.clip(fx / max(1.0, w * fade), 0.0, 1.0)
    fr = np.clip((w - 1 - fx) / max(1.0, w * fade), 0.0, 1.0)
    prof = (np.minimum(fl, fr) * 255.0 * a).clip(0, 255).astype(np.uint8)
    al[:, :] = prof[None, :]
    # slight cyan->indigo ramp along the rule for a bit of life
    t = (fx / max(1.0, w - 1.0))[None, :, None]
    c0 = np.asarray(CYAN, dtype=np.float32)[None, None, :]
    c1 = np.asarray(color, dtype=np.float32)[None, None, :]
    rgb[:] = np.clip(c0 * (1 - t) + c1 * t, 0, 255).astype(np.uint8)
    layer = Image.fromarray(np.dstack([rgb, al]), "RGBA")
    _composite(img, layer, (int(round(x)), int(round(y))))
    return (int(round(x)), int(round(y)), int(round(x)) + w, int(round(y)) + h)


# ----------------------------------------------------------------------------
# Stat number
# ----------------------------------------------------------------------------

def stat_number(img, xy, text: str = "1,000,000", size: int = 168, t: float = 1.0, *,
                anchor: str = "mm",
                weight: float = 700.0,
                tracking: Optional[int] = None,
                delay: float = 0.0,
                offset_y: float = 44.0,
                damping: float = DEF_DAMPING,
                stiffness: float = DEF_STIFFNESS,
                stagger_window: float = DEF_STAGGER_WINDOW,
                stagger_dur: float = 0.62,
                top_color: Color = ACC_TOP,
                bot_color: Color = ACC_BOT,
                rim: Optional[Color] = (235, 253, 255),
                rim_strength: float = 0.85,
                glow: Color = CYAN,
                glow_wide: float = 0.30,
                glow_tight: float = 0.34,
                separation: float = 0.40,
                plate: Union[None, str, float, dict] = None,
                tighten_commas: float = 0.62,
                alpha: float = 1.0) -> BBox:
    """Hero numeral in Space Grotesk, cyan-tinted with a wide halo.

    ``tighten_commas`` scales the advance each comma claims (Space Grotesk's
    full-width comma advance leaves the "1,000,000" looking gappy at display
    size — squeezing it gives the tight numeral rhythm designers use).
    """
    text = str(text)
    if not text:
        return (0, 0, 0, 0)
    size = float(size)
    factors = np.ones(len(text), dtype=np.float64)
    if tighten_commas and tighten_commas != 1.0:
        for i, c in enumerate(text):
            if c in ",.":
                factors[i] = float(tighten_commas)
    L = _layout(text, size, "grotesk", weight, tracking, anchor, "numeral",
                factors=factors)
    font, sp = L["font"], L["sp"]
    total_h, origin = L["total_h"], L["origin"]
    cw, ch = L["canvas"]
    dest = L["dest"]

    a, dy = entrance(t, delay, offset_y, damping, stiffness)
    a *= clamp01(alpha)
    if a <= 0.004:
        return (0, 0, 0, 0)
    cas = _char_alphas(text, t, delay, stagger_window, stagger_dur)
    if float(cas.max(initial=0.0)) <= 0.004:
        return (0, 0, 0, 0)

    mask = _paint_mask((cw, ch), text, font, sp, cas, 1.0, 0.0, 0.0, origin,
                       factors)
    if int(mask.max()) < 3:
        return (0, 0, 0, 0)
    ox = int(round(xy[0] + dest[0]))
    oy = int(round(xy[1] + dy + dest[1]))

    pk = _as_plate_kwargs(plate)
    if pk["enabled"]:
        _plate(img, _rest_bbox(L, xy, dy), strength=pk.get("strength"),
               auto=pk.get("auto", False), pad=(40, 30), radius=30)

    under: Optional[Image.Image] = None
    ring = separation_ring(mask, (3, 4, 10), separation, max(2.0, size * 0.026))
    gl = two_stage_glow(mask, size, glow, glow_wide, glow_tight, INDIGO)
    if ring is not None:
        under = ring
    if gl is not None:
        under = gl if under is None else Image.alpha_composite(under, gl)
    if under is not None:
        _composite(img, under, (ox, oy))
    core = gradient_core(mask, top_color, bot_color)
    if rim is not None:
        band = rim_light(mask, rim, rim_strength, max(1.0, size * 0.020))
        if int(band.max()) > 3:
            core = Image.alpha_composite(core, _tinted(band, rim))
    _composite(img, core, (ox, oy))
    return _rest_bbox(L, xy, dy)


# ----------------------------------------------------------------------------
# Eyebrow
# ----------------------------------------------------------------------------

def eyebrow(img, xy, text: str, size: int = 21, t: float = 1.0, *,
            anchor: str = "lm",
            tracking: int = 8,
            delay: float = 0.0,
            fill: Color = EYEBROW_FILL,
            upper: bool = True,
            glow: Optional[Color] = CYAN,
            glow_alpha: float = 0.40,
            glow_radius: Optional[int] = None,
            separation: float = 0.55,
            plate: Union[None, str, float, dict] = None,
            offset_y: float = 16.0,
            stagger_window: float = 0.45,
            stagger_dur: float = 0.42,
            alpha: float = 1.0,
            rule: bool = False,
            rule_gap: int = 18,
            rule_color: Color = INDIGO) -> BBox:
    """Small uppercase mono kicker (IBM Plex Mono SemiBold) with +8 tracking."""
    text = str(text)
    if not text:
        return (0, 0, 0, 0)
    if upper:
        text = text.upper()
    size = float(size)
    try:
        font = _mono_font(size)
    except Exception:
        font = _font("mono", int(size), None)
    sp = float(tracking)
    a, dy = entrance(t, delay, offset_y, DEF_DAMPING * 1.15, DEF_STIFFNESS * 1.15)
    a *= clamp01(alpha)
    if a <= 0.004:
        return (0, 0, 0, 0)
    cas = _char_alphas(text, t, delay, stagger_window, stagger_dur)
    if float(cas.max(initial=0.0)) <= 0.004:
        return (0, 0, 0, 0)

    adv, total, asc, desc = _line_box(text, font, sp)
    ax, ay, total_h = _anchor_offsets(anchor, total, asc, desc)
    ib = measure(text, font, sp)
    lx = xy[0] - total * ax
    ty = xy[1] + dy - total_h * ay
    bx0 = int(lx) + ib[0]; by0 = int(round(ty)) + ib[1]
    bx1 = bx0 + (ib[2] - ib[0]); by1 = by0 + (ib[3] - ib[1])

    pk = _as_plate_kwargs(plate)
    if pk["enabled"]:
        _plate(img, (bx0, by0 - (int(rule_gap) + 6 if rule else 0), bx1, by1),
              strength=pk.get("strength"), auto=pk.get("auto", False),
              pad=(26, 16), radius=14, feather=26.0)

    if separation > 0.004:
        pad = int(math.ceil(size * 0.9)) + 8
        m = _paint_mask((int(total) + 2 * pad, int(total_h) + 2 * pad), text, font, sp,
                        cas, 1.0, 0.0, 0.0, (pad - total * ax, pad - total_h * ay))
        ring = separation_ring(m, (3, 3, 8), separation, max(1.0, size * 0.10))
        if ring is not None:
            _composite(img, ring, (int(round(lx)) - pad, int(round(ty)) - pad))

    gr = int(glow_radius) if glow_radius else max(3, int(size * 0.30))
    E.draw_text(img, (xy[0], xy[1] + dy), text, font=font, size=int(size),
                fill=fill, anchor=anchor, glow=glow, glow_radius=gr,
                glow_alpha=glow_alpha, alpha=a, spacing=sp, char_alphas=cas)
    if rule:
        _draw_rule(img, lx, ty - float(rule_gap), total, a, rule_color, 0.30)
    return (bx0, by0, bx1, by1)


# ----------------------------------------------------------------------------
# QA samples
# ----------------------------------------------------------------------------

W, H = 1280, 720
INDIGO_GRADE = np.array([1.12, 0.93, 1.10], dtype=np.float32)


def _bg(kind: str, t: float, energy: float = 0.7, frame: Optional[int] = None) -> Image.Image:
    fn = {"tunnel": E.bg_tunnel, "rings": E.bg_rings, "solid": E.bg_solid,
          "grid": E.bg_grid, "waves": E.bg_waves, "particles": E.bg_particles}[kind]
    arr = fn(W, H, t=t, energy=float(energy), frame=int(frame if frame is not None else t * 30))
    graded = np.clip(arr.astype(np.float32) * INDIGO_GRADE, 0, 255).astype(np.uint8)
    return Image.fromarray(graded, "RGB").convert("RGBA")


def _save(img, path: str) -> str:
    img.convert("RGB").save(path, quality=94, subsampling=1)
    return path


def _contrast(img, bbox: BBox, fill: Color) -> Tuple[float, float, float]:
    """(mean text luminance, mean local bg luminance, WCAG-ish ratio)."""
    base, _ = E._as_rgba(img)
    arr = np.asarray(base.convert("RGB"), dtype=np.float32)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(arr.shape[1], x1); y1 = min(arr.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0
    reg = arr[y0:y1, x0:x1]
    lum = 0.2126 * reg[:, :, 0] + 0.7152 * reg[:, :, 1] + 0.0722 * reg[:, :, 2]
    fl = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
    mean = float(lum.mean()); p95 = float(np.percentile(lum, 95))
    l1, l2 = (fl + 5.0) / 5.0, (mean + 5.0) / 5.0
    lo, hi = (l1, l2) if l1 > l2 else (l2, l1)
    return fl, mean, (hi + 0.05) / (lo + 0.05)


def _lum(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def _readability(img, box: BBox) -> Tuple[float, float, float]:
    """(glyph luma, surround luma, ratio) measured on the *finished* image.

    A percentile over the whole box is meaningless — the brightest dot is
    almost always in the gap between letters, not behind one. What predicts
    whether a line is readable is how bright the glyph core is against the
    pixels immediately around it, so that is what we measure: pixels well
    above the local median are the glyphs, the rest is its surround.
    """
    base, _ = E._as_rgba(img)
    arr = np.asarray(base.convert("RGB"), dtype=np.float32)
    W, H = arr.shape[1], arr.shape[0]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return 0.0, 0.0, 0.0
    lum = _lum(arr[y0:y1, x0:x1])
    thr = float(np.median(lum)) + 55.0
    glyph = lum > thr
    if glyph.sum() < 12:
        return float(lum.max()), float(np.median(lum)), 0.0
    fg = float(np.percentile(lum[glyph], 60))
    bg = float(np.percentile(lum[~glyph], 80)) if (~glyph).any() else 0.0
    l1, l2 = (fg + 5.0) / 5.0, (bg + 5.0) / 5.0
    lo, hi = (l1, l2) if l1 <= l2 else (l2, l1)
    return fg, bg, (hi + 0.05) / (lo + 0.05)


def _bg_stats(img, box: BBox) -> Tuple[float, float]:
    """(mean, p95) luminance of the *background* under a text box."""
    return _region_luma(img, box)


def qa_samples(out_dir: str = HERE) -> Dict[str, object]:
    """Render the QA stills and report measured legibility numbers."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    findings: List[str] = []

    # -- (a) 3-line headline + sub over the dot tunnel ----------------------
    img = _bg("tunnel", t=7.35, energy=0.82, frame=220)
    lines = [("ONE BILLION", "BILLION", "none"),
             ("IMPRESSIONS.", None, "wipe-left"),
             ("ONE DECISION.", "ONE", "none")]
    y0, step, size = 248, 116, 104
    sub_y = y0 + 2 * step + 128
    boxes = [headline_box((120, y0 + i * step), t_, size, anchor="lm")
             for i, (t_, _a, _r) in enumerate(lines)]
    sbox = sub_box((120, sub_y), "You approve the spend. Alpha ships the work.",
                   size=30, rule=True)
    info_a = plate_all(img, boxes + [sbox], auto=True, pad=(40, 30),
                       radius=30, shape="ellipse")
    for i, (txt, acc, rev) in enumerate(lines):
        kw = {"accent": acc, "accent_color": ACCENT_WARM if acc == "ONE" else CYAN} if acc else {}
        headline(img, (120, y0 + i * step), txt, size, t=1.6, delay=i * 0.15,
                 reveal=rev, **kw)
    sub_line(img, (120, sub_y), "You approve the spend. Alpha ships the work.",
             size=30, t=2.2, rule=True)
    paths.append(_save(img, os.path.join(out_dir, "_typo_a_headline.jpg")))
    fg, bgl, r = _readability(img, sbox)
    findings.append(
        f"(a) tunnel, merged plate {info_a['strength']:.2f} -> sub glyph "
        f"{fg:.0f} vs surround {bgl:.0f} = {r:.1f}:1.")

    # -- (b) stat_number + eyebrow + sub over the rings ---------------------
    img = _bg("rings", t=12.6, energy=0.74, frame=378)
    eb = eyebrow_box((120, 178), "LIVE COST", size=22, rule=True)
    sn = stat_box((120, 372), "1,000,000", size=176, anchor="lm")
    sb = sub_box((120, 560), "Creative sessions run this quarter. No rounding up.",
                 size=31)
    info_b = plate_all(img, [eb, sn, sb], auto=True, pad=(40, 30), radius=30,
                       shape="ellipse")
    eyebrow(img, (120, 178), "LIVE COST", size=22, t=1.1, rule=True)
    stat_number(img, (120, 372), "1,000,000", size=176, t=1.6, anchor="lm")
    sub_line(img, (120, 560), "Creative sessions run this quarter. No rounding up.",
             size=31, t=2.1, accent="this quarter", accent_color=ACCENT_WARM)
    paths.append(_save(img, os.path.join(out_dir, "_typo_b_stat.jpg")))
    fg, bgl, r = _readability(img, sb)
    fgn, bgn, rn = _readability(img, sn)
    findings.append(
        f"(b) rings, merged plate {info_b['strength']:.2f} -> sub {fg:.0f} vs "
        f"{bgl:.0f} = {r:.1f}:1; cyan numeral {fgn:.0f} vs {bgn:.0f} = {rn:.1f}:1.")

    # -- (c) sub_line + rule, with and without the plate --------------------
    txt = "Spend only moves after you say yes."
    probe = sub_box((120, 372), txt, size=32, rule=True)

    img_a = _bg("tunnel", t=4.05, energy=1.0, frame=122)
    raw_mean, raw_p95 = _bg_stats(img_a, probe)
    sub_line(img_a, (120, 372), txt, size=32, t=1.4, rule=True)
    fg_a, bg_a, r_a = _readability(img_a, probe)
    paths.append(_save(img_a, os.path.join(out_dir, "_typo_c_sub_noplate.jpg")))

    img_b = _bg("tunnel", t=4.05, energy=1.0, frame=122)
    info_c = plate(img_b, probe, auto=True, pad=(36, 28), radius=26)
    sub_line(img_b, (120, 372), txt, size=32, t=1.4, rule=True)
    paths.append(_save(img_b, os.path.join(out_dir, "_typo_c_sub_plate.jpg")))
    pl_mean, pl_p95 = _bg_stats(img_b, probe)
    fg_b, bg_b, r_b = _readability(img_b, probe)
    findings.append(
        f"(c) brightest tunnel: NO plate -> glyph {fg_a:.0f} vs surround {bg_a:.0f} "
        f"= {r_a:.1f}:1 (box p95 {raw_p95:.0f}). plate='auto' {info_c['strength']:.2f} "
        f"-> {fg_b:.0f} vs {bg_b:.0f} = {r_b:.1f}:1 (box p95 {pl_p95:.0f}).")
    findings.append(
        f"    separation ring alone: {_readability(img_a, probe)[2]:.1f}:1 with no "
        f"scrim at all, vs 0.0 plate -> the ring is the cheap fix, the plate is "
        f"the one that survives the brightest beat.")

    # -- (d) every treatment on a solid base -------------------------------
    img = _bg("solid", t=2.0, energy=0.35, frame=60)
    eyebrow(img, (120, 128), "PROOF", size=20, t=1.2, rule=True)
    headline(img, (120, 214), "FLAT WHITE", 88, t=1.4)
    headline(img, (120, 316), "GRADIENT BODY", 88, t=1.4, delay=0.12,
             reveal="blur", accent="GRADIENT", accent_color=CYAN)
    stat_number(img, (640, 492), "1,000,000", size=130, t=1.7, anchor="mm",
                delay=0.2)
    sub_line(img, (120, 636), "Sub-line, +3 tracking, muted lavender.", size=30,
             t=2.1, rule=True)
    paths.append(_save(img, os.path.join(out_dir, "_typo_d_solid.jpg")))

    # -- (e) reveal variants strip -----------------------------------------
    img = _bg("solid", t=2.0, energy=0.3, frame=60)
    for i, rv in enumerate(("none", "wipe-left", "blur")):
        eyebrow(img, (300, 92 + i * 168), rv.replace("-", " "), size=17, t=0.35,
                anchor="mm")
        headline(img, (640, 152 + i * 168), rv.upper(), 62, t=0.5,
                 anchor="mm", reveal=rv)
    paths.append(_save(img, os.path.join(out_dir, "_typo_e_reveals.jpg")))

    return {"paths": paths, "findings": findings}


if __name__ == "__main__":
    res = qa_samples()
    print("built typography.py — refined text-colour system")
    print("samples:")
    for p in res["paths"]:
        print("   ", p)
    print("findings:")
    for f in res["findings"]:
        print("   ", f)