#!/usr/bin/env python3
"""ALPHA promo — endcard / logo outro (22.12s -> 25.00s, 86 frames @ 30 fps).

Public API
----------
``render_endcard_frame(frame_idx, total=86, fps=30) -> PIL.Image`` (RGB 1280x720)

Everything is a pure function of the clamped frame index, so the last 24
frames (0.8 s) are byte-identical for a clean hold.

CLI:  ``python3 endcard.py qa``      -> 6 QA stills + contact sheet
      ``python3 endcard.py render``  -> full 86-frame PNG sequence
      ``python3 endcard.py info``    -> layout / timing dump

NOTE: this module never mutates ``engine.get_font``'s cache (variable-font
instances are shared process-wide), so weighted Inter Tight faces are built
from the on-disk TTFs privately.
"""

from __future__ import annotations

import glob
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (  # noqa: E402  (path set above)
    BASE, CYAN, FONT_DIR, interpolate, spring, stagger_alphas,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INDIGO: Tuple[int, int, int] = (108, 75, 255)      # brand indigo  #6C4BFF
INDIGO_LT: Tuple[int, int, int] = (139, 107, 255)  # tifted indigo #8B6BFF
NEAR_WHITE: Tuple[int, int, int] = (237, 235, 255)  # #EDEBFF
FINE: Tuple[int, int, int] = (154, 151, 176)      # #9A97B0

W, H = 1280, 720
FPS = 30.0
TOTAL_DEFAULT = 86

# ---------------------------------------------------------------------------
# Asset locations
# ---------------------------------------------------------------------------
BLINK_DIR = "/tmp/opencode/logo/eye_blink"
LOGO_DIR = "/tmp/opencode/rebuild/logo"

# Crop box of the 60 blink frames inside their 1280x720 plate — identical to
# the box used for eye_open.png / eye_closed.png, so the eye optical centre
# maps to plate (640, 360) -> asset centre (400, 250) of the resized layer.
EYE_CROP = (240, 110, 1040, 610)   # 800 x 500

# ---------------------------------------------------------------------------
# LAYOUT (1280 x 720)  — all values are final settled positions
# ---------------------------------------------------------------------------
LAYOUT: Dict[str, object] = {
    # box coordinates (x0, y0, x1, y1) and centre
    "eye":        {"box": (440, 40, 840, 290), "size": (400, 250),
                   "centre": (640, 165), "visible": (488, 63, 791, 255)},
    "wordmark":   {"box": (505, 300, 775, 525), "size": (270, 225),
                   "centre": (640, 412.5), "visible_bottom": 522},
    "rule":       {"centre": (640, 534), "size": (64, 2), "peak_alpha": 0.70},
    # text: value is the BASELINE y (PIL 'ls' anchor); sizes are px
    "tagline":    {"text": "Say it. It's built.", "baseline": 576,
                   "size": 36, "weight": 600, "fill": NEAR_WHITE,
                   "glyph_top": 548, "glyph_bottom": 582},
    "domain":     {"text": "alpha.studio", "baseline": 622,
                   "size": 27, "weight": 500, "fill": INDIGO_LT,
                   "tracking": 3, "glyph_top": 602, "glyph_bottom": 627},
    "fineprint":  {"text": "You approve everything. It never spends on its own.",
                   "baseline": 670, "size": 20, "weight": 400, "fill": FINE,
                   "glyph_top": 655, "glyph_bottom": 673},
}

EYE_X, EYE_Y, EYE_X1, EYE_Y1 = LAYOUT["eye"]["box"]
EYE_W, EYE_H = LAYOUT["eye"]["size"]
WM_X, WM_Y, WM_X1, WM_Y1 = LAYOUT["wordmark"]["box"]
WM_W, WM_H = LAYOUT["wordmark"]["size"]
assert (EYE_X1 - EYE_X, EYE_Y1 - EYE_Y) == (EYE_W, EYE_H)
assert (WM_X1 - WM_X, WM_Y1 - WM_Y) == (WM_W, WM_H)
RULE_CX, RULE_CY = LAYOUT["rule"]["centre"]
RULE_W, RULE_H = LAYOUT["rule"]["size"]

TAG_TEXT = LAYOUT["tagline"]["text"]
DOM_TEXT = LAYOUT["domain"]["text"]
FINE_TEXT = LAYOUT["fineprint"]["text"]

# ---------------------------------------------------------------------------
# TIMING (frames @ 30 fps, 86 total)
# ---------------------------------------------------------------------------
FREEZE_FRAMES = 24                 # 0.8 s
FREEZE_START = TOTAL_DEFAULT - FREEZE_FRAMES   # frame 62 -> t = 2.0667 s

# Beat grid inside the outro. The cut is beat-locked to the downbeat at
# abs 22.1227 s; bpm 119.68 -> 0.5013 s/beat, so beats land on local frames
# 0, 15, 30, 45, 60, 75. Both blinks put the fully-closed pose on a beat.
BEAT_FRAMES = [0, 15, 30, 45, 60, 75]
BLINK_APEX = [15, 45]              # <- the two blinks
BLINK_SCALE = 0.641                # source frames consumed per outro frame
BLINK_SRC_LO, BLINK_SRC_HI = 9, 48   # source range of one trimmed blink
BLINK_SRC_APEX = 28                 # middle of the source closed-hold (25-31)
BLINK_SPAN_L = (BLINK_SRC_APEX - BLINK_SRC_LO) * BLINK_SCALE   # 12.18
BLINK_SPAN_R = (BLINK_SRC_HI - BLINK_SRC_APEX) * BLINK_SCALE   # 12.82

# Entrance schedule (frame numbers)
# The outro opens on a beat-locked hard cut, so frame 0 must already carry the
# eye at (near) full strength — the fade window deliberately starts *before*
# f0 and lands at a=0.98 on the cut, then a scale settle supplies the life.
F_EYE_IN = (-8, 3)                 # a(0) = 0.98 — no flash-to-black
F_EYE_SCALE = (0, 12)              # 1.07 -> 1.00
F_WM = 3                           # wordmark spring start
F_TAG = 16                         # tagline stagger start (0.62 s)
F_RULE = (22, 32)                  # rule width ramp
F_DOM = (24, 36)
F_FINE = (28, 42)

SWEEP_PERIOD = 4.0                 # s — slow light-sweep cycle (one there-and-back)
SWEEP_AMOUNT = 0.28                # peak additive brightness (20-30% of full scale)
SWEEP_SIGMA = 150.0                # px
SWEEP_TILT = 0.42                  # band slope (dy/dx)
SWEEP_SPAN = 820.0                 # px the band centre travels either side of x=640
SWEEP_FADE_OUT = 10                # ramp sweep out over the last 10 frames
# Band centre: cx(t) = 640 - SPAN*cos(2*pi*t/PERIOD). Period is exactly 4.0 s.
# cx(0) = -180 (off-frame left, so the beat-locked cut lands on a clean frame),
# the band is dead centre over the eye/wordmark block at t = 1.0 s (frame 30,
# beat 3), and it has travelled off-frame right by t = 2.0 s — leaving the
# marks untouched long before the still hold at frame 62.
GRAIN = 1.2

# ---------------------------------------------------------------------------
# Private font handling (never touches engine's shared variable-font cache)
# ---------------------------------------------------------------------------
_FONT_FILES = {
    "inter": "InterTight[wght].ttf",
    "grotesk": "SpaceGrotesk[wght].ttf",
    "mono": "IBMPlexMono-Regular.ttf",
}
_FONT_CACHE: Dict[Tuple[str, int, float], object] = {}


def _font(name: str = "inter", size: int = 36, weight: Optional[float] = None):
    """Load a private font instance; ``weight`` sets the variable-font axis."""
    key = (name, int(size), float(weight) if weight is not None else -1.0)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    from PIL import ImageFont  # local import keeps module import cheap
    path = None
    for pat in (os.path.join(FONT_DIR, _FONT_FILES.get(name, "")),):
        if pat and os.path.exists(pat):
            path = pat
            break
    if path is None:  # fall back to any matching ttf in the font dir
        hits = sorted(glob.glob(os.path.join(FONT_DIR, "*.ttf")))
        want = "InterTight" if name == "inter" else name
        path = next((p for p in hits if want.lower() in os.path.basename(p).lower()),
                    hits[0] if hits else None)
    font = ImageFont.truetype(path, int(size)) if path else ImageFont.load_default(size=size)
    if weight is not None:
        try:
            font.set_variation_by_axes([float(weight)])
        except Exception:
            pass
    _FONT_CACHE[key] = font
    return font


# ---------------------------------------------------------------------------
# Asset caches
# ---------------------------------------------------------------------------
_BLINK: Dict[int, Image.Image] = {}
_WORDMARK: Optional[Image.Image] = None


def _blink_frame(src: int) -> Image.Image:
    """src index (0..59) of the blink sequence -> RGBA 400x250 layer."""
    src = int(np.clip(src, 0, 59))
    if src in _BLINK:
        return _BLINK[src]
    im = Image.open(f"{BLINK_DIR}/eye{src:03d}.png").convert("RGBA")
    im = im.crop(EYE_CROP).resize((EYE_W, EYE_H), Image.LANCZOS)
    _BLINK[src] = im
    return im


def _wordmark() -> Image.Image:
    global _WORDMARK
    if _WORDMARK is None:
        _WORDMARK = Image.open(f"{LOGO_DIR}/wordmark.png").convert("RGBA") \
            .resize((WM_W, WM_H), Image.LANCZOS)
    return _WORDMARK


def blink_source_frame(frame_idx: int) -> int:
    """Map an outro frame index to a source blink-sequence frame index.

    Two blinks; each window is [apex-12.18, apex+12.82] and inside the window
    the source is traversed at 0.641 source-frames per outro frame (source 28
    = fully closed = apex). Outside every window the eye is in the open hold.
    """
    for apex in BLINK_APEX:
        d = frame_idx - apex
        if -BLINK_SPAN_L <= d <= BLINK_SPAN_R:
            src = BLINK_SRC_APEX + d / BLINK_SCALE
            return int(np.clip(round(src), BLINK_SRC_LO, BLINK_SRC_HI))
    return 0


def blink_opacity(frame_idx: int) -> float:
    """Rough lid-closure proxy for QA (1 = shut). Derived from the source
    frame index using the documented 10-24 close / 25-31 hold / 32-46 open."""
    src = blink_source_frame(frame_idx)
    if src <= 9 or src >= 47:
        return 0.0
    if src <= 24:
        return float(np.clip((src - 9) / 15.0, 0, 1))
    if src <= 31:
        return 1.0
    return float(np.clip(1.0 - (src - 31) / 16.0, 0, 1))


# ---------------------------------------------------------------------------
# Text helper: per-char alpha + per-char y offset + tracking + glow
# ---------------------------------------------------------------------------
def text_layer(text: str, font, fill, baseline: float, centre_x: float = W / 2,
               tracking: float = 0.0, char_alpha=None, char_dy=None,
               glow=None, glow_radius: int = 0, glow_alpha: float = 0.5) -> Image.Image:
    """Render one fully-transparent RGBA layer holding ``text``."""
    n = len(text)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if n == 0:
        return layer
    d = ImageDraw.Draw(layer)
    try:
        asc, desc = font.getmetrics()
    except Exception:
        asc, desc = int(font.size * 0.8), int(font.size * 0.2)
    adv = [float(d.textlength(ch, font=font)) for ch in text]
    total = sum(adv) + tracking * max(n - 1, 0)
    x0 = centre_x - total / 2.0
    ca = np.ones(n) if char_alpha is None else np.clip(np.asarray(char_alpha, float), 0, 1)
    cd = np.zeros(n) if char_dy is None else np.asarray(char_dy, float)

    def _paint(target, colour, extra):
        dd = ImageDraw.Draw(target)
        cx = x0
        for i, ch in enumerate(text):
            a = float(ca[i]) * float(extra)
            if a > 0.003:
                dd.text((cx, baseline + float(cd[i])), ch, font=font,
                        fill=tuple(colour) + (int(round(255 * min(a, 1.0))),),
                        anchor="ls")
            cx += adv[i] + tracking

    if glow is not None:
        gl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        _paint(gl, glow, glow_alpha)
        gl = gl.filter(ImageFilter.GaussianBlur(glow_radius or max(2, font.size // 10)))
        layer = Image.alpha_composite(layer, gl)
    core = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    _paint(core, fill, 1.0)
    return Image.alpha_composite(layer, core)


# ---------------------------------------------------------------------------
# Background: base + subtle radial indigo glow behind the eye + vignette
# ---------------------------------------------------------------------------
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)
_RMAX = float(np.sqrt((W * 0.5) ** 2 + (H * 0.5) ** 2))
_R = np.sqrt((_XX - W * 0.5) ** 2 + (_YY - H * 0.5) ** 2).astype(np.float32)
_GLOW = np.exp(-((( _XX - 640.0) / 340.0) ** 2 + ((_YY - 165.0) / 265.0) ** 2)).astype(np.float32)
_VIG = np.clip(1.0 - 0.42 * (_R / _RMAX) ** 2, 0.55, 1.0).astype(np.float32)


def _background(t: float, frame_seed: int) -> np.ndarray:
    img = np.empty((H, W, 3), np.float32)
    img[:, :, 0] = BASE[0]
    img[:, :, 1] = BASE[1]
    img[:, :, 2] = BASE[2]
    breathe = 1.0 + 0.16 * np.sin(2.0 * np.pi * (t % SWEEP_PERIOD) / SWEEP_PERIOD)
    img += _GLOW[:, :, None] * (np.array(INDIGO, np.float32) * (0.105 * breathe))
    img *= _VIG[:, :, None]
    rng = np.random.RandomState((int(frame_seed) & 0xFFFFFFFF) ^ 0x9E3779B9)
    img += rng.randn(H, W, 1).astype(np.float32) * GRAIN
    return img


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------
def render_endcard_frame(frame_idx: int, total: int = TOTAL_DEFAULT,
                         fps: float = FPS) -> Image.Image:
    """Render outro frame ``frame_idx`` of ``total`` (30 fps) as 1280x720 RGB."""
    frame_idx = int(np.clip(int(frame_idx), 0, int(total) - 1))
    freeze_start = max(0, int(total) - FREEZE_FRAMES)

    # ---- clamp all time-varying terms -> last 0.8 s is perfectly still ----
    f = min(frame_idx, freeze_start)
    t = f / float(fps)

    # ---- 1. background ---------------------------------------------------
    canvas = Image.fromarray(np.clip(_background(t, f), 0, 255).astype(np.uint8), "RGB")
    mark = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # ---- 2. eye (real 60-frame blink, resampled to the outro) -----------
    eye_a = float(interpolate(f, F_EYE_IN, (0.0, 1.0), "ease_out"))
    eye_s = 1.07 - 0.07 * float(interpolate(f, F_EYE_SCALE, (0.0, 1.0), "ease_out"))
    eye = _blink_frame(blink_source_frame(f))
    if abs(eye_s - 1.0) > 1e-4:
        ew, eh = max(1, int(round(EYE_W * eye_s))), max(1, int(round(EYE_H * eye_s)))
        eye = eye.resize((ew, eh), Image.LANCZOS)
        ex = EYE_X + (EYE_W - ew) // 2
        ey = EYE_Y + (EYE_H - eh) // 2
    else:
        ex, ey = EYE_X, EYE_Y
    if eye_a < 0.999:
        eye = eye.copy()
        a = np.asarray(eye).astype(np.float32)
        a[:, :, 3] *= eye_a
        eye = Image.fromarray(a.astype(np.uint8), "RGBA")
    mark.alpha_composite(eye, (ex, ey))

    # ---- 3. wordmark rising in with spring overshoot --------------------
    s = float(spring(max(t - F_WM / fps, 0.0), damping=9.5, stiffness=110.0, mass=1.0))
    rise = 46.0
    wm_y = int(round(WM_Y + rise * (1.0 - s)))
    wm_a = float(np.clip((t - F_WM / fps) / 0.16, 0.0, 1.0))
    if wm_a > 0.004:
        wm = _wordmark()
        if wm_a < 0.999:
            wm = wm.copy()
            a = np.asarray(wm).astype(np.float32)
            a[:, :, 3] *= wm_a
            wm = Image.fromarray(a.astype(np.uint8), "RGBA")
        mark.alpha_composite(wm, (WM_X, wm_y))

    # ---- 4. slow indigo light sweep across eye + wordmark ----------------
    phase = (t % SWEEP_PERIOD) / SWEEP_PERIOD
    env = (float(interpolate(f, (0, 4), (0.0, 1.0), "ease_out"))
           * float(interpolate(f, (freeze_start - SWEEP_FADE_OUT, freeze_start),
                               (1.0, 0.0), "ease_in")))
    if env > 0.002:
        arr = np.asarray(mark).astype(np.float32)
        cx = 640.0 - SWEEP_SPAN * math.cos(2.0 * math.pi * phase)
        prof = np.exp(-((_XX + SWEEP_TILT * (_YY - 300.0) - cx) ** 2)
                      / (2.0 * SWEEP_SIGMA ** 2)).astype(np.float32) * env
        col = (np.array(INDIGO_LT, np.float32) * 0.62
               + np.array(CYAN, np.float32) * 0.38)
        # Screen blend: b is a unitless 0..SWEEP_AMOUNT gain on an already
        # 0..255 colour, so the peak added brightness is SWEEP_AMOUNT (28%)
        # of full scale. Screen never clips and never darkens, so the
        # wordmark's silver speculars ride through the sweep cleanly.
        b = (prof * SWEEP_AMOUNT)[:, :, None] * col
        src = arr[:, :, :3] / 255.0
        arr[:, :, :3] = np.clip((1.0 - (1.0 - src) * (1.0 - b / 255.0)) * 255.0, 0, 255)
        mark = Image.fromarray(arr.astype(np.uint8), "RGBA")

    base = np.asarray(canvas, np.float32)
    mk = np.asarray(mark).astype(np.float32)
    al = mk[:, :, 3:4] / 255.0
    out = base * (1.0 - al) + mk[:, :, :3] * al
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    # ---- 5. fine indigo rule --------------------------------------------
    rw = float(interpolate(f, F_RULE, (0.0, 1.0), "ease_out"))
    if rw > 0.004:
        rule = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        rd = ImageDraw.Draw(rule)
        full = RULE_W * rw
        x0 = int(round(RULE_CX - full / 2))
        x1 = int(round(RULE_CX + full / 2))
        rd.rectangle([x0, RULE_CY, x1, RULE_CY + RULE_H - 1],
                     fill=tuple(INDIGO_LT) + (int(round(255 * LAYOUT["rule"]["peak_alpha"])),))
        # taper the ends so the rule reads as a hairline, not a bar
        ra = np.asarray(rule).astype(np.float32)
        xs = np.arange(W, dtype=np.float32)
        taper = np.clip(np.minimum(xs - x0, x1 - xs) / 16.0, 0.0, 1.0)
        taper = taper * taper * (3.0 - 2.0 * taper)
        ra[:, :, 3] *= taper[None, :]
        img = Image.alpha_composite(img.convert("RGBA"),
                                    Image.fromarray(ra.astype(np.uint8), "RGBA")).convert("RGB")

    # ---- 6. tagline (per-char stagger), domain, fineprint ---------------
    for lay in _text_layers(f).values():
        if lay is not None:
            img = Image.alpha_composite(img.convert("RGBA"), lay).convert("RGB")

    return img.convert("RGB")


def _text_layers(f: int) -> Dict[str, Optional[Image.Image]]:
    """The three text layers exactly as composited at clamped frame ``f``."""
    out: Dict[str, Optional[Image.Image]] = {}
    tag_font = _font("inter", int(LAYOUT["tagline"]["size"]),
                     float(LAYOUT["tagline"]["weight"]))
    dom_font = _font("inter", int(LAYOUT["domain"]["size"]),
                     float(LAYOUT["domain"]["weight"]))
    fine_font = _font("inter", int(LAYOUT["fineprint"]["size"]),
                      float(LAYOUT["fineprint"]["weight"]))

    # tagline: per-char stagger + per-char 14 px rise
    p = float(np.clip((f - F_TAG) / 18.6, 0.0, 1.0))          # 0.62 s
    ca = stagger_alphas(len(TAG_TEXT), p, window=0.55)
    out["tagline"] = text_layer(
        TAG_TEXT, tag_font, LAYOUT["tagline"]["fill"],
        float(LAYOUT["tagline"]["baseline"]), char_alpha=ca,
        char_dy=14.0 * (1.0 - ca),
        glow=INDIGO, glow_radius=10, glow_alpha=0.30)

    pa = float(interpolate(f, F_DOM, (0.0, 1.0), "ease_out"))
    out["domain"] = None
    if pa > 0.004:
        out["domain"] = text_layer(
            DOM_TEXT, dom_font, LAYOUT["domain"]["fill"],
            float(LAYOUT["domain"]["baseline"]), tracking=3.0,
            char_alpha=np.full(len(DOM_TEXT), pa),
            char_dy=np.full(len(DOM_TEXT), 11.0 * (1.0 - pa)),
            glow=INDIGO, glow_radius=12, glow_alpha=0.40)

    fa = float(interpolate(f, F_FINE, (0.0, 1.0), "ease_out"))
    out["fineprint"] = None
    if fa > 0.004:
        out["fineprint"] = text_layer(
            FINE_TEXT, fine_font, LAYOUT["fineprint"]["fill"],
            float(LAYOUT["fineprint"]["baseline"]),
            char_alpha=np.full(len(FINE_TEXT), fa),
            char_dy=np.full(len(FINE_TEXT), 9.0 * (1.0 - fa)))
    return out


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------
QA_FRAMES = [0, 20, 43, 50, 70, 85]


def overlap(b1, b2) -> bool:
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or
                b1[3] <= b2[1] or b2[3] <= b1[1])


def qa(out_dir: str = "/tmp/opencode/rebuild") -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    imgs = []
    for i in QA_FRAMES:
        im = render_endcard_frame(i)
        p = os.path.join(out_dir, f"_ec_qa_{i:02d}.jpg")
        im.save(p, quality=93, subsampling=0)
        paths.append(p)
        imgs.append(im)
    # contact sheet 3 x 2 at 50%
    cw, ch = W // 2, H // 2
    sheet = Image.new("RGB", (cw * 3, ch * 2), (0, 0, 0))
    for k, im in enumerate(imgs):
        sheet.paste(im.resize((cw, ch), Image.LANCZOS), ((k % 3) * cw, (k // 3) * ch))
    sp = os.path.join(out_dir, "_ec_sheet.jpg")
    sheet.save(sp, quality=92)
    paths.append(sp)
    return paths


def _boxes(frame_idx: int) -> Dict[str, tuple]:
    """Resolved (settled) boxes used by the overlap audit."""
    s = float(spring(max(frame_idx / FPS - F_WM / FPS, 0.0), 9.5, 110.0, 1.0))
    wm_y = WM_Y + 46.0 * (1.0 - s)
    L = LAYOUT
    return {
        "eye": L["eye"]["box"],
        "wordmark": (WM_X, int(round(wm_y)), WM_X + WM_W, int(round(wm_y)) + WM_H),
        "rule": (RULE_CX - RULE_W // 2, RULE_CY, RULE_CX + RULE_W // 2, RULE_CY + RULE_H),
        "tagline": (0, L["tagline"]["glyph_top"], W, L["tagline"]["glyph_bottom"]),
        "domain": (0, L["domain"]["glyph_top"], W, L["domain"]["glyph_bottom"]),
        "fineprint": (0, L["fineprint"]["glyph_top"], W, L["fineprint"]["glyph_bottom"]),
    }


def audit() -> Dict[str, object]:
    res: Dict[str, object] = {}
    boxes = _boxes(999)          # fully settled
    names = list(boxes)
    bad = [(names[i], names[j]) for i in range(len(names))
           for j in range(i + 1, len(names)) if overlap(boxes[names[i]], boxes[names[j]])]
    res["overlap_pairs"] = bad
    ys = [(n, boxes[n][1], boxes[n][3]) for n in names]
    res["boxes"] = {n: boxes[n] for n in names}
    res["gaps"] = []
    order = ["eye", "wordmark", "rule", "tagline", "domain", "fineprint"]
    for a, b in zip(order, order[1:]):
        res["gaps"].append((f"{a}->{b}", boxes[b][1] - boxes[a][3]))
    res["margins"] = {"top": boxes["eye"][1], "bottom": H - boxes["fineprint"][3]}
    # freeze integrity
    a = np.asarray(render_endcard_frame(62), np.int16)
    same = all(np.array_equal(a, np.asarray(render_endcard_frame(i), np.int16))
               for i in range(62, 86))
    res["freeze_identical_62_85"] = bool(same)
    res["blink"] = {i: (blink_source_frame(i), round(blink_opacity(i), 3))
                    for i in range(0, 86, 1) if blink_opacity(i) > 0.0}
    res["dyn_overlap"] = dynamic_overlap_audit()
    return res


def dynamic_overlap_audit(total: int = TOTAL_DEFAULT) -> List[str]:
    """Ink-level overlap audit across EVERY frame (not just the settled pose).

    Uses each layer's real alpha>8 bounding box at that frame, so the wordmark
    spring overshoot and the per-char stagger offsets are all accounted for.
    """
    bad: List[str] = []
    for fi in range(total):
        f = min(fi, FREEZE_START)
        boxes = {}
        # eye (real rendered alpha incl. its scale settle)
        src = blink_source_frame(f)
        ea = float(interpolate(f, F_EYE_IN, (0.0, 1.0), "ease_out"))
        es = 1.07 - 0.07 * float(interpolate(f, F_EYE_SCALE, (0.0, 1.0), "ease_out"))
        eye = _blink_frame(src)
        if abs(es - 1.0) > 1e-4:
            ew, eh = int(round(EYE_W * es)), int(round(EYE_H * es))
            eye = eye.resize((ew, eh), Image.LANCZOS)
        else:
            ew, eh = EYE_W, EYE_H
        a = np.asarray(eye)[..., 3]
        ys, xs = np.where((a > 8) & (np.asarray(eye)[..., 3] * ea > 8))
        if len(ys):
            boxes["eye"] = (EYE_X + int(xs.min()), EYE_Y + int(ys.min()),
                            EYE_X + int(xs.max()), EYE_Y + int(ys.max()))
        # wordmark
        s = float(spring(max(f / FPS - F_WM / FPS, 0.0), 9.5, 110.0, 1.0))
        wy = WM_Y + 46.0 * (1.0 - s)
        wa = float(np.clip((f / FPS - F_WM / FPS) / 0.16, 0.0, 1.0))
        if wa > 0.031:
            ys, xs = np.where(np.asarray(_wordmark())[..., 3] * wa > 8)
            boxes["wordmark"] = (WM_X + int(xs.min()), int(round(wy)) + int(ys.min()),
                                 WM_X + int(xs.max()), int(round(wy)) + int(ys.max()))
        # rule + text (real ink boxes from the actual rendered layers)
        rw = float(interpolate(f, F_RULE, (0.0, 1.0), "ease_out"))
        if rw > 0.031:
            boxes["rule"] = (RULE_CX - RULE_W * rw / 2, RULE_CY,
                             RULE_CX + RULE_W * rw / 2, RULE_CY + RULE_H)
        for tkey in ("tagline", "domain", "fineprint"):
            lay = _text_layers(f)
            if lay[tkey] is None:
                continue
            ys, xs = np.where(np.asarray(lay[tkey])[..., 3] > 8)
            if len(ys):
                boxes[tkey] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        nm = list(boxes)
        for i in range(len(nm)):
            for j in range(i + 1, len(nm)):
                if overlap(boxes[nm[i]], boxes[nm[j]]):
                    bad.append(f"f{fi}: {nm[i]} x {nm[j]} {boxes[nm[i]]} {boxes[nm[j]]}")
    return bad


def info() -> Dict[str, object]:
    return {
        "layout": LAYOUT,
        "beat_frames": BEAT_FRAMES,
        "blink_apex_frames": BLINK_APEX,
        "blink_window": [[a - BLINK_SPAN_L, a + BLINK_SPAN_R] for a in BLINK_APEX],
        "freeze_start_frame": FREEZE_START,
        "audit": audit(),
    }


def halo_verify(out_dir: str = LOGO_DIR) -> Dict[str, object]:
    """Re-verify the wordmark + eye halo (magenta and black comps, edge scan).

    Method: straight-alpha composite each RGBA asset over pure magenta and pure
    black, then inspect semi-transparent edge pixels (0.05 < a < 0.95). A white
    halo/fringe over magenta shows up as edge pixels whose min(R,G,B) ~= 250.
    """
    os.makedirs(out_dir, exist_ok=True)
    res: Dict[str, object] = {}
    targets = {
        "wordmark": Image.open(f"{LOGO_DIR}/wordmark.png").convert("RGBA"),
        "eye_open": _blink_frame(0),
        "eye_closed": _blink_frame(28),
    }
    for name, im in targets.items():
        arr = np.asarray(im).astype(np.float32)
        a = arr[..., 3] / 255.0
        comps = {}
        for bg_name, bgc in (("magenta", (255, 0, 255)), ("black", (0, 0, 0))):
            comp = arr[..., :3] * a[..., None] + np.array(bgc, np.float32) * (1 - a[..., None])
            ci = np.clip(comp, 0, 255).astype(np.uint8)
            Image.fromarray(ci, "RGB").save(f"{out_dir}/_v2_{name}_{bg_name}.png")
            edge = (a > 0.05) & (a < 0.95)
            n = int(edge.sum())
            minch = ci.min(axis=2)[edge] if n else np.array([0])
            comps[bg_name] = {
                "edge_px": n,
                "max_min_channel": int(minch.max()) if n else 0,
                "p99_min_channel": int(np.percentile(minch, 99)) if n else 0,
                "n_over_245": int((minch > 245).sum()) if n else 0,
            }
        # crop-border clipping check (glow must not be cut off)
        al = (arr[..., 3] > 0.003)
        border = max(int(arr[0, :, 3].max()), int(arr[-1, :, 3].max()),
                     int(arr[:, 0, 3].max()), int(arr[:, -1, 3].max()))
        ys, xs = np.where(al)
        res[name] = {
            "size": list(im.size),
            "max_border_alpha": border,
            "glow_extent_a0.003": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "margin_to_border": [int(xs.min()), int(ys.min()),
                                 int(im.size[0] - 1 - xs.max()),
                                 int(im.size[1] - 1 - ys.max())],
            **comps,
        }
    # endcard-space eye: the exact 400x250 layer the outro composites
    res["endcard_eye_layer"] = {"size": [EYE_W, EYE_H],
                                "placed_box": [EYE_X, EYE_Y, EYE_X1, EYE_Y1]}
    return res


def render_sequence(out_dir: str = "/tmp/opencode/rebuild/frames_endcard",
                    total: int = TOTAL_DEFAULT) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for i in range(total):
        p = os.path.join(out_dir, f"endcard_{i:03d}.png")
        render_endcard_frame(i, total=total).save(p)
        out.append(p)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "qa"
    if mode == "qa":
        for p in qa():
            print("wrote", p)
    elif mode == "render":
        for p in render_sequence():
            print("wrote", p)
    elif mode == "halo":
        import json
        print(json.dumps(halo_verify(), indent=2, default=str))
    else:
        import json
        print(json.dumps(info(), indent=2, default=str))