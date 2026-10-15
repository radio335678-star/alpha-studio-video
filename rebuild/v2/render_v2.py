"""ALPHA promo v2 — beat-locked procedural renderer. 1280x720@30, 750 frames.

Every pixel is generated: backgrounds (halftone tunnel/rings/grid/waves/particles),
colour grade (opponent-space indigo pull), typography (gradient core + rim light +
two-stage glow + separation ring). Cuts land on downbeats/hits measured from the
track; line reveals are snapped to the beat grid. Audio is muxed untouched.
"""
import json, os, sys, math
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E
import beatgrid as B
import grade as G
import typography as T
import endcard as EC

W, H, FPS, NFRAMES = 1280, 720, 30, 750
OUT_DIR = os.path.join(HERE, "frames_v2")
TRANS = 9  # overlap frames (0.3 beat) — whip travel ~ half a beat

BG = {"tunnel": E.bg_tunnel, "rings": E.bg_rings, "grid": E.bg_grid,
      "particles": E.bg_particles, "solid": E.bg_solid, "waves": E.bg_waves}

SCENES = json.load(open(os.path.join(HERE, "scenes_v2.json")))
WHITE = (245, 243, 255)

# ── per-scene layout spec ───────────────────────────────────────────────────
# hero_first: first line gets the display size, rest the body size (the stat scene)
# eyebrow: optional small mono kicker above the block
# stat: big Space Grotesk numeral as line 0
LAYOUTS = {
    "hook":            dict(size=92,  lh=1.14, cy=330, x=640,  anchor="mm"),
    "hero":            dict(size=94,  lh=1.14, cy=330, x=640,  anchor="mm"),
    "sees":            dict(size=88,  lh=1.16, cy=340, x=120,  anchor="lm"),
    "plans":           dict(size=88,  lh=1.16, cy=340, x=120,  anchor="lm"),
    "always-on":       dict(size=90,  lh=1.15, cy=330, x=640,  anchor="mm"),
    "connect":         dict(size=84,  lh=1.16, cy=330, x=640,  anchor="mm"),
    "proof":           dict(size=92,  lh=1.16, cy=330, x=640,  anchor="mm", stat=True),
    "triplet":         dict(size=88,  lh=1.16, cy=330, x=640,  anchor="mm"),
    "while-you-sleep": dict(size=84,  lh=1.16, cy=340, x=120,  anchor="lm"),
    "closer":          dict(size=100, lh=1.14, cy=330, x=640,  anchor="mm"),
}
EYEBROWS = {"connect": "PLUG IN ONCE", "proof": "PROOF", "triplet": "ONE NIGHT SHIFT",
            "always-on": "ALWAYS ON"}

def clamp01(x):
    return float(max(0.0, min(1.0, x)))

def scene_index_at(t):
    for i, sc in enumerate(SCENES):
        if sc["start"] <= t < sc["end"]:
            return i
    return len(SCENES) - 1

def scene_index_at_frame(f):
    t = f / FPS
    for i, sc in enumerate(SCENES):
        f0 = int(round(sc["start"] * FPS)); f1 = int(round(sc["end"] * FPS))
        if f0 <= f < f1:
            return i
    return len(SCENES) - 1

def snap_f(scene_f0, i):
    """Line-reveal frame offsets locked to the beat grid after scene start."""
    return B.snapped_onsets(scene_f0 + 1, spacing=1, count=5)

def text_content(img, idx, t_abs):
    sc = SCENES[idx]
    lay = LAYOUTS.get(sc["id"], dict(size=88, lh=1.16, cy=340, x=640, anchor="mm"))
    f0 = int(round(sc["start"] * FPS))
    local_f = max(0, int(round((t_abs - sc["start"]) * FPS)))
    local = local_f / FPS
    n = len(sc["lines"])
    f0 = int(round(sc["start"] * FPS))
    size0 = lay["size"]
    lines = sc["lines"]
    stat = lay.get("stat", False)
    lh = size0 * lay["lh"]
    ey = sc.get("eyebrow") or EYEBROWS.get(sc["id"])
    cy = lay["cy"] - (34 if ey else 0)
    y_top = cy - (n - 1) * lh / 2
    if stat:
        y_top -= 26  # the display numeral is much taller than its line box
    x, anchor = lay["x"], lay["anchor"]
    centered = anchor == "mm"
    accent_anchor = "mm" if centered else "lm"

    # ── plate (luminance-adaptive scrim) under the whole block ──────────────
    if ey:
        T.eyebrow(img, (x, y_top - 74), ey, size=20, t=local, delay=0.0, anchor=accent_anchor)
    if stat:
        boxes = [T.stat_box((x, y_top - 6), lines[0], size=150, anchor=accent_anchor)]
        for i in range(1, n):
            boxes.append(T.headline_box((x, y_top + i * lh - 6), lines[i], size=int(size0), anchor=accent_anchor))
    else:
        boxes = T.headline_block_boxes((x, y_top), lines,
                                       sizes=[size0] * n,
                                       line_height=lay["lh"], anchor=accent_anchor)
    yb = max(b[3] for b in boxes)
    if sc.get("sub"):
        boxes = boxes + [T.sub_box((x, yb + 52), sc["sub"], size=30, anchor=accent_anchor)]
    T.plate_all(img, boxes, auto=True, ceil=0.55, target_p95=88.0, pad=(70, 44))

    # ── lines, reveals snapped to beats (first line reveals on frame 0) ────
    # Beats sit on frames ≡ 2 (mod 15). Every scene starts on a beat frame, so
    # line i reveals exactly i beats (15 frames) after the cut — no drift.
    onsets = [(i * 15 + 2 - f0) % 15 for i in range(n)]
    for i, line in enumerate(lines):
        d_f = onsets[i] if i < len(onsets) else i * 15
        tt = local - d_f / FPS
        if tt <= 0:
            continue
        y_c = y_top + i * lh - 6
        if stat and i == 0:
            T.stat_number(img, (x, y_c), line, size=150, t=tt, anchor=accent_anchor)
            continue
        accent_words = ["Alpha"] if "Alpha" in line else None
        T.headline(img, (x, y_c), line, size=int(size0), t=tt, anchor=accent_anchor,
                   delay=0.0, accent=accent_words, stagger_dur=0.26, stagger_window=0.45)
    # sub last, one beat after the final line
    if sc.get("sub") and local > 0.45:
        d_sub = onsets[min(n - 1, len(onsets) - 1)] + 15
        T.sub_line(img, (x, yb + 52), sc["sub"], size=30,
                   t=local - d_sub / FPS, anchor=accent_anchor,
                   rule=centered, delay=0.0)
    return img

def render_content(idx, t_abs):
    sc = SCENES[idx]
    f = int(round(t_abs * FPS)); f = max(0, min(NFRAMES - 1, f))
    e = float(B.energy_at(f))
    if sc["id"] == "endcard":
        f0 = int(round(sc["start"] * FPS))
        local = max(0, f - f0)
        total = int(round(sc["end"] * FPS)) - f0
        return np.asarray(EC.render_endcard_frame(local, total=total, fps=FPS).convert("RGB"))
    raw = BG[sc["bg"]](W, H, t=t_abs, energy=e, frame=f)
    img = Image.fromarray(G.indigo_look(raw, t_abs, e))
    return np.asarray(text_content(img, idx, t_abs).convert("RGB"))

def render_frame(f):
    t = f / FPS
    idx = scene_index_at_frame(f)
    sc = SCENES[idx]
    f0 = int(round(sc["start"] * FPS))
    local_f = f - f0
    if idx > 0 and local_f < TRANS:
        k = local_f / TRANS
        prev = SCENES[idx - 1]
        t_a = f / FPS - k * (TRANS / FPS)
        a = render_content(idx - 1, t_a)
        b = render_content(idx, t)
        accent = bool(sc.get("accent")) or bool(prev.get("accent")) or bool(B.hit_pulse(f, 30) > 0.02)
        kind = "whip" if accent else "crossfade"
        kk = (1 - (1 - k) ** (1 / 3)) if kind == "whip" else k
        frame = E.transition_blend(a, b, kk, kind=kind)
    else:
        frame = render_content(idx, t)
    # ── beat-synced camera: downbeat breathe, beat micro-push, hit impact ──
    bar, beat, hit = B.bar_pulse(f), B.beat_pulse(f), B.hit_pulse(f)
    e = float(B.energy_at(f))
    sec = B.section_at(t)
    z = 1 + 0.016 * bar + 0.007 * beat + 0.030 * hit + 0.010 * e
    if z > 1.0015:
        frame = E._zoom_crop(frame, z)
    fl = 0.10 * hit + 0.02 * bar
    if fl > 0.004:
        frame = np.clip(frame.astype(np.float32) * (1 - fl) + 255 * fl, 0, 255).astype(np.uint8)
    # scene-wide punch on accent scenes: ~3% contrast kick that eases out over ~0.7s
    if sc.get("accent"):
        tl = max(0.0, (t - sc["start"]))
        k = math.exp(-tl / 0.40) * 0.030 + 0.028 * hit
        if k > 0.003:
            m = frame.astype(np.float32).mean()
            frame = np.clip((frame.astype(np.float32) - m) * (1 + 3 * k) + m, 0, 255).astype(np.uint8)
    return frame

def one_frame(f):
    import warnings; warnings.filterwarnings("ignore")
    Image.fromarray(render_frame(f)).save(os.path.join(OUT_DIR, f"f{f:04d}.jpg"), quality=93)
    return f

if __name__ == "__main__":
    import time
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    only = sys.argv[1:] and [int(x) for x in sys.argv[1:]]
    if only:
        for f in only:
            one_frame(f)
        print("single frames done")
    else:
        with ProcessPoolExecutor(max_workers=4) as ex:
            done = 0
            for _ in ex.map(one_frame, range(NFRAMES), chunksize=6):
                done += 1
                if done % 75 == 0:
                    print(f"  {done}/{NFRAMES} ({time.time()-t0:.0f}s)", flush=True)
        print(f"ALL {NFRAMES} frames in {time.time()-t0:.0f}s")