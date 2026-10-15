"""ALPHA promo rebuild — master renderer. Every pixel procedural, no source pixels.
1280x720 @ 30fps, 750 frames. Beat-synced camera + entrances. Run: python3 render.py"""
import json, os, sys, math
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E

W, H, FPS, NFRAMES = 1280, 720, 30, 750
OUT_DIR = os.path.join(HERE, "frames")
FF = "/usr/local/lib/python3.10/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

WHITE = (245, 243, 255)
ALPHA_CYAN = (150, 220, 255)
SUB_FILL = (196, 186, 255)
DOMAIN_FILL = (150, 120, 255)
FINE_FILL = (154, 151, 176)
TRANS = 10  # overlap frames

BG = {"tunnel": E.bg_tunnel, "rings": E.bg_rings, "grid": E.bg_grid,
      "particles": E.bg_particles, "solid": E.bg_solid, "waves": E.bg_waves}

SCENES = json.load(open(os.path.join(HERE, "scenes.json")))
BEATS = json.load(open(os.path.join(HERE, "beats.json")))
ENERGY = BEATS["energy"]
HIT_F = [int(round(h * FPS)) for h in BEATS["hits"]]
BEAT_F = [int(round(b * FPS)) for b in BEATS["beats"]]

LOGO = os.path.join(HERE, "logo")
_eye_open = _eye_closed = _wordmark = None

def assets():
    global _eye_open, _eye_closed, _wordmark
    if _eye_open is None:
        _eye_open = Image.open(os.path.join(LOGO, "eye_open.png")).convert("RGBA").resize((400, 250), Image.LANCZOS)
        _eye_closed = Image.open(os.path.join(LOGO, "eye_closed.png")).convert("RGBA").resize((400, 250), Image.LANCZOS)
        _wordmark = Image.open(os.path.join(LOGO, "wordmark.png")).convert("RGBA").resize((270, 225), Image.LANCZOS)
    return _eye_open, _eye_closed, _wordmark

def clamp01(x):
    return max(0.0, min(1.0, x))

def pulse(f, centers, decay):
    p = 0.0
    for c in centers:
        d = f - c
        if 0 <= d <= 24:
            p = max(p, math.exp(-d / decay))
    return p

INDIGO_GRADE = np.array([1.12, 0.93, 1.10], dtype=np.float32)

def grade_bg(frame):
    g = frame.astype(np.float32) * INDIGO_GRADE
    return np.clip(g, 0, 255).astype(np.uint8)

def text_block(img, lines, local, layout, font="tight", base_size=None, line_delay=0.15, hero_first=False):
    """Draw headline block with cascaded per-line entrances. Returns img."""
    n = len(lines)
    def _sz(i):
        if base_size:
            return base_size
        if hero_first and i == 0:
            return 108
        return {1: 124, 2: 100, 3: 82}[min(n, 3)]
    size0 = _sz(0)
    lh = size0 * 1.16
    cy = 335 if n == 3 else 350
    y_top = cy - (n - 1) * lh / 2
    x = 640 if layout == "center" else 120
    anchor = "mm" if layout == "center" else "lm"
    scr_c = (640, 360) if layout == "center" else (430, 360)
    E.add_scrim(img, center=scr_c, axes=(560, 200), strength=0.62)
    yb = y_top
    for i, line in enumerate(lines):
        lt = local - i * line_delay
        if lt <= 0:
            continue
        size = _sz(i)
        y_c = y_top + i * lh + (size0 - size) / 2
        yb = y_c + size / 2
        s = E.spring(lt, damping=14, stiffness=170, mass=1)  # snappy, slight overshoot
        a = clamp01(s)
        y_off = 60 * (1 - clamp01(s))
        prog = clamp01(lt / 0.45)
        ca = E.stagger_alphas(len(line), prog, window=0.55)
        fill = ALPHA_CYAN if "Alpha" in line else WHITE
        glow = E.CYAN if "Alpha" in line else E.INDIGO
        fnt = "grotesk" if (len(line) > 12 and any(ch.isdigit() for ch in line)) else font
        E.draw_text(img, (x, y_c + y_off), line, font=fnt, size=size,
                    fill=fill, anchor=anchor, glow=glow, alpha=a, char_alphas=ca)
    return yb

def draw_sub(img, sub, local, layout, y_bottom):
    if not sub or local < 0.85:
        return
    a = clamp01((local - 0.85) / 0.35)
    x = 640 if layout == "center" else 120
    anchor = "mm" if layout == "center" else "lm"
    E.draw_text(img, (x, y_bottom + 64), sub, font="inter", size=33,
                fill=SUB_FILL, anchor=anchor, glow=E.INDIGO, alpha=a, spacing=2)

def render_content(idx, t_abs):
    """Full scene content (bg + text) at absolute time t_abs. No camera/flash."""
    sc = SCENES[idx]
    local = t_abs - sc["start"]
    e = ENERGY[max(0, min(NFRAMES - 1, int(round(t_abs * FPS))))]
    frame = grade_bg(BG[sc["bg"]](W, H, t=t_abs, energy=float(e), frame=int(round(t_abs * FPS))))
    img = Image.fromarray(frame)
    if sc["id"] == "endcard":
        return render_endcard(img, t_abs, local, e)
    yb = text_block(img, sc["lines"], local, sc["layout"], hero_first=(sc["id"] == "proof"))
    draw_sub(img, sc.get("sub"), local, sc["layout"], yb)
    out = np.asarray(img.convert("RGB"))
    return out

def render_endcard(img, t_abs, local, e):
    eye_o, eye_c, wm = assets()
    base = img.convert("RGBA")
    # eye with blink dips at 22.35 and 23.65
    blink = 0.0
    for bc in (22.35, 23.65):
        d = abs(t_abs - bc)
        if d < 0.16:
            blink = max(blink, 1 - d / 0.16)
    ea = clamp01(E.spring(max(0.0, local), damping=10, stiffness=100))
    eye = Image.blend(eye_o, eye_c, clamp01(blink))
    if ea < 1.0:
        eye = eye.copy()
        ea_arr = eye.split()[3].point(lambda v: int(v * ea))
        eye.putalpha(ea_arr)
    base.alpha_composite(eye, (440, 36))
    # wordmark rise
    wl = local - 0.2
    if wl > 0:
        s = E.spring(wl, damping=14, stiffness=170)
        w = wm.copy()
        wa = w.split()[3].point(lambda v: int(v * clamp01(s)))
        w.putalpha(wa)
        base.alpha_composite(w, (505, 300 + int(40 * (1 - clamp01(s)))))
    # tagline stagger
    tl = local - 0.5
    if tl > 0:
        ca = E.stagger_alphas(len("Say it. It's built."), clamp01(tl / 0.6), 0.55)
        E.draw_text(base, (640, 562), "Say it. It's built.", font="inter", size=36,
                    fill=WHITE, anchor="mm", alpha=clamp01(tl / 0.4), char_alphas=ca)
    if local > 0.9:
        E.draw_text(base, (640, 610), "alpha.studio", font="inter", size=27,
                    fill=DOMAIN_FILL, anchor="mm", glow=E.INDIGO, alpha=clamp01((local - 0.9) / 0.4), spacing=3)
    if local > 1.3:
        E.draw_text(base, (640, 662), "You approve everything. It never spends on its own.",
                    font="inter", size=20, fill=FINE_FILL, anchor="mm", alpha=clamp01((local - 1.3) / 0.4))
    return np.asarray(base.convert("RGB"))

def scene_at(t):
    for i, sc in enumerate(SCENES):
        if sc["start"] <= t < sc["end"] or (i == len(SCENES) - 1 and t >= sc["start"]):
            return i
    return len(SCENES) - 1

def render_frame(f):
    t = f / FPS
    idx = scene_at(t)
    sc = SCENES[idx]
    local_f = int(round((t - sc["start"]) * FPS))
    if idx > 0 and local_f < TRANS:
        k = local_f / TRANS
        sc_prev, sc_now = SCENES[idx - 1], sc
        t_a = sc_prev["end"] - (TRANS - local_f) / FPS
        a = render_content(idx - 1, t_a)
        b = render_content(idx, t)
        kind = "whip" if (sc_prev.get("accent") or sc_now.get("accent")) else "crossfade"
        try:
            kk = (1 - (1 - k) ** (1 / 3)) if kind == "whip" else k  # linearize whip travel
            frame = E.transition_blend(a, b, kk, kind=kind)
        except TypeError:
            frame = E.transition_blend(a, b, k)
    else:
        frame = render_content(idx, t)
    # beat-synced camera pulse + hit flash
    ph, pb = pulse(f, HIT_F, 5.0), pulse(f, BEAT_F, 4.0)
    e = ENERGY[f]
    z = 1 + 0.030 * ph + 0.012 * pb + 0.008 * e
    if z > 1.001:
        frame = E._zoom_crop(frame, z)
    fl = 0.12 * pulse(f, HIT_F, 1.5)
    if fl > 0.004:
        frame = np.clip(frame.astype(np.float32) * (1 - fl) + 255 * fl, 0, 255).astype(np.uint8)
    return frame

def one_frame(f):
    import warnings
    warnings.filterwarnings("ignore")
    frame = render_frame(f)
    Image.fromarray(frame).save(os.path.join(OUT_DIR, f"f{f:04d}.jpg"), quality=93)
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
            for _ in ex.map(one_frame, range(NFRAMES), chunksize=8):
                done += 1
                if done % 75 == 0:
                    print(f"  {done}/{NFRAMES} ({time.time()-t0:.0f}s)", flush=True)
        print(f"ALL {NFRAMES} frames in {time.time()-t0:.0f}s")
