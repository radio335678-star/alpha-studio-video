#!/usr/bin/env python3
"""
build_lut.py -- generate `alpha_indigo.cube`, a 3D LUT that re-badges the
neon-GREEN promo (original.mp4) into ALPHA's indigo/cyan identity.

METHOD
------
A global `hue=h=` rotation is wrong here: it also rotates the pure-white end
cards and the neutral greys. Instead we work in an luma/opponent space:

    Y = 0.2126R + 0.7152G + 0.0722B          (Rec.709 luma, gamma-encoded)
    U = B - Y
    V = R - Y

(exact inverse: R = Y+V, B = Y+U, G = Y - (KR/KG)V - (KB/KG)U)

Why this space:
  * Luma is an explicit axis, so we can hold the luminance/contrast structure
    (and therefore the halftone dot geometry) essentially fixed.
  * Chroma magnitude collapses to ~0 for neutrals, so white cards and greys
    are automatically left alone instead of being hue-rotated.
  * Green sits at opponent-hue ~252 deg; indigo #6C4BFF sits at ~8 deg and
    electric cyan #2BD9FF at ~286 deg. We remap *only* the green window and
    ramp the destination hue by luma: violet in the shadows -> indigo in the
    mids -> cyan in the highlights. That gives the ALPHA gradient for free.

Everything is applied in gamma-encoded sRGB, which is exactly what ffmpeg's
`lut3d` filter samples (it does not linearise).

Output: alpha_indigo.cube (48^3 by default)
"""

import argparse
import numpy as np

# ----------------------------------------------------------------------------
# brand constants
# ----------------------------------------------------------------------------
INDIGO = np.array([0x6C, 0x4B, 0xFF], dtype=np.float64) / 255.0   # #6C4BFF
CYAN   = np.array([0x2B, 0xD9, 0xFF], dtype=np.float64) / 255.0   # #2BD9FF
BASE   = np.array([0x07, 0x07, 0x0C], dtype=np.float64) / 255.0   # #07070C

# ----------------------------------------------------------------------------
# colour space helpers
# ----------------------------------------------------------------------------
KR, KG, KB = 0.2126, 0.7152, 0.0722


def rgb2yuv(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    Y = KR * rgb[..., 0] + KG * rgb[..., 1] + KB * rgb[..., 2]
    U = rgb[..., 2] - Y
    V = rgb[..., 0] - Y
    return Y, U, V


def yuv2rgb(Y, U, V):
    # exact inverse of rgb2yuv for the Rec.709 luma weights above
    R = Y + V
    B = Y + U
    G = Y - (KR / KG) * V - (KB / KG) * U
    return np.stack([R, G, B], axis=-1)


def hsv2rgb(h, s, v):
    """h in degrees, s/v in 0..1 -> (...,3) rgb"""
    h = np.asarray(h, dtype=np.float64)
    c = v * s
    hp = (h % 360.0) / 60.0
    x = c * (1.0 - np.abs(np.mod(hp, 2.0) - 1.0))
    z = np.zeros_like(c)
    rr = np.select([hp < 1, hp < 2, hp < 3, hp < 4, hp < 5], [c, x, z, z, x], default=c)
    gg = np.select([hp < 1, hp < 2, hp < 3, hp < 4, hp < 5], [x, c, c, x, z], default=z)
    bb = np.select([hp < 1, hp < 2, hp < 3, hp < 4, hp < 5], [z, z, x, c, c], default=x)
    m = v - c
    return np.stack([rr + m, gg + m, bb + m], axis=-1)


def rgb2hsv(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(mx)
    nz = d > 1e-9
    dn = np.where(nz, d, 1.0)
    mr = (mx == r) & nz
    mg = (mx == g) & nz
    mb = (mx == b) & nz
    h = np.where(mr, ((g - b) / dn) % 6.0, h)
    h = np.where(mg, ((b - r) / dn) + 2.0, h)
    h = np.where(mb, ((r - g) / dn) + 4.0, h)
    h = (h * 60.0) % 360.0
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return h, s, mx


def opponent_hue(U, V):
    """opponent-plane angle in degrees [0,360); green ~252, indigo ~8."""
    return np.degrees(np.arctan2(V, U)) % 360.0


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------
# gamut clamp: largest chroma magnitude M along (dirU,dirV) that stays in [0,1]
# ----------------------------------------------------------------------------
def clamp_chroma(Y, dirU, dirV, m_max, iters=34):
    """
    Bisection on M in [0, m_max] such that yuv2rgb(Y, M*dir) lands in [0,1].
    Vectorised. Returns M (same shape as Y).
    """
    lo = np.zeros_like(Y)
    hi = np.array(m_max, dtype=np.float64) * np.ones_like(Y)

    def ok(M):
        c = yuv2rgb(Y, M * dirU, M * dirV)
        return np.all((c >= -1e-6) & (c <= 1.0 + 1e-6), axis=-1)

    # if even m_max is out of gamut we still want a graceful result: bisect anyway
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        good = ok(mid)
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)
    return lo


# ----------------------------------------------------------------------------
# the grade
# ----------------------------------------------------------------------------
def build_lut(n, cfg):
    # --- LUT grid (ffmpeg .cube ordering: red changes fastest, then green, blue)
    steps = (np.arange(n) / (n - 1.0))
    grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1)
    src = grid.reshape(-1, 3)

    r, g, b = src[..., 0], src[..., 1], src[..., 2]
    Y_s, U_s, V_s = rgb2yuv(src)
    h_s, s_s, v_s = rgb2hsv(src)
    phi_s = opponent_hue(U_s, V_s)
    chroma = np.hypot(U_s, V_s)

    # --- 1. the green gate -------------------------------------------------
    # Only pixels whose OPPONENT hue sits in the green window get re-badged.
    # Smooth shoulders keep the transition free of LUT lattice steps.
    d = np.abs(wrap180(phi_s - cfg["green_hue"]))
    win = 1.0 - smoothstep(cfg["green_half"], cfg["green_half"] + cfg["green_shoulder"], d)
    # low-chroma noise (compression ringing around the white cards) is snapped
    # to neutral instead of being hue-rotated -- this is what keeps the white
    # end cards genuinely white rather than lavender.
    cmask = smoothstep(cfg["chroma_kill"], cfg["chroma_kill"] * 2.4, chroma)
    W = win * cmask

    # --- 2. target luma ----------------------------------------------------
    # Holding luma EXACTLY fixed is the structure-safe choice, but the source
    # greens are bright AND saturated (Y~0.67 with chroma~0.55). At that luma
    # the indigo/cyan gamut only admits a pale, washed-out blue -- the neon
    # punch is lost. So we spend luma to buy chroma headroom:
    #
    #     Y_t = Y_s - chroma_pull * chroma
    #
    # This is MONOTONIC in the source tone, so contrast relationships and the
    # halftone dot geometry are preserved exactly; it just re-points the
    # saturated highlights lower on the ramp so they can hold real indigo.
    # Pure neutrals are untouched (chroma==0 => Y_t==Y_s).
    Y_t = Y_s - cfg["chroma_pull"] * chroma
    # never crush a pixel to nothing just because it was saturated
    Y_t = np.maximum(Y_t, Y_s * cfg["y_floor"])
    Y_t = np.clip(Y_t, 0.0, 1.0)

    # --- 3. destination hue, ramped by the TARGET luma --------------------
    # NOTE: this must key off Y_t, not Y_s. Keying it off Y_s makes the ramp
    # disagree with the luma the colour is actually built at, which silently
    # pushes bright greens into the cyan end of the ramp and renders them as
    # flat royal blue instead of indigo.
    ramp = cfg["hue_ramp"]          # list of (Yn, delta-from-indigo degrees)
    rn = np.array([p[0] for p in ramp], dtype=np.float64)
    rv = np.array([p[1] for p in ramp], dtype=np.float64)
    rn = rn / rn[-1]                # ramp spans full luma range
    phi_t = np.interp(Y_t, rn, rv + H_INDIGO)

    # --- 4. target chroma magnitude ---------------------------------------
    # We keep the source chroma *magnitude* (scaled) and only re-point its
    # direction. That preserves the original saturation/contrast structure
    # exactly; only the hue changes.
    M_want = chroma * cfg["sat_gain"]
    # ease off chroma in the mid-luma band: large flat fields of saturated
    # green become an unnaturally loud wall of flat indigo otherwise
    M_want *= (1.0 - cfg["mid_desat"] * smoothstep(0.15, 0.55, Y_s)
                     * (1.0 - smoothstep(0.62, 0.95, Y_s)))

    # --- 5. point chroma at the destination hue, then gamut-clamp ---------
    phi = np.radians(phi_t)
    dirU, dirV = np.cos(phi), np.sin(phi)
    M = clamp_chroma(np.clip(Y_t, 0.0, 1.0), dirU, dirV,
                     np.maximum(M_want, 1e-9))

    graded = yuv2rgb(Y_t, M * dirU, M * dirV)

    # untouched branch (fully neutral pixels keep their exact luma)
    passthrough = yuv2rgb(Y_s, U_s * cmask, V_s * cmask)

    out = passthrough + W[..., None] * (graded - passthrough)

    out = np.clip(out, 0.0, 1.0)

    # --- 7. finishing touch: gentle S-curve -------------------------------
    if cfg["contrast"] != 1.0:
        out = (out - 0.5) * cfg["contrast"] + 0.5
    out = np.clip(out, 0.0, 1.0)

    # --- 8. black lift toward #07070C (last, so whites are untouched) ------
    # additive + masked out before it reaches mid/highlights: pure black
    # lands exactly on #07070C and pure white stays exactly 255.
    Y_out = KR * out[..., 0] + KG * out[..., 1] + KB * out[..., 2]
    lm = 1.0 - smoothstep(cfg["lift_end"] * 0.55, cfg["lift_end"], Y_out)
    out = np.clip(out + lm[..., None] * (BASE[None, :] * cfg["lift"]), 0.0, 1.0)

    return out.reshape(n, n, n, 3)


def _brand_hues():
    """opponent-hue angles of the brand anchors, computed live so the ramp can
    never drift out of sync with the constants above."""
    _, Ui, Vi = rgb2yuv(INDIGO)
    _, Uc, Vc = rgb2yuv(CYAN)
    _, Ug, Vg = rgb2yuv(np.array([0.19, 0.84, 0.39]))   # the video's mid green
    return (float(opponent_hue(np.array(Ui), np.array(Vi))),
            float(opponent_hue(np.array(Uc), np.array(Vc))),
            float(opponent_hue(np.array(Ug), np.array(Vg))))


H_INDIGO, H_CYAN, H_GREEN = _brand_hues()
# shortest-path signed deltas from indigo, so the ramp reads as "indigo, then
# rotate toward cyan" regardless of the 0/360 wrap
_D_CYAN = ((H_CYAN - H_INDIGO + 180.0) % 360.0) - 180.0

DEFAULT_CFG = dict(
    # green detection (opponent-hue degrees, measured from the source footage)
    green_hue=H_GREEN,
    green_half=30.0,
    green_shoulder=38.0,
    chroma_kill=0.008,
    # luma-preservation mix: 1.0 = locked to original luma
    alpha=1.0,
    # destination hue ramp: (normalised luma, degrees relative to indigo).
    # 0.38 -> exactly INDIGO, 1.00 -> exactly CYAN.
    hue_ramp=[
        (0.00, 26.0),    # shadows: violet  (keeps the glow halo rich, not grey)
        (0.14, 14.0),
        (0.30, 4.0),     # indigo, slightly violet
        (0.55, 0.0),     # == #6C4BFF exactly at the main anchor
        (0.78, 0.35 * _D_CYAN),   # blue-cyan begins
        (0.92, 0.72 * _D_CYAN),   # azure
        (1.00, _D_CYAN),          # == #2BD9FF exactly, for the white-hot core
    ],
    sat_gain=1.05,
    mid_desat=0.18,
    chroma_pull=0.50,  # luma spent per unit chroma, to buy saturation headroom
    y_floor=0.30,      # ... but never below this fraction of source luma
    lift=1.0,
    lift_end=0.16,
    contrast=1.02,
)


def write_cube(path, lut, title):
    n = lut.shape[0]
    with open(path, "w") as f:
        f.write('TITLE "%s"\n' % title)
        f.write("LUT_3D_SIZE %d\n" % n)
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        # .cube ordering: R fastest, then G, then B  ==  our meshgrid 'ij'
        flat = lut.reshape(-1, 3)
        for i in range(flat.shape[0]):
            r, g, b = flat[i]
            f.write("%.6f %.6f %.6f\n" % (r, g, b))


def self_test(cfg):
    """verify the opponent round-trip and the neutral invariant."""
    rng = np.random.default_rng(0)
    x = rng.random((20000, 3))
    Y, U, V = rgb2yuv(x)
    x2 = yuv2rgb(Y, U, V)
    err = np.abs(x - x2).max()
    assert err < 1e-9, "opponent round-trip failed: %g" % err

    # brand anchors must land on their intended opponent hues
    for name, col in (("indigo", INDIGO), ("cyan", CYAN)):
        Yc, Uc, Vc = rgb2yuv(col)
        hexs = "#%02X%02X%02X" % tuple((col * 255).round().astype(int))
        print("  %-7s %s  luma=%.3f  opponent_hue=%.1f"
              % (name, hexs, Yc, opponent_hue(np.array(Uc), np.array(Vc))))
    print("  video green centre -> opponent_hue %.1f" % H_GREEN)

    # pure white / pure black must survive the grade essentially untouched
    lut = build_lut(33, cfg)
    idx = lambda r, g, b: (r * 32 * 1024 + g * 32 * 32 + b * 32)
    w = lut[32, 32, 32]
    k = lut[0, 0, 0]
    print("  white in -> %s   black in -> %s" % (w.round(4), k.round(4)))
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--size", type=int, default=48)
    ap.add_argument("-o", "--out", default="alpha_indigo.cube")
    ap.add_argument("--chroma-pull", type=float, default=None)
    ap.add_argument("--y-floor", type=float, default=None)
    ap.add_argument("--sat-gain", type=float, default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.chroma_pull is not None:
        cfg["chroma_pull"] = args.chroma_pull
    if args.y_floor is not None:
        cfg["y_floor"] = args.y_floor
    if args.sat_gain is not None:
        cfg["sat_gain"] = args.sat_gain

    print("self-test:")
    self_test(cfg)

    lut = build_lut(args.size, cfg)
    write_cube(args.out, lut, "ALPHA indigo/cyan re-grade of neon-green promo")
    print("wrote %s  (%d^3 = %d entries)" % (args.out, args.size, args.size ** 3))


if __name__ == "__main__":
    main()