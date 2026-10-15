# Procedural Promo Motion System — Research (1280×720, 30fps, 750f / 25s, numpy+PIL)

Sources verified from Remotion repo source (`packages/core/src/spring/*`, `easing.ts`, docs for
`interpolate`/`spring`/`Sequence`/`TransitionSeries`). Beat/type/pacing numbers are standard
motion-design practice values, chosen to be directly implementable.

## 1. Remotion core math (exact, port these)

### 1.1 `spring()` — analytic damped-oscillator stepped per frame
Defaults: `mass=1, damping=10, stiffness=100, overshootClamping=false, from=0, to=1`.
Per step (`t` in seconds, `dt = min(now-last,64)/1000`):
```
zeta   = c / (2*sqrt(k*m))        # damping ratio
omega0 = sqrt(k/m)                # undamped angular freq (rad/s)
omega1 = omega0 * sqrt(1-zeta^2)
x0 = to-current; v0 = -velocity
UNDERDAMPED (zeta<1):
  env = exp(-zeta*omega0*t); s=sin(omega1*t); c1=cos(omega1*t)
  frag = env * (s*(v0+zeta*omega0*x0)/omega1 + x0*c1)
  pos = to - frag
  vel = zeta*omega0*frag - env*(c1*(v0+zeta*omega0*x0) - omega1*x0*s)
CRITICAL (zeta>=1 — NOTE: Remotion uses this branch for ALL zeta>=1, i.e. overdamped
  input is clamped to critically-damped response, it never goes slower):
  env = exp(-omega0*t)
  pos = to - env*(x0 + (v0+omega0*x0)*t)
  vel = env*(v0*(t*omega0-1) + t*x0*omega0^2)
```
Integration loop: start `{current:from, velocity:0}`, step once per integer frame
`time=(f/fps)*1000` for `f=0..floor(frame)`, plus one fractional step if `frame%1>0`.
`delay=d` ⇒ evaluate at `frame-d` (clamped ≥0). `durationInFrames=N` ⇒ time-stretch input
`frame*N_natural/N` (needs `measureSpring` first). `reverse` ⇒ mirror around N.
`overshootClamping=true` ⇒ clamp output to `to` (kills bounce, keeps fast attack).

Settle times @30fps (`measureSpring`, threshold 0.005, ≈ frames to stay in ±0.5% for 20f):

| preset | m / c / k | zeta | overshoot | settle |
|---|---|---|---|---|
| default | 1 / 10 / 100 | 0.50 | ~16% | ~35f |
| snappy UI (entrances) | 1 / 26 / 170 | 1.0 | 0% | ~15f |
| poppy (scale pops, text land) | 1 / 14 / 170 | 0.54 | ~13% | ~23f |
| transition ease (docs example) | 1 / 200 / 100 | →critical branch | 0% | 23f |
| slow luxurious | 1 / 18 / 60 | 1.16→critical | 0% | ~30f |

Rule: entrances = snappy/poppy spring 12–20f; never use default zeta=0.5 for text (too boingy).
Overshoot amount formula (zeta<1): `exp(-pi*zeta/sqrt(1-zeta^2))`.

Minimal numpy port: precompute the 1-D spring curve once per (config, scene-len) into a
lookup array `S[0..N]`, then every animated property = `interpolate(S[f], [0,1], [a,b])`.
Cost is negligible; no per-pixel ODE.

### 1.2 `interpolate()` — piecewise-linear + per-segment easing (numpy: `np.interp` per segment)
`interpolate(x, inR, outR, easing=E, extrapolate='clamp')`: find segment `i`, `u=(x-inR[i])/(inR[i+1]-inR[i])`,
`y=outR[i]+(outR[i+1]-outR[i])*E(u)`; clamp outside. Our renderer needs exactly this helper
(`ease()` applied to normalized segment progress, not global time). Support easing-as-list
(one function per segment).

### 1.3 Easing formulas (exact, from Remotion/React-Native source)
```
linear t | quad t^2 | cubic t^3 | poly(n) t^n
sin:    1-cos(t*pi/2)
circle: 1-sqrt(1-t^2)
exp:    2^(10*(t-1))            # note exp(0)=1/1024, not 0
elastic(b=1): 1 - cos(t*pi/2)^3 * cos(t*b*pi)
back(s=1.70158): t^2*((s+1)*t - s)
bounce: piecewise 7.5625 u^2 offsets [.75/.9375/.984375] at 1/2.75,2/2.75,2.5/2.75
bezier(x1,y1,x2,y2): CSS cubic-bezier on progress
out(E)(t)=1-E(1-t) | inOut(E): E(2t)/2 then 1-E(2(1-t))/2
```
Workhorse curves for promos (use these 90% of the time):
entrances `bezier(0.16,1,0.3,1)` (expo-out, fast attack + long soft tail);
exits `bezier(0.7,0,0.84,0)` or in-quad (accelerate away, 8–12f);
in-out moves `bezier(0.85,0,0.15,1)`. Never use linear for position/opacity/scale.

### 1.4 `Sequence` / `TransitionSeries` semantics (timeline model to copy)
Local clock: `local = global - from`; nesting adds offsets. Total duration =
`sum(seq) - sum(transition)` (transitions overlap both sides). Transitions 10–20f:
`fade` = opacity ramp; `wipe/slide` = translate 0→W with expo-out + 4–6 subframe motion
blur; `zoom-blur` = scale 1→1.15 + blur 0→8 on exit, reverse on entry.
Our port: each scene gets `(start, dur)`; transitions are overlap windows, not gaps.

## 2. Kinetic typography (per-character system)
- Stagger: char delay **2f** (67ms), word delay **4f**, line delay **6f**; total cascade span ≤ **24f**.
- Entrance per char (15f, expo-out): `y: +0.35em→0`, `opacity 0→1` (first 10f), `blur 8→0px`
  (first 12f), `scale 1.12→1.0` via poppy spring for hero words only (body text: no scale).
- Overshoot: land at 1.0 with ≤13% spring overshoot on scale only; never overshoot position.
- Word-by-word opacity cascade for sentences: 4f stagger, 12f fade each.
- Exit: 10f, `y→-0.25em`, opacity→0, in-quad; exits overlap next scene entry by 6–8f.
- Readability: any card holds full text ≥ **45f** (1.5s); hero word on screen ≥ 60f total.
- Emphasis: 1 keyword/card in brand color + 4–8% scale-up, lands 4f after the rest (delayed pop).
- Blur-to-sharp = render text layer, Gaussian blur radius `8*(1-u)`, cheap at 720p, do it.

## 3. Beat-synced editing (transient-driven animation)
- Detect hits offline: spectral-flux onset envelope (`librosa.onset.onset_strength`,
  or numpy fallback: rectified L1 diff of log-mel/STFT magnitudes), peak-pick ≥0.35s apart,
  quantize to frames. Precompute arrays: `hit[f]∈{0,1}`, `energy[f]∈[0,1]` (smoothed envelope).
- Vocabulary (one per hit, rotate to avoid monotony):
  - scale pop: anticipation −3f ease-in to 0.97 → impact +2f out to **1.08–1.12** → poppy-spring
    settle ~15f. Kick drum → this.
  - cut/whip-pan: 8f translate **0.25–0.35×W** expo-out + directional 5-tap blur. Snare → this.
  - impact frame: hold 2f (duplicate frame) + white flash 2f @ alpha 0.20–0.25 + blur pulse.
  - hats/16ths: micro jitter ±2px or 0.5% scale ticks, never full pops (visual clutter rule:
    full-strength move only on kicks/snares, ≤1 per ~15f).
- Background pulse (halftone gain, ring speed, vignette) follows `energy[f]` continuously —
  this sells sync even between cuts. Smooth energy with attack 3f / release 15f.

## 4. Halftone / dot-matrix tunnel (brand color on black)
- Base grid: spacing `s=18px` (720p), dot radius `r = 0.42*s*b(x,y,t)`, `b` = pattern brightness
  0..1; skip dots with `r<0.6px`. Add ±10% hash jitter to radius to kill banding.
- Patterns (layer 1–2 max + vignette): (a) radial rings `b=0.5+0.5*sin(k*r_px − w*f)` with
  `k≈0.05/px`, `w` scaled by energy; (b) tunnel: `N=14` rings, `z=frac(i/N + f*speed/30)`,
  radius=`Rmax*(1-z)^2.2`, alpha=`z`, width pulse `1+0.6*energy`; `speed≈0.5/s` idle, `×3` on hits.
- Pulse: global gain `g=1+0.6*E[f]` multiplies radii; on kick frames add expanding shockwave ring
  (radius 0→700px over 18f, alpha 0.5→0).
- Depth: two grids (far s=12px @40% alpha, near s=26px) drifting in opposite directions
  (parallax 6–10px/s). Foreground text always ≥60% screen-center contrast: fade dots to black
  under text via radial mask.

## 5. "Hyperframe" smooth slow-mo in a procedural renderer
- Motion blur = frame accumulation: render `K=6` subframes at `f+(j/K−0.5)*shutter`,
  `shutter=0.5f` (180°), box-average. Apply ONLY during transitions/pans/pops (gate on
  velocity), not globally — 6× cost only on ~15% of frames.
- Speed-ramped zooms: `scale(f)=1+A*smoothstep(u)+0.03*E[f]`, `A=0.06–0.10` per scene, always
  eased; direction alternates push/pull per scene for variety.
- Slow-mo hit: warp local clock — 0.25× speed for 6f around a big drop, recover over 12f
  (implement as remapped subframe sampling; cheap since it's a time remap, not re-render).
- No linear moves anywhere; every camera move gets ≥8f ease-in AND ease-out.

## 6. 25s promo pacing (750f @30fps — 7 scenes)
| # | scene | frames | time | notes |
|---|---|---|---|---|
| 1 | hook (logo + punchline) | 0–90 | 0.0–3.0s | fastest energy, biggest type, hit on f0+first kick |
| 2 | card A | 90–195 | 3.0–6.5s | 105f |
| 3 | card B | 195–300 | 6.5–10.0s | 105f |
| 4 | card C | 300–405 | 10.0–13.5s | 105f |
| 5 | card D | 405–510 | 13.5–17.0s | 105f |
| 6 | card E / proof | 510–615 | 17.0–20.5s | 105f |
| 7 | CTA + logo hold | 615–750 | 20.5–25.0s | 135f, 2.5s static-ish hold at end for readability |
- Transitions 12f overlaps at each boundary (total math: 90+5×105+135 − 6×12 = 750 ⇒ trim
  scene durs by overlap when scheduling; implement overlaps, not gaps).
- Entry/exit rhythm per card: enter 15f (staggered chars) → hold ≥45f readable → exit 10f,
  exit overlaps next entry. New accent (pop/pan/flash) every 15–30f; nothing static >30f
  except final CTA hold. Cut on kicks, move on snares, shimmer on hats.
- Text budget: ≤7 words/card, hero ≤3 words; 1280×720, title 90–140px bold, 8% safe margins.
