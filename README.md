# ALPHA — Autonomous Studio | Promo

25s brand film. Same cuts, same motion, **bit-identical soundtrack**. Only the words,
the colour and the closing mark changed.

## Master
`alpha_full.mp4` — 25.00s, 1280x720, 30fps, 750 frames, h264 + AAC 48kHz stereo.

## Brand identity — "Luminous Indigo"

| Role | Hex | Use |
| :--- | :--- | :--- |
| Primary | `#6C4BFF` indigo | the studio's presence, headlines |
| Secondary | `#2BD9FF` cyan | live money, approval edges, iris |
| Base | `#07070C` near-black | grounds, end card |

Why not orange / teal / green: those read as a status light first and a brand second.
Indigo holds a dark frame instead of shouting over it, which suits a product that works
while you sleep, and the cyan edge on cost and approval is what earns trust as quickly
for a shopkeeper as for an engineer.

## What changed

**Colour.** The whole film was green. Rebuilt in an opponent space (`Y` luma, `U = B−Y`,
`V = R−Y`) via a 48³ LUT, so luminance structure and the halftone dot geometry are held
fixed and neutrals are never rotated. Whites stay white, blacks lift to `#07070C`.
`chroma_pull` was re-tuned 0.50 → 0.62 after visual A/B: 0.50 read as washed lavender in
the large flat fields, 0.62 restores true black inside the dots and lifts local contrast.

**Words.** Every card rewritten in plain language, max 4 words per line, no jargon — so it
lands for people who know nothing about software *and* for developers.

| Card | Was | Now |
| :--- | :--- | :--- |
| boot | `FILE M LC-2.5-P` | `ALPHA 2.5 P` |
| | `Read the repo. Fix the flaky test.` | `Say what you want. / Alpha builds it.` |
| hero | `LongCat-2.5` | `Alpha-2.5` |
| | `AGENTIC` | `A REAL TEAM` |
| | `THINKS` | `PLANS` |
| | `1.6T` | `24/7` |
| | `DUAL API` | `EASY CONNECT` |
| | `TOKENS IN / OUT` | `JOBS DONE` / `$0.42 · COST SHOWN LIVE` |
| | `ACT.` | `BUILD. / Works while you sleep.` |
| | `Plugs into the harness you already use.` | `Plug your tools in once.` |
| | tool list (Claude Code, OpenCode…) | `Shop · Payments · Video · Email · Data` |
| | `Long-horizon. Multimodal. Agentic.` | `You say it once. / It's built. / By morning.` |

**Closing mark.** LongCat's cat icon and its white end card are gone. The last 3.6s is a
near-black card with a luminous indigo eye that blinks twice, the Alpha wordmark, and
`Say it. It's built. / alpha.studio / You approve everything. It never spends on its own.`

## Brand assets (`brand/`)
- `alpha-wordmark.png` — 1400×1165 RGBA, cut from the supplied logo. Matte is min-channel
  based (`a = 255 − min(R,G,B)`) with a smoothstep ramp, then un-matted against white so
  the glow carries no halo. Verified over black and magenta.
- `alpha-mark.png` — 512×512 badge.
- `alpha-eye.png` — the end-card presence, fully open.
- `alpha-indigo.cube` — the 48³ LUT. `build_lut.py` regenerates it; `filter.txt` is the
  ffmpeg fragment: `format=gbrp,lut3d=alpha-indigo.cube,format=yuv420p`.

## Rebuild
```bash
# 1. grade
ffmpeg -i original.mp4 -vf "$(cat brand/filter.txt)" -c:v libx264 -crf 16 -c:a copy base.mp4
# 2. overlay the per-card patches (each gated with between(t,start,end))
ffmpeg -i base.mp4 -i patches/*.png -filter_complex "<overlay chain>" ...
# 3. append the end card, then mux the ORIGINAL audio untouched
ffmpeg -f concat -i concat.txt -c:v libx264 -r 30 part.mp4
ffmpeg -i part.mp4 -i original.mp4 -map 0:v -map 1:a -c copy final.mp4
```

Text was replaced by inpainting each measured text box with a blurred, brightness-matched
copy of the same frame, so the halftone texture survives instead of leaving a flat box.