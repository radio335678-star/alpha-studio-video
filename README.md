# ALPHA Studio Video

Full Alpha rebrand: same cuts, same sound, all words rewritten for ALPHA.

## Files
- `alpha_full.mp4` — 25s, 1280x720, 30fps. Full Alpha script, silver grade (white/grey/black), blended 3D Alpha logo at end. Audio copied bit-identical.
- `edited_alpha_silver.mp4` — earlier pass (name-swap only, kept LongCat lines).

## What changed in alpha_full.mp4
Silver grade (`hue=s=0`), no green. All cards rewritten short + plain (normal people + devs):
- boot: `> boot alpha-2.5`, `MEITUAN · ALPHA`
- prompt: `Found store bug` / `Fixed, preview ready - tap to approve`
- hero: `Alpha-2.5 PREVIEW`
- stingers: `Crew builds / not a chat box`, `See everything live / cost, who, what`, `Talk it through / plan before build`, `1.6T dreams powered / shops, clips, apps`, `Easy Connect / plug store once`
- tokens: `1,000,000 / Pay as you build`
- triptych: `See everything live`, `Talk it through`, `You approve all / never sends alone`
- harness: `Five rooms, one crew / THINK BUILD STUDIO AUTOMATE CONNECT / talk, make, sell - runs all night` + `EASY CONNECT · LIVE COST SHOWN`
- positioning: `Works while you sleep. / selling, posting, helping`
- end: blended Alpha mark + `Alpha / Alpha-2.5-Preview / alpha.chat` + `© 2026 ALPHA - ALL RIGHTS RESERVED`

## Build
Base: `base_silver.mp4` + 16 PNG patches (`/tmp/opencode/A`, `/tmp/opencode/B`) + `patch_logo_blend.png` (feathered cutout, shadow + silver glow), ffmpeg overlay chain, `libx264 crf 18`, `-c:a copy`.
