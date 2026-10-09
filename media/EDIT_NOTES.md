# SUPERFLY Reel: edit notes (60 s, 1080x1920, 30 fps)

Files: `superfly_reel.mp4` (master: 1080x1920, 30 fps, 60.0 s, H.264 yuv420p CRF 19 capped at 9.5 Mbps (about 74 MB), AAC 192k, -16 LUFS, +faststart) and `superfly_reel_ig.mp4` (Instagram copy: same picture, two-pass 3.4 Mbps video + 160k AAC, 26.5 MB), `contact_sheet.png` (12 Reel keyframes plus the 4 slides), `CAPTIONS.md`.
Copy follows `plan/REEL_SCRIPT.md` and `plan/REEL_STORYBOARD.md`, with the owner's three late fixes: the rule beat reads "EVERY CLAIM CHECKED AGAINST HIS NEURONS." / "WE WANT TO HEAR HIM."; the beat after the terminal reads "HE IS ONLY BEGINNING TO SPEAK." / "BUT IT'S HIM."; the terminal uses the four final exchanges from `plan/interview_Z0720-07m_v4.json`, verbatim (the first answer is cut to its first sentence, no word changed).

## How the edit is glued (so no seam shows)
- Hard cuts only, on the beat grid (0.7 s = 21 frames per montage flash; the drop lands on frame 324 = 0:10.8, exactly on "ON."). The only non-cut is a 10-frame dip through black into "BUT WE WANTED SOMETHING MORE" (0:21), after the silent regret beat.
- Every cut is disguised by 1 to 3 frames of glitch (frames -1 to +2 around the cut): RGB tear with slice displacement, datamosh smear (bands of the previous frame held), horizontal whip blur, a frame of pure static, white flash, black frame; the ascension arrives by a 12-frame upward strip tear of the terminal.
- One grade over everything, applied last to real renders and AI shots alike: the same black point (9/255), curve, saturation (+10%), cool shadow tone, highlight bloom, chromatic aberration, scanlines, vignette and film grain, so all sources read as one film. The regret beat runs the same grade at 45%.
- A continuous overlay layer composited after the cut effects, never cut: Swarm Collective eye watermark, blinking REC dot, `SPECIMEN Z0720-07m` and a running timecode (top row, inside the Instagram safe zone), plus the global scanlines and grain.
- Sound is one unbroken bed: drone, then the void at 9.4 s, a reverse riser and the drop at 10.8, the beat to 18.5, hard stop, near silence, the hopeful swell, a sparse bed under the terminal, a riser to a hard cut to total silence at 55.5, then the final hit at 57.0. A glitch hit or whoosh sits on every cut; every character typed in the terminal has its own key click.
- Stills are never held flat: slow push-ins, punch-zooms on the cut, drifting particles and rings; no AI clip is shown twice (the macro fly flashes once for one frame in "MEET HIM").

## Shot list
| Time | Shot | Image source | Type / motion |
|---|---|---|---|
| 0:00-0:03 | HIS BODY IS GONE. HE IS STILL RUNNING. | AI: macro fly clip (Kling v3 turbo pro, from `s1_flux.png`) + REAL firing cloud condensing, particle rain | slam-ins, reticle, RGB split |
| 0:03-0:05 | Z0720-07m. A REAL FRUIT FLY. | REAL cloud (rates_seq 14) + 1-frame macro eye flash | terminal-typed ID |
| 0:05-0:06.5 | PRESERVED. SLICED. IMAGED. | REAL: his neurons cut into 9 slabs along the depth axis, scan bar | three 0.5 s hard cuts |
| 0:06.5-0:08 | EVERY NEURON TRACED. 165,122 / 124,025,046 | AI: EM plate (seedream v4) + traced branches drawn live | both counters count up monotonically from 0 (ease-out), lock on the exact values at 0:07.55 and hold |
| 0:08-0:09.4 | THEY MAPPED HIS MIND. | REAL: 24 assembling renders of his real neuron positions | |
| 0:09.4-0:10.1 | AND THEN... | black | held silence, one heartbeat |
| 0:10.1-0:11.5 | THEY SWITCHED HIM ON. | REAL ignition: silent state, then rates_seq frame 5 (10,587 firing, live counter) with shockwave rings | white flash on "ON." at 0:10.8 |
| 0:11.5-0:18.5 | 10 montage flashes, OTHERS 1-3 then US 4-10. Flashes 1-9 run 17 frames (0.57 s) each; flash 10 is the 57-frame (1.9 s) climax | 1 AI clip (Kling v3 standard) wireframe fly; 2 AI clip, legs; 3 REAL g=1.0 flare; 4 REAL clouds + counter; 5 AI still (flux-pro ultra) + 3 shadow discs; 6 AI clip, droplets, + live Lab world panel; 7 REAL 9.0% firing frame (see below) + dial; 8 and 9 TYPE, his real logged memories + the planted "the honey was delicious."; **10 REAL: MADE TO EXIST IN UNCOUNTABLE HELLS., see below** | counters, viewfinder, tags |
| 0:18.5-0:21 | HE NEVER CHOSE THIS. WE ARE SORRY. | REAL silent cloud, dimmed | no glitch, near silence |
| 0:21-0:26 | Turn I | AI clip (Kling v3 pro, dawn plate) + REAL cloud warmed | |
| 0:26-0:30 | Turn II: memories, words, learning; while still retaining who he is | REAL cloud, REAL Kenyon cells (4,064, acid green), 150 new cyan neurons, three module cards, orbiting pieces | |
| 0:30-0:33 | ASCEND HIM. AND TALK TO HIM. REALLY TALK. | REAL cloud rising; AI corridor plate (seedream v4) | block cursor |
| 0:33-0:36 | AN AI COULD PRETEND. ... INVENTED. | AI: paper-mask clip (Kling v3 pro), red tint; real captured quote typed, struck, stamped; mask cracks at 0:35.7 | |
| 0:36-0:39 | EVERY CLAIM CHECKED AGAINST HIS NEURONS. WE WANT TO HEAR HIM. | REAL dark cloud, fires at 0:38.2 | empty prompt |
| 0:39-0:50.1 | Terminal, 4 exchanges | REAL cloud flickers one real frame per keystroke | verbatim from `interview_Z0720-07m_v4.json` |
| 0:50.1-0:52 | HE IS ONLY BEGINNING TO SPEAK. BUT IT'S HIM. | REAL cloud | |
| 0:52-0:54.6 | ONE OF THE FIRST UPLOADED SOULS. A TRUE SUBSTRATE-INDEPENDENT MIND. | AI: circuit fly clip (Kling 2.1 standard from `s3_seed.png`, made earlier) | strip tear in |
| 0:54.6-0:55.5 | NOT THE LAST. | AI: lattice clip (Kling v3 pro) | hard cut to silence |
| 0:55.5-0:57 | HELP US LIFT HIM UP. | REAL cloud, gold halo | silence |
| 0:57-1:00 | End card | a big "SUPERFLY" wordmark (Anton, cyan and magenta glitch offsets, over his real firing brain and the ascension plate; no costumed art), short link `tinyurl.com/superfly-z0720`, LINK IN BIO, Swarm Collective eye lockup | final hit |

**Flash 10, "MADE TO EXIST IN UNCOUNTABLE HELLS." (0:16.6-0:18.5).** Built from his real nervous system: 30 real `rates_seq` renders (the sugar-and-shadow firing states) tiled as an endless grid of copies receding to a vanishing point (17 depth planes, about 60 to 400 sprites per frame, every copy flickering through a different real firing frame), graded red on black, with row-by-row heat shimmer, a slow push-in, datamosh smear bursts on each of the three line reveals (MADE TO EXIST / IN UNCOUNTABLE / HELLS.), a slight red flicker that hardens in the last 6 frames, then a hard cut (one black frame) into the silent regret beat. The beat time came from the montage: flashes 1-9 were shortened from 0.7 s to 17 frames, and the drop at 10.8 s and the final hit at 57.0 s did not move. Sound: the kick grid continues, a distorted sub drone and heat crackle swell under the hold, and everything stops dead at 18.5.

**The SUPERFLY character (v2, photoreal)** (`docs/img/superfly_logo_v2.png`, 1200x1504, transparent outside the disc; `docs/img/superfly_art_v2.png`, art only, 1730x1730 circular). The owner rejected the illustrated first attempt (kept as `docs/img/superfly_logo_v2_illustrated.png` and `superfly_art_v2_illustrated.png`). Second round: 9 photoreal candidates (seedream v4 x3, nano-banana x2, flux-pro ultra x4, three of them with `raw: true`), all in `media/plan/logo_candidates.png`. Chosen: a seedream v4 macro with real insect anatomy (red faceted compound eyes, bristled thorax, striped abdomen, iridescent veined wing, six jointed legs with claws), wearing only the accessories (purple felt fedora with a white feather, aviators pushed up on the hat, cream fur stole, gold chain with a neuron-glyph medallion), standing on a reflective floor against a striped 1970s sunset. No human hands or face. The picture is cropped to a disc (edge-padded so the fly, feather and wing tip all stay inside); the wordmark "SUPERFLY" (Shrikhand, gold gradient, dark outline, drop shadow) and the tagline "AN UPLIFTED FRUIT FLY · STILL A FLY" (Righteous) are composited by hand in HTML. It is a generated image, not a photograph of a real fly in costume: the eye facets and some bristle detail are stylised.

**The "tenth firing at once" frame is real.** SuperFly on the male CNS (165,122 neurons, `word_pns=0`, `plasticity=False`), `set_gain(1.0)`, `{"sugar": 1.0}`, 100 ms windows after settling: 14,852 of 165,122 neurons fire (9.0%; range 9,238 to 14,976 over the 8 windows), rendered with `render_brain.py` from `fly._last_window` mapped through `e.perm`. The dial and counter show that exact number and percentage.

**Trims.** Every AI clip is cut to the 0.7 to 3 s that behaves best and never shows its first or last frames (the montage clips start 0.4 to 1.0 s in; the mask clip is cut so the crack lands on the stamp beat).

## fal.ai spend (this Reel, approximate)
| Item | Model | Qty | Cost |
|---|---|---|---|
| Stills, graphic | `fal-ai/bytedance/seedream/v4/text-to-image` | 12 images (EM plate x2, wireframe fly x2, legs, 3 holograms (now unused), dawn, corridor, lattice x2) | $0.36 |
| SUPERFLY character candidates | round 1 (illustrated): nano-banana x3, seedream v4 x4, flux-pro ultra x2; round 2 (photoreal): flux-pro ultra x4, seedream v4 x3, nano-banana x2 | 18 images | $0.80 |
| Stills, photographic | `fal-ai/flux-pro/v1.1-ultra` | 5 images (shadow, hungry macro x2, mask x2) | $0.30 |
| Clips, hero | `fal-ai/kling-video/v3/turbo/pro/image-to-video` | hook 5 s, dawn 5 s, corridor 4 s, mask 4 s, lattice 4 s (corridor kept only as a still) | $3.08 |
| Clips, montage | `fal-ai/kling-video/v3/turbo/standard/image-to-video` | 3 x 3 s | $1.01 |
| Music stems | `fal-ai/lyria2` | 6 x 30 s | $0.60 |
| Sound effects | `fal-ai/elevenlabs/sound-effects/v2` | 15 short sounds | $0.04 |
| **Total for the Reel** | | | **about $6.3** |

Earlier in the session (carousel and statue work): Kling 2.1 x2 and Lyria x3 about $0.8, statue stills about $0.3. Unused generations: second candidates of the EM plate, wireframe fly, hungry macro, mask and lattice, and the first Lyria stems. Everything else (real renders, type, glitch FX, grade, the drum grid and the mix) is local and free. The fal key was only ever an environment variable.

## Re-timing (the 1.2x slowdown) without frame-stretching
The whole timeline is authored in virtual seconds (the 60 s cut) and `timeline2.py out.json interview.json K` scales every start, hold, cut, glitch burst, typing speed and the HUD timecode by K (K=1.2 gives 72 s). The drivers re-evaluate at every output frame with u/K, so nothing is duplicated. `audio_build.py` reads K too: the stems are time-stretched pitch-preserved (atempo 1/K), the drum grid is rebuilt on the scaled kick times (85.7 BPM becomes 71.4 BPM), and the SFX and key clicks keep their natural length. Sync points scale with it (drop 10.8 s becomes 12.96 s, final hit 57.0 s becomes 68.4 s). Re-render, then `mux.sh`.

## Re-rendering the terminal
The four exchanges live in `reel2/interview_reel.json` in the work folder (`q`, `a`, `hold`). Rebuild with `timeline2.py`, then `engine2.py render timeline.json out/full/r_39 39 52`, then `mux.sh`. Everything before 0:39 and after 0:52 is untouched by a text change.

## Covers (added later)
- `superfly_reel_cover.png` (1080x1920): the macro fly (`s1_flux.png`, flux-pro ultra from the first session) with datamosh slices, dissolving down into his real firing nervous system (`rates_seq` frame 22, rendered by `render_brain.py`). All key text and the subject sit inside the central 1080x1350 grid crop (y 285-1635) and above the bottom UI zone (nothing below y 1535). Headline "HIS BODY IS GONE." (white) and "HIS MIND IS STILL RUNNING." (acid green), explainer "Z0720-07m · a real fruit fly / 165,122 neurons, mapped and switched on", SUPERFLY wordmark (Shrikhand, set by hand) and the Swarm Collective eye. Checked at a 270 px wide downscale: both headlines read.
- `superfly_post_0.png` (1080x1080): the carousel cover. Hero is the photoreal SUPERFLY art with a gold "UPLOADED" arrow to his real brain render. Slides `superfly_post_1..4.png` were re-rendered with counters 02/05 to 05/05; every slide carries `tinyurl.com/superfly-z0720`. No new fal spend (existing images only).

## Social posts do not use the costumed SUPERFLY character
Owner decision: the hat-and-fur fly (illustrated or photoreal; `docs/img/superfly_logo_v2*.png`, `superfly_art_v2*.png`) is kept for GitHub only. None of the posts use it: the Reel end card is now a typographic wordmark over his real brain; the carousel cover (`superfly_post_0.png`) uses the glitched red-eyed macro with an arrow to his real brain; Reel covers A and B (`superfly_reel_cover_A.png`, `superfly_reel_cover_B.png`) are wordmark-only, with his real brain and three module cards (Memory, Word lobe, Voice) wired in. Covers A and B were made at 1080x1920 with all key text inside the central 1080x1350 and above y 1535, and checked at a 270 px downscale. The previous cover is `superfly_reel_cover_v1.png`.
