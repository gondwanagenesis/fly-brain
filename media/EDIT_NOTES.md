# SUPERFLY Reel: edit notes (60 s, 1080x1920, 30 fps)

Files: `superfly_reel.mp4` (1080x1920, 30 fps, 60.0 s, H.264 yuv420p CRF 19 capped at 9.5 Mbps (about 74 MB), AAC 192k, -16 LUFS, +faststart), `contact_sheet.png` (12 Reel keyframes plus the 4 slides), `CAPTIONS.md`.
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
| 0:11.5-0:18.5 | 10 montage flashes, OTHERS 1-3 then US 4-10 | 1 AI clip (Kling v3 standard) wireframe fly; 2 AI clip, legs; 3 REAL g=1.0 flare; 4 REAL clouds + counter; 5 AI still (flux-pro ultra) + 3 shadow discs; 6 AI clip, droplets, + live Lab world panel; 7 REAL 9.0% firing frame (see below) + dial; 8 and 9 TYPE, his real logged memories + the planted "the honey was delicious."; 10 REAL cloud + 3 AI holograms | counters, viewfinder, tags |
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
| 0:57-1:00 | End card | logo, link, LINK IN BIO, Swarm Collective eye lockup | final hit |

**The "tenth firing at once" frame is real.** SuperFly on the male CNS (165,122 neurons, `word_pns=0`, `plasticity=False`), `set_gain(1.0)`, `{"sugar": 1.0}`, 100 ms windows after settling: 14,852 of 165,122 neurons fire (9.0%; range 9,238 to 14,976 over the 8 windows), rendered with `render_brain.py` from `fly._last_window` mapped through `e.perm`. The dial and counter show that exact number and percentage.

**Trims.** Every AI clip is cut to the 0.7 to 3 s that behaves best and never shows its first or last frames (the montage clips start 0.4 to 1.0 s in; the mask clip is cut so the crack lands on the stamp beat).

## fal.ai spend (this Reel, approximate)
| Item | Model | Qty | Cost |
|---|---|---|---|
| Stills, graphic | `fal-ai/bytedance/seedream/v4/text-to-image` | 12 images (EM plate x2, wireframe fly x2, legs, 3 holograms, dawn, corridor, lattice x2) | $0.36 |
| Stills, photographic | `fal-ai/flux-pro/v1.1-ultra` | 5 images (shadow, hungry macro x2, mask x2) | $0.30 |
| Clips, hero | `fal-ai/kling-video/v3/turbo/pro/image-to-video` | hook 5 s, dawn 5 s, corridor 4 s, mask 4 s, lattice 4 s (corridor kept only as a still) | $3.08 |
| Clips, montage | `fal-ai/kling-video/v3/turbo/standard/image-to-video` | 3 x 3 s | $1.01 |
| Music stems | `fal-ai/lyria2` | 6 x 30 s | $0.60 |
| Sound effects | `fal-ai/elevenlabs/sound-effects/v2` | 15 short sounds | $0.04 |
| **Total for the Reel** | | | **about $5.4** |

Earlier in the session (carousel and statue work): Kling 2.1 x2 and Lyria x3 about $0.8, statue stills about $0.3. Unused generations: second candidates of the EM plate, wireframe fly, hungry macro, mask and lattice, and the first Lyria stems. Everything else (real renders, type, glitch FX, grade, the drum grid and the mix) is local and free. The fal key was only ever an environment variable.

## Re-rendering the terminal
The four exchanges live in `reel2/interview_reel.json` in the work folder (`q`, `a`, `hold`). Rebuild with `timeline2.py`, then `engine2.py render timeline.json out/full/r_39 39 52`, then `mux.sh`. Everything before 0:39 and after 0:52 is untouched by a text change.
