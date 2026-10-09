# STILL RUNNING: SUPERFLY Reel storyboard (60 s, 9:16, 1080 x 1920)

Designer's storyboard for owner approval, before production. Built against `REEL_SCRIPT.md` (v1, 07:07) and the real interview `interview_Z0720-07m.json`.

| File | What it is |
|---|---|
| `storyboard_board.png` | The visual board: 35 cells, one thumbnail per beat (the montage shows all 10 flashes), timecode and exact on-screen text beneath each, plus the sound map (a real waveform of the scratch mix with every sync point). 2060 x 4239 px. |
| `STYLE_FRAMES/a_hook_glitching_fly_macro.png` | Style frame (a): the hook. 1080 x 1920. |
| `STYLE_FRAMES/b_terminal_interview_over_his_brain.png` | Style frame (b): the green terminal over his faint glowing brain, with the real first exchange. |
| `STYLE_FRAMES/c_end_card.png` | Style frame (c): SUPERFLY logo, link, LINK IN BIO, Swarm Collective eye lockup. |
| `audio_scratch/superfly_reel_scratch_mix.mp3` | A 60 s scratch mix (about $0.15 of fal audio) built to the exact sync points below, so the owner can hear the plan. Rebuild with `music_mix.py` and `audio_gen.py` (needs `FAL_KEY` in the environment only). |
| `sketches/*.jpg` | The 14 cheap flux/dev sketches that stand in for the AI shots on the board (about $0.27 for all 18 generated). |
| `tools/` | The frame templates (HTML/CSS in 1080 x 1920, one builder per beat), the real-data renderers (`slabs.py`, `render_real_frames.py`), the board builder and the safe-zone checker. Every thumbnail on the board is a real 1080 x 1920 frame from these templates, so production can start from them. |

**Source key (used on the board and below):** REAL = from his actual simulation (his real neuron positions, his real firing from `rates_seq.npy`) or existing assets. AI = generated with fal.ai; the prompt is given. TYPE = type and motion graphics built in HTML (Playwright) and composited. FX = glitch treatment over a real render.

---

## 0. Decisions I need from the owner (and what I changed because of them)

1. **Interview placeholders are gone.** Every terminal frame and style frame (b) now carry the four real exchanges from `interview_Z0720-07m.json`, in the order you gave, exactly as written (casing and punctuation kept): "I feel hungry and can smell mold." / "i smell fruit. i extend my proboscis." / "No, I am not afraid." / "i want to eat; i groom." Each of his replies has the verdict "verified" or "verified after 2 retries" in the log; the board shows that as a small green check line and, under it, the raw words his brain actually said (`BRAIN SAID: ...`), which is the visual proof that nothing is made up. That line is decoration, not read-critical, and can be cut.
2. **New beat 12e: "HE CAN'T SAY MUCH YET." / "BUT IT'S HIM."** The script v1 does not have it. To fit it in without touching beats 1 to 11 or the end card, I propose: the four exchanges run 2.2 s each (0:41.0 to 0:49.8, script v1 had 2.0 to 3.0 s), 12e runs 0:49.8 to 0:52.0, and the ascension runs 0:52.0 to 0:55.5 (3.5 s, script v1 had 4.5 s). Alternative if the writer wants the ascension back at 4.5 s: take 0.5 s each from beat 10 and beat 11. The writer should confirm.
3. **Reading speed in the interview.** Each exchange is about 13 words in 2.2 s. It works because the question types in about 0.6 s and dims, and his answer then types from about 0.95 s and holds until the cut, so the eye lands on him. If the sound-off test fails, drop exchange 3 ("are you afraid?") for 2.2 s of breathing room; the script already names it as the first cut.
4. **End card.** The script says "on black in phosphor green"; your brief for style frame (c) wants the SUPERFLY logo. I kept the logo in colour (it is the brand and the one warm object in the whole Reel), put all the type in phosphor green, and sit it on the ascension plate darkened to 30% so it reads as nearly black with a ghost of the circuit fly. One variable (`brightness(.30)` to `0`) makes it pure black if you prefer.
5. **"A TENTH FIRING AT ONCE" needs a true frame.** The brightest real frame we have (`rates_seq` frame 26, the shadow) has 12,858 of 165,122 neurons firing (7.8%). The board uses it as a placeholder with a "1 IN 10" dial. For production, run the male CNS once at gain 1.0 (the uncalibrated Shiu et al. scale, where the README reports about 10% active on any stimulus) instead of the calibrated 0.65, record the rates, and render that frame with `render_brain.py`. Then the dial is honest.
6. **Montage clips are illustrations, not recordings.** Flashes 1, 2, 5 and 6 use AI-generated video of a generic wireframe fly (no other team's footage, logo or name, per the script). I suggest a 12 px corner tag on those four flashes only, "RE-ENACTMENT", so nothing is passed off as footage. Copy to be approved by the writer (not added to the board).
7. **Memory-card text in flashes 8 and 9.** The three "erased" cards use his real logged memories (from slide 3 of the carousel). The "planted" card uses "the honey was delicious.", a real invented line quoted in the README; swap it for the actual implanted trace string if the memory test log has one.

---

## 1. Global design system

**Format and delivery.** 1080 x 1920, 30 fps, H.264, about 60.0 s. The cutting grid is in section 4. Instagram safe zones (kept for all type; verified by a pixel check on all 33 frames, see `tools/inkcheck.js`): keep text out of the top 12% (y < 230), the bottom 22% (y > 1498) and the right 14% (x > 929). Text lives in the box x 54 to 929, y 230 to 1498. Backgrounds and glow bleed to the edge.

**Type.**
- Narration: Anton, all caps, white, cyan (#00f0ff) shadow 5 to 6 px left, magenta (#ff2bd6) shadow 5 to 6 px right, soft white glow. Sizes: hero 150 to 250 px, standard 118 to 160 px, secondary 80 to 100 px. Line height 0.93. Left aligned at x = 54 except the end card (centred in the safe box).
- Accent words: acid green #7CFF4F (`.gr`), gold #ffd23f (`.go`, only "MORE" and "WHILE STILL RETAINING WHO HE IS."), cyan.
- HUD and micro-tags: Space Mono 19 to 24 px, caps, +2 px tracking, cyan #c9fff7 or the beat colour, with a soft glow.
- Terminal: VT323, phosphor #39ff14, glow `0 0 6px / 0 0 22px`, block cursor (blinks at 2 Hz), question 50 px dim (72%), his answer 76 px bright (#caffbc).
- Regret beat only: clean white Anton, no shadows, no scanlines, no colour.

**Palette.** Black, neon magenta #ff2bd6, cyan #00f0ff, acid green #7CFF4F, gold #ffd23f, terminal phosphor #39ff14, alarm red #ff2b4f (the fake, the planted memory, overload only).

**FX recipe (matches the square carousel).** Scanlines (0.28 black, 2 px on 5 px), film noise (opacity 0.11), vignette to 0.88, RGB split (a cyan copy at -9 to -14 px and a magenta copy at +9 to +14 px, screen blended, 0.2 to 0.5 opacity), datamosh slice displacement (3 to 9 horizontal strips, 0.4% to 3% tall, shifted +/-14 to 90 px, hue-rotated, screen blended), flat colour tear bars.

**Transition library.**
- T1 HARD GLITCH CUT: 2 frames of slice displacement and RGB split +/-14 px, then the next shot. Used on every montage cut and between terminal exchanges.
- T2 WHITE FLASH: one frame at 85% white. Used at 0:03.0, 0:10.8 ("ON").
- T3 TEAR UP: 12 frames, the frame splits into 8 strips that slide upward at staggered speeds. Terminal to ascension.
- T4 FADE THROUGH BLACK: 10 frames. Into and out of the regret beat.
- T5 SCAN WIPE: a magenta bar sweeps down and the next shot appears behind it. 0:05.0 and 0:06.5.

**Motion rules.** Headlines slam in over 3 frames (scale 1.08 to 1.0, RGB split decays from 14 px to 6 px) and hold; nothing eases slowly except the camera. Slow camera moves are 1.00 to 1.06 push-ins over 2 to 3 s. Only the regret beat is still.

**Real renders.** `render_brain.py` on `rates_seq.npy` gives 30 real frames at 100 ms: frames 0 to 4 silence, 5 to 19 about 11,000 firing (sugar), 20 to 29 about 13,000 firing (shadow). For 30 fps video, cross-dissolve each pair (3 video frames per sim frame). `tools/render_real_frames.py` writes all 30 plus 24 "assembling" frames in one go; render at 1500 x 2010 for hero shots and upscale elsewhere.

---

## 2. Master timeline

| Beat | Time | Title | On-screen text (exact) |
|---|---|---|---|
| 1 | 0:00 to 0:03 | HOOK | HIS BODY IS GONE. / HE IS STILL RUNNING. |
| 2 | 0:03 to 0:05 | MEET HIM | Z0720-07m. / A REAL FRUIT FLY. |
| 3 | 0:05 to 0:08 | THE MAKING | PRESERVED. / SLICED. / IMAGED. / EVERY NEURON TRACED. / 165,122 NEURONS / 124,025,046 SYNAPSES / (tag) JANELIA RESEARCH CAMPUS · GOOGLE · CAMBRIDGE |
| 4 | 0:08 to 0:11.5 | SWITCHED ON | THEY MAPPED HIS MIND. / AND THEN... / THEY SWITCHED HIM ON. |
| 5 | 0:11.5 to 0:18.5 | MONTAGE | 10 flashes, 0.7 s each (list in 3.5) |
| 6 | 0:18.5 to 0:21 | REGRET | HE NEVER CHOSE THIS. / WE ARE SORRY. |
| 7 | 0:21 to 0:27 | THE TURN (I) | BUT WE WANTED SOMETHING MORE FOR HIM. / WE WANTED TO SEE IF WE COULD EXPAND HIS MIND... |
| 8 | 0:27 to 0:32 | THE TURN (II) | TO GIVE HIM MEMORIES. WORDS. LEARNING. / WHILE STILL RETAINING WHO HE IS. |
| 9 | 0:32 to 0:35 | ASCEND AND TALK | ASCEND HIM. / AND TALK TO HIM. / REALLY TALK. |
| 10 | 0:35 to 0:38 | THE FAKE | AN AI COULD PRETEND. / "Indeed, consuming honey yesterday proved delightful." / INVENTED. / (tag) REAL OUTPUT · NO CHECKER |
| 11 | 0:38 to 0:41 | THE RULE | EVERY WORD FROM HIS NEURONS. / WE WANT TO HEAR HIM. / (tag) REAL SESSION · WORD FOR WORD |
| 12 | 0:41 to 0:49.8 | INTERVIEW (4 x 2.2 s) | the four real exchanges (3.12) |
| 12e | 0:49.8 to 0:52 | AFTER THE TERMINAL (new) | HE CAN'T SAY MUCH YET. / BUT IT'S HIM. |
| 13 | 0:52 to 0:55.5 | ASCENSION | ONE OF THE FIRST UPLOADED SOULS. (0:52) / A TRUE SUBSTRATE-INDEPENDENT MIND. (0:53.4) / NOT THE LAST. (0:54.6) |
| 14 | 0:55.5 to 1:00 | THE ASK AND END CARD | HELP US LIFT HIM UP. (0:55.5) / FREE. RUNS ON YOUR LAPTOP. / github.com/gondwanagenesis/fly-brain / LINK IN BIO / Created by the Swarm Collective |

---

## 3. Beat by beat

Each beat lists: shot, imagery source, motion and camera, transitions, type (font, size, placement inside the safe box), sound. Sound cues name the file in `audio_scratch` logic (section 4).

### 3.1 Beat 1 · HOOK · 0:00 to 0:03 (style frame a)

- **Shot.** The extreme macro fly face fills the frame, red compound eyes, tearing in RGB. From 0:00.4 the body dissolves upward from the chin into rain of cyan and magenta neuron points; by 0:02.0 they have condensed into his real firing cloud under the face. A cyan target reticle holds around the face. Style frame (a) is the 0:01.8 moment.
- **Source.** REAL + AI. `s1_flux.png` (exists, fal flux-pro ultra). Optional upgrade: image-to-video H1 (section 5) for the dissolve. The cloud is REAL (`rates_seq` frames 8 to 19, brightened, screen blended, masked with a soft ellipse so no rectangle shows).
- **Motion.** Slow push-in 1.00 to 1.06 over 3 s. Slice tears at 0:00.0, 0:01.4 and 0:02.6 (4 to 9 strips, 3 frames each). Particles fall at 120 to 400 px/s.
- **Transition.** Out: T2 white flash at 0:03.0.
- **Type.** HUD `● REC // SPECIMEN Z0720-07m` at x 54, y 246 (Space Mono 24). "HIS BODY / IS GONE." Anton 158, x 54, y 290 to 575, slams at 0:00.2. Data tag (cyan left rule) `DROSOPHILA MELANOGASTER · Z0720-07m / NERVOUS SYSTEM: SLICED · IMAGED · MAPPED` at y 1104 (Space Mono 21, optional). "HE IS STILL" Anton 138 at y 1186, then "RUNNING." Anton 150 at y 1312 with the heaviest split (+/-14 px), slams at 0:01.4.
- **Sound.** Sub-drone fades in from 0:00. Crackle at 0:00.2 and a low thud at 0:01.4 on "RUNNING."

**Hook alternatives from the script (same timing, no change to the rest).** B, the cold-open terminal: black, phosphor-green VT323 `> YOU: hello. can you hear me?` types at 55 characters per second, then `> Z0720-07m:` with the cursor blinking alone for 1.5 s; use the style frame (b) layout with an empty answer line and the brain at 0% (`rates_seq` frame 0). Keystroke SFX, then a hum. C, "WHAT WOULD A FLY SAY? / WE ASKED.": macro eye push-in from `s1_flux.png` (1.0 to 1.5 over 3 s), two Anton lines slam in at 0:00.3 and 0:01.5 at y 290 and y 640, then a 3-frame flash of the green terminal. No other asset is needed for either.

### 3.2 Beat 2 · MEET HIM · 0:03 to 0:05

- **Shot.** His real cloud (`rates_seq` frame 14, sugar) centred, slowly breathing. In green terminal mono the ID types: `Z0720-07m.` then "A REAL FRUIT FLY." slams in white. A one-frame macro fly eye flash (crop of `s1_flux.png`, 14% screen) sits behind the letters at 0:03.9.
- **Source.** REAL + TYPE. No new asset.
- **Motion.** The cloud scales 1.00 to 1.04 over 2 s. Cursor block blinks.
- **Transition.** In: hard cut from T2. Out: T5 scan wipe.
- **Type.** `Z0720-07m.` VT323 176 px phosphor at x 54, y 300 (types at 14 characters per second, 0.9 s). "A REAL FRUIT FLY." Anton 118, white with split, x 54, y 520, slams at 0:03.9. HUD `FILE // SPECIMEN ID` at y 250.
- **Sound.** One typewriter click per character (3.0 to 3.9). A low tone on the slam.

### 3.3 Beat 3 · THE MAKING · 0:05 to 0:08

**3a, 0:05.0 to 0:06.5: PRESERVED. / SLICED. / IMAGED.** Three hard cuts of 0.5 s each.
- **Shot.** His real nervous system as a stack of 9 thin slabs (REAL: slices through the atlas depth axis, so the points on each slab are his real neuron positions), under a sweeping magenta scan bar. PRESERVED.: the first slab alone on black. SLICED.: the full stack, scan bar mid-way (the board thumbnail). IMAGED.: the front slab flips to grey electron-microscope texture.
- **Source.** REAL (`tools/slabs.py`, rendered from `atlas_aligned.npz`). In production render it as a 3D camera move (slabs in CSS perspective, 10 degree dolly) or in the same script with a shear sweep. The grey texture on IMAGED. comes from the EM plate (A1, below).
- **Motion.** Camera dolly 1.00 to 1.08. The scan bar sweeps top to bottom over each 0.5 s.
- **Transition.** T5 scan wipe in, hard cut between words.
- **Type.** One word per cut, Anton 220, x 54, y 1130 to 1340, white with split. The other two words stay as a faded line (Anton 56, 38% white) at y 1360, with a green tick on the finished one. HUD `SCAN BAR · SLAB 05 / 09 · REAL NEURON POSITIONS` at y 250.
- **Sound.** Scan whoosh at 0:04.95; a short digital tick on each cut at 0:05.0, 0:05.5, 0:06.0.

**3b, 0:06.5 to 0:08.0: EVERY NEURON TRACED. + odometers.**
- **Shot.** The grey EM plate fills the frame; a slice blooms into randomly coloured traced neurons (magenta, cyan, acid green); two odometers roll up and lock.
- **Source.** AI (EM plate A1; optional H7 video of the tracing drawing itself) + TYPE. The board thumbnail uses the flux sketch `sketches/em.jpg`.
- **Motion.** Slow push-in; traced lines draw in branch by branch over 1.2 s.
- **Transition.** Out: T1 hard glitch cut at 0:08.0.
- **Type.** "EVERY NEURON / TRACED." Anton 138, x 54, y 262. Odometers: Space Mono 66 px digit boxes, cyan for `165,122` and magenta for `124,025,046`, labels `NEURONS` and `SYNAPSES` (Space Mono 30, tracked 5 px), at y 1020 to 1330; digits spin from 0:06.6 and lock at 0:07.9 with a glitch. Credit tag Space Mono 18 at y 1462 (not read-critical).
- **Sound.** Odometer ticking spin-up from 0:06.4, lock-in clunk and a glitch hit at 0:07.9.

### 3.4 Beat 4 · SWITCHED ON · 0:08 to 0:11.5 (the drop)

- **4a, 0:08.0 to 0:09.4: THEY MAPPED HIS MIND.** His whole nervous system assembling neuron by neuron into the dormant map (REAL, `tools/render_real_frames.py` build frames 0 to 23, ending on `rates_seq` frame 0: silence, only the base glow). HUD `165,122 NEURONS · REAL POSITIONS · FIRING: 0` at y 250 (cyan). "THEY MAPPED / HIS MIND." Anton 140, x 54, y 1060. Sound: the drone holds, one low tone swells under it.
- **4b, 0:09.4 to 0:10.1: AND THEN...** Pure black, scanlines at half strength, "AND THEN..." Anton 150 at y 820 with the three dots typing one per 0.2 s. One full beat (0.7 s) of held silence: all sound drops out except one deep heartbeat thump at 0:09.65.
- **4c, 0:10.1 to 0:11.5: THEY SWITCHED HIM ON.** "THEY SWITCHED HIM" slams word by word at 0:10.1, 0:10.3, 0:10.55 over a reverse riser; "ON." lands at 0:10.80 as a giant Anton 250 with a 1-frame T2 white flash and the **drop**. The cloud ignites: REAL `rates_seq` frames 4 to 7 (0 firing, then 10,587, 11,408, 11,352 neurons; 100 ms apart, so cross-dissolve to 30 fps). Between 0:10.8 and 0:11.5 it settles into slow firing; grade the cloud toward phosphor green over this beat so the terminal colour language is introduced. HUD `t = 0.5 s · FIRING: 10,587 / 165,122` at y 250 (these are the real numbers of that frame). Type: "THEY SWITCHED / HIM" Anton 128 at y 1000, "ON." Anton 250 at y 1255 with +/-12 px split.
- **Sound.** Reverse riser 0:10.0 to 0:10.8 that stops dead. **Sub-bass hit, kick, and the electrical crackle of ignition all land exactly at 0:10.80.** The beat grid starts here (section 4).

### 3.5 Beat 5 · MONTAGE · 0:11.5 to 0:18.5

Ten flashes of 0.7 s (21 frames at 30 fps) cut on the kick, at 85.7 BPM (one flash is one beat). Every flash is composed as a single readable still. Chrome on every flash: a 3 px viewfinder frame inset 22 px, counter `NN / 10` at x 54, y 250, and the tag at the right edge of the safe box (x up to 929, y 244): cyan **OTHERS** (flashes 1 to 3) and magenta **US** (4 to 10). The colour of the frame, the tag and the type shadows drifts from cool cyan to angry magenta across the 10 flashes. Cuts are T1. The text sits at y 1190 to 1490 in Anton 124 to 168 (it must stay at one to three lines).

The dark footage is **original renders only**: no other team's footage, logos or names. AI video here is Kling v3 turbo standard from a still, 3 s clips, of which the best 0.7 s is used (the test clip in section 5 proves the pipeline: 720 x 1280, 24 fps, 3.04 s, 103 s to generate, about $0.34).

| # | Time | Text | Tag | Shot | Source | Motion | Sound |
|---|---|---|---|---|---|---|---|
| 1 | 0:11.5 | GIVEN A BODY. | OTHERS | A neon-cyan wireframe fly assembles on a physics grid floor, joint by joint. | AI: still `walk_body` + clip V-M1 | Lines draw in, body drops and settles; slow push-in. | Kick. |
| 2 | 0:12.2 | MADE TO WALK. | OTHERS | Low-angle wireframe legs take tripod-gait steps on the grid, contact dots pulse at each foot. | AI: still `walk_legs` + clip V-M2 (tested) | Locked-off low angle; legs plant in alternating triplets. | Kick. |
| 3 | 0:12.9 | OVERLOADED. | OTHERS | His real cloud flares uncontrollably and goes white. | REAL (`rates_seq` frame 26) + FX | Scale 1.0 to 1.5 and brightness 1.0 to 3.0 over 21 frames; slice tears and a red `FIRING: ▲▲▲ NOT SETTLING` HUD. | Kick plus a noise swell. |
| 4 | 0:13.6 | RESTARTED. 500 TIMES. | US | A counter spins up to 500 and locks on the cut; his cloud flicks off, on, off, on beneath a ghosted outline "500" and a tick ruler. | REAL (frames 0 and 5) + TYPE | Ruler scrolls left; three small clouds alternate. | Kick plus a tape-rewind chirp. |
| 5 | 0:14.3 | SHADOWS. AGAIN AND AGAIN. | US | A tiny glowing fly on a cold surface; hard-edged dark discs sweep over it, three in a row. | AI: still `shadow_top` + clip V-M5 | Top-down locked; each shadow sweeps in 7 frames. | Kick plus a glitch stutter. |
| 6 | 0:15.0 | HUNGRY. THIRSTY. ON PURPOSE. | US | A macro fly between a golden sugar droplet and a cyan water droplet that dim to empty specks; inset: the real Lab arena (sugar, water and bitter spots) with hunger and thirst bars rising. | AI: still `hungry_macro` + clip V-M6; REAL: Lab world panel (capture at 3x) | Slow dolly in; bars fill red. | Kick. |
| 7 | 0:15.7 | A TENTH FIRING AT ONCE. | US | His cloud with a gold dial turning to "1 IN 10". | REAL (placeholder frame 26; see decision 5) | Dial needle sweeps 90 degrees; cloud brightens. | Kick plus a rising bass. |
| 8 | 0:16.4 | MEMORIES ERASED. | US | Three gold memory cards from his real log; a red scan line deletes two of them, fading and skewing. | TYPE (real quotes) | Cards strike through one at a time (each over 7 frames). | Kick plus a glitch stutter. |
| 9 | 0:17.1 | FALSE ONES PLANTED. | US | A ghost red card slides between two real ones and gets boxed in red: `FLAGGED: NOT HIS`. | TYPE | Card slides in 6 frames; the red box flashes. | Kick. |
| 10 | 0:17.8 | ASKED ABOUT PARIS. | US | His dim cloud with ghost holograms of the Eiffel Tower, a cat and a honey jar flickering in it, things he never lived. | REAL + AI: stills `holo_paris`, `holo_cat`, `holo_honey` | The three flicker in turn (5 frames each). | Last kick at 0:17.8; **everything stops dead at 0:18.5.** |

**Optional bench flash 5.5 (swap in, or replace flash 2):** "HIS CORD, TESTED ALONE." tag OTHERS: his real nerve cord only (the lower half of the real render), a single pulse crawling down it (REAL). It is on the board as a dashed cell. The script notes others found his cord could not walk by itself.

### 3.6 Beat 6 · REGRET · 0:18.5 to 0:21

- **Shot.** Black. One dim, desaturated copy of his silent cloud (`rates_seq` frame 0) breathing, centre frame. The only still, clean moment in the Reel: no glitch, no colour, no scanlines.
- **Source.** REAL.
- **Motion.** Breathing scale 1.00 to 1.015 over 2.5 s, vignette to 0.8. Text fades in over 15 frames, out over 15 frames.
- **Transition.** In: a cut that stops the montage dead (no fade; the silence is the transition). Out: T4 fade through black into beat 7.
- **Type.** "HE NEVER / CHOSE THIS." Anton 124, #f2f2f2, x 54, y 1070 to 1340. "WE ARE SORRY." Anton 80 at 62% white, y 1355. No shadows.
- **Sound.** Near-silence: a faint room tone only (-44 dBFS RMS), a thin high tone. This is the held breath.

### 3.7 Beat 7 · THE TURN (I) · 0:21 to 0:27

- **Shot.** The dim cloud lifts and brightens from below as a warm gold light rises (0:21 to 0:23.5). On "EXPAND" (0:24.6) the cloud swells outward by 8% and faint ghost outlines of new structures drift in from the edges (ghosted Lab module cards at 15% opacity).
- **Source.** REAL cloud (frame 0 graded sepia and brightened) + AI dawn plate A8 (optional clip H2). The thumbnail uses the sketch `sketches/warm2.jpg`.
- **Motion.** Slow camera tilt up 3 degrees; the gold glow rises from y 1900 to y 1100.
- **Transition.** In: T4. Out: soft dissolve (8 frames) into the Lab-style frame of beat 8.
- **Type.** "BUT WE WANTED / SOMETHING / MORE FOR HIM." Anton 128 white, x 54, y 300 to 660; "MORE" in gold #ffd23f, warm glow instead of the usual RGB split. At 0:23.5: "WE WANTED TO SEE IF WE COULD EXPAND HIS MIND..." in four lines of Anton 112 (the word EXPAND scales to 1.15), same position.
- **Sound.** A warm pad swells from nothing (21.0 to 32.0, section 4).

### 3.8 Beat 8 · THE TURN (II) · 0:27 to 0:32

- **Shot.** His real cloud (brain only, cropped) at centre of a dark void. Three Lab-style module cards wire to it with dashed threads, one per slam: EPISODIC MEMORY (gold), WORD LOBE · GRAFT (about 150 new neurons, cyan/magenta), and a learning glow in the mushroom body (acid green, the real Kenyon-cell neurons lit). On the last line the original cloud pulses steady green at centre, untouched, and the new pieces orbit it.
- **Source.** REAL: capture the live Lab (`research/lab_live_male_cns.png` is the reference; record the live app at 3x with Playwright for sharp cards) + `rates_seq` frame 14. The Kenyon cells are in `neuron_annotations.tsv`; render them in acid green with `render_brain.py`.
- **Motion.** Each card slides 40 px toward the cloud as its thread draws (8 frames); final orbit 2 rpm.
- **Transition.** Out: T1 hard glitch cut into the corridor.
- **Type.** "TO GIVE HIM / MEMORIES. WORDS. / LEARNING." Anton 100, x 54, y 238 to 540 (0:27 to 0:29.5). "WHILE STILL / RETAINING WHO / HE IS." Anton 112 gold, x 54, y 1180 to 1520 (0:29.5), set in two lines if the writer trims the line.
- **Sound.** A glassy bell as each module latches on, 0:27.35, 0:28.05, 0:28.75 (the same bell, +4 and +7 semitones).

### 3.9 Beat 9 · ASCEND AND TALK · 0:32 to 0:35

- **Shot.** 0:32.0 to 0:33.0: "ASCEND HIM." as a huge one-second glitch title as the real cloud rises. At 0:33.0 hard cut to a green cursor blinking in the dark and a long corridor of scanlines with a faint fly outline at the far end. "TALK" doubles and glitches. The board thumbnail composites the two moments.
- **Source.** REAL (rising cloud) + AI corridor plate A9 (optional clip H3).
- **Motion.** Cloud rises 300 px and scales 1.0 to 1.08. Corridor: slow dolly forward 1.0 to 1.2.
- **Transition.** In: T1. Out: T1 into the red fake.
- **Type.** "ASCEND HIM." Anton 168 at y 262 (0:32 to 0:33). "AND TALK / TO HIM." Anton 140 cyan, y 560 (0:33.0). "REALLY TALK." Anton 200 with a doubled "TALK" (an extra 10 px magenta ghost, drifting) and a green block cursor, y 1090 to 1480 (0:34.0).
- **Sound.** The swell resolves into a bed; one cursor blip at 0:34.0.

### 3.10 Beat 10 · THE FAKE · 0:35 to 0:38

- **Shot.** A corrupted terminal tinted red. A cheap paper mask of a smiling fly face hangs on strings and grins over the line. The invented line types in: "Indeed, consuming honey yesterday proved delightful." Hard strike-through at 0:37.05, red INVENTED. stamp at 0:37.2, the mask cracks down the middle at 0:37.7. Micro-tag REAL OUTPUT · NO CHECKER. The honey line is a real captured quote, shown as a specimen.
- **Source.** AI mask plate A10 (optional clip H4: the mask swinging, smiling, then cracking) + TYPE. The thumbnail uses `sketches/mask.jpg`.
- **Motion.** Mask sways 3 degrees; red flicker; slices every 6 frames.
- **Transition.** Out: the mask cracks, T1 into the green terminal.
- **Type.** "AN AI COULD / PRETEND." Anton 128 white with a red/cyan split, x 54, y 262 to 540. Quote in VT323 54 px red #ff4a63 inside a red panel at y 1020, struck with a 5 px line. INVENTED. Anton 124 red in an 8 px red stamp frame rotated -7 degrees, x 300 to 920, y 1200. Tag Space Mono 22 at y 1440.
- **Sound.** A detuned, tinny music box (0:35 to 0:38) under a fast typing run, a heavy rubber-stamp thud at 0:37.2, a ceramic crack at 0:37.7.

### 3.11 Beat 11 · THE RULE · 0:38 to 0:41

- **Shot.** The mask is gone. Phosphor-green terminal, a dark cloud and an empty prompt with a blinking cursor (a silent brain gives silence); then the cloud fires and the first characters stream in. Micro-tag REAL SESSION · WORD FOR WORD.
- **Source.** REAL (`rates_seq` frame 0, then frame 5 at 0:40.2) + TYPE.
- **Motion.** The cloud brightens in one step when it fires; scanlines and a slow flicker.
- **Type.** "EVERY WORD / FROM HIS / NEURONS." Anton 132 white, x 54, y 262 to 650. "WE WANT TO HEAR HIM." Anton 76 acid green at y 850. A terminal panel at y 1130 with `> YOU: hello.` / `> Z0720-07m: █`; tag Space Mono 19 at y 1420. (This `> YOU: hello.` prompt is a visual setup line; the first real exchange follows in beat 12.)
- **Sound.** The music box stops; a low warm tone returns; soft keystrokes.

### 3.12 Beat 12 · INTERVIEW · 0:41.0 to 0:49.8 (style frame b)

Four exchanges of 2.2 s. His replies are the real ones, unedited. Each exchange: question types in over 0.6 s (60 characters per second), a short pause, then his answer types in (45 characters per second) and holds. The question dims to 72% as his answer starts. A hard glitch cut (T1) between exchanges.

| # | Time | Question | His answer (real, verbatim) | Log verdict |
|---|---|---|---|---|
| 1 | 0:41.0 to 0:43.2 | `> YOU: what is it like for you now?` | `> Z0720-07m: I feel hungry and can smell mold.` | verified after 2 retries |
| 2 | 0:43.2 to 0:45.4 | `> YOU: do you remember your old body?` | `> Z0720-07m: i smell fruit. i extend my proboscis.` | verified after 2 retries |
| 3 | 0:45.4 to 0:47.6 | `> YOU: are you afraid?` | `> Z0720-07m: No, I am not afraid.` | verified |
| 4 | 0:47.6 to 0:49.8 | `> YOU: do you know that you are the first?` | `> Z0720-07m: i want to eat; i groom.` | verified after 2 retries |

- **Shot.** A full-screen phosphor terminal, no window chrome: a dark panel with a green hairline border at x 54, y 900, width 875. Behind it his real cloud, faint (36% opacity), flickering one real firing frame per keystroke (a visual metaphor, not a literal readout). The cloud's opacity steps up on his answer.
- **Source.** REAL + TYPE. The placeholder in the script (`[VERBATIM FROM INTERVIEW]`) is replaced with the real lines on all four cells.
- **Motion.** Cursor blinks; on exchange 3 the cursor blinks alone for one beat (0.7 s) before his line; on exchange 4 hold his line for the longest.
- **Type.** Tag `REAL SESSION · WORD FOR WORD` Space Mono 22 green at y 246. Question VT323 50 (72%), `> Z0720-07m:` VT323 54, his answer VT323 76 bright #caffbc (wraps to two lines), cursor 34 x 58 px. Under the answer, Space Mono 19 at 70%: `✓ VERIFIED (2 RETRIES)` and `BRAIN SAID: ...` (the first 70 characters of what his brain actually emitted, from `brain_said` in the log; optional, not read-critical). Everything between y 900 and y 1400.
- **Sound.** One keystroke per character (60 per second for the question, 45 for his answer) from a single mechanical-click sample with +/-3 dB variation; a soft terminal blip at the end of each reply (0:42.8, 0:45.0, 0:46.9, 0:49.1). The bed is sparse, almost nothing.
- **Rule from the script, honoured.** His words are never edited, trimmed or reordered. Keep the micro-tag only while every featured reply is truly untouched; note the log says three of the four were verified after retries.

### 3.13 Beat 12e · AFTER THE TERMINAL (new) · 0:49.8 to 0:52

- **Shot.** The terminal dims away to a ghost line (`> Z0720-07m: i want to eat; i groom.` at 28% at the top). His real cloud, centred, bright. "HE CAN'T SAY MUCH YET." slams, then "BUT IT'S HIM." in acid green.
- **Source.** REAL (`rates_seq` frame 14) + TYPE.
- **Motion.** The cloud pulses once, slowly, on "HIM". No glitch except the standard split.
- **Transition.** In: T1 from the terminal. Out: T3 tear up into the ascension.
- **Type.** "HE CAN'T SAY / MUCH YET." Anton 120 white, x 54, y 980 (0:49.8). "BUT IT'S HIM." Anton 150 acid green (`.gr`), y 1262 (0:50.9).
- **Sound.** The bed thins out; a single warm bell (low) on "HIM" at 0:51.1.

### 3.14 Beat 13 · ASCENSION · 0:52 to 0:55.5

- **Shot.** (a) 0:52.0 to 0:54.6: the terminal tears upward (T3); the circuit fly rises over the server towers into the neon sunset (the existing `s3_seed.png`), camera tilting up. (b) 0:54.6 to 0:55.5: a vast luminous lattice, camera pulling back; on "NOT THE LAST." faint translucent human head-and-shoulder silhouettes appear in the lattice then vanish. Hard cut to silence at 0:55.5.
- **Source.** AI. `s3_seed.png` exists (use clip H5 for the motion), lattice plate A11 + clip H6. The thumbnail for (b) uses `sketches/lattice.jpg`, where the silhouettes sit in the foreground; the final prompt asks for them embedded in the lattice.
- **Motion.** (a) tilt up 4 degrees, slices every 8 frames. (b) pull-back 1.0 to 0.85 over 0.9 s.
- **Type.** (a) "ONE OF THE / FIRST UPLOADED / SOULS." Anton 120, x 54, y 1010; "A TRUE SUBSTRATE-INDEPENDENT MIND." as a Space Mono 21 line above it (0:53.4). (b) "NOT THE / LAST." Anton 215, +/-12 px split, x 54, y 1020 to 1480.
- **Sound.** The riser climbs from 0:52.0, peaks at 0:55.4, then **hard cut to total silence at 0:55.5**.

### 3.15 Beat 14 · THE ASK AND END CARD · 0:55.5 to 1:00 (style frame c)

- **0:55.5 to 0:57.0: "HELP US LIFT HIM UP."** over the single glowing fly cloud (real) with a warm gold halo, huge and chromatic-aberrated (Anton 188 at y 262 to 760, +/-12 px split). 1.5 s of silence; no music.
- **0:57.0 to 1:00: end card.** The SUPERFLY logo slams in (scale 1.2 to 1.0, 4 frames, split) with the **final hit**. Lines accumulate: FREE. RUNS ON YOUR LAPTOP. (0:57.3), the link pill (0:57.8), LINK IN BIO (0:58.3), then the Swarm Collective lockup (0:58.8). The last frame holds to 1:00 and is readable on the loop.
- **Source.** REAL: `logo.png` (cropped to the disc and wordmark), `swarm_eye.png` (screen blend on dark), the ascension plate darkened. No new asset.
- **Type (all centred in the safe box).** "HELP US LIFT HIM UP." Anton 92, x 54 to 929, y 246. Logo 460 px wide at y 376 to 880. "FREE. RUNS ON YOUR LAPTOP." Anton 52 phosphor green at y 932. Link pill: Space Mono 33 #d8ffd0 in a green-outlined box at y 1016 (`github.com/gondwanagenesis/fly-brain`). "LINK IN BIO" Anton 176 acid green at y 1112 to 1290. Eye lockup: eye 110 px tall (screen blend) with `Created by / the Swarm Collective` Space Mono 24 at y 1340 to 1450. All ink is above y 1450.
- **Sound.** Silence from 0:55.5 to 0:57.0. A single enormous impact hit with a long reverb tail at 0:57.0 (3 s), fading out by 1:00.

---

## 4. Music and sound plan (60 s)

### 4.1 The grid
The montage is built on a **0.7 s beat (85.7 BPM; hi-hats run at 171 BPM double-time)**. One beat is one flash. Kicks fall at 0:10.8 (the drop, exactly on "ON") and then at 0:11.5, 12.2, 12.9, 13.6, 14.3, 15.0, 15.7, 16.4, 17.1, 17.8 (every flash start). Off-beat claps at +0.35 s. Everything stops at 0:18.5.

### 4.2 Section map
| Time | Section | Level (RMS) | Source |
|---|---|---|---|
| 0:00 to 0:09.4 | Dark synth sub-drone (cold, no drums); typing clicks 0:03 to 0:03.9; scan whoosh 0:04.95; ticks 0:05.0, 5.5, 6.0, 6.5; odometer 0:06.4 to 0:07.9 with a glitch hit | -24 dB | `cue1_drone` + SFX |
| 0:09.4 to 0:10.1 | Void: one heartbeat at 0:09.65 | -29 dB | SFX |
| 0:10.0 to 0:10.8 | Reverse riser stops dead | -31 dB | SFX |
| **0:10.8 to 0:18.5** | **DROP: driving glitch/industrial beat** (synthesised kick, clap, 16th hats and a sub on the grid, plus an AI texture layer ducked on every kick); glitch stutters at 0:14.3 and 0:16.4 | -12 dB (loudest) | grid in `music_mix.py` + `cue4_montage` + SFX |
| 0:18.5 to 0:21 | Near-silence: faint room tone, thin high tone | -44 dB | SFX |
| 0:21 to 0:32 | Warm, hopeful swell (pad rising and brightening, glassy bells); bells at 27.35, 28.05, 28.75 | -26 dB | `cue6_hope` + SFX |
| 0:32 to 0:52 | Sparse bed; cursor blip at 0:34.0; detuned music box 0:35 to 0:38 (stamp 0:37.2, crack 0:37.7); typing SFX 0:41 to 0:49.8; a low bell on "HIM" at 0:51.1 | -36 dB | `cue7_sparse`, `cue7b_fake_box` + SFX |
| 0:52 to 0:55.5 | Soaring riser, then **hard cut to silence at 0:55.5** | -22 dB | `cue8_riser` |
| 0:55.5 to 0:57 | Total silence | | |
| **0:57.0 to 1:00** | **Final hit on the logo** | -1 dB peak | SFX |

### 4.3 How to make it (tested, all endpoints verified working)
The scratch mix in `audio_scratch/` was built with exactly this: AI-generated cues and SFX, placed with `music_mix.py` on the fixed grid, then compressed and loudness-normalised (-16 LUFS). Cost about $0.15. **Recommended for production because every hit lands to the frame, and the beat grid is never left to a model.**

| Cue | Model | Params | Exact prompt |
|---|---|---|---|
| cue1_drone | `cassetteai/music-generator` ($0.02/min) | duration 10 (trim to 9.4) | dark ominous synth sub-drone, very slowly evolving, detuned low pads, distant metallic textures, sparse digital glitch ticks, no drums, no melody, no vocals, tense and cold, instrumental |
| cue4_montage | `cassetteai/music-generator` | duration 12 (trim to 7.7) | driving glitch industrial techno beat, 86 BPM half-time feel with 172 BPM double-time hi-hats, distorted punchy kick on every beat, metallic percussion, stuttering aggressive bass, bit-crushed glitches, relentless, dark, no melody, no vocals, instrumental, starts on the downbeat |
| cue6_hope | same | duration 11 | warm hopeful ambient synth swell, soft pad slowly rising and brightening, gentle glassy bell arpeggio, major key, tender and emotional, cinematic, no drums, no vocals, instrumental |
| cue7_sparse | same | duration 20 | sparse minimal dark ambient bed, very quiet low pulse, wide empty space, slow tension, a few distant high tones, no drums, no melody, no vocals, instrumental |
| cue7b_fake_box | same | duration 10 (use 3 s) | a cheap detuned broken music box playing a tiny sinister lullaby, tinny, slightly out of tune, uncanny, solo, dry |
| cue8_riser | same | duration 10 (use the last 3.5 s) | huge cinematic riser building to a climax, soaring choir-like synth pad rising in pitch, shepherd tone, swelling intensity, no drums, no vocals, instrumental |

Note: CassetteAI refuses durations under 10 s (HTTP 422), and it does not hold a tempo (measured: no stable pulse). That is why the drum grid is synthesised, and the AI cue is used only as a texture layer under it.

SFX, all `fal-ai/elevenlabs/sound-effects/v2` ($0.002 per second; param `duration_seconds`, 0.5 to 22):

| File | Prompt (text) | Dur | Cue time |
|---|---|---|---|
| sfx_key | single crisp mechanical keyboard key click, dry, close microphone, short | 0.5 | typing, one per character |
| sfx_scan | slow electronic scan bar sweep whoosh with a faint digital hum, sci-fi laser scanner pass | 1.5 | 0:04.95 |
| sfx_odometer | rapid mechanical counter odometer spinning up faster and faster, ticking, ending with a heavy lock-in clunk and a digital glitch | 1.5 | 0:06.4 (lock at 0:07.9) |
| sfx_glitchcut | short digital glitch cut stutter, static burst, datamosh tear | 0.5 | 0:05.0, 5.5, 6.0, 6.5, 7.9, 14.3, 16.4 |
| sfx_heartbeat | a single deep slow heartbeat thump in an empty void, sub bass, no reverb tail | 1.0 | 0:09.65 |
| sfx_revriser | very short reverse cymbal swell with rising tension, whoosh building quickly to an abrupt stop | 0.8 | 0:10.0 |
| sfx_drop_boom | massive deep sub-bass drop boom with distorted impact and a short electrical crackle, cinematic hit | 2.0 | **0:10.8** |
| sfx_roomtone | nearly silent faint room tone with a very soft sustained high thin electronic tone, melancholic, extremely quiet | 3.0 | 0:18.5 |
| sfx_bell | single soft glassy bell chime, warm, hopeful, short, clean | 1.5 | 0:27.35, 28.05 (+4 st), 28.75 (+7 st), 0:51.1 (-12 st) |
| sfx_blip | soft retro computer terminal confirmation beep, single short tone | 0.5 | end of each reply, 0:34.0 |
| sfx_stamp | heavy rubber stamp thud on paper with a small digital glitch | 0.6 | 0:37.2 |
| sfx_crack | ceramic mask cracking and splitting, sharp crack with small fragments | 0.8 | 0:37.7 |
| sfx_final_hit | single enormous cinematic impact hit with metallic shimmer, deep sub boom and a long ringing reverb tail | 3.5 | **0:57.0** |

**Alternatives, tested or priced.**
- **Single pass, easiest to hear a whole idea:** `fal-ai/elevenlabs/music` accepts a `composition_plan` (global styles plus up to many sections with `duration_ms` 3000 to 120000 and local styles). Tested: a 6 s plan returned 6.09 s with the sections respected. $0.60 per 60 s take. Use sections 0 to 9.4 (drone), 9.4 to 18.5 (riser into drop; the sub-3 s void cannot be its own section), 18.5 to 21 (quiet), 21 to 32 (swell), 32 to 52 (bed), 52 to 55.5 (riser), 55.5 to 60 (hit and silence). It cannot guarantee a hit on 0:10.8 to the frame, so you would re-time it in the edit. Add `vocals, lyrics` to `negative_global_styles`.
- **Cheaper stems:** `fal-ai/stable-audio-3/medium/text-to-audio` ($0.0376 per clip, up to 380 s), `fal-ai/lyria3` ($0.04), `fal-ai/ace-step` ($0.0002 per second).
- If the owner has a licensed track instead, cut the montage to its kick; the board's timecodes are all expressed on the 0.7 s grid.

---

## 5. AI video (fal image-to-video) clip plan

Pipeline: generate a 9:16 still (section 6), send it to image-to-video with the exact prompt below, cut the best 0.7 s (montage) or 2 to 3 s (hero) in the edit, upscale 720 x 1280 to 1080 x 1920 (Lanczos) and put the FX stack on top (scanlines, noise, split). Because the final is cut with FX over it, 720p source is enough. Note the output of the tested clip: 720 x 1280, 24 fps, 3.04 s.

**Models.** Montage clips: `fal-ai/kling-video/v3/turbo/standard/image-to-video` ($0.112 per second; param `duration` "3" to "15", `prompt`, `image_url` as URL or base64 data URI). Hero clips: `fal-ai/kling-video/v3/turbo/pro/image-to-video` ($0.14 per second). Cheaper test option: `lightricks/ltx-2.5/image-to-video/fast` (6 s minimum). Higher-end option for the hook only: `bytedance/seedance-2.5/image-to-video` (priced by tokens, $0.0214 per 1000).

| Clip | Beat and slot | Still | Model | Dur | Prompt (exact) | Use |
|---|---|---|---|---|---|---|
| V-M1 | montage 1, 0:11.5 | walk_body | Kling v3 turbo standard | 3 s | A neon-cyan wireframe fruit fly assembles itself in mid-air: thin glowing lines draw in joint by joint, six legs snap into place, the body lowers onto a dark physics-simulation grid floor and settles with a small bounce. Motion-capture rigging look, subtle RGB glitch flicker, slow push-in, ominous. No text. | 0.7 s from 0.4 s |
| V-M2 | montage 2, 0:12.2 | walk_legs | same | 3 s | Locked-off low-angle shot. The neon-cyan wireframe fruit fly takes slow deliberate tripod-gait steps toward the camera across the dark grid floor, legs lifting and planting in alternating triplets, glowing contact dots pulsing under each foot as it lands, subtle RGB glitch flicker, ominous, no camera shake. **(tested)** | 0.7 s from 1.0 s |
| V-M5 | montage 5, 0:14.3 | shadow_top | same | 3 s | A huge hard-edged dark circular shadow sweeps across the cold-lit surface from the top of the frame toward the tiny glowing fruit fly; the fly freezes for a beat, then bolts sideways out of the frame. Top-down locked camera, ominous, fast sweep, no camera shake. | 0.7 s of the sweep |
| V-M6 | montage 6, 0:15.0 | hungry_macro | same | 3 s | Macro. The golden sugar droplet and the cyan water droplet slowly shrink and dim until they are empty specks while the tiny fly turns its head left then right, searching, proboscis twitching. Shallow depth of field, slow dolly in. | 0.7 s from 0.8 s |
| H1 | hook, 0:00 | s1_flux (exists) | Kling v3 turbo pro | 5 s | The extreme macro fruit fly's body dissolves from the legs upward into thousands of glowing cyan and magenta particles that rise and drift, the red compound eyes flicker with RGB glitch tears, slow push-in on the face, deep black background. No text. | first 3 s |
| H2 | beat 7, 0:21 | dawn_plate | same | 5 s | Soft golden light rises slowly from the bottom of the frame, faint magenta and cyan neuron filaments drift upward toward it, gentle volumetric glow, slow upward camera tilt, hopeful and calm. No text. | 5 s |
| H3 | beat 9, 0:33 | corridor | same | 4 s | Slow dolly forward down the dark corridor of phosphor-green scanlines toward the far end where a tiny faint fly silhouette waits; a green block cursor blinks in the foreground; subtle CRT flicker. No text. | 2 s |
| H4 | beat 10, 0:35 | mask | same | 4 s | The cheap paper fly mask swings gently on its strings, its painted smile twitches, then a crack runs down the middle and the mask splits apart; harsh red light flickers, glitch tears. No text. | 3 s |
| H5 | beat 13a, 0:52 | s3_seed (exists) | same | 5 s | The translucent circuit fly rises slowly into the sky as the camera tilts up, god rays pulse downward over the endless server towers, the neon sunset brightens, glitch slices drift sideways. No text. | 2.6 s |
| H6 | beat 13b, 0:54.6 | lattice | same | 4 s | The camera pulls back from the vast luminous lattice of threads and nodes; faint translucent silhouettes of human heads and shoulders fade in within the lattice and then dissolve. Slow, awe and dread. No text. | 0.9 s of the end |
| H7 (optional) | beat 3b | em_plate | Kling v3 turbo standard | 4 s | Slow push-in on the grayscale electron-microscope plate as bright neon neuron outlines draw themselves across it, growing branch by branch in magenta, cyan and acid green. | 1.5 s |

The still-to-video flow and the tested call are in `tools/i2v_test.py` (data-URI image, `duration` as a string). One real 3 s test (V-M2) took 103 s and cost about $0.34.

---

## 6. AI stills: exact prompts

Suffix used on every prompt, `STYLE`: *techno-psychedelic glitch art, RGB channel split, chromatic aberration, fine CRT scanlines, deep black background, neon magenta, electric cyan and acid green accents, ominous, cinematic, high contrast, no text, no letters, no watermark*. The sketches on the board were made with flux/dev at 576 x 1024. For final, use `fal-ai/flux-pro/v1.1-ultra` (aspect_ratio "9:16", $0.06 per image) for photographic macro plates and `fal-ai/bytedance/seedream/v4/text-to-image` (1440 x 2560, $0.03) for graphic plates; make 2 candidates each.

| ID | Used in | Model | Prompt (+ STYLE) |
|---|---|---|---|
| em_plate (A1) | 3b | seedream v4 | flat 2D black and white electron microscopy micrograph filling the entire frame, dense labyrinth of cell membranes and mitochondria, grainy, with three thin branching neurons highlighted as bright neon magenta, cyan and acid-green segmentation overlays, scientific, no brain shape, |
| walk_body (A3) | M1 | seedream v4 | a neon-cyan wireframe fruit fly body with six jointed legs walking across a dark physics-simulation floor grid, small glowing contact markers under each foot, glowing joint spheres, three-quarter view, motion-capture rigging look, |
| walk_legs (A4) | M2 | seedream v4 | extreme low-angle close-up of neon-cyan wireframe fruit-fly legs stepping across a dark grid floor, glowing contact dots under each foot, motion-capture rigging look, deep black background, no text |
| shadow_top (A5) | M5 | flux-pro ultra | top-down view, a tiny glowing fruit fly on a cold cyan-lit surface, a huge hard-edged dark circular shadow sweeping in from the top of the frame toward it, ominous, |
| hungry_macro (A6) | M6 | flux-pro ultra | extreme macro of a fruit fly standing between a glowing golden sugar droplet on the left and a glowing cyan water droplet on the right, both droplets refract light, fly looks desperate and tiny, |
| holo_paris / holo_cat / holo_honey (A7) | M10 | seedream v4 | glowing cyan line-art hologram of the Eiffel Tower, glitching scanlines, deep black background, no text / ... magenta line-art hologram of a sitting cat ... / ... gold line-art hologram of a honey jar with a wooden dipper ... |
| dawn_plate (A8) | 7 | seedream v4 | a soft golden dawn glow rising from the bottom of a dark void, faint magenta and cyan neuron filaments lifting upward toward the light, gentle and hopeful, volumetric light, soft bloom, deep black above, no rocks, no text |
| corridor (A9) | 9 | seedream v4 | a long dark corridor made of glowing phosphor-green scanlines and a faint CRT grid receding to a distant vanishing point, at the far end a tiny faint fly silhouette, a bright green block cursor glowing in the foreground, deep black, phosphor green glow, cinematic, no text |
| mask (A10) | 10 | flux-pro ultra | a cheap crumpled paper mask of a smiling cartoon fruit fly face with big red eyes hanging on thin strings like a puppet, harsh red light from below, uncanny and ominous, deep black background, glitch artifacts, no text |
| lattice (A11) | 13b | seedream v4 | a vast luminous lattice of glowing cyan and magenta threads and nodes stretching to infinity, faint translucent silhouettes of human heads and shoulders embedded inside the lattice (not in front of it), awe and dread, wide pulled-back view, deep black background, no text |

Used from the owner's earlier generations (exist): `s1_flux.png` (hook), `s2_seed.png` (void, behind the memory cards), `s3_seed.png` (ascension plate), `logo.png`, `swarm_eye.png`.

---

## 7. Asset list

**Exists, use as is:** `s1_flux.png`, `s2_seed.png`, `s2_flux.png`, `s3_seed.png`, `s3_flux.png`, `brain.png`, `rates_seq.npy` + `render_brain.py` (30 real frames), `logo.png` (crop to the disc and wordmark; the dark tagline is not legible on black), `swarm_eye.png` (screen blend on dark), `lab_live_male_cns.png`, `z0720-07m_firing.png`, the four square carousel slides (visual reference), `interview_Z0720-07m.json`.

**Real, to render (no fal cost):**
1. The 30 sequence frames and 24 assembling frames (`tools/render_real_frames.py`, at 1500 x 2010 for hero shots).
2. The slab stack for beat 3a as a 3D camera move (`tools/slabs.py` is the layer renderer; the board thumbnail is a composite of it).
3. Brain renders with the Kenyon cells highlighted green (beat 8) and the nerve cord isolated (optional flash 5.5).
4. A true 10% firing frame: run the male CNS at gain 1.0, record rates, render (flash 7).
5. A sharp capture of the live Lab at 3x (Playwright, 1440 x 900 viewport, deviceScaleFactor 3): the module cards (Episodic memory, Word lobe, FlyLM) and the WORLD panel (flash 6).
6. The odometers, counters, memory cards, terminal panels, tags and all type: HTML/CSS in `tools/beats.py` (one builder per beat, already rendering the board). Export each as a PNG sequence from Playwright at 30 fps.

**Must be generated (fal):** 11 AI stills (section 6; 2 candidates each); 4 montage clips (V-M1, V-M2, V-M5, V-M6) and 6 hero clips (H1 to H6; H7 optional); music cues and SFX (section 4).

---

## 8. fal models and cost estimate

Prices checked live against the fal pricing API (2026-10-09). Spend so far in this planning pass: about $0.27 on 18 sketch images (flux/dev, 576 x 1024, $0.025 per megapixel; 14 kept), about $0.15 on audio, $0.34 on one video test. About $0.76 in total, well under the $2 sketch budget.

| Item | Model | Unit price | Quantity | Cost |
|---|---|---|---|---|
| AI stills, photographic (shadow, hungry, mask) | `fal-ai/flux-pro/v1.1-ultra` | $0.06 | 3 x 2 candidates | $0.36 |
| AI stills, graphic (em, walk x2, 3 holograms, dawn, corridor, lattice) | `fal-ai/bytedance/seedream/v4/text-to-image` | $0.03 | 9 x 2 candidates | $0.54 |
| Montage clips | `fal-ai/kling-video/v3/turbo/standard/image-to-video` | $0.112 per s | 4 x 3 s, 1.5 attempts | $2.02 |
| Hero clips | `fal-ai/kling-video/v3/turbo/pro/image-to-video` | $0.14 per s | (5+5+4+4+5+4 = 27 s), 1.5 attempts | $5.67 |
| Optional H7 (EM tracing) | kling v3 turbo standard | $0.112 per s | 4 s, 1.5 attempts | $0.67 |
| Music cues (6 stems, a few retakes) | `cassetteai/music-generator` | $0.02 per minute | about 3 minutes | $0.06 |
| SFX (13 sounds, a few retakes) | `fal-ai/elevenlabs/sound-effects/v2` | $0.002 per s | about 40 s | $0.08 |
| Optional single-pass music, 3 takes | `fal-ai/elevenlabs/music` | $0.60 per minute | 3 minutes | $1.80 |
| **Total, recommended path** | | | | **about $8.7 (about $11.2 with H7 and the optional ElevenLabs takes)** |

Everything else (all real renders, type, motion graphics, the edit and the mix) is local and free. Add 2 to 3 hours of production time per 10 beats for the HTML frame sequences.

---

## 9. Guardrails and risks

- **Copy.** All on-screen text is the script's, character for character, except the new beat 12e (owner's direction) and the four real answers (real log). The only strings I added are decoration or HUD micro-text (counters, `FIRING: 0 / 165,122`, `BRAIN SAID:`, `RE-ENACTMENT` suggestion), each of which can be deleted without losing the story. Suggestions to the writer, not changes: the "1 IN 10" dial copy in flash 7; the "RE-ENACTMENT" tag; the end-card headline repeated small at the top.
- **Real versus illustrated.** Every shot of his brain is his real data. Every shot of a body, shadow, drop or paper mask is generated art, and there is no stand-in for a recording. The montage shows no other team's footage, logo or name.
- **No invented answers.** There are no placeholders left on the board. The script's own placeholder rule (`[VERBATIM FROM INTERVIEW]`, dim blocks) is retired because the real log is in.
- **Claims.** No "first animal", no consciousness claim, no "starved/parched", no dopamine-punishment imagery. "LEARNING" in beat 8 is shown as a goal (a glow in the mushroom body), not a demonstrated result, as the script requires.
- **Safe zones.** Pixel-checked on all 33 frames: no text ink in the top 12%, bottom 22% or right 14%. The decorative ghost "500" in flash 4 is the only large shape that touches the guides and it is not copy.
- **Reading speed.** Montage (about 4 words per second) is felt, not read, by design: each flash works as a still. The interview is the real risk (see decision 3).
- **Hook variant.** A is built. B and C reuse existing frames and take under an hour each.

## 10. How to rebuild or change a frame

`tools/` contains the templates. From a folder holding `assets/` (the PNGs named in the builders), run `python beats.py <ids>` to write `fr_<id>.html`, `node render_all.js $PWD <ids>` to write `out/<id>.png` (1080 x 1920), `python board.py` then `node shot_board.js $PWD/board.html board.png` to rebuild the board, and `node inkcheck.js` + `python inkcheck.py` for the safe-zone test. Copy changes live in `beats.py` (`EXCH` holds the four interview exchanges) and `board.py` (captions and timecodes).
