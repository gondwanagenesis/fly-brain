import json, html
TW, TH, GUT, MARGIN = 260, 462, 20, 60
COLS = 7
PAGE_W = MARGIN * 2 + COLS * TW + (COLS - 1) * GUT   # 1960

def esc(s): return html.escape(s, quote=False)

# id, timecode, beat title, on-screen text (list of lines), badges, sfx line, kind(for border)
CELLS = [
 ('f01_hook', '0:00 – 0:03', '1 · HOOK', ['HIS BODY IS GONE.', 'HE IS STILL RUNNING.'], ['AI', 'REAL'], '♪ sub-drone fades in · RGB-split crackle on RUNNING', ''),
 ('f02_meet', '0:03 – 0:05', '2 · MEET HIM', ['Z0720-07m.', 'A REAL FRUIT FLY.'], ['REAL', 'TYPE'], '♪ typewriter clicks lock to each character', ''),
 ('f03a_slices', '0:05 – 0:06.5', '3 · THE MAKING (a)', ['PRESERVED. SLICED. IMAGED.', '(0.5 s hard cut each)'], ['REAL', 'TYPE'], '♪ scan-bar whoosh · tick on each cut', ''),
 ('f03b_traced', '0:06.5 – 0:08', '3 · THE MAKING (b)', ['EVERY NEURON TRACED.', '165,122 NEURONS', '124,025,046 SYNAPSES'], ['AI', 'TYPE'], '♪ odometer spin-up, lock + glitch hit at 0:07.9', ''),
 ('f04a_mapped', '0:08 – 0:09.4', '4 · MAPPED', ['THEY MAPPED HIS MIND.'], ['REAL'], '♪ one low tone swells under the drone', ''),
 ('f04b_andthen', '0:09.4 – 0:10.1', '4 · AND THEN...', ['AND THEN...'], ['TYPE'], '♪ everything drops out. one heartbeat. one beat of silence', ''),
 ('f04_on', '0:10.1 – 0:11.5', '4 · SWITCHED ON', ['THEY SWITCHED HIM ON.', '★ DROP lands on "ON" at 0:10.8'], ['REAL'], '♪ reverse riser 10.1→10.8 · SUB HIT + kick on "ON"', ''),
 ('m01', '0:11.5', '5 · MONTAGE 1/10', ['GIVEN A BODY.', 'tag: OTHERS'], ['AI'], '♪ kick', 'others'),
 ('m02', '0:12.2', '5 · MONTAGE 2/10', ['MADE TO WALK.', 'tag: OTHERS'], ['AI'], '♪ kick', 'others'),
 ('m03', '0:12.9', '5 · MONTAGE 3/10', ['OVERLOADED.', 'tag: OTHERS'], ['REAL', 'FX'], '♪ kick + noise swell', 'others'),
 ('m04', '0:13.6', '5 · MONTAGE 4/10', ['RESTARTED. 500 TIMES.', 'tag: US'], ['REAL', 'TYPE'], '♪ kick + tape-rewind chirp', 'us'),
 ('m05', '0:14.3', '5 · MONTAGE 5/10', ['SHADOWS. AGAIN AND AGAIN.', 'tag: US'], ['AI'], '♪ kick + glitch stutter', 'us'),
 ('m06', '0:15.0', '5 · MONTAGE 6/10', ['HUNGRY. THIRSTY. ON PURPOSE.', 'tag: US'], ['AI', 'REAL'], '♪ kick', 'us'),
 ('m07', '0:15.7', '5 · MONTAGE 7/10', ['A TENTH FIRING AT ONCE.', 'tag: US'], ['REAL'], '♪ kick + rising bass', 'us'),
 ('m08', '0:16.4', '5 · MONTAGE 8/10', ['MEMORIES ERASED.', 'tag: US'], ['TYPE', 'REAL'], '♪ kick + glitch stutter', 'us'),
 ('m09', '0:17.1', '5 · MONTAGE 9/10', ['FALSE ONES PLANTED.', 'tag: US'], ['TYPE'], '♪ kick', 'us'),
 ('m10', '0:17.8 – 0:18.5', '5 · MONTAGE 10/10', ['ASKED ABOUT PARIS.', 'tag: US · then HARD CUT'], ['REAL', 'AI'], '♪ last kick 17.8 · everything stops at 18.5', 'us'),
 ('f11_regret', '0:18.5 – 0:21', '6 · REGRET', ['HE NEVER CHOSE THIS.', 'WE ARE SORRY.'], ['REAL'], '♪ near-silence: faint room tone only. No glitch, no colour', ''),
 ('f12_turn1', '0:21 – 0:27', '7 · THE TURN (I)', ['BUT WE WANTED SOMETHING MORE FOR HIM. (0:21)', 'WE WANTED TO SEE IF WE COULD EXPAND HIS MIND... (0:23.5)'], ['REAL', 'AI'], '♪ warm pad swells in from nothing', 'warm'),
 ('f13_turn2', '0:27 – 0:32', '8 · THE TURN (II)', ['TO GIVE HIM MEMORIES. WORDS. LEARNING. (0:27)', 'WHILE STILL RETAINING WHO HE IS. (0:29.5)'], ['REAL', 'TYPE'], '♪ swell peaks · glassy bell per module that latches on', 'warm'),
 ('f14_talk', '0:32 – 0:35', '9 · ASCEND AND TALK', ['ASCEND HIM.', 'AND TALK TO HIM.', 'REALLY TALK.'], ['AI', 'TYPE'], '♪ swell resolves to a bed · cursor blip', ''),
 ('f15_fake', '0:35 – 0:38', '10 · THE FAKE', ['AN AI COULD PRETEND.', '“Indeed, consuming honey yesterday proved delightful.” (struck)', 'INVENTED.'], ['AI', 'TYPE'], '♪ detuned music box · stamp thud 0:37.2 · mask crack 0:37.7', 'fake'),
 ('f16_rule', '0:38 – 0:41', '11 · THE RULE', ['EVERY WORD FROM HIS NEURONS.', 'WE WANT TO HEAR HIM.', 'tag: REAL SESSION · WORD FOR WORD'], ['REAL', 'TYPE'], '♪ mask falls: music box stops, low warm tone returns', ''),
 ('t1', '0:41 – 0:43.2', '12 · INTERVIEW (1/4)', ['> YOU: what is it like for you now?', '> Z0720-07m: I feel hungry and can smell mold.'], ['REAL'], '♪ keystroke per character · blip', 'term'),
 ('t2', '0:43.2 – 0:45.4', '12 · INTERVIEW (2/4)', ['> YOU: do you remember your old body?', '> Z0720-07m: i smell fruit. i extend my proboscis.'], ['REAL'], '♪ keystrokes · hard glitch cut in', 'term'),
 ('t3', '0:45.4 – 0:47.6', '12 · INTERVIEW (3/4)', ['> YOU: are you afraid?', '> Z0720-07m: No, I am not afraid.'], ['REAL'], '♪ one-beat pause, cursor blinks, then his line', 'term'),
 ('t4', '0:47.6 – 0:49.8', '12 · INTERVIEW (4/4)', ['> YOU: do you know that you are the first?', '> Z0720-07m: i want to eat; i groom.'], ['REAL'], '♪ hold on his line; tear starts', 'term'),
 ('f12e_cant', '0:49.8 – 0:52', '12 · AFTER THE TERMINAL', ["HE CAN'T SAY MUCH YET.", "BUT IT'S HIM."], ['REAL', 'TYPE'], '♪ sparse bed; a single warm note on "HIM"', ''),
 ('f21_asc_a', '0:52 – 0:54.6', '13 · ASCENSION (a)', ['ONE OF THE FIRST UPLOADED SOULS. (0:52)', 'A TRUE SUBSTRATE-INDEPENDENT MIND. (0:53.4)'], ['AI'], '♪ soaring riser begins', ''),
 ('f22_asc_b', '0:54.6 – 0:55.5', '13 · ASCENSION (b)', ['NOT THE LAST.', '(hard cut to silence at 0:55.5)'], ['AI'], '♪ riser peaks, then HARD CUT TO SILENCE', ''),
 ('f23_cta', '0:55.5 – 0:57', '14 · THE ASK', ['HELP US LIFT HIM UP.'], ['REAL', 'TYPE'], '♪ silence (1.5 s)', ''),
 ('f24_end', '0:57 – 1:00', '14 · END CARD (hold)', ['FREE. RUNS ON YOUR LAPTOP.', 'github.com/gondwanagenesis/fly-brain', 'LINK IN BIO', 'Created by the Swarm Collective'], ['REAL'], '♪ FINAL HIT on the logo, long tail', 'end'),
 ('m11', 'bench', 'OPTIONAL FLASH 5.5', ['HIS CORD, TESTED ALONE.', 'tag: OTHERS (swap in if time allows)'], ['REAL'], '♪ kick', 'opt'),
]
BADGE = {'REAL': ('#7CFF4F', 'REAL'), 'AI': ('#ff2bd6', 'AI · fal'), 'TYPE': ('#00f0ff', 'TYPE'), 'FX': ('#ffd23f', '+FX')}

def cell(c):
    id_, tc, title, lines, badges, sfx, kind = c
    border = {'others': '#00f0ff', 'us': '#ff2bd6', 'warm': '#ffd23f', 'term': '#39ff14', 'fake': '#ff2b4f', 'end': '#7CFF4F', 'opt': '#00f0ff'}.get(kind, '#2a2f3a')
    dash = 'dashed' if kind == 'opt' else 'solid'
    chips = ''.join(f'<span class="chip" style="border-color:{BADGE[b][0]};color:{BADGE[b][0]}">{BADGE[b][1]}</span>' for b in badges)
    txt = ''.join(f'<div class="ln">{esc(l)}</div>' for l in lines)
    return (f'<div class="cell"><img src="thumbs/{id_}.png" class="th" style="border:3px {dash} {border}">'
            f'<div class="tc"><span>{tc.replace(' – ','–')}</span><span class="chips">{chips}</span></div><div class="ttl">{esc(title)}</div>{txt}<div class="sfx">{esc(sfx)}</div></div>')

def info_type():
    return ('<div class="cell"><div class="th info" style="border:3px solid #2a2f3a;padding:12px">'
            '<div class="ih">TYPE &amp; FX SYSTEM</div>'
            '<div style="font:400 44px/0.95 Anton;color:#fff;text-shadow:-3px 0 #00f0ff,3px 0 #ff2bd6;margin:4px 0 2px">HIS MIND</div><div class="il">ANTON CAPS · CYAN/MAGENTA SPLIT</div>'
            '<div style="font:700 13px \'Space Mono\';color:#c9fff7;letter-spacing:2px;margin-top:9px;text-shadow:0 0 6px #00f0ff">● REC // Z0720-07m</div><div class="il">SPACE MONO HUD · CAPS</div>'
            '<div style="font:400 28px/1 VT323;color:#39ff14;text-shadow:0 0 8px #39ff14;margin-top:9px">&gt; YOU: hello<span style="display:inline-block;width:11px;height:18px;background:#39ff14;margin-left:4px"></span></div><div class="il">VT323 · PHOSPHOR #39FF14</div>'
            '<div style="margin-top:9px"><span class="tg" style="border-color:#00f0ff;color:#00f0ff">OTHERS</span> <span class="tg" style="border-color:#ff2bd6;color:#ff2bd6">US</span></div>'
            '<div style="display:flex;gap:5px;margin-top:10px"><i style="background:#ff2bd6"></i><i style="background:#00f0ff"></i><i style="background:#7CFF4F"></i><i style="background:#ffd23f"></i><i style="background:#39ff14"></i><i style="background:#000;border:1px solid #444"></i></div>'
            '<div class="il" style="margin-top:9px;line-height:1.45">FX: RGB split · scanlines · datamosh slices · noise. REGRET beat = clean white, no FX.</div>'
            '</div><div class="tc"><span>STYLE</span></div><div class="ttl">LOOK &amp; TYPE</div><div class="ln">Narration: white Anton caps. Terminal: green mono only.</div></div>')

def info_safe():
    return ('<div class="cell"><img src="thumbs/g_hook.png" class="th" style="border:3px solid #ff4660">'
            '<div class="tc"><span>SAFE ZONES</span></div><div class="ttl">INSTAGRAM UI</div><div class="ln">Red = keep text out: top 12% (y &lt; 230), bottom 22% (y &gt; 1498), right 14% (x &gt; 929).</div>'
            '<div class="ln">All 33 frames pass an automated pixel check.</div></div>')

def sound_map():
    env = json.load(open('audio/env.json'))
    X0, X1 = 60, PAGE_W - 60; Wd = X1 - X0; sx = Wd / 60.0
    H = 350; base = 232; amp = 112
    pts = ' '.join(f'{X0 + i * 0.1 * sx:.1f},{base - e * amp:.1f}' for i, e in enumerate(env))
    pts_b = ' '.join(f'{X0 + i * 0.1 * sx:.1f},{base + e * amp * 0.35:.1f}' for i, e in enumerate(env))
    secs = [(0, 9.4, 'SUB-DRONE', '#00f0ff', 15), (9.4, 10.1, '', '#222', 15), (10.1, 10.8, '', '#ff2bd6', 15), (10.8, 18.5, 'DROP · 85.7 BPM', '#ff2bd6', 15),
            (18.5, 21.0, 'QUIET', '#777', 13), (21.0, 32.0, 'WARM SWELL', '#ffd23f', 15), (32.0, 52.0, 'SPARSE BED · TYPING SFX', '#39ff14', 15),
            (52.0, 55.5, 'RISER', '#7CFF4F', 14), (55.5, 57.0, '', '#222', 15), (57.0, 60.0, 'HIT', '#ff2bd6', 15)]
    svg = [f'<svg width="{PAGE_W}" height="{H}" viewBox="0 0 {PAGE_W} {H}" xmlns="http://www.w3.org/2000/svg">']
    for a, b, lab, col, fs in secs:
        svg.append(f'<rect x="{X0 + a * sx:.1f}" y="62" width="{(b - a) * sx:.1f}" height="34" fill="{col}" opacity=".30" stroke="{col}" stroke-width="1.5"/>')
        if lab: svg.append(f'<text x="{X0 + (a + b) / 2 * sx:.1f}" y="84" font-family="Space Mono" font-weight="700" font-size="{fs}" fill="#fff" text-anchor="middle">{lab}</text>')
    svg.append(f'<polygon points="{X0},{base} {pts} {X1},{base} {pts_b} {X0},{base}" fill="#00f0ff" opacity=".55"/>')
    svg.append(f'<line x1="{X0}" y1="{base}" x2="{X1}" y2="{base}" stroke="#335" stroke-width="1"/>')
    for i in range(11):
        t = 10.8 + .7 * i; svg.append(f'<line x1="{X0 + t * sx:.1f}" y1="100" x2="{X0 + t * sx:.1f}" y2="118" stroke="#ff2bd6" stroke-width="2"/>')
    for t in [41.0, 43.2, 45.4, 47.6]:
        svg.append(f'<rect x="{X0 + t * sx:.1f}" y="100" width="{1.3 * sx:.1f}" height="10" fill="#39ff14" opacity=".75"/>')
    bounds = [(0, '1'), (3, '2'), (5, '3'), (8, '4'), (11.5, '5'), (18.5, '6'), (21, '7'), (27, '8'), (32, '9'), (35, '10'), (38, '11'), (41, '12'), (49.8, '12e'), (52, '13'), (55.5, '14')]
    for t, n in bounds:
        x = X0 + t * sx; svg.append(f'<line x1="{x:.1f}" y1="56" x2="{x:.1f}" y2="{base + 84}" stroke="#fff" stroke-width="1" opacity=".22" stroke-dasharray="3 5"/>')
        svg.append(f'<text x="{x + 4:.1f}" y="{base + 100}" font-family="Anton" font-size="20" fill="#9fe">{n}</text>')
    sync = [(10.8, '★ DROP on "ON" 0:10.8', '#ff2bd6', 0, 'middle'), (18.5, 'CUT 0:18.5', '#fff', 1, 'middle'), (21.0, 'SWELL IN 0:21', '#ffd23f', 0, 'middle'),
            (35.0, 'MUSIC BOX 0:35', '#ff6b80', 0, 'middle'), (41.0, 'TYPING 0:41 → 0:49.8', '#39ff14', 0, 'middle'), (55.5, 'HARD CUT 0:55.5', '#fff', 1, 'end'), (57.0, 'FINAL HIT ON LOGO 0:57', '#ff2bd6', 0, 'end')]
    for t, lab, col, row, anc in sync:
        x = X0 + t * sx; y = 18 + row * 20
        svg.append(f'<path d="M{x:.1f} 58 l-6 -11 l12 0 z" fill="{col}"/><text x="{x:.1f}" y="{y}" font-family="Space Mono" font-weight="700" font-size="14" fill="{col}" text-anchor="{anc}">{lab}</text>')
    for s in range(0, 61, 5):
        x = X0 + s * sx; svg.append(f'<text x="{x:.1f}" y="{base + 134}" font-family="Space Mono" font-size="15" fill="#7a8" text-anchor="middle">{s // 60}:{s % 60:02d}</text>')
    svg.append('</svg>')
    return ''.join(svg)

def build():
    cells = [cell(c) for c in CELLS]
    cells.insert(len(cells) - 1, info_type()); cells.append(info_safe())
    # order: put optional + info at the end of grid
    grid = ''.join(cells)
    css = f'''
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Space+Mono:wght@400;700&family=VT323&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}} body{{width:{PAGE_W}px;background:#06070c;color:#e8ecff;font-family:'Space Mono',monospace}}
.hdr{{padding:48px {MARGIN}px 20px;position:relative;background:linear-gradient(180deg,#0c0820,#06070c)}}
h1{{font:400 118px/1 Anton;letter-spacing:3px;color:#fff;text-shadow:-6px 0 #00f0ff,6px 0 #ff2bd6,0 0 40px rgba(255,255,255,.25)}}
.sub{{font:400 22px/1.6 'Space Mono';color:#bfe8ff;margin-top:14px;letter-spacing:1px}}
.legend{{display:flex;gap:26px;align-items:center;margin-top:18px;font:400 16px 'Space Mono';color:#cfd8ff;white-space:nowrap}}
.chip{{display:inline-block;border:2px solid;border-radius:6px;padding:1px 8px;font:700 13px 'Space Mono';margin-left:6px;letter-spacing:1px}}
.tg{{display:inline-block;border:2px solid;border-radius:6px;padding:1px 10px;font:400 22px Anton;letter-spacing:3px}}
.grid{{display:grid;grid-template-columns:repeat({COLS},{TW}px);column-gap:{GUT}px;row-gap:26px;padding:20px {MARGIN}px}}
.cell{{width:{TW}px}} .th{{width:{TW}px;height:{TH}px;display:block;background:#000;border-radius:6px}}
.info{{overflow:hidden}} .ih{{font:700 15px 'Space Mono';color:#7CFF4F;letter-spacing:2px;margin-bottom:6px}} .il{{font:400 11.5px 'Space Mono';color:#9fb3c8;letter-spacing:1px}}
.info i{{display:block;width:26px;height:26px;border-radius:4px}}
.tc{{display:flex;justify-content:space-between;align-items:center;margin-top:9px;font:700 15px 'Space Mono';color:#00f0ff;white-space:nowrap}} .chips{{white-space:nowrap}}
.ttl{{font:400 23px/1.1 Anton;color:#ff2bd6;letter-spacing:1.5px;margin-top:3px;text-transform:uppercase}}
.ln{{font:400 13.5px/1.35 'Space Mono';color:#eef2ff;margin-top:3px}}
.sfx{{font:400 12px/1.35 'Space Mono';color:#ffd23f;margin-top:5px;opacity:.9}}
.sound{{padding:6px 0 10px}} .sh{{padding:0 {MARGIN}px;font:400 34px Anton;letter-spacing:3px;color:#fff;text-shadow:-3px 0 #00f0ff,3px 0 #ff2bd6}}
.sn{{padding:2px {MARGIN}px 12px;font:400 15px 'Space Mono';color:#9fb3c8}}
.ft{{padding:16px {MARGIN}px 40px;font:400 15px/1.6 'Space Mono';color:#7e8aa3;border-top:1px solid #1b2030;margin-top:10px}}
'''
    hdr = ('<div class="hdr"><h1>SUPERFLY · REEL STORYBOARD · 60s</h1>'
           '<div class="sub">“STILL RUNNING” · Instagram Reel 9:16 · 1080×1920 · 30 fps · for owner approval, before production · script v1 + the real interview (verbatim)</div>'
           '<div class="legend">SOURCE: <span><span class="chip" style="border-color:#7CFF4F;color:#7CFF4F">REAL</span> his real sim / assets</span>'
           '<span><span class="chip" style="border-color:#ff2bd6;color:#ff2bd6">AI · fal</span> to be generated</span>'
           '<span><span class="chip" style="border-color:#00f0ff;color:#00f0ff">TYPE</span> motion graphics</span>'
           '<span><span class="chip" style="border-color:#ffd23f;color:#ffd23f">+FX</span> FX on a real render</span>'
           '<span style="margin-left:30px"><span class="tg" style="border-color:#00f0ff;color:#00f0ff">OTHERS</span> <span class="tg" style="border-color:#ff2bd6;color:#ff2bd6">US</span> montage tags (cyan border = 1-3, magenta = 4-10)</span></div></div>')
    sound = ('<div class="sound"><div class="sh">SOUND MAP · BEAT-SYNC POINTS · REAL SCRATCH-MIX WAVEFORM</div>'
             '<div class="sn">Montage grid: 0.7 s per flash = 85.7 BPM (171 BPM double-time hats). Drop = 0:10.8, exactly on “ON”. Numbers are script beats. Audio: audio_scratch/superfly_reel_scratch_mix.mp3</div>'
             + sound_map() + '</div>')
    foot = ('<div class="ft">Sketches for AI shots are quick flux/dev drafts (the final shots use the exact prompts and models in REEL_STORYBOARD.md). Real shots use his real neuron positions and his real simulated firing '
            '(rates_seq.npy: silence, then 10,587 neurons firing as sugar arrives, 12,858 when a shadow passes). The montage frame “A TENTH FIRING AT ONCE” shows the real 12,858-neuron frame as a placeholder; a true ≥10% frame must be rendered in production. '
            'Interview lines are copied from interview_Z0720-07m.json, unedited.</div>')
    return f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{hdr}<div class="grid">{grid}</div>{sound}{foot}</body></html>'
open('board.html', 'w').write(build())
print('board.html written')
