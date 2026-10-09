"""SUPERFLY Reel scratch mix: places AI cues + SFX at the storyboard's sync points (60 s, 44.1 kHz stereo).
The montage drum grid is synthesised on an exact 0.7 s beat (85.714 BPM) so every flash lands on a kick.
usage: python music_mix.py <audio_dir> <out.wav>"""
import sys, os, numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
SR = 44100; D = 60.0
A = sys.argv[1]; OUT = sys.argv[2]
mix = np.zeros((int(SR * D), 2), np.float32)
def load(name, trim=None, start=0.0):
    sr, x = wavfile.read(os.path.join(A, name + '.wav')); x = x.astype(np.float32) / 32768
    if x.ndim == 1: x = np.stack([x, x], 1)
    x = x[int(start * SR):]
    if trim: x = x[:int(trim * SR)]
    return x
def db(d): return 10 ** (d / 20)
def put(x, t, gain_db=0.0, fi=0.0, fo=0.0):
    x = x.copy() * db(gain_db); n = len(x)
    if fi > 0: x[:int(fi * SR)] *= np.linspace(0, 1, int(fi * SR))[:, None][:len(x[:int(fi*SR)])]
    if fo > 0:
        k = min(int(fo * SR), n); x[n - k:] *= np.linspace(1, 0, k)[:, None]
    i = int(t * SR); j = min(i + n, len(mix))
    if i < len(mix): mix[i:j] += x[:j - i]
def putn(x, t, target, fi=0.0, fo=0.0):
    """normalise x to a target RMS (dBFS) before placing, so cue levels are predictable"""
    r = np.sqrt((x ** 2).mean()) + 1e-9
    put(x, t, target - 20 * np.log10(r), fi, fo)
def mono(x): return np.stack([x, x], 1).astype(np.float32)
rng = np.random.default_rng(3)
# ---- synthesised drum grid ------------------------------------------------
def kick():
    t = np.arange(int(0.42 * SR)) / SR; f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR; y = np.sin(ph) * np.exp(-t * 7.5)
    y += 0.5 * np.exp(-t * 400) * rng.standard_normal(len(t))
    return mono(np.tanh(y * 2.4) * 0.9)
def clap():
    t = np.arange(int(0.22 * SR)) / SR; n = rng.standard_normal(len(t))
    sos = butter(4, [900, 6000], btype='band', fs=SR, output='sos'); y = sosfilt(sos, n) * np.exp(-t * 20)
    return mono(np.tanh(y * 2.0) * 0.55)
def hat():
    t = np.arange(int(0.05 * SR)) / SR; n = rng.standard_normal(len(t))
    sos = butter(4, 7000, btype='high', fs=SR, output='sos'); return mono(sosfilt(sos, n) * np.exp(-t * 90) * 0.28)
def sub(dur, f=41.2):
    t = np.arange(int(dur * SR)) / SR; y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t)
    y *= np.minimum(1, t * 200) * np.exp(-t * 3.0); return mono(np.tanh(y * 1.8) * 0.5)
BEAT = 0.7; DROP = 10.8; END = 18.5
K, C, H, S = kick(), clap(), hat(), sub(0.34)
t = DROP; i = 0
while t < END - 0.02:
    put(K, t, -1); put(S, t, -4)
    put(C, t + BEAT / 2, -6); put(S, t + BEAT / 2, -9)
    for k in range(4): put(H, t + k * BEAT / 4, -10 if k % 2 else -7)
    t += BEAT; i += 1
# AI texture layer under the grid (cut hard at 18.5)
tex = load('cue4_montage', trim=END - DROP)
duck = np.ones(len(tex), np.float32)
for b in np.arange(0, END - DROP, BEAT):
    i0 = int(b * SR); n = int(0.28 * SR); duck[i0:i0 + n] *= np.linspace(0.25, 1, min(n, len(duck) - i0))
putn(tex * duck[:, None], DROP, -19, fi=0.0, fo=0.04)
# ---- cues ------------------------------------------------------------------
putn(load('cue1_drone', trim=9.4), 0.0, -24, fi=0.8, fo=0.5)                    # cold open drone 0-9.4
put(load('sfx_scan'), 4.95, -8)                                               # scan bar sweep
put(load('sfx_odometer'), 6.4, -8)                                            # odometers spin up, lock ~7.9
put(load('sfx_glitchcut'), 7.9, -6)                                           # numbers lock (glitch)
put(load('sfx_heartbeat'), 9.65, -7)                                          # AND THEN... (held breath)
put(load('sfx_revriser'), 10.0, -5)                                           # THEY SWITCHED HIM ... riser, stops at 10.8
put(load('sfx_drop_boom'), DROP, -2, fo=0.05)                                 # DROP on "ON"
for t0 in [5.0, 5.5, 6.0, 6.5]: put(load('sfx_glitchcut'), t0, -13)     # hard-cut ticks: PRESERVED / SLICED / IMAGED / TRACED
put(load('sfx_glitchcut'), 14.3, -8); put(load('sfx_glitchcut'), 16.4, -8)    # flash 5 and 8 stutters
putn(load('sfx_roomtone', trim=3.0), 18.5, -44, fi=0.3, fo=0.5)                # regret: near silence
putn(load('cue6_hope', trim=11.0), 21.0, -26, fi=4.0, fo=1.0)                   # hopeful swell 21-32
putn(load('cue7_sparse', trim=20.0), 32.0, -36, fi=1.5, fo=1.0)                # sparse bed 32-52
putn(load('cue7b_fake_box', trim=3.0), 35.0, -28, fi=0.1, fo=0.15)              # the fake: detuned music box 35-38
def pitched(x, r):
    n = int(len(x) / r); idx = np.linspace(0, len(x) - 1, n); return np.stack([np.interp(idx, np.arange(len(x)), x[:, c]) for c in (0, 1)], 1).astype(np.float32)
bell = load('sfx_bell')
for t0, r in [(27.35, 1.0), (28.05, 1.26), (28.75, 1.5)]: put(pitched(bell, r), t0, -14)   # a bell as each new module latches on
put(pitched(bell, 0.5), 51.1, -12)                                            # one warm low note on "HIM"
put(load('sfx_blip'), 34.0, -16)                                              # REALLY TALK. cursor blip
put(load('sfx_stamp'), 37.2, -4); put(load('sfx_crack'), 37.7, -5)            # INVENTED stamp, mask cracks
putn(load('cue8_riser', trim=3.5, start=6.5), 52.0, -22, fi=1.2, fo=0.02)       # ascension riser 52-55.5, hard cut
put(load('sfx_final_hit', trim=3.0), 57.0, -1, fo=0.4)                        # final hit on the logo
# ---- typing clicks (one keystroke sample, random gain/pitch-free) -----------
key = load('sfx_key', trim=0.25)
def typing(start, n, cps, gain=-12):
    for k in range(n): put(key, start + k / cps + rng.uniform(-.004, .004), gain + rng.uniform(-3, 2))
typing(3.0, 10, 14)          # Z0720-07m.  (beat 2)
typing(35.75, 52, 40)        # the invented specimen line (beat 10)
# real interview, 4 exchanges x 2.2 s (41.0-49.8): question typed fast, pause, answer typed, blip
for s0, nq, a0, na in [(41.00, 35, 41.95, 33), (43.20, 31, 44.05, 37), (45.40, 22, 46.40, 20), (47.60, 35, 48.55, 23)]:
    typing(s0, nq, 60, -9); typing(a0, na, 45, -7); put(load('sfx_blip'), a0 + na / 45 + 0.08, -16)
# ---- master ---------------------------------------------------------------
peak = np.abs(mix).max(); mix *= db(-2.5) / peak
wavfile.write(OUT, SR, (mix * 32767).astype(np.int16)); print('wrote', OUT, 'peak', peak)
