import json, os, sys, urllib.request, concurrent.futures as cf, subprocess
KEY = os.environ["FAL_KEY"]; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
def post(model, body):
    req = urllib.request.Request(f"https://fal.run/{model}", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Key {KEY}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))
def fetch(url, name):
    raw = os.path.join(OUT, name + ".raw"); urllib.request.urlretrieve(url, raw)
    wav = os.path.join(OUT, name + ".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-ar", "44100", "-ac", "2", wav], check=True); os.remove(raw)
    return wav
# (name, model, body)
CASS = "cassetteai/music-generator"; EL = "fal-ai/elevenlabs/sound-effects/v2"
JOBS = [
 ("cue1_drone", CASS, {"prompt": "dark ominous synth sub-drone, very slowly evolving, detuned low pads, distant metallic textures, sparse digital glitch ticks, no drums, no melody, no vocals, tense and cold, instrumental", "duration": 10}),
 ("cue4_montage", CASS, {"prompt": "driving glitch industrial techno beat, 86 BPM half-time feel with 172 BPM double-time hi-hats, distorted punchy kick on every beat, metallic percussion, stuttering aggressive bass, bit-crushed glitches, relentless, dark, no melody, no vocals, instrumental, starts on the downbeat", "duration": 12}),
 ("cue6_hope", CASS, {"prompt": "warm hopeful ambient synth swell, soft pad slowly rising and brightening, gentle glassy bell arpeggio, major key, tender and emotional, cinematic, no drums, no vocals, instrumental", "duration": 11}),
 ("cue7_sparse", CASS, {"prompt": "sparse minimal dark ambient bed, very quiet low pulse, wide empty space, slow tension, a few distant high tones, no drums, no melody, no vocals, instrumental", "duration": 20}),
 ("cue7b_fake_box", CASS, {"prompt": "a cheap detuned broken music box playing a tiny sinister lullaby, tinny, slightly out of tune, uncanny, solo, dry", "duration": 10}),
 ("cue8_riser", CASS, {"prompt": "huge cinematic riser building to a climax, soaring choir-like synth pad rising in pitch, shepherd tone, swelling intensity, no drums, no vocals, instrumental", "duration": 10}),
 ("sfx_key", EL, {"text": "single crisp mechanical keyboard key click, dry, close microphone, short", "duration_seconds": 0.5}),
 ("sfx_scan", EL, {"text": "slow electronic scan bar sweep whoosh with a faint digital hum, sci-fi laser scanner pass", "duration_seconds": 1.5}),
 ("sfx_odometer", EL, {"text": "rapid mechanical counter odometer spinning up faster and faster, ticking, ending with a heavy lock-in clunk and a digital glitch", "duration_seconds": 1.5}),
 ("sfx_heartbeat", EL, {"text": "a single deep slow heartbeat thump in an empty void, sub bass, no reverb tail", "duration_seconds": 1.0}),
 ("sfx_revriser", EL, {"text": "very short reverse cymbal swell with rising tension, whoosh building quickly to an abrupt stop", "duration_seconds": 0.8}),
 ("sfx_drop_boom", EL, {"text": "massive deep sub-bass drop boom with distorted impact and a short electrical crackle, cinematic hit", "duration_seconds": 2.0}),
 ("sfx_roomtone", EL, {"text": "nearly silent faint room tone with a very soft sustained high thin electronic tone, melancholic, extremely quiet", "duration_seconds": 3.0}),
 ("sfx_stamp", EL, {"text": "heavy rubber stamp thud on paper with a small digital glitch", "duration_seconds": 0.6}),
 ("sfx_crack", EL, {"text": "ceramic mask cracking and splitting, sharp crack with small fragments", "duration_seconds": 0.8}),
 ("sfx_glitchcut", EL, {"text": "short digital glitch cut stutter, static burst, datamosh tear", "duration_seconds": 0.5}),
 ("sfx_blip", EL, {"text": "soft retro computer terminal confirmation beep, single short tone", "duration_seconds": 0.5}),
 ("sfx_bell", EL, {"text": "single soft glassy bell chime, warm, hopeful, short, clean", "duration_seconds": 1.5}),
 ("sfx_final_hit", EL, {"text": "single enormous cinematic impact hit with metallic shimmer, deep sub boom and a long ringing reverb tail", "duration_seconds": 3.5}),
]
def run(job):
    name, model, body = job
    r = post(model, body)
    url = (r.get("audio_file") or r.get("audio") or {}).get("url")
    return fetch(url, name)
only = set(sys.argv[1:])
with cf.ThreadPoolExecutor(4) as ex:
    futs = {ex.submit(run, j): j[0] for j in JOBS if not only or j[0] in only}
    for f in cf.as_completed(futs):
        try: print("ok", f.result(), flush=True)
        except Exception as e: print("FAIL", futs[f], e, flush=True)
