import json, os, sys, urllib.request, concurrent.futures as cf
KEY = os.environ["FAL_KEY"]; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sketch")
os.makedirs(OUT, exist_ok=True)
STYLE = ("techno-psychedelic glitch art, RGB channel split, chromatic aberration, fine CRT scanlines, deep black background, "
         "neon magenta, electric cyan and acid green accents, ominous, cinematic, high contrast, no text, no letters, no watermark")
JOBS = {
 "slices": "a long row of dozens of thin translucent glass-like cyan slabs standing in sequence receding into depth, the silhouette of a fruit fly sliced through all of them so each slab holds one thin cross-section of the fly, glowing slab edges, a magenta scan line crossing, macro lab feel, " + STYLE,
 "em": "flat 2D black and white electron microscopy micrograph filling the entire frame, dense labyrinth of cell membranes and mitochondria, grainy, "
       "with three thin branching neurons highlighted as bright neon magenta, cyan and acid-green segmentation overlays, scientific, no brain shape, " + STYLE,
 "walk": "a neon-cyan wireframe fruit fly body with six jointed legs walking across a dark physics-simulation floor grid, small glowing contact markers under each foot, "
         "glowing joint spheres, three-quarter view, motion-capture rigging look, " + STYLE,
 "shadow": "top-down view, a tiny glowing fruit fly on a cold cyan-lit surface, a huge hard-edged dark circular shadow sweeping in from the top of the frame toward it, ominous, " + STYLE,
 "bitter": "extreme macro of a fruit fly proboscis recoiling from a droplet of dark toxic acid-green bitter liquid, refractive droplet, red compound eye blurred in background, dramatic rim light, " + STYLE,
 "hungry": "extreme macro of a fruit fly standing between a glowing golden sugar droplet on the left and a glowing cyan water droplet on the right, both droplets refract light, fly looks desperate and tiny, " + STYLE,
 "warm": "dawn of warm golden light breaking over a dark neural void, god rays, faint luminous magenta and cyan neuron filaments reaching toward the light, hopeful, soft bloom, sacred and gentle, " + STYLE,
 "regret": "a single tiny fruit fly silhouette resting on a dark glowing windowsill edge in near total darkness, faint cyan rim light, quiet, melancholic, vast empty space, " + STYLE,
}

JOBS.update({
 "corridor": "a long dark corridor made of glowing phosphor-green scanlines and a faint CRT grid receding to a distant vanishing point, at the far end a tiny faint fly silhouette, a bright green block cursor glowing in the foreground, deep black, phosphor green glow, cinematic, no text",
 "mask": "a cheap crumpled paper mask of a smiling cartoon fruit fly face with big red eyes hanging on thin strings like a puppet, harsh red light from below, uncanny and ominous, deep black background, glitch artifacts, no text",
 "lattice": "a vast luminous lattice of glowing cyan and magenta threads and nodes stretching to infinity, faint translucent silhouettes of human heads and shoulders embedded in the lattice, awe and dread, wide pulled-back view, deep black background, no text",
 "paris": "glowing cyan line-art hologram of the Eiffel Tower, glitching scanlines, deep black background, no text",
 "cat": "glowing magenta line-art hologram of a sitting cat, glitching scanlines, deep black background, no text",
 "honey": "glowing gold line-art hologram of a honey jar with a wooden dipper, glitching scanlines, deep black background, no text",
 "warm2": "a soft golden dawn glow rising from the bottom of a dark void, faint magenta and cyan neuron filaments lifting upward toward the light, gentle and hopeful, volumetric light, soft bloom, deep black above, no rocks, no text",
 "legs": "extreme low-angle close-up of neon-cyan wireframe fruit-fly legs stepping across a dark grid floor, glowing contact dots under each foot, motion-capture rigging look, deep black background, no text",
})
MODEL = "fal-ai/flux/dev"
def run(name):
    body = {"prompt": JOBS[name], "image_size": {"width": 576, "height": 1024}, "num_images": 1, "num_inference_steps": 28,
            "enable_safety_checker": False, "output_format": "png"}
    req = urllib.request.Request(f"https://fal.run/{MODEL}", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Key {KEY}", "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    path = os.path.join(OUT, f"{name}.png"); urllib.request.urlretrieve(r["images"][0]["url"], path); return path
names = sys.argv[1:] or list(JOBS)
with cf.ThreadPoolExecutor(4) as ex:
    futs = {ex.submit(run, n): n for n in names}
    for f in cf.as_completed(futs):
        try: print("ok", f.result(), flush=True)
        except Exception as e: print("FAIL", futs[f], e, flush=True)
