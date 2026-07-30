"""FlyBrain Studio -- a live 3D interface to the whole-brain simulation.

    .venv\\Scripts\\python.exe flyloop\\studio.py
    then open http://127.0.0.1:8765

What it is: the 138,639-neuron connectome rendered at its real FlyWire
coordinates, driven by the native kernel in a background thread, with the
membrane model switchable at run time. Every point is a real neuron and lights
when that neuron actually spikes -- the render is the simulation's output, not
an animation of it.

DESIGN NOTES
------------
* Only the standard library serves it. No Flask, no websockets, no build step:
  the whole thing is one Python file plus one HTML file, and it runs from the
  same venv as the simulation. A tool that needs its own install is a tool that
  stops working six months later.

* The simulation runs in its own thread and the HTTP handlers never touch the
  engine directly. `engine.step()` is a ctypes call, which releases the GIL for
  its whole duration, so the sim thread and the server genuinely overlap rather
  than time-slicing.

* Activity decays once per FRAME, not once per step. Applying an exponential
  decay to a 138,639-element array 10,000 times a second in Python would cost
  far more than the simulation it is displaying. Spikes are accumulated over
  the frame and the decay applied once, which is exact for display purposes and
  three orders of magnitude cheaper.

* Switching model does NOT rebuild the engine (see NativeBrainEngine.set_model).
  The connectome, the delay ring, the tiling and the permutation are properties
  of the network, not of the membrane equation, so a switch costs milliseconds.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

import models as nrn_models                      # noqa: E402
from native_engine import NativeBrainEngine      # noqa: E402

HERE = Path(__file__).resolve().parent
WEB = HERE / "studio"
DATA = ROOT / "data"

# The two protocols the published paper actually runs.
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940639259967, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]
P9 = [720575940627652358, 720575940635872101]

REAL_TIME_MS_PER_STEP = 0.1     # dt; 1.0x real time == 10,000 steps/second


class Sim:
    """The simulation thread and everything the browser can see of it."""

    def __init__(self, data_dir=DATA, model="lif_euler", threads=4):
        self.lock = threading.Lock()
        atlas = np.load(DATA / "flywire_meta" / "atlas_aligned.npz",
                        allow_pickle=False)
        self.xyz = atlas["xyz"]
        self.cat = atlas["cat"]
        self.region = atlas["region"]
        self.cat_labels = [str(x) for x in atlas["cat_labels"]]
        self.cat_colors = [str(x) for x in atlas["cat_colors"]]
        self.region_labels = [str(x) for x in atlas["region_labels"]]
        self.n_regions = len(self.region_labels)

        print("loading connectome ...", flush=True)
        t0 = time.perf_counter()
        self.engine = NativeBrainEngine(
            data_dir=str(data_dir), stim_ids=SUGAR, seed=1234,
            threads=threads, reorder="cell_type", model=model)
        self.N = self.engine.N
        print(f"  {self.N} neurons, {time.perf_counter()-t0:.1f}s, "
              f"ISA={self.engine.isa}", flush=True)

        # The atlas is in ORIGINAL csv order; the engine renumbers neurons by
        # cell_type for tile locality. Map once, here, so the render never has
        # to think about it: perm[i] is the original index of engine slot i.
        p = self.engine.perm
        if p is not None:
            self.xyz = self.xyz[p]
            self.cat = self.cat[p]
            self.region = self.region[p]

        self.act = np.zeros(self.N, dtype=np.float32)
        self.act_u8 = np.zeros(self.N, dtype=np.uint8)
        self.region_of = self.region.astype(np.int64)

        self.running = True
        self.speed = 1.0            # multiples of real time; 0 = unlimited
        self.rate_hz = 200.0
        self.protocol = "sugar"
        self.decay_ms = 40.0        # visual persistence of a spike

        self.ms_per_step = 0.0
        self.spikes_per_step = 0.0
        self.live_frac = 100.0
        self.sim_ms = 0.0
        self.wall_speed = 0.0
        self.region_rate = np.zeros(self.n_regions, dtype=np.float32)
        self.history = []           # (sim_ms, spikes/step) for the raster strip

        self.engine.inject(self.rate_hz)
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------- thread
    def _loop(self):
        target_fps = 30.0
        steps = 64
        while not self._stop:
            if not self.running:
                time.sleep(0.02)
                continue

            with self.lock:
                eng = self.engine
                t0 = time.perf_counter()
                hits = []
                nsp = 0
                for _ in range(steps):
                    n = eng.step()
                    nsp += n
                    if n:
                        hits.append(eng.spike_idx.copy())
                wall = time.perf_counter() - t0

                # One decay for the whole frame. The per-step form would be
                # 138,639 multiplies ten thousand times a second, which costs
                # more than the simulation.
                d = float(np.exp(-(steps * REAL_TIME_MS_PER_STEP) / self.decay_ms))
                self.act *= d
                if hits:
                    idx = np.concatenate(hits)
                    self.act[idx] = 1.0
                    counts = np.bincount(self.region_of[idx],
                                         minlength=self.n_regions)
                else:
                    counts = np.zeros(self.n_regions)

                np.clip(self.act * 255.0, 0, 255, out=self.act_u8,
                        casting="unsafe")

                self.ms_per_step = wall / steps * 1e3
                self.spikes_per_step = nsp / steps
                self.sim_ms = eng.t_ms
                self.wall_speed = REAL_TIME_MS_PER_STEP / max(self.ms_per_step, 1e-9)
                bits = np.unpackbits(eng.tile_live.view(np.uint8),
                                     bitorder="little")
                self.live_frac = 100.0 * int(bits[:eng.n_tiles].sum()) / eng.n_tiles
                sim_ms_elapsed = steps * REAL_TIME_MS_PER_STEP
                self.region_rate = (counts / (sim_ms_elapsed / 1000.0)).astype(
                    np.float32)
                self.history.append((self.sim_ms, self.spikes_per_step))
                if len(self.history) > 600:
                    del self.history[:len(self.history) - 600]

            # Pace to the requested multiple of real time. speed == 0 means
            # "as fast as the machine allows", which is the mode that answers
            # "can this run faster than the fly does".
            if self.speed > 0:
                want = steps * REAL_TIME_MS_PER_STEP / 1000.0 / self.speed
                slack = want - wall
                if slack > 0:
                    time.sleep(slack)
            # Keep the frame near 1/30 s so the browser gets smooth updates
            # without the sim thread being dominated by Python overhead.
            if wall < 1.0 / target_fps * 0.6:
                steps = min(steps * 2, 16384)
            elif wall > 1.0 / target_fps * 1.6:
                steps = max(steps // 2, 1)

    # --------------------------------------------------------------- control
    def apply(self, cmd):
        with self.lock:
            if "running" in cmd:
                self.running = bool(cmd["running"])
            if "speed" in cmd:
                self.speed = max(0.0, float(cmd["speed"]))
            if "decay_ms" in cmd:
                self.decay_ms = max(1.0, float(cmd["decay_ms"]))
            if "rate_hz" in cmd:
                self.rate_hz = max(0.0, float(cmd["rate_hz"]))
                self.engine.inject(self.rate_hz)
            if cmd.get("reset"):
                self.engine.reset()
                self.engine.inject(self.rate_hz)
                self.act[:] = 0
                self.history.clear()
            if "protocol" in cmd:
                self._set_protocol(cmd["protocol"])
            if "model" in cmd and cmd["model"] != self.engine.model:
                self.engine.set_model(cmd["model"])
                self.engine.inject(self.rate_hz)
                self.act[:] = 0
                self.history.clear()
        return self.state()

    def _set_protocol(self, name):
        eng = self.engine
        rng = np.random.default_rng(0)
        pool = eng.i2flyid
        if name == "sugar":
            ids = SUGAR
        elif name == "p9":
            ids = P9
        elif name == "silent":
            ids = SUGAR
        elif name.startswith("broad"):
            n = int(name[5:] or 100)
            ids = rng.choice(pool, n, replace=False).tolist()
        else:
            return
        self.protocol = name
        eng.set_stim_neurons(ids)
        eng.reset()
        eng.inject(0.0 if name == "silent" else self.rate_hz)
        self.act[:] = 0
        self.history.clear()

    def state(self):
        spec = self.engine.spec
        return {
            "model": self.engine.model,
            "protocol": self.protocol,
            "running": self.running,
            "speed": self.speed,
            "rate_hz": self.rate_hz,
            "decay_ms": self.decay_ms,
            "n": int(self.N),
            "isa": self.engine.isa,
            "threads": int(self.engine.threads),
            "sim_ms": round(self.sim_ms, 2),
            "ms_per_step": round(self.ms_per_step, 5),
            "realtime": round(self.wall_speed, 3),
            "spikes_per_step": round(self.spikes_per_step, 3),
            "live_frac": round(self.live_frac, 2),
            "n_stim": int(self.engine.stim_idx.size),
            "region_rate": [round(float(x), 1) for x in self.region_rate],
            "history": [[round(a, 1), round(b, 3)] for a, b in self.history[-240:]],
            "spec": {
                "label": spec.label, "citation": spec.citation,
                "equation": spec.equation, "note": spec.note,
                "n_aux": spec.n_aux, "aux": list(spec.aux_names),
                "flops": spec.flops, "exact": spec.exact,
                "k_in": spec.extra.get("k_in"),
                "rest": float(spec.rest[0]),
                "skip": spec.extra.get("can_skip_tiles"),
            },
        }


def model_catalogue():
    out = []
    for key in nrn_models.ALL_MODELS:
        s = nrn_models.build(key)
        out.append({
            "key": key, "label": s.label, "citation": s.citation,
            "equation": s.equation, "note": s.note, "n_aux": s.n_aux,
            "aux": list(s.aux_names), "flops": s.flops, "exact": s.exact,
        })
    return out


SIM: Sim | None = None
CATALOGUE = None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass          # the console belongs to the simulation, not the server

    def _send(self, body, ctype, extra=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send((WEB / "index.html").read_bytes(),
                              "text/html; charset=utf-8")
        if path == "/atlas.bin":
            # xyz int16 x3, then cat uint8, then region uint8
            body = (SIM.xyz.astype(np.int16).tobytes()
                    + SIM.cat.astype(np.uint8).tobytes()
                    + SIM.region.astype(np.uint8).tobytes())
            return self._send(body, "application/octet-stream")
        if path == "/meta":
            body = json.dumps({
                "n": int(SIM.N),
                "cats": [{"label": l, "color": c}
                         for l, c in zip(SIM.cat_labels, SIM.cat_colors)],
                "regions": SIM.region_labels,
                "models": CATALOGUE,
                "state": SIM.state(),
            }).encode()
            return self._send(body, "application/json")
        if path == "/frame.bin":
            # [uint32 json length][json][activity bytes]. The state used to ride
            # in a response header, which works right up until the raster
            # history makes the header a few kilobytes and a proxy or a browser
            # quietly truncates it. The body has no such limit.
            with SIM.lock:
                act = SIM.act_u8.tobytes()
                st = json.dumps(SIM.state()).encode()
            body = len(st).to_bytes(4, "little") + st + act
            return self._send(body, "application/octet-stream")
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        if self.path == "/control":
            return self._send(json.dumps(SIM.apply(json.loads(raw))).encode(),
                              "application/json")
        if self.path == "/snapshot":
            # Figure capture. The browser can save a PNG on its own, but not
            # with a filename that records which model and protocol produced it
            # -- and an unlabelled render of a brain is worth very little.
            import base64
            import re as _re
            body = json.loads(raw)
            data = _re.sub(r"^data:image/png;base64,", "", body["png"])
            out = ROOT / "data" / "results" / "studio"
            out.mkdir(parents=True, exist_ok=True)
            st = SIM.state()
            name = (f"{st['model']}_{st['protocol']}_"
                    f"{int(st['sim_ms'])}ms.png")
            (out / name).write_bytes(base64.b64decode(data))
            return self._send(json.dumps({"path": str(out / name)}).encode(),
                              "application/json")
        return self.send_error(404)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global SIM, CATALOGUE
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("FlyBrain Studio")
    CATALOGUE = model_catalogue()
    SIM = Sim()
    with Server(("127.0.0.1", port), Handler) as srv:
        print(f"\n  ->  http://127.0.0.1:{port}\n")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
