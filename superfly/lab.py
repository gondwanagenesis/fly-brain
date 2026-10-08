"""SUPERFLY Lab: the uplifted fly, live -- brain, modules, world and conversation.

    python -m superfly.lab                 # the real fly (brain + world + voice + LM)
    python -m superfly.lab --no-lm         # without the conversational LM
    python -m superfly.lab --mock          # synthetic data, for front-end work
    then open http://127.0.0.1:8770

The page shows the 138,639 FlyWire neurons at their real coordinates, at the
centre, lit by their real spikes. Around them, the added modules (senses,
interoception, word lobe, motor readout, the grounded voice FlyLM, episodic
memory, the conversational LM), wired to the brain regions they actually
connect to, each with its own live activity. Beside it, the world the fly
lives in, and the conversation: the fly's spontaneous remarks and its replies.

API (all JSON except the .bin endpoints)
  GET  /                 the page (superfly/lab/index.html)
  GET  /atlas.bin        int16 xyz[N*3] | uint8 cat[N] | uint8 region[N]
                         (engine slot order; grafted neurons placed by the MB)
  GET  /layout           static description: cats, regions, modules, wires, world
  GET  /frame.bin?since=K
                         uint32 LE json length | json | uint8 activity[N]
  POST /api/say          {"text": "..."} -> {"ok": true}; the reply arrives in frames
"""
from __future__ import annotations

import http.server
import json
import math
import socketserver
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "lab"
ATLAS = ROOT / "data" / "flywire_meta" / "atlas_aligned.npz"

SENSE_KEYS = ["sugar", "bitter", "water", "salt", "umami", "co2", "geosmin",
              "male_pheromone", "fruit_odor", "sound", "wind", "heat", "cold",
              "dry", "humid", "shadow"]
INTERO_KEYS = ["intero.hunger", "intero.satiety", "intero.ISN", "intero.thirst"]
MOTOR_KEYS = ["walk_forward", "turn_left", "turn_right", "walk_backward",
              "escape", "groom", "feed"]

MODULES = [
    {"id": "senses", "label": "Senses", "kind": "port", "units": len(SENSE_KEYS) * 2,
     "unit_labels": [f"{k} {s}" for k in SENSE_KEYS for s in ("L", "R")],
     "desc": "World -> Poisson drive on the fly's own receptor neurons (taste, smell, "
             "wind/sound, temperature, humidity, LC4/LPLC2 loom), per side",
     "place": "left"},
    {"id": "intero", "label": "Interoception", "kind": "port", "units": len(INTERO_KEYS),
     "unit_labels": ["hunger->MBON11", "satiety->PPL101", "ISN", "thirst->ITP"],
     "desc": "Body state -> the fly's interoceptive neurons (review 06)", "place": "bottom-left"},
    {"id": "wordlobe", "label": "Word lobe (grafted)", "kind": "graft", "units": 150,
     "desc": "150 grafted projection neurons onto real Kenyon cells; words heard "
             "as letter-trigram codes", "place": "top-left"},
    {"id": "motor", "label": "Motor readout", "kind": "port", "units": len(MOTOR_KEYS),
     "unit_labels": MOTOR_KEYS,
     "desc": "The fly's own descending neurons and MN9 -> the body (no other "
             "source of action)", "place": "bottom"},
    {"id": "body", "label": "Body + nerve cord (stand-in)", "kind": "world", "units": 6,
     "unit_labels": ["walk", "stop", "groom", "feed", "escape", "sleep"],
     "desc": "Minimal VNC: walking rhythm modulated by descending commands", "place": "bottom-right"},
    {"id": "flylm", "label": "FlyLM (grounded voice)", "kind": "nn", "layers": 4, "units": 48,
     "desc": "2.9M-parameter transformer: central-brain activity -> neural tokens -> "
             "inner speech; hearing heads -> senses", "place": "right"},
    {"id": "memory", "label": "Episodic memory", "kind": "memory", "units": 24,
     "desc": "Stored brain states (KC code + central features), hash-chained; "
             "recalled by re-evocation", "place": "top-right"},
    {"id": "mind", "label": "Conversational LM", "kind": "nn", "layers": 28, "units": 48,
     "desc": "Qwen2.5-1.5B-Instruct; speaks only from the fly's records; every "
             "first-person claim checked", "place": "far-right"},
]
WIRES = [
    {"from": "senses", "to": "brain:Antennal lobe", "label": "smell"},
    {"from": "senses", "to": "brain:SEZ / motor out", "label": "taste"},
    {"from": "senses", "to": "brain:Optic lobe", "label": "loom"},
    {"from": "intero", "to": "brain:Mushroom body", "label": "MBON11 / PPL101"},
    {"from": "intero", "to": "brain:SEZ / motor out", "label": "ISN / ITP"},
    {"from": "wordlobe", "to": "brain:Mushroom body", "label": "PN -> KC"},
    {"from": "brain:SEZ / motor out", "to": "motor", "label": "DN / MN9"},
    {"from": "motor", "to": "body", "label": "commands"},
    {"from": "body", "to": "senses", "label": "the world"},
    {"from": "brain:Mushroom body", "to": "flylm", "label": "central brain -> neural tokens"},
    {"from": "brain:Central complex", "to": "flylm", "label": ""},
    {"from": "flylm", "to": "memory", "label": "episodes"},
    {"from": "brain:Mushroom body", "to": "memory", "label": "KC code"},
    {"from": "memory", "to": "mind", "label": "recalled records"},
    {"from": "flylm", "to": "mind", "label": "inner speech"},
    {"from": "flylm", "to": "senses", "label": "hearing"},
]


class Lab:
    """Owns the life, the mind, and everything the browser sees."""

    def __init__(self, mock=False, lm=True, warm_s=0.0):
        a = np.load(ATLAS, allow_pickle=False)
        self.xyz, self.cat, self.region = a["xyz"], a["cat"], a["region"]
        self.cat_labels = [str(x) for x in a["cat_labels"]]
        self.cat_colors = [str(x) for x in a["cat_colors"]]
        self.region_labels = [str(x) for x in a["region_labels"]]
        self.mock = mock
        self.lock = threading.Lock()
        self.chat = []                       # {id, who, text, t, verdict}
        self.phase = "living"
        self.life = self.mind = None
        if mock:
            self.N = self.xyz.shape[0] + 150
            self.n_native = self.xyz.shape[0]
            self.perm = None
        else:
            sys.path.insert(0, str(ROOT))
            from superfly.life import Life
            from superfly.mind import Mind, Talker, LayerTap
            self.life = Life()
            e = self.life.fly.e
            self.N, self.n_native, self.perm = e.N, e.n_native, e.perm
            v = self.life.voice
            self.flylm_tap = LayerTap(list(v.m.blocks), v.cfg.d)
            MODULES[5]["layers"] = len(v.m.blocks)
            talker = Talker(threads=4) if lm else None
            if talker is not None:
                MODULES[7]["layers"] = talker.n_layers
            else:
                MODULES[7]["label"] += " (off)"
            self.mind = Mind(self.life, talker)
            self.life.listeners.append(self._on_event)
        self._place_grafted()
        if self.perm is not None:
            p = self.perm
            self.xyz, self.cat, self.region = self.xyz[p], self.cat[p], self.region[p]
        self.act = np.zeros(self.N, np.float32)
        self.act_u8 = np.zeros(self.N, np.uint8)
        self.decay_ms = 60.0
        self.region_rate = {}
        self._stop = False
        if not mock:
            self.life.on_tick = self._on_tick
            if warm_s:
                self.life.live(warm_s)
        threading.Thread(target=self._loop, daemon=True).start()

    def _place_grafted(self):
        n_graft = self.N - self.xyz.shape[0]
        if n_graft <= 0:
            return
        mb = self.region == self.region_labels.index("Mushroom body")
        c = self.xyz[mb].astype(np.float32).mean(0)
        rng = np.random.default_rng(0)
        pts = (c + np.array([-90.0, 0.0, 60.0]) + rng.normal(0, 25.0, (n_graft, 3))).astype(np.int16)
        self.cat_labels.append("Grafted (SUPERFLY)")
        self.cat_colors.append("#f472b6")
        self.region_labels.append("Grafted (SUPERFLY)")
        self.xyz = np.concatenate([self.xyz, pts])
        self.cat = np.concatenate([self.cat, np.full(n_graft, len(self.cat_labels) - 1, self.cat.dtype)])
        self.region = np.concatenate([self.region, np.full(n_graft, len(self.region_labels) - 1, self.region.dtype)])

    # ------------------------------------------------------------ live data
    def _on_event(self, kind, p):
        with self.lock:
            if kind == "say":
                self.chat.append({"id": len(self.chat), "who": "fly", "text": p["text"], "t": p["t"]})
            elif kind == "reply":
                self.chat.append({"id": len(self.chat), "who": "fly-reply", "text": p["text"],
                                  "t": p["t"], "verdict": p["verdict"]})

    def _on_tick(self, life):
        idx, cnt = life.frame()
        with self.lock:
            self.act *= math.exp(-life.DT_MS / self.decay_ms)
            self.act[idx] = 1.0
            np.clip(self.act * 255.0, 0, 255, out=self.act_u8, casting="unsafe")

    def _loop(self):
        while not self._stop:
            if self.mock:
                self._mock_tick()
                time.sleep(0.05)
                continue
            if self.mind.lock.locked():          # a conversation turn owns the fly
                time.sleep(0.02)
                continue
            self.life.tick()

    def _mock_tick(self):
        rng = np.random.default_rng()
        with self.lock:
            self.act *= 0.6
            self.act[rng.integers(0, self.N, 3000)] = 1.0
            np.clip(self.act * 255.0, 0, 255, out=self.act_u8, casting="unsafe")
            if rng.random() < 0.01:
                self.chat.append({"id": len(self.chat), "who": "fly", "t": time.time() % 1000,
                                  "text": str(rng.choice(["i smell fruit a little.", "a breeze. i turn.",
                                                          "something sweet. i want to eat."]))})

    def say(self, text):
        with self.lock:
            self.chat.append({"id": len(self.chat), "who": "you", "text": text,
                              "t": 0 if self.mock else round(self.life.world.t, 2)})
        if self.mock:
            def later():
                time.sleep(2)
                self._on_event("reply", {"text": f"(mock) i heard you say: {text}", "t": 0,
                                         "verdict": "mock"})
            threading.Thread(target=later, daemon=True).start()
            return {"ok": True}
        threading.Thread(target=self.mind.respond, args=(text,), daemon=True).start()
        return {"ok": True}

    # ------------------------------------------------------------ JSON
    def layout(self):
        world = None
        if not self.mock:
            s = self.life.world.state()
            world = {"arena": s["arena"], "objects": s["objects"], "lamp": s["lamp"]}
        else:
            world = {"arena": 100.0, "lamp": [22.0, 22.0],
                     "objects": [{"kind": "sugar", "x": 25, "y": 72, "r": 3, "odour": "fruit_odor"},
                                 {"kind": "water", "x": 76, "y": 70, "r": 3, "odour": ""},
                                 {"kind": "bitter", "x": 72, "y": 22, "r": 3, "odour": "geosmin"}]}
        xyz = self.xyz.astype(np.float32)
        return {"n": int(self.N), "n_native": int(self.n_native),
                "cats": [{"label": l, "color": c} for l, c in zip(self.cat_labels, self.cat_colors)],
                "regions": self.region_labels,
                "region_centroids": {r: xyz[self.region == i].mean(0).round(1).tolist()
                                     for i, r in enumerate(self.region_labels) if (self.region == i).any()},
                "bbox": {"min": xyz.min(0).tolist(), "max": xyz.max(0).tolist()},
                "modules": MODULES, "wires": WIRES, "world": world,
                "sense_keys": SENSE_KEYS, "intero_keys": INTERO_KEYS, "motor_keys": MOTOR_KEYS,
                "mock": self.mock}

    def frame(self, since=0):
        with self.lock:
            act = self.act_u8.tobytes()
            chat = [c for c in self.chat if c["id"] >= since]
        st = self._mock_state() if self.mock else self._state()
        st["chat"] = chat
        st["chat_next"] = len(self.chat)
        js = json.dumps(st).encode()
        return len(js).to_bytes(4, "little") + js + act

    def _state(self):
        L, M = self.life, self.mind
        w = L.world
        sense = w.sense() if hasattr(w, "sense") else {}
        senses = [round(float(sense.get((k, s), 0.0)), 1) for k in SENSE_KEYS for s in ("left", "right")]
        from superfly.life import intero_rates
        ir = intero_rates(w.needs)
        last = L.moments[-1] if L.moments else None

        def u8(frames):
            if not frames:
                return None
            f = np.asarray(frames[-1], np.float32)
            return np.clip(f / 4.0 * 255, 0, 255).astype(int).tolist()
        mem = L.memory
        rec = (M.last or {}).get("recalled", []) if M else []
        phase = "living"
        if M and M.lock.locked():
            phase = getattr(M, "phase", "thinking")
        wordlobe = self.act_u8[self.n_native:].astype(int).tolist()
        body_modes = ["walk", "stop", "groom", "feed", "escape", "sleep"]
        return {
            "t": round(w.t, 2), "realtime": round(w.t / max(L.wall_s, 1e-9), 3), "phase": phase,
            "world": w.state(),
            "modules": {
                "senses": senses,
                "intero": [round(ir[k], 1) for k in INTERO_KEYS],
                "wordlobe": wordlobe,
                "motor": [round(L.motor.get(k, 0.0), 1) for k in MOTOR_KEYS],
                "body": [255 if w.body.mode == m else 0 for m in body_modes],
                "flylm": u8(self.flylm_tap.frames),
                "memory": [{"i": e.i, "t": round(e.t, 1), "speech": e.speech, "event": e.event,
                            "place": e.place, "hash": e.hash} for e in mem.eps[-24:]],
                "mind": u8(M.talker.tap.frames) if M and M.talker else None,
            },
            "memory_n": len(mem.eps), "recalled": [r["i"] for r in rec],
            "inner": last.said if last else "", "inner_labels": last.labels if last else [],
        }

    def _mock_state(self):
        rng = np.random.default_rng()
        t = time.time() % 10000
        return {"t": round(t, 2), "realtime": 0.33, "phase": "living",
                "world": {"t": t, "light": 0.5 + 0.5 * math.cos(t / 60),
                          "fly": {"x": 50 + 30 * math.cos(t / 9), "y": 50 + 30 * math.sin(t / 7),
                                  "heading": t / 3 % 6.28, "mode": "walk", "on": ""},
                          "wind": {"dir": 1.0, "speed": 0.3}, "loom": 0.0,
                          "needs": {"energy": 0.6, "water": 0.7, "sleep": 0.1, "arousal": 0.0,
                                    "content": 0.0, "hunger": 0.4, "thirst": 0.3, "asleep": False},
                          "objects": [], "lamp": [22, 22], "arena": 100},
                "modules": {"senses": rng.uniform(0, 20, len(SENSE_KEYS) * 2).round(1).tolist(),
                            "intero": [24.0, 36.0, 30.0, 18.0],
                            "wordlobe": rng.integers(0, 255, 150).tolist(),
                            "motor": rng.uniform(0, 30, len(MOTOR_KEYS)).round(1).tolist(),
                            "body": [255, 0, 0, 0, 0, 0],
                            "flylm": rng.integers(0, 255, (4, 48)).tolist(),
                            "memory": [{"i": i, "t": i * 5.0, "speech": "i smell fruit.", "event": "",
                                        "place": "in the open", "hash": "%016x" % i} for i in range(12)],
                            "mind": rng.integers(0, 255, (28, 48)).tolist()},
                "memory_n": 12, "recalled": [3], "inner": "i smell fruit a little. i turn.",
                "inner_labels": [["percept", "fruit_odor"], ["action", "turn"]]}


LAB: Lab | None = None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, q = self.path.partition("?")
        if path in ("/", "/index.html"):
            return self._send((WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/atlas.bin":
            body = (LAB.xyz.astype(np.int16).tobytes() + LAB.cat.astype(np.uint8).tobytes()
                    + LAB.region.astype(np.uint8).tobytes())
            return self._send(body, "application/octet-stream")
        if path == "/layout":
            return self._send(json.dumps(LAB.layout()).encode(), "application/json")
        if path == "/frame.bin":
            since = 0
            for kv in q.split("&"):
                if kv.startswith("since="):
                    since = int(kv[6:] or 0)
            return self._send(LAB.frame(since), "application/octet-stream")
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        if self.path == "/api/say":
            text = str(json.loads(raw).get("text", ""))[:300].strip()
            if not text:
                return self._send(b'{"ok": false}', "application/json")
            return self._send(json.dumps(LAB.say(text)).encode(), "application/json")
        self.send_error(404)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global LAB
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    port = int(args[0]) if args else 8770
    LAB = Lab(mock="--mock" in sys.argv, lm="--no-lm" not in sys.argv)
    with Server(("127.0.0.1", port), Handler) as srv:
        print(f"\n  SUPERFLY Lab  ->  http://127.0.0.1:{port}\n", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
