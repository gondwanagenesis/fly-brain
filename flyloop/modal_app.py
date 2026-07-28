"""Run the whole embodied fly (connectome brain + MuJoCo body) on Modal.

Usage
-----
  modal run  flyloop/modal_app.py::upload_data      # one-time: push connectome (~1 GB)
  modal run  flyloop/modal_app.py::run_loop         # closed-loop sugar experiment -> mp4
  modal run  flyloop/modal_app.py::bench            # engine benchmark on cloud CPU/GPU
  modal deploy flyloop/modal_app.py                 # persistent endpoint

Headless rendering uses EGL (no display needed). Data lives in a Modal Volume so
containers start instantly and scale to zero when idle.
"""
import modal

app = modal.App("embodied-fly")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1", "libglew-dev", "libegl1", "libgles2",
        "libosmesa6", "libglfw3", "ffmpeg", "patchelf",
    )
    .pip_install(
        "torch --index-url https://download.pytorch.org/whl/cpu",
        "flygym==2.1.0", "mujoco==3.9.0",
        "pandas", "pyarrow", "numpy<3", "imageio", "imageio-ffmpeg",
    )
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
    .add_local_dir(
        "flyloop", remote_path="/root/flyloop",
        ignore=["__pycache__", "*.pyc"],
    )
)

vol = modal.Volume.from_name("flybrain-data", create_if_missing=True)
DATA = "/data"

# Files the brain needs.
#   fanout_csc.pt          -- the ONLY synapse table event-driven mode needs
#   weight_csr.pkl (289MB) -- OPTIONAL, only for the event_mode=False fallback
REQUIRED = [
    "2025_Completeness_783.csv",          # flywire id <-> index
    "fanout_csc.pt",                      # event-driven fan-out (CSC)
    "flywire_meta/neuron_annotations.tsv",  # super_class -> GATE2 motor ids
]
OPTIONAL = [
    "2025_Connectivity_783.parquet",      # only to rebuild fanout from scratch
    "weight_csr.pkl",                     # only for dense-spmv comparison
]


@app.function(image=image, volumes={DATA: vol}, timeout=3600)
def _put(relpath: str, blob: bytes):
    from pathlib import Path
    p = Path(DATA) / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    vol.commit()
    return f"{relpath}: {len(blob)/1e6:.1f} MB"


@app.function(image=image, volumes={DATA: vol}, timeout=600)
def _verify():
    """List what actually landed in the volume, with sizes."""
    from pathlib import Path
    out = {}
    for p in sorted(Path(DATA).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(DATA)).replace("\\", "/")] = p.stat().st_size
    return out


@app.local_entrypoint()
def upload_data(local_dir: str = "data", include_optional: bool = False):
    """Push connectome files into the Modal Volume, then VERIFY they landed.

    The previous shell-loop version hid failures behind `>/dev/null`; this one
    checks the remote listing and reports any file that did not arrive.
    """
    from pathlib import Path
    root = Path(local_dir)
    want = list(REQUIRED) + (list(OPTIONAL) if include_optional else [])
    for rel in want:
        f = root / rel
        if not f.exists():
            print(f"  SKIP (missing locally): {rel}")
            continue
        mb = f.stat().st_size / 1e6
        print(f"  uploading {rel} ({mb:.1f} MB) ...", flush=True)
        try:
            print("   ", _put.remote(rel, f.read_bytes()))
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")

    print("\nverifying remote volume ...")
    remote = _verify.remote()
    ok = True
    for rel in REQUIRED:
        if rel in remote:
            print(f"  OK      {rel}  ({remote[rel]/1e6:.1f} MB)")
        else:
            ok = False
            print(f"  MISSING {rel}   <-- required")
    extra = set(remote) - set(REQUIRED) - set(OPTIONAL)
    for e in sorted(extra):
        print(f"  (extra) {e}  ({remote[e]/1e6:.1f} MB)")
    print("\nvolume ready." if ok else "\nINCOMPLETE - rerun for the missing files.")


@app.function(image=image, volumes={DATA: vol}, timeout=3600, cpu=8.0,
              memory=16384)
def _run_loop(duration_ms: float = 400.0, render: bool = True):
    """Closed sensorimotor loop in the cloud; returns (mp4_bytes, csv_text)."""
    import sys, subprocess, os
    from pathlib import Path
    sys.path.insert(0, "/root/flyloop")
    out = Path("/tmp/out"); out.mkdir(exist_ok=True)
    cmd = [sys.executable, "/root/flyloop/closed_loop.py",
           "--duration-ms", str(duration_ms),
           "--data", DATA,
           "--out", str(out / "closed_loop.mp4")]
    if not render:
        cmd.append("--headless")
    env = dict(os.environ, PYTHONPATH="/root/flyloop", MUJOCO_GL="egl")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(r.stdout[-4000:]);  print(r.stderr[-2000:])
    mp4 = (out / "closed_loop.mp4")
    csv = (out / "closed_loop.csv")
    return (mp4.read_bytes() if mp4.exists() else b"",
            csv.read_text() if csv.exists() else "")


@app.local_entrypoint()
def run_loop(duration_ms: float = 400.0, out: str = "virtualfly/modal_loop.mp4"):
    from pathlib import Path
    mp4, csv = _run_loop.remote(duration_ms=duration_ms)
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    if mp4:
        p.write_bytes(mp4); print(f"saved {p} ({len(mp4)/1e6:.1f} MB)")
    if csv:
        p.with_suffix(".csv").write_text(csv); print(f"saved {p.with_suffix('.csv')}")


@app.function(image=image, volumes={DATA: vol}, timeout=1800, cpu=8.0,
              memory=16384)
def _bench(steps: int = 1000):
    import sys, time, torch
    sys.path.insert(0, "/root/flyloop")
    from brain_engine import BrainEngine
    from closed_loop import SUGAR_GRNS
    res = {}
    for mode in (True, False):
        e = BrainEngine(data_dir=DATA, stim_ids=SUGAR_GRNS, seed=0)
        e.event_mode = mode; e.inject(200.0)
        for _ in range(50): e.step()
        t0 = time.perf_counter()
        n = sum(int(e.step().sum().item()) for _ in range(steps))
        res["event" if mode else "dense"] = (time.perf_counter() - t0, n)
    return res


@app.local_entrypoint()
def bench(steps: int = 1000):
    r = _bench.remote(steps=steps)
    for k, (dt, n) in r.items():
        print(f"{k:6s} {dt:7.3f}s / {steps} steps = {dt/steps*1000:6.3f} ms/step  "
              f"spikes={n}")
    if "dense" in r and "event" in r:
        print(f"speedup: {r['dense'][0]/r['event'][0]:.1f}x")
