"""Run SUPERFLY on your own computer with one command.

    python -m superfly.quickstart              # the male CNS fly, opens the Lab in your browser
    python -m superfly.quickstart --flywire    # the FlyWire (female brain) fly instead
    python -m superfly.quickstart --no-lm      # skip the conversational model (saves ~3 GB)
    python -m superfly.quickstart --big-lm     # Qwen2.5-3B instead of 1.5B (slower, more fluent)

The first run prepares, once:
  1. the FlyWire annotation table (55 MB)
  2. the connectome: FlyWire's fan-out from the parquet in this repo, or the
     Janelia male CNS (566 MB download, converted in about a minute)
  3. the conversational model from Hugging Face into ./models/
Then it starts the Lab (http://127.0.0.1:8770) and opens it. The trained
voices ship in the repo (data/superfly_cache*/flylm_tiny_world.pt), so nothing
needs training.

Needs: Python 3.10+, an x86-64 CPU (the native kernel uses AVX2/AVX-512 when
present), a C compiler (gcc/clang on Linux and Intel macOS, MSVC Build Tools
on Windows), ~10 GB free disk, and the packages in requirements-superfly.txt.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "flywire_meta" / "neuron_annotations.tsv"
ANN_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
           "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
LMS = {"small": "Qwen/Qwen2.5-1.5B-Instruct", "big": "Qwen/Qwen2.5-3B-Instruct"}


def step(msg):
    print(f"\n== {msg}", flush=True)


def run(args, env):
    r = subprocess.call([sys.executable, *args], cwd=ROOT, env=env)
    if r:
        sys.exit(f"failed: {' '.join(args)}")


def main():
    argv = sys.argv[1:]
    connectome = "flywire783" if "--flywire" in argv else "male_cns"
    env = dict(os.environ, PYTHONPATH=str(ROOT), SUPERFLY_CONNECTOME=connectome)
    step("annotations")
    if not ANN.exists():
        ANN.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", ANN_URL)
        urllib.request.urlretrieve(ANN_URL, ANN)
    print("ok")
    step(f"connectome: {connectome}")
    if connectome == "flywire783":
        if not (ROOT / "data" / "fanout_csc.pt").exists():
            run(["code/build_fanout_csc.py"], env)
    elif not (ROOT / "data" / "male_cns" / "fanout_csc.pt").exists():
        run(["-m", "superfly.connectomes.male_cns"], env)
    print("ok")
    lab_args = ["-m", "superfly.lab"]
    if "--no-lm" in argv:
        lab_args.append("--no-lm")
    else:
        repo = LMS["big" if "--big-lm" in argv else "small"]
        dest = ROOT / "models" / repo.split("/")[1]
        step(f"conversational model: {repo}")
        if not (dest / "config.json").exists():
            from huggingface_hub import snapshot_download
            snapshot_download(repo, local_dir=str(dest),
                              allow_patterns=["*.json", "*.safetensors", "*.txt"])
        env["SUPERFLY_TALKER"] = str(dest)
        print("ok")
    step("starting the Lab (first start builds the native kernel and takes ~1 min)")
    url = "http://127.0.0.1:8770"

    def open_later():
        time.sleep(45)
        webbrowser.open(url)
    threading.Thread(target=open_later, daemon=True).start()
    print(f"open {url} if the browser does not open by itself; Ctrl+C to stop")
    subprocess.call([sys.executable, *lab_args], cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
