"""Build data/fanout_csc.pt straight from the connectivity parquet.

The engines normally derive it from weight_csr.pkl (code/run_pytorch.py), which
needs the dense 289 MB pickle first. This goes direct: group the edges by
presynaptic neuron, targets ascending within each group, duplicates summed
exactly as torch's coalesce() would. Weights stay float32 holding exact
integers (synapse count x sign), which is what NativeBrainEngine's int16
narrowing asserts.

    python code/build_fanout_csc.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"


def main():
    comp = pd.read_csv(D / "2025_Completeness_783.csv", index_col=0)
    N = len(comp)
    c = pd.read_parquet(D / "2025_Connectivity_783.parquet",
                        columns=["Presynaptic_Index", "Postsynaptic_Index",
                                 "Excitatory x Connectivity"])
    pre = c.Presynaptic_Index.to_numpy(np.int64)
    post = c.Postsynaptic_Index.to_numpy(np.int64)
    w = c["Excitatory x Connectivity"].to_numpy(np.int64)
    key = pre * N + post
    order = np.argsort(key, kind="stable")
    key, w = key[order], w[order]
    uk, start = np.unique(key, return_index=True)
    ws = np.add.reduceat(w, start)               # sums duplicate (pre, post)
    pre_u, post_u = uk // N, uk % N
    crow = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre_u, minlength=N), out=crow[1:])
    fo = {"crow": torch.from_numpy(crow),
          "post": torch.from_numpy(post_u.astype(np.int64)),
          "val": torch.from_numpy(ws.astype(np.float32))}
    torch.save(fo, D / "fanout_csc.pt")
    print(f"N={N} edges={len(uk)} (dups merged: {len(key) - len(uk)}) "
          f"|w|max={int(np.abs(ws).max())} -> {D / 'fanout_csc.pt'}")


if __name__ == "__main__":
    main()
