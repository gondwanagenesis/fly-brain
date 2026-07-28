"""Extract the real synaptic sub-network AMONG the 394 active neurons,
indexed to match playback.json's neuron order, so the artifact can animate
signal travelling from neuron to neuron along true connections.
"""
import pandas as pd, numpy as np, pyarrow.parquet as pq, base64, json

ann = pd.read_csv("data/flywire_meta/neuron_annotations.tsv", sep="\t", low_memory=False)
ann = ann.dropna(subset=["pos_x", "pos_y", "pos_z", "root_id"]).copy()
ann["root_id"] = ann.root_id.astype("int64")
pos_ids = set(ann.root_id)

sp = pd.read_parquet("data/results/pytorch_t1.0s_n1.parquet")
active_ids = sp.flywire_id.astype("int64").unique()          # order = playback neuron order
have = [i for i in active_ids if i in pos_ids]
id2idx = {rid: k for k, rid in enumerate(have)}
active = set(id2idx)
print("active neurons:", len(have))

pf = pq.ParquetFile("data/2025_Connectivity_783.parquet")
chunks = []
for b in pf.iter_batches(columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"],
                         batch_size=1_000_000):
    d = b.to_pandas()
    m = d.Presynaptic_ID.isin(active) & d.Postsynaptic_ID.isin(active)
    if m.any():
        chunks.append(d[m])
E = pd.concat(chunks, ignore_index=True)
print("induced edges:", len(E))

# keep the strongest connections for a legible network
CAP = 6500
E = E.sort_values("Connectivity", ascending=False).head(CAP).reset_index(drop=True)
src = E.Presynaptic_ID.map(id2idx).to_numpy().astype(np.uint16)
dst = E.Postsynaptic_ID.map(id2idx).to_numpy().astype(np.uint16)
exc = np.where(E.Excitatory.to_numpy() > 0, 1, 0).astype(np.uint8)
# quantized weight 0..255 (log-scaled) for line thickness / pulse size
w = E.Connectivity.to_numpy().astype(float)
wq = np.clip((np.log1p(w) / np.log1p(w.max()) * 255), 0, 255).astype(np.uint8)

def b64(a): return base64.b64encode(a.tobytes()).decode()
out = {"n": int(len(E)), "src": b64(src), "dst": b64(dst), "exc": b64(exc), "w": b64(wq),
       "totalInduced": int(sum(len(c) for c in chunks)), "excFrac": float(exc.mean())}
open("data/flywire_meta/edges.json", "w").write(json.dumps(out))
print("kept", len(E), "edges | exc frac", round(float(exc.mean()), 2),
      "| size", round(len(json.dumps(out)) / 1024), "KB")
