"""Pack the real 1-second simulation spikes into a playback dataset:
faint context brain + the 394 active neurons at their true 3-D positions +
every spike (time, neuron) so the artifact can animate the second.
"""
import pandas as pd, numpy as np, base64, json

ann = pd.read_csv("data/flywire_meta/neuron_annotations.tsv", sep="\t", low_memory=False)
ann = ann.dropna(subset=["pos_x", "pos_y", "pos_z", "root_id"]).copy()
ann["root_id"] = ann.root_id.astype("int64")

# --- shared coordinate transform (identical to the atlas) ---
ann["X"] = ann.pos_x * 4 / 1000.0
ann["Y"] = ann.pos_y * 4 / 1000.0
ann["Z"] = ann.pos_z * 40 / 1000.0
cx, cy, cz = ann.X.median(), ann.Y.median(), ann.Z.median()
def to_view(X, Y, Z):
    return (X - cx), -(Y - cy), (Z - cz)
vx, vy, vz = to_view(ann.X, ann.Y, ann.Z)
span = np.percentile(np.abs(np.r_[vx, vy, vz]), 99.5)
S = 1000.0 / span
ann["vx"], ann["vy"], ann["vz"] = vx * S, vy * S, vz * S

CATS = [
    ("Optic lobe", "#22d3ee", ["optic"]),
    ("Central brain", "#c084fc", ["central"]),
    ("Sensory input", "#35d39a", ["sensory", "sensory_ascending"]),
    ("Visual projection", "#2dd4bf", ["visual_projection", "visual_centrifugal"]),
    ("Ascending", "#f5b638", ["ascending"]),
    ("Descending", "#fb923c", ["descending"]),
    ("Motor / endocrine", "#fb7185", ["motor", "endocrine"]),
]
sc2cat = {sc: i for i, (_n, _c, scs) in enumerate(CATS) for sc in scs}
ann["cat"] = ann.super_class.map(sc2cat).fillna(1).astype(int)

# --- context cloud (faint), stratified subsample ~30k ---
rng = np.random.default_rng(3)
keep = []
for i in range(len(CATS)):
    idx = ann.index[ann.cat == i].to_numpy()
    n = len(idx) if len(idx) <= 4000 else max(3000, int(30000 * len(idx) ** 0.6 /
        sum((ann.cat == j).sum() ** 0.6 for j in range(len(CATS)))))
    keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
ctx = ann.loc[np.concatenate(keep)]
ctx_xyz = np.stack([ctx.vx, ctx.vy, ctx.vz], 1).round().astype(np.int16)
ctx_cat = ctx.cat.to_numpy().astype(np.uint8)

# --- spikes ---
sp = pd.read_parquet("data/results/pytorch_t1.0s_n1.parquet")
tcol = "time_ms" if "time_ms" in sp.columns else "t"
sp = sp[["flywire_id", tcol]].rename(columns={tcol: "t"}).copy()
sp["flywire_id"] = sp.flywire_id.astype("int64")

STIM = {720575940624963786,720575940630233916,720575940637568838,720575940638202345,
720575940617000768,720575940630797113,720575940632889389,720575940621754367,
720575940621502051,720575940640649691,720575940639332736,720575940616885538,
720575940639198653,720575940639259967,720575940617937543,720575940632425919,
720575940633143833,720575940612670570,720575940628853239,720575940629176663,
720575940611875570}

pos = ann.set_index("root_id")[["vx", "vy", "vz", "cat"]]
active_ids = sp.flywire_id.unique()
have = [i for i in active_ids if i in pos.index]
missing = len(active_ids) - len(have)
id2idx = {rid: k for k, rid in enumerate(have)}

neu_xyz = np.stack([pos.loc[have].vx, pos.loc[have].vy, pos.loc[have].vz], 1).round().astype(np.int16)
neu_cat = pos.loc[have].cat.to_numpy().astype(np.uint8)
neu_stim = np.array([1 if r in STIM else 0 for r in have], np.uint8)

sp = sp[sp.flywire_id.isin(id2idx)].sort_values("t")
spk_t = np.clip((sp.t.to_numpy() * 10).round(), 0, 10000).astype(np.uint16)  # 0.1ms units
spk_n = sp.flywire_id.map(id2idx).to_numpy().astype(np.uint16)
spk_count = np.bincount(spk_n, minlength=len(have)).astype(np.uint16)

# population activity histogram (spikes per 5ms bin, 200 bins over 1000ms)
hist, _ = np.histogram(sp.t.to_numpy(), bins=200, range=(0, 1000))

def b64(a): return base64.b64encode(a.tobytes()).decode()
out = {
    "ctx": {"n": int(len(ctx)), "xyz": b64(ctx_xyz), "cat": b64(ctx_cat)},
    "neu": {"n": int(len(have)), "xyz": b64(neu_xyz), "cat": b64(neu_cat), "stim": b64(neu_stim),
            "count": b64(spk_count)},
    "spk": {"n": int(len(sp)), "t": b64(spk_t), "ni": b64(spk_n)},
    "hist": hist.astype(int).tolist(),
    "cats": [{"name": n, "color": c} for n, c, _s in CATS],
    "meta": {"totalNeurons": int(len(ann)), "activeNeurons": int(len(have)),
             "totalSpikes": int(len(sp)), "stim": int(neu_stim.sum()),
             "durationMs": 1000, "missing": int(missing)},
}
open("data/flywire_meta/playback.json", "w").write(json.dumps(out))
print("active:", len(have), "| stim matched:", int(neu_stim.sum()),
      "| spikes:", len(sp), "| ctx:", len(ctx), "| missing pos:", missing)
print("peak 5ms-bin spikes:", int(hist.max()), "| size:",
      round(len(json.dumps(out)) / 1024), "KB")
