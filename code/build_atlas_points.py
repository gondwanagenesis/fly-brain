"""Turn the real FlyWire neuron annotation table into a compact 3D point cloud
+ real region-centroid labels for the pannable atlas artifact.

Coordinates: FlyWire 'pos' anchors are in FAFB voxels (4nm x, 4nm y, 40nm z).
We convert to isotropic microns, center, and scale into a normalized cube.
Output: a JSON (base64-packed int16 xyz + uint8 category) for inline embedding.
"""
import pandas as pd, numpy as np, base64, json

df = pd.read_csv("data/flywire_meta/neuron_annotations.tsv", sep="\t", low_memory=False)
df = df.dropna(subset=["pos_x", "pos_y", "pos_z"]).copy()

# voxel -> micron (isotropic)
df["X"] = df.pos_x * 4 / 1000.0
df["Y"] = df.pos_y * 4 / 1000.0
df["Z"] = df.pos_z * 40 / 1000.0

# center on median, view frontal: screen up = dorsal (-Y)
cx, cy, cz = df.X.median(), df.Y.median(), df.Z.median()
df["vx"] = df.X - cx
df["vy"] = -(df.Y - cy)
df["vz"] = df.Z - cz
# normalize to ~[-1000,1000]
span = np.percentile(np.abs(np.r_[df.vx, df.vy, df.vz]), 99.5)
S = 1000.0 / span
for a in ["vx", "vy", "vz"]:
    df[a] = df[a] * S

# ---- category (real super_class, grouped for a clean legend) ----
CATS = [
    ("optic",       "Optic lobe",          "#22d3ee", ["optic"]),
    ("central",     "Central brain",       "#c084fc", ["central"]),
    ("sensory",     "Sensory input",       "#35d39a", ["sensory", "sensory_ascending"]),
    ("visualproj",  "Visual projection",   "#2dd4bf", ["visual_projection", "visual_centrifugal"]),
    ("ascending",   "Ascending",           "#f5b638", ["ascending"]),
    ("descending",  "Descending",          "#fb923c", ["descending"]),
    ("motor",       "Motor / endocrine",   "#fb7185", ["motor", "endocrine"]),
]
sc2cat = {}
for i, (_k, _n, _c, scs) in enumerate(CATS):
    for sc in scs:
        sc2cat[sc] = i
df["cat"] = df.super_class.map(sc2cat).fillna(1).astype(int)

# ---- stratified subsample (keep small classes whole, thin the big ones) ----
TARGET = 52000
keep = []
rng = np.random.default_rng(7)
for i, (_k, _n, _c, _scs) in enumerate(CATS):
    idx = df.index[df.cat == i].to_numpy()
    # proportional but capped; keep everything under 6k
    if len(idx) <= 6000:
        keep.append(idx)
    else:
        frac = TARGET * (len(idx) ** 0.62) / sum(
            (df.cat == j).sum() ** 0.62 for j in range(len(CATS)))
        n = min(len(idx), max(4000, int(frac)))
        keep.append(rng.choice(idx, size=n, replace=False))
sub = df.loc[np.concatenate(keep)]
print("subsampled points:", len(sub))

# ---- pack ----
xyz = np.stack([sub.vx, sub.vy, sub.vz], axis=1).round().astype(np.int16)
cat = sub.cat.to_numpy().astype(np.uint8)
b64_xyz = base64.b64encode(xyz.tobytes()).decode()
b64_cat = base64.b64encode(cat.tobytes()).decode()

# ---- real region-centroid labels (selected by real cell_class / super_class) ----
def centroid(mask, minn=15):
    d = df[mask]
    if len(d) < minn: return None
    return [float(d.vx.mean()), float(d.vy.mean()), float(d.vz.mean()), int(len(d))]

def sides(mask, name, cat, fn, paper):
    out = []
    for s, tag in [("left", " (L)"), ("right", " (R)")]:
        c = centroid(mask & (df.side == s))
        if c: out.append({"name": name + tag, "x": c[0], "y": c[1], "z": c[2],
                          "n": c[3], "cat": cat, "fn": fn, "paper": paper})
    return out

labels = []
labels += sides(df.super_class == "optic", "Optic lobe", 0,
    "~77,500 neurons — the fly's entire visual system: lamina, medulla, lobula and lobula plate stacked retinotopically.",
    "Fischbach & Dittrich 1989 · Maisak et al. 2013, Nature")
labels += sides(df.cell_class == "Kenyon_Cell", "Mushroom body", 1,
    "Kenyon cells — the associative learning & memory centre where dopamine rewrites odour value.",
    "Aso et al. 2014, eLife")
cx_c = centroid(df.cell_class == "CX")
if cx_c: labels.append({"name": "Central complex", "x": cx_c[0], "y": cx_c[1], "z": cx_c[2],
    "n": cx_c[3], "cat": 1,
    "fn": "Navigation hub — ellipsoid body (heading compass), fan-shaped body, protocerebral bridge and noduli.",
    "paper": "Seelig & Jayaraman 2015, Nature · Hulse et al. 2021, eLife"})
labels += sides(df.cell_class.isin(["ALPN", "ALLN", "ALIN", "ALON"]), "Antennal lobe", 2,
    "Primary olfactory centre — ~50 glomeruli where odour receptor neurons meet projection neurons.",
    "Wilson 2013, Annu. Rev. Neurosci.")
labels += sides(df.cell_class.isin(["LHLN", "LHCENT"]), "Lateral horn", 1,
    "Innate olfactory behaviour — assigns hard-wired value to odours.",
    "Dolan et al. 2019, eLife")
mo = centroid(df.super_class.isin(["motor", "descending"]))
if mo: labels.append({"name": "SEZ / motor output", "x": mo[0], "y": mo[1], "z": mo[2],
    "n": mo[3], "cat": 6,
    "fn": "Subesophageal zone & descending pathways: feeding motor control and commands to the nerve cord. Our sugar experiment's 21 gustatory neurons drive the proboscis-extension circuit here.",
    "paper": "Sterne et al. 2021, eLife"})

out = {
    "n": int(len(sub)),
    "xyz": b64_xyz,
    "cat": b64_cat,
    "cats": [{"key": k, "name": n, "color": c, "count": int((df.cat == i).sum())}
             for i, (k, n, c, _s) in enumerate(CATS)],
    "labels": labels,
    "totalNeurons": int(len(df)),
}
with open("data/flywire_meta/atlas_points.json", "w") as f:
    json.dump(out, f)
print("labels:", [l["name"] for l in labels])
print("wrote data/flywire_meta/atlas_points.json",
      round(len(json.dumps(out)) / 1024), "KB")
