"""Build a 3D atlas aligned to the SIMULATION's neuron indices.

data/flywire_meta/atlas_points.json was a stratified SUBSAMPLE built for a
static illustration, so it cannot be driven by the simulation: there is no way
to say "neuron 91,428 just spiked, light up that point". This builds the
aligned version -- one entry per simulated neuron, in the exact row order of
2025_Completeness_783.csv, which is the order every engine indexes by.

Outputs data/flywire_meta/atlas_aligned.npz:

    xyz       (N, 3) int16    position, normalised to a [-1000, 1000] cube
    cat       (N,)   uint8    coarse class, for colouring and legend filters
    region    (N,)   uint8    named region, for the per-region activity readout
    has_pos   (N,)   bool     whether a real coordinate was found

COORDINATES
-----------
FlyWire `pos` anchors are FAFB voxels at 4 x 4 x 40 nm, so they are converted
to isotropic microns before anything else -- skipping that leaves the brain
squashed tenfold along z, which looks plausible enough in a render to go
unnoticed. Neurons with no annotation row keep has_pos = False and are placed
at the centroid of their class rather than at the origin, so they do not pile
up in a spike at the middle of the brain.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "flywire_meta"

CATS = [
    ("optic",      "Optic lobe",        "#22d3ee", ["optic"]),
    ("central",    "Central brain",     "#c084fc", ["central"]),
    ("sensory",    "Sensory input",     "#35d39a", ["sensory", "sensory_ascending"]),
    ("visualproj", "Visual projection", "#2dd4bf", ["visual_projection",
                                                    "visual_centrifugal"]),
    ("ascending",  "Ascending",         "#f5b638", ["ascending"]),
    ("descending", "Descending",        "#fb923c", ["descending"]),
    ("motor",      "Motor / endocrine", "#fb7185", ["motor", "endocrine"]),
]

# Named regions, selected the same way the static atlas selected them, but kept
# per neuron so the UI can report live firing rates by brain region.
REGIONS = [
    ("unassigned",     "Unassigned",        None),
    ("optic",          "Optic lobe",        ("super_class", ["optic"])),
    ("mushroom_body",  "Mushroom body",     ("cell_class", ["Kenyon_Cell"])),
    ("central_complex", "Central complex",  ("cell_class", ["CX"])),
    ("antennal_lobe",  "Antennal lobe",     ("cell_class", ["ALPN", "ALLN",
                                                            "ALIN", "ALON"])),
    ("lateral_horn",   "Lateral horn",      ("cell_class", ["LHLN", "LHCENT"])),
    ("sez_motor",      "SEZ / motor out",   ("super_class", ["motor",
                                                             "descending"])),
    ("visual_proj",    "Visual projection", ("super_class",
                                             ["visual_projection",
                                              "visual_centrifugal"])),
    ("ascending",      "Ascending",         ("super_class", ["ascending"])),
]


def main():
    comp = pd.read_csv(ROOT / "data" / "2025_Completeness_783.csv", index_col=0)
    ids = np.asarray(comp.index, dtype=np.int64)
    N = len(ids)
    print(f"simulated neurons: {N}")

    ann = pd.read_csv(META / "neuron_annotations.tsv", sep="\t", low_memory=False)
    ann = ann.drop_duplicates(subset="root_id").set_index("root_id")
    hit = ann.reindex(ids)
    found = hit.pos_x.notna().to_numpy()
    print(f"with a FlyWire position: {found.sum()} ({100*found.mean():.2f}%)")

    # voxel -> isotropic micron
    X = hit.pos_x.to_numpy(dtype=np.float64) * 4 / 1000.0
    Y = hit.pos_y.to_numpy(dtype=np.float64) * 4 / 1000.0
    Z = hit.pos_z.to_numpy(dtype=np.float64) * 40 / 1000.0

    cx, cy, cz = (np.nanmedian(X), np.nanmedian(Y), np.nanmedian(Z))
    vx, vy, vz = X - cx, -(Y - cy), Z - cz      # screen up = dorsal
    span = np.nanpercentile(np.abs(np.concatenate([vx, vy, vz])), 99.5)
    s = 1000.0 / span
    vx, vy, vz = vx * s, vy * s, vz * s

    sc2cat = {}
    for i, (_k, _n, _c, scs) in enumerate(CATS):
        for sc in scs:
            sc2cat[sc] = i
    cat = hit.super_class.map(sc2cat).fillna(1).astype(np.uint8).to_numpy()

    region = np.zeros(N, dtype=np.uint8)
    for ri, (_key, _label, sel) in enumerate(REGIONS):
        if sel is None:
            continue
        col, vals = sel
        m = hit[col].isin(vals).to_numpy()
        region[m & (region == 0)] = ri

    # Neurons without a coordinate go to their class centroid, not the origin:
    # a few thousand points stacked at (0,0,0) reads as a bright artificial
    # nucleus in the middle of the render and is the first thing anyone asks
    # about.
    for c in range(len(CATS)):
        m = (cat == c)
        if not m.any():
            continue
        miss = m & ~found
        if not miss.any():
            continue
        good = m & found
        src = good if good.any() else found
        for arr in (vx, vy, vz):
            arr[miss] = np.nanmean(arr[src])

    xyz = np.stack([vx, vy, vz], 1)
    xyz = np.nan_to_num(xyz, nan=0.0)
    xyz = np.clip(np.round(xyz), -32000, 32000).astype(np.int16)

    out = META / "atlas_aligned.npz"
    np.savez_compressed(
        out, xyz=xyz, cat=cat, region=region, has_pos=found,
        cat_keys=np.array([c[0] for c in CATS]),
        cat_labels=np.array([c[1] for c in CATS]),
        cat_colors=np.array([c[2] for c in CATS]),
        region_keys=np.array([r[0] for r in REGIONS]),
        region_labels=np.array([r[1] for r in REGIONS]),
    )
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    for ri, (_k, label, _s) in enumerate(REGIONS):
        n = int((region == ri).sum())
        if n:
            print(f"  {label:<20} {n:7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
