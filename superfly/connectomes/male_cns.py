"""Convert the Janelia male CNS connectome (v1.0) to this repo's file layout.

The male CNS is one animal's brain, optic lobes AND ventral nerve cord
(Berg et al., Cell 2026; male-cns.janelia.org, CC-BY). Public flat files:

    https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
      body-annotations-male-cns-v1.0-minconf-0.5.feather          14 MB
      body-neurotransmitters-male-cns-v1.0.feather                  43 MB
      connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather  508 MB

This writes, under data/male_cns/, the same files the engine reads for
FlyWire, so every tool runs unchanged with SUPERFLY_CONNECTOME=male_cns:

    2025_Completeness_783.csv       the simulated neurons (status == Traced)
    2025_Connectivity_783.parquet   edges: synapse count x sign
    fanout_csc.pt                   the engine's fan-out
    flywire_meta/neuron_annotations.tsv   FlyWire-schema annotations
    flywire_meta/atlas_aligned.npz  3D positions + categories for the Lab

Choices (documented for the paper):
  * Neurons: the 165,122 bodies with status "Traced". Edges: all traced-to-
    traced weights (no threshold; the FlyWire file used by Shiu et al. has
    none either).
  * Sign: the rule found in the FlyWire file Shiu et al. used -- GABA and
    glutamate inhibitory, everything else excitatory -- plus histamine
    inhibitory (histamine-gated chloride channels; absent from FlyWire's
    brain-only set). Per-body consensus_nt, else predicted_nt.
  * Labels: FlyWire super_class names; cell_class / cell_sub_class taken
    from FlyWire by cell-type match (the male type when FlyWire has a type
    of that name -- it is often finer, LB3c vs LB3 -- else flywireType), else the male CNS's own class and
    subclass. cell_type is the male type; the FlyWire match is kept.
  * Positions: soma location (8 nm voxels -> um). Neurons without a soma in
    the volume (most sensory neurons) are placed at the synapse-weighted mean
    of their partners' somata (marked approximate in the npz).

    python -m superfly.connectomes.male_cns
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "male_cns"
RAW = OUT / "raw"
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {"ann": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
         "nt": "body-neurotransmitters-male-cns-v1.0.feather",
         "w": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"}
SUPER = {"cb_intrinsic": "central", "ol_intrinsic": "optic",
         "visual_projection": "visual_projection", "visual_centrifugal": "visual_centrifugal",
         "cb_sensory": "sensory", "ol_sensory": "sensory", "vnc_sensory": "sensory",
         "sensory_ascending": "sensory_ascending", "sensory_descending": "sensory",
         "cb_sensory_tbc": "sensory", "vnc_sensory_tbc": "sensory",
         "descending_neuron": "descending", "efferent_descending": "descending",
         "ascending_neuron": "ascending", "efferent_ascending": "ascending",
         "cb_motor": "motor", "vnc_motor": "motor",
         "cb_endocrine": "endocrine", "vnc_endocrine": "endocrine",
         "vnc_intrinsic": "vnc_intrinsic", "vnc_efferent": "efferent",
         "cb_efferent": "efferent", "ENS": "efferent", "vnc_tbc": "vnc_intrinsic"}
INHIBITORY = {"gaba", "glutamate", "histamine"}
CATS = [("Optic lobe", "#22d3ee"), ("Central brain", "#c084fc"), ("Sensory input", "#35d39a"),
        ("Visual projection", "#2dd4bf"), ("Ascending", "#f5b638"), ("Descending", "#fb923c"),
        ("Motor / endocrine", "#fb7185"), ("Nerve cord", "#94a3b8")]
REGIONS = ["Unassigned", "Optic lobe", "Mushroom body", "Central complex", "Antennal lobe",
           "Lateral horn", "SEZ / motor out", "Visual projection", "Ascending", "Nerve cord (VNC)"]


def fetch():
    import urllib.request
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES.values():
        p = RAW / f
        if not p.exists():
            print("downloading", f, flush=True)
            urllib.request.urlretrieve(BASE + f, p)


def main():
    t0 = time.perf_counter()
    fetch()
    a = pd.read_feather(RAW / FILES["ann"])
    a = a[a.status == "Traced"].drop_duplicates("bodyId").reset_index(drop=True)
    ids = a.bodyId.to_numpy(np.int64)
    N = len(ids)
    print(f"{N} traced neurons", flush=True)
    # ---- neurotransmitters and signs
    nt = pd.read_feather(RAW / FILES["nt"], columns=["body", "consensus_nt", "predicted_nt"])
    nt = nt.drop_duplicates("body").set_index("body").reindex(ids)
    nt_s = nt.consensus_nt.fillna(nt.predicted_nt).fillna("unknown").astype(str).str.lower()
    sign = np.where(nt_s.isin(INHIBITORY).to_numpy(), -1, 1).astype(np.int64)
    # ---- FlyWire-schema labels by type transfer
    fw = pd.read_csv(ROOT / "data" / "flywire_meta" / "neuron_annotations.tsv", sep="\t",
                     low_memory=False, usecols=["cell_type", "super_class", "cell_class", "cell_sub_class"])
    fw = fw[fw.cell_type.notna()]
    fw_by_type = fw.groupby("cell_type").agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    fwt = a.flywireType.astype("string")
    mt = a.type.astype("string")       # the male type is often finer (LB3c vs flywireType LB3)
    key = mt.where(mt.isin(fw_by_type.index), fwt.where(fwt.isin(fw_by_type.index)))
    borrowed = fw_by_type.reindex(key.fillna("__none__").to_numpy())
    super_class = a.superclass.map(SUPER).fillna("")
    cell_class = pd.Series(borrowed.cell_class.to_numpy(), dtype="string").fillna(
        a["class"].astype("string")).fillna("")
    cell_sub = pd.Series(borrowed.cell_sub_class.to_numpy(), dtype="string").fillna(
        a.subclass.astype("string")).fillna("")
    side = a.somaSide.fillna(a.rootSide).map({"L": "left", "R": "right", "M": "center"}).fillna("na")
    # ---- positions: soma, else partner-weighted mean (needs the edges)
    print("reading weights ...", flush=True)
    w = pd.read_feather(RAW / FILES["w"], columns=["body_pre", "body_post", "weight"])
    idx = pd.Series(np.arange(N), index=ids)
    pre = idx.reindex(w.body_pre.to_numpy()).to_numpy()
    post = idx.reindex(w.body_post.to_numpy()).to_numpy()
    ok = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[ok].astype(np.int64), post[ok].astype(np.int64)
    cnt = w.weight.to_numpy(np.int64)[ok]
    del w
    print(f"{ok.sum()} edges among traced neurons ({(~ok).sum()} dropped), "
          f"{cnt.sum()} synapses", flush=True)
    soma = np.full((N, 3), np.nan)
    has = a.somaLocation.notna().to_numpy()
    soma[has] = np.stack(a.somaLocation[has].to_numpy()).astype(float) * 0.008   # um
    approx = ~has
    for _ in range(2):                       # partners' somata, twice (chains)
        known = ~np.isnan(soma[:, 0])
        acc = np.zeros((N, 3))
        wsum = np.zeros(N)
        for s, d in ((pre, post), (post, pre)):
            m = known[d] & ~known[s]
            for j in range(3):
                acc[:, j] += np.bincount(s[m], weights=soma[d[m], j] * cnt[m], minlength=N)
            wsum += np.bincount(s[m], weights=cnt[m], minlength=N)
        fill = (~known) & (wsum > 0)
        soma[fill] = acc[fill] / wsum[fill, None]
    soma[np.isnan(soma)] = np.nanmean(soma, 0)[None].repeat(N, 0)[np.isnan(soma)]
    # ---- files
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flywire_meta").mkdir(exist_ok=True)
    pd.DataFrame({"Completed": True}, index=pd.Index(ids, name="")).to_csv(OUT / "2025_Completeness_783.csv")
    ann = pd.DataFrame({
        "root_id": ids, "super_class": super_class, "cell_class": cell_class,
        "cell_sub_class": cell_sub, "cell_type": a.type.astype("string").fillna(""),
        "flywire_type": fwt.fillna(""), "hemibrain_type": a.hemibrainType.astype("string").fillna(""),
        "soma_x": soma[:, 0].round(1), "soma_y": soma[:, 1].round(1), "soma_z": soma[:, 2].round(1),
        "top_nt": nt_s.to_numpy(), "side": side, "status": "Traced",
        "position_approx": approx, "mcns_superclass": a.superclass.astype("string").fillna("")})
    ann.to_csv(OUT / "flywire_meta" / "neuron_annotations.tsv", sep="\t", index=False)
    wsg = cnt * sign[pre]
    pd.DataFrame({"Presynaptic_ID": ids[pre], "Postsynaptic_ID": ids[post],
                  "Presynaptic_Index": pre, "Postsynaptic_Index": post,
                  "Connectivity": cnt, "Excitatory": sign[pre], "Excitatory x Connectivity": wsg}
                 ).to_parquet(OUT / "2025_Connectivity_783.parquet", index=False)
    k = pre * N + post
    o = np.argsort(k, kind="stable")
    k, ws = k[o], wsg[o]
    uk, st = np.unique(k, return_index=True)
    ws = np.add.reduceat(ws, st)
    crow = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(uk // N, minlength=N), out=crow[1:])
    torch.save({"crow": torch.from_numpy(crow), "post": torch.from_numpy((uk % N).astype(np.int64)),
                "val": torch.from_numpy(ws.astype(np.float32))}, OUT / "fanout_csc.pt")
    # ---- Lab atlas
    sc, cc = super_class.to_numpy(), cell_class.to_numpy()
    cat = np.full(N, 1, np.uint8)
    for lab, m in [(0, sc == "optic"), (2, np.isin(sc, ["sensory", "sensory_ascending"])),
                   (3, np.isin(sc, ["visual_projection", "visual_centrifugal"])), (4, sc == "ascending"),
                   (5, sc == "descending"), (6, np.isin(sc, ["motor", "endocrine", "efferent"])),
                   (7, sc == "vnc_intrinsic")]:
        cat[m] = lab
    vnc = a.superclass.astype(str).str.startswith("vnc").to_numpy()
    reg = np.zeros(N, np.uint8)
    for lab, m in [(1, sc == "optic"), (2, np.isin(cc, ["Kenyon_Cell", "MBON", "DAN"])),
                   (3, cc == "CX"), (4, np.isin(cc, ["ALPN", "ALLN", "olfactory"])),
                   (5, np.isin(cc, ["LHLN", "LHCENT"])),
                   (6, ((sc == "descending") | (cc == "gustatory") | (sc == "motor")) & ~vnc),
                   (7, np.isin(sc, ["visual_projection", "visual_centrifugal"])), (8, sc == "ascending"),
                   (9, vnc)]:
        reg[m] = lab
    xyz = soma - soma.mean(0)
    xyz = np.clip(xyz * 2.0, -32000, 32000).astype(np.int16)        # 0.5 um units
    np.savez_compressed(OUT / "flywire_meta" / "atlas_aligned.npz", xyz=xyz, cat=cat, region=reg,
                        has_pos=~approx, cat_keys=np.array([c[0] for c in CATS]),
                        cat_labels=np.array([c[0] for c in CATS]),
                        cat_colors=np.array([c[1] for c in CATS]),
                        region_keys=np.array(REGIONS), region_labels=np.array(REGIONS))
    print(f"N={N} edges={len(uk)} synapses={cnt.sum()} |w|max={int(np.abs(ws).max())} "
          f"inhibitory share={np.mean(sign < 0):.3f}; {time.perf_counter() - t0:.0f}s -> {OUT}", flush=True)
    print("super_class:", pd.Series(sc).value_counts().to_dict())


if __name__ == "__main__":
    main()
