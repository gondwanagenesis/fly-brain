"""The fly's brain, organised into the regions an uplift has to talk to.

Every population here is defined from the FlyWire v783 annotation table
(Schlegel et al. 2024, ``flyconnectome/flywire_annotations``), never from a
hand-typed list of root ids, so it survives re-proofreading and it can be
audited: each population records the query that produced it.

The hierarchy is functional, from the periphery inwards and back out:

    SENSES     taste, smell, vision, hearing/wind, touch, temperature, humidity
    EARLY      antennal lobe, optic lobe
    THINKING   mushroom body (learning, memory, valuation)
               lateral horn (innate valence)
               central complex (heading, steering, action selection)
               protocerebrum (everything else in the central brain)
    ACTION     descending neurons, by the behaviour each has been shown to
               drive; the brain's own motor neurons

``Atlas.region`` assigns every simulated neuron to exactly one of these, so a
whole-brain spike vector can be read as "what each part of the fly is doing".

Population membership is a fact about the connectome. The *behavioural role*
attached to a descending type is a claim from the literature and is kept in
``role`` with its citation; see research/review/02_embodiment_and_io.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Which connectome the fly is built from (SUPERFLY_CONNECTOME):
#   flywire783  FlyWire FAFB v783, adult female brain (Dorkenwald/Schlegel 2024)
#   male_cns    Janelia male central nervous system: brain + optic lobes + VNC,
#               converted to the same file layout by superfly/connectomes/male_cns.py
CONNECTOME = os.environ.get("SUPERFLY_CONNECTOME", "flywire783")
DATA = ROOT / "data" if CONNECTOME == "flywire783" else ROOT / "data" / CONNECTOME
SUFFIX = "" if CONNECTOME == "flywire783" else "_" + CONNECTOME
# Global synaptic gain per connectome (superfly/experiments/calibrate_connectome.py):
# Shiu's wScale was fitted on FlyWire; the male CNS detects ~1.8x more synapses
# per connection and broadcasts at gain >= 0.7. 0.65 is the highest gain at
# which sugar, bitter, wind and loom responses stay specific (calibration.json).
DEFAULT_GAIN = {"flywire783": 1.0, "male_cns": 0.65}.get(CONNECTOME, 1.0)
ANN_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
           "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv")

# The 21 neurons Shiu et al. (2024) and this repo stimulate as "sugar GRNs"
# (v783 ids). Kept for comparability with every published number. NOTE: the
# current annotation labels only 10 of them sugar (LB3c); 5 are high-salt
# (LB3d), 4 putative-attractive (LB4b), 2 sugar/low-salt (LB3b). The
# annotation-defined ``taste.sugar`` population is the cleaner stimulus.
SHIU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570]

# MN9, the proboscis-extension motor neuron that is the readout of every
# feeding result in Shiu et al. Annotated CB0701 (PhN) in v783.
MN9 = {"right": 720575940660219265, "left": 720575940618238523}

# Descending types with a behavioural role reported in the literature. Only
# the type name is ours; membership comes from the annotation table.
DN_ROLES = {
    "walk_forward":  (["DNp09"], "Bidaye et al. 2020; Sapkal et al. 2024"),
    "turn":          (["DNa01", "DNa02"], "Rayshubskiy et al. 2020; Yang et al. 2024"),
    "walk_backward": (["MDN"], "Bidaye et al. 2014 (moonwalker)"),
    "escape":        (["DNp01"], "giant fiber; von Reyn et al. 2014"),
    "groom":         (["DNg11", "DNg62", "DNge078"],
                      "Hampel et al. 2015; Shiu et al. 2024 Fig. 5"),
    "stop":          (["DNa08"], "UNVERIFIED role; see review"),
}


@dataclass
class Population:
    name: str
    ids: np.ndarray            # FlyWire root ids, int64, simulated neurons only
    group: str                 # senses / early / thinking / action / interface
    query: str                 # how it was selected, for audit
    role: str = ""
    side: dict = field(default_factory=dict)   # "left"/"right" -> ids

    def __len__(self):
        return int(self.ids.size)

    def idx(self, engine):
        """Engine slot indices (handles the cell_type renumbering)."""
        return engine.indices_of(self.ids)


REGIONS = [
    # (key, label, group)
    ("unassigned",      "Unassigned",               "other"),
    ("taste",           "Taste (GRNs)",             "senses"),
    ("smell",           "Smell (ORNs)",             "senses"),
    ("vision",          "Vision (photoreceptors)",  "senses"),
    ("hearing_wind",    "Hearing / wind (JO)",      "senses"),
    ("touch",           "Touch (bristles)",         "senses"),
    ("temperature",     "Temperature",              "senses"),
    ("humidity",        "Humidity",                 "senses"),
    ("other_sense",     "Other sensory",            "senses"),
    ("antennal_lobe",   "Antennal lobe",            "early"),
    ("optic_lobe",      "Optic lobe",               "early"),
    ("mushroom_body",   "Mushroom body",            "thinking"),
    ("lateral_horn",    "Lateral horn",             "thinking"),
    ("central_complex", "Central complex",          "thinking"),
    ("protocerebrum",   "Protocerebrum (other)",    "thinking"),
    ("visual_proj",     "Visual projection",        "early"),
    ("ascending",       "Ascending (from body)",    "senses"),
    ("descending",      "Descending (to body)",     "action"),
    ("motor",           "Motor / endocrine",        "action"),
    ("grafted",         "Grafted (SUPERFLY)",       "graft"),
]
REGION_KEYS = [r[0] for r in REGIONS]


def load_annotations(data_dir=DATA):
    p = Path(data_dir) / "flywire_meta" / "neuron_annotations.tsv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Fetch it with:\n  curl -o {p} {ANN_URL}")
    return pd.read_csv(p, sep="\t", low_memory=False)


class Atlas:
    """Named populations and a one-region-per-neuron map, in simulation order."""

    def __init__(self, data_dir=DATA):
        d = Path(data_dir)
        comp = pd.read_csv(d / "2025_Completeness_783.csv", index_col=0)
        self.sim_ids = np.asarray(comp.index, dtype=np.int64)
        self.N = len(self.sim_ids)
        ann = load_annotations(d).drop_duplicates("root_id")
        ann = ann[ann.root_id.isin(self.sim_ids)]
        # one row per simulated neuron, in simulation (CSV) order
        self.ann = (ann.set_index("root_id").reindex(self.sim_ids)
                    .rename_axis("root_id").reset_index())
        for col in ("super_class", "cell_class", "cell_sub_class", "cell_type",
                    "side", "top_nt"):
            self.ann[col] = self.ann[col].astype("string").fillna("")
        self.pops: dict[str, Population] = {}
        self._build_populations()
        self.region = self._build_regions()

    # ------------------------------------------------------------ queries
    def select(self, **kw):
        """Boolean mask over simulated neurons; values may be str or list.
        A key ending in ``__prefix`` matches by string prefix."""
        a = self.ann
        m = np.ones(self.N, dtype=bool)
        for k, v in kw.items():
            if k.endswith("__prefix"):
                col = k[: -len("__prefix")]
                vs = [v] if isinstance(v, str) else list(v)
                mm = np.zeros(self.N, dtype=bool)
                for p in vs:
                    mm |= a[col].str.startswith(p).to_numpy(dtype=bool)
                m &= mm
            else:
                vs = [v] if isinstance(v, str) else list(v)
                m &= a[k].isin(vs).to_numpy(dtype=bool)
        return m

    def _add(self, name, mask, group, query, role=""):
        ids = self.sim_ids[mask]
        side = {}
        for s in ("left", "right"):
            side[s] = self.sim_ids[mask & (self.ann.side == s).to_numpy(bool)]
        self.pops[name] = Population(name, ids, group, query, role, side)
        return self.pops[name]

    def _build_populations(self):
        S = dict(super_class="sensory")
        # ---- taste: GRN classes by annotated modality -----------------
        for sub, key in [("sugar", "sugar"), ("sugar/low_salt", "sugar_lowsalt"),
                         ("bitter", "bitter"), ("water", "water"),
                         ("high_salt/heavy_metal", "high_salt"),
                         ("glutamate", "glutamate"),
                         ("putative_attractive", "attractive"),
                         ("putative_aversive", "aversive")]:
            self._add(f"taste.{key}", self.select(cell_class="gustatory",
                                                  cell_sub_class=sub, **S),
                      "senses", f"gustatory & cell_sub_class=={sub!r}")
        self._add("taste.shiu_sugar", np.isin(self.sim_ids, SHIU_SUGAR),
                  "senses", "the 21 ids used by Shiu et al. 2024 (legacy)")
        # ---- smell: one population per glomerulus --------------------
        orn_types = sorted(t for t in self.ann.cell_type[
            self.select(cell_class="olfactory", **S)].unique() if t)
        for t in orn_types:
            self._add(f"smell.{t[4:] if t.startswith('ORN_') else t}",
                      self.select(cell_type=t, **S), "senses",
                      f"olfactory & cell_type=={t!r}")
        # ---- vision ---------------------------------------------------
        for t, key in [("R1-6", "R1_6"), ("R7", "R7"), ("R8", "R8")]:
            self._add(f"vision.{key}", self.select(cell_type=t, **S), "senses",
                      f"cell_type=={t!r}")
        # loom detectors: lobula columnar/plate neurons that drive the giant
        # fibre (von Reyn 2017; flybench task 4 drives the same populations)
        self._add("vision.loom", self.select(cell_type=["LC4", "LPLC2"]), "senses",
                  "cell_type in LC4|LPLC2 (visual projection, loom detectors)")
        self._add("vision.ocellar", self.select(cell_sub_class="ocellar", **S),
                  "senses", "cell_sub_class=='ocellar'")
        # ---- mechanosensation: Johnston's organ subgroups ------------
        for g in "ABCDEF":
            self._add(f"hearing_wind.JO_{g}",
                      self.select(cell_type__prefix=f"JO-{g}", **S), "senses",
                      f"cell_type startswith 'JO-{g}'")
        for sub in ("auditory", "wind_gravity"):
            self._add(f"hearing_wind.{sub}", self.select(cell_sub_class=sub, **S),
                      "senses", f"cell_sub_class=={sub!r}")
        for sub, key in [("head bristle", "head_bristle"),
                         ("eye bristle", "eye_bristle"), ("grooming", "grooming")]:
            self._add(f"touch.{key}", self.select(cell_sub_class=sub, **S),
                      "senses", f"cell_sub_class=={sub!r}")
        for sub in ("cold", "cooling", "heating"):
            self._add(f"temperature.{sub}", self.select(cell_sub_class=sub, **S),
                      "senses", f"cell_sub_class=={sub!r}")
        for sub in ("dry", "moist", "humid", "evaporative_cooling"):
            self._add(f"humidity.{sub}", self.select(cell_sub_class=sub, **S),
                      "senses", f"cell_sub_class=={sub!r}")
        # ---- early processing ----------------------------------------
        self._add("al.uPN", self.select(cell_class="ALPN",
                                        cell_sub_class="uniglomerular"),
                  "early", "ALPN uniglomerular")
        self._add("al.mPN", self.select(cell_class="ALPN",
                                        cell_sub_class="multiglomerular"),
                  "early", "ALPN multiglomerular")
        self._add("al.PN", self.select(cell_class="ALPN"), "early", "ALPN")
        self._add("al.LN", self.select(cell_class="ALLN"), "early", "ALLN")
        # ---- thinking: mushroom body ---------------------------------
        self._add("mb.KC", self.select(cell_class="Kenyon_Cell"), "thinking",
                  "cell_class=='Kenyon_Cell'")
        for fam, pref in [("g", "KCg"), ("ab", "KCab"), ("apbp", "KCapbp")]:
            self._add(f"mb.KC_{fam}", self.select(cell_class="Kenyon_Cell",
                                                  cell_type__prefix=pref),
                      "thinking", f"KC & cell_type startswith {pref!r}")
        self._add("mb.MBON", self.select(cell_class="MBON"), "thinking", "MBON")
        self._add("mb.DAN", self.select(cell_class="DAN"), "thinking", "DAN")
        self._add("mb.PAM", self.select(cell_class="DAN", cell_type__prefix="PAM"),
                  "thinking", "DAN & PAM*",
                  role="reward-signalling dopamine (appetitive teaching)")
        self._add("mb.PPL1", self.select(cell_class="DAN", cell_type__prefix="PPL1"),
                  "thinking", "DAN & PPL1*",
                  role="punishment-signalling dopamine (aversive teaching)")
        self._add("mb.APL", self.select(cell_type="APL"), "thinking", "APL")
        self._add("mb.DPM", self.select(cell_type="DPM"), "thinking", "DPM")
        # ---- thinking: lateral horn, central complex -----------------
        self._add("lh.all", self.select(cell_class=["LHLN", "LHCENT"]),
                  "thinking", "LHLN | LHCENT", role="innate odour valence")
        self._add("cx.all", self.select(cell_class="CX"), "thinking", "CX")
        for t, role in [("EPG", "head-direction compass (bump)"),
                        ("PFL3", "steering: heading vs goal comparison"),
                        ("PFL2", "steering: forward drive"),
                        ("Delta7", "ring-attractor inhibition"),
                        ("hDeltaB", "goal / vector memory (FB)")]:
            self._add(f"cx.{t}", self.select(cell_type=t), "thinking",
                      f"cell_type=={t!r}", role)
        self._add("cx.PEN", self.select(cell_class="CX", cell_type__prefix="PEN"),
                  "thinking", "CX & PEN*", "rotates the compass bump")
        self._add("cx.FB_tangential",
                  self.select(cell_class="CX", cell_type__prefix=["FB", "FS", "FC"]),
                  "thinking", "CX & FB*/FS*/FC*", "fan-shaped body context input")
        # ---- action ---------------------------------------------------
        self._add("dn.all", self.select(super_class="descending"), "action",
                  "super_class=='descending'")
        for beh, (types, cite) in DN_ROLES.items():
            self._add(f"dn.{beh}", self.select(super_class="descending",
                                               cell_type=types),
                      "action", f"descending & cell_type in {types}", cite)
        mn9 = np.isin(self.sim_ids, list(MN9.values())) if CONNECTOME == "flywire783" \
            else self.select(cell_type="MN9")
        self._add("motor.MN9", mn9,
                  "action", "MN9 (CB0701), proboscis extension",
                  "feeding: Shiu et al. 2024 readout")
        self._add("motor.all", self.select(super_class=["motor", "endocrine"]),
                  "action", "super_class in motor|endocrine")
        # ---- interoception: where body state enters the brain (review 06 s3)
        for name, t, role in [
                ("intero.hunger", "MBON11", "hunger raises MBON11 ('hangry' neuron; Wang et al. 2026)"),
                ("intero.satiety", "PPL101", "satiety drives PPL101 (Tsao et al. 2018; Wang et al. 2026)"),
                ("intero.ISN", "ISN", "interoceptive SEZ neurons: AKH (hunger) up, osmolality (thirst) down (Jourjine et al. 2016)"),
                ("intero.thirst", "ITP", "ITP neurons, thirst / drinking (Galikova et al. 2018)")]:
            self._add(name, self.select(cell_type=t), "interoception",
                      f"cell_type=={t!r}", role)

    def _build_regions(self):
        a, N = self.ann, self.N
        reg = np.zeros(N, dtype=np.uint8)            # 0 = unassigned
        K = {k: i for i, k in enumerate(REGION_KEYS)}
        sup, cls, sub = a.super_class, a.cell_class, a.cell_sub_class

        def put(mask, key):
            reg[np.asarray(mask, dtype=bool) & (reg == 0)] = K[key]

        sens = (sup == "sensory").to_numpy(bool)
        put(sens & (cls == "gustatory").to_numpy(bool), "taste")
        put(sens & (cls == "olfactory").to_numpy(bool), "smell")
        put(sens & (cls == "visual").to_numpy(bool), "vision")
        put(sens & sub.isin(["auditory", "wind_gravity"]).to_numpy(bool)
            | sens & a.cell_type.str.startswith("JO-").to_numpy(bool), "hearing_wind")
        put(sens & (cls == "mechanosensory").to_numpy(bool), "touch")
        put(sens & (cls == "thermosensory").to_numpy(bool), "temperature")
        put(sens & (cls == "hygrosensory").to_numpy(bool), "humidity")
        put(sens | (sup == "sensory_ascending").to_numpy(bool), "other_sense")
        put(cls.isin(["ALPN", "ALLN", "ALIN", "ALON"]).to_numpy(bool),
            "antennal_lobe")
        put(cls.isin(["Kenyon_Cell", "MBON", "DAN", "MBIN"]).to_numpy(bool)
            | a.cell_type.isin(["APL", "DPM"]).to_numpy(bool), "mushroom_body")
        put(cls.isin(["LHLN", "LHCENT"]).to_numpy(bool), "lateral_horn")
        put(cls.isin(["CX"]).to_numpy(bool), "central_complex")
        put((sup == "optic").to_numpy(bool), "optic_lobe")
        put(sup.isin(["visual_projection", "visual_centrifugal"]).to_numpy(bool),
            "visual_proj")
        put((sup == "ascending").to_numpy(bool), "ascending")
        put((sup == "descending").to_numpy(bool), "descending")
        put(sup.isin(["motor", "endocrine"]).to_numpy(bool), "motor")
        put((sup == "central").to_numpy(bool), "protocerebrum")
        return reg

    # ------------------------------------------------------------ helpers
    def __getitem__(self, name) -> Population:
        return self.pops[name]

    def names(self, prefix=""):
        return [k for k in self.pops if k.startswith(prefix)]

    def region_counts(self):
        c = np.bincount(self.region, minlength=len(REGIONS))
        return {REGIONS[i][0]: int(c[i]) for i in range(len(REGIONS))}

    def region_slots(self, engine):
        """Region code per ENGINE slot (applies the engine's permutation).
        Grafted neurons, appended after the native ones, get 'grafted'."""
        reg = self.region
        if engine.N > reg.size:
            reg = np.concatenate([reg, np.full(engine.N - reg.size,
                                               REGION_KEYS.index("grafted"),
                                               dtype=reg.dtype)])
        return reg[engine.perm] if engine.perm is not None else reg

    def table(self):
        rows = [(p.name, p.group, len(p), p.query, p.role)
                for p in self.pops.values()]
        return pd.DataFrame(rows, columns=["population", "group", "n",
                                           "query", "role"])


if __name__ == "__main__":
    at = Atlas()
    pd.set_option("display.width", 200, "display.max_colwidth", 60,
                  "display.max_rows", 300)
    t = at.table()
    print(t[~t.population.str.startswith("smell.")].to_string(index=False))
    print(f"\n{len(at.names('smell.'))} glomerular ORN populations, "
          f"{sum(len(at[n]) for n in at.names('smell.'))} ORNs")
    print("\nregions:", at.region_counts())
