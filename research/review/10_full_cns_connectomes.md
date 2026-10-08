# Full-CNS connectomes (MaleCNS, BANC, MANC) as a real nerve cord for SUPERFLY

Researched 2026-10-08 from the sandbox (HTTPS via the agent proxy). "Verified" means I fetched it with curl or pyarrow
from this machine that day. Nothing larger than 50 MB was downloaded; bigger tables were read by HTTP range
requests (schema, a few record batches, or one column). Anything I could not check is marked **UNVERIFIED**.

## Bottom line

- **The male CNS is the practical choice, and its data is public with no login.** Three Feather files totalling
  about 566 MB give you the annotations, neurotransmitters and all neuron-to-neuron weights for brain, optic lobes
  and nerve cord of one animal. Full URLs and sizes are in section 1.
- **It is cross-matched to FlyWire.** The annotation table has a `flywireType` column. DNp01, DNa02, MBON11,
  PPL101, LC4, LPLC2 and the ORNs match by name. MN9 matches under FlyWire's label `CB0701`. The gustatory
  receptor neurons are only partly matched.
- **Cost is modest, about 1.2 to 1.7 times FlyWire per model second (my estimate, not measured).** The real risk is
  dynamics. The stock Shiu parameters make the male cord ignite, and one pre-registered test found no walking
  rhythm from it.
- **Switching substrate does not remove the need for a gait layer.** Real motor neurons can replace the hand-written
  readout now. A real central pattern generator (CPG) is not yet demonstrated in a leaky integrate-and-fire (LIF) cord.

## 1. Male CNS (Janelia / Cambridge / Google; Berg et al., Cell 2026)

Citation as given by [flycns](https://github.com/fsantibanezleal/CAOS_FlyCNS): Cell 189:5504-5526, doi
10.1016/j.cell.2026.08.015. I could not open the Cell page, so the headline "166,700 neurons" quoted by several
project READMEs is **UNVERIFIED** against the paper. What I measured from the data is below. Release history:
v0.9 on 2025-10-05 and v1.0 on 2026-06-08 ([release notes](https://male-cns.janelia.org/release/)). The license is
CC-BY ([download page](https://male-cns.janelia.org/download/)).

### 1.1 Download URLs that work without a token

Bucket root: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`. Listing it
returns HTTP 200 anonymously, and `Range:` requests work. The file descriptions are on the
[download page](https://male-cns.janelia.org/download/) and in the bucket's
[README](https://storage.googleapis.com/flyem-male-cns/README_RELEASE_BUCKET.md), which still describes v0.9 but has
the same layout.

| File (prefix `...-male-cns-v1.0`) | Bytes | Rows x cols | What I did |
|---|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | 211,577 x 36 | downloaded |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | 1,835,518 x 10 | downloaded |
| `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` | 508,025,642 | 25,563,197 x 5 | range-read, weights column scanned |
| `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` | 502,169,298 | 25,568,639 x 5 | same. Almost identical to traced-only. **UNVERIFIED** what "significant" means |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (all segments) | 1,051,241,946 | 151,856,684 x 3 | range-read |
| `body-stats-...-minconf-0.5.feather` | 778 MB | 1,349 batches | schema only |
| `syn-partners-...` (all / traced-only / significant-only) | 6.78 GB / 2.97 GB / 2.97 GB | | listing only |
| `syn-points-...`, `tbar-neurotransmitters-...` | 13.06 GB, 2.65 GB | | listing only |

- **Weights table.** Columns are `body_pre`, `body_post`, `weight` (synapse count). The traced-only and
  significant-only variants add `type_pre` and `type_post`. Rows are sorted by weight, descending.
- **Which weights file to use.** Use traced-only: both ends are traced neurons. flymsg reports 95.1 % of traced
  neurons' input synapses come from other traced neurons ([flymsg data notes](https://github.com/gianlucamazza/flymsg/blob/main/docs/data.md)).
- **Skeletons and meshes.** The same bucket holds skeletons (SWC, 8 nm units) under `v1.0/segmentation/`
  ([README](https://storage.googleapis.com/flyem-male-cns/README_RELEASE_BUCKET.md)). I did not list them.
- **neuPrint.** `https://neuprint.janelia.org/api/dbmeta/datasets` answers anonymously and lists `male-cns:v1.0`,
  but a Cypher query returns HTTP 401 without a token. The bulk files avoid that.
- **Unneeded.** `neo4j` database dumps are listed on the download page. I did not test them.

### 1.2 Schema

**Annotations** (`bodyId` is unique, 211,577 rows). Columns:
- identity and classification: `bodyId`, `type`, `instance`, `group`, `supertype`, `superclass`, `class`, `subclass`;
- sidedness and position: `somaSide` (L/R/M), `rootSide`, `somaNeuromere`, `entryNerve`, `exitNerve`,
  `somaLocation` (list of 3 ints, 8 nm voxels, present for 140,024 of the 165,122 traced neurons),
  `tosomaLocation`;
- tracing state: `status`, `statusLabel`;
- cross-dataset names: `flywireType`, `hemibrainType`, `mancType`, `mancBodyid`, `vfbId`;
- developmental and sex labels: `itoleeHl`, `trumanHl`, `birthtime`, `dimorphism`, `fruDsx`;
- others: `receptorType`, `synonyms`, `matchingNotes`, `serialMotif`.

There is no ROI or neuropil column. Per-synapse ROIs are only in the 13 GB `syn-points` table and in neuPrint
(token). For 2D plotting use `somaLocation`.

**Neurotransmitters.** Per body: `predicted_nt`, `predicted_nt_confidence` (median 0.93 over traced neurons),
`ground_truth`, `celltype_predicted_nt` and `consensus_nt`. Counts of `consensus_nt` over the 165,122 traced neurons:

| consensus_nt | neurons |
|---|---|
| acetylcholine | 103,718 |
| glutamate | 29,296 |
| GABA | 22,055 |
| histamine | 5,910 |
| unclear | 3,100 |
| dopamine / octopamine / serotonin | 392 / 101 / 48 |
| no row | 502 |

**Counts (from the files).** 165,122 neurons have `status == "Traced"`. Their weights sum to 124,025,046 synapses.

| superclass (traced) | n |
|---|---|
| optic-lobe intrinsic | 89,390 |
| central-brain intrinsic | 32,160 |
| VNC intrinsic | 13,151 |
| visual projection | 9,201 |
| VNC sensory | 6,365 |
| central-brain sensory | 4,868 |
| optic-lobe sensory | 4,114 |
| ascending | 1,846 |
| descending | 1,314 |
| VNC motor | 708 |
| central-brain motor | 107 |

### 1.3 Cross-match to FlyWire

I looked up each requested type in `type` and `flywireType`:

| FlyWire type | MaleCNS result |
|---|---|
| DNp01 (giant fibre) | 2 neurons, bodyIds 10001 (R) and 10010 (L). `type == flywireType` |
| DNa02 | 2 neurons, 10360 (R) and 523769 (L) |
| MBON11 | 2 neurons, 10704 (L) and 11402 (R) |
| PPL101 | 2 neurons, 11327 (R) and 11900 (L) |
| LC4 / LPLC2 | 126 / 185 neurons |
| MN9 | 2 neurons, 10331 (L) and 16949 (R). **`type` is `MN9` but `flywireType` is `CB0701`.** This repo's own FlyWire annotations label the Shiu MN9 pair (roots 720575940660219265 and 720575940618238523) `CB0701`, so a join on `flywireType` works |
| Olfactory receptor neurons | class `olfactory`: 2,639 neurons, 2,635 with a `flywireType` such as `ORN_DA1` (204 neurons) |
| Gustatory receptor neurons | class `gustatory`: 1,428 neurons, only 274 with a `flywireType` (`LB3`, `claw_tpGRN`, ...). 1,416 carry male-only names (`LgLG1a`, `WG1`, ...). **Sugar and bitter sets must be rebuilt** |

Coverage over the traced neurons:
- 141,169 of 165,122 (85 %) have a `flywireType`.
- Descending: 1,304 of 1,314. Ascending: 1,249 of 1,846.
- VNC motor neurons: 0 of 708 (they use MANC muscle-style names instead).
- `flywireType` can hold comma-separated groups (for example `VS1,...,VS8`), so split before joining.

Against this repo's `data/flywire_meta/neuron_annotations.tsv`, 132,250 of 139,248 FlyWire neurons (95.0 %) have a
`cell_type` that appears in the male `flywireType` set. 456 of 473 FlyWire descending types appear in the male
descending set.

### 1.4 Caveats from people who ran it

- **Synapse density.** Male neurons carry a median 1.81 times the synapses of their FAFB counterparts (7,327 matched
  types). At Shiu's `w_syn = 0.275`, the whole male CNS falls into self-sustained activity after a strong stimulus.
  flymsg uses 0.192 (0.275 / 1.43), chosen by its own model selection
  ([flymsg model notes](https://github.com/gianlucamazza/flymsg/blob/main/docs/model.md)).
- **Why this matters for edge counts.** The same density difference shows up in the table in section 5.
- **Transmitters.** They are machine predictions, and 3,100 neurons are "unclear".

## 2. BANC (female brain and nerve cord; Bates et al.), kept short

Sources: [BANC README](https://raw.githubusercontent.com/htem/BANC-project/main/README.md), the [Dataverse
deposit](https://doi.org/10.7910/DVN/7WTH1N) and the bucket's
[README](https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/README.txt). The paper is
[Nature 2026](https://doi.org/10.1038/s41586-026-10735-w), preprint bioRxiv 10.1101/2025.07.31.667571
([abstract via API](https://api.biorxiv.org/details/biorxiv/10.1101/2025.07.31.667571)). I could not open Nature or
bioRxiv pages.

- **Counts.**
  - `banc_888_meta.feather` has 188,508 rows today (188,162 when documented). These are segments, including glia,
    trachea and fragments, so the README's "approximately 188,000 neurons" overstates neurons.
  - 150,952 are flagged `proofread`. About 146,769 rows carry a neuronal `super_class`.
  - The README reports 199 million predicted synapses (all segments; **UNVERIFIED** for neurons only).
- **Regions.** Brain, optic lobes and VNC are all present. The preprint-era release notes say the left optic lobe was
  not fully proofread ([changelog](https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/CHANGELOG.md)).
- **Annotations.** Cell-type hierarchy, `hemilineage`, `side`, `nerve`, `neuromere`, `flow`, predicted and verified
  neurotransmitter, and cross-dataset type columns for FAFB, MANC, maleCNS, hemibrain and FANC
  ([column documentation](https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/documentation/banc_888_meta.md)).
- **Matching to FlyWire (counted from the meta table).**
  - Descending neurons: 1,316, all with an FAFB type.
  - Ascending neurons: 1,849, of which 1,556 have an FAFB type and 1,830 a MANC type.
  - Motor neurons: 805, of which 106 have an FAFB type.
- **Download, no login.** The bucket `gs://lee-lab_brain-and-nerve-cord-fly-connectome` is anonymously readable over
  HTTPS, and the same files are on Dataverse (379 files, 536 GB total). Dataverse `/api/access/datafile/<id>`
  redirects to a signed S3 URL, and a range request returned HTTP 206.

| File | Size | Rows |
|---|---|---|
| `compiled_data/banc_888/banc_888_meta.feather` | 57.5 MB | 188,508 |
| `compiled_data/banc_888/banc_888_edgelist_simple_v2.feather` (size >= 5) | 305 MB | 11,510,975 |
| `compiled_data/banc_888/banc_888_edgelist_simple_v3.feather` (size >= 10) | 359 MB | 13,507,098 |
| `compiled_data/banc_888/banc_888_neurotransmitter_prediction_v2.csv` | 21 MB | |

- **Codex.** The [codex.flywire.ai/banc](https://codex.flywire.ai/banc) page returns HTTP 200. I did not test its
  export endpoints, so whether they need a login is **UNVERIFIED**. The public CAVE datastack
  `brain_and_nerve_cord_public` is named in the [codex_annotations notes](https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/documentation/codex_annotations.md).
  Whether it needs a token is **UNVERIFIED**.

## 3. MANC (male nerve cord only)

- **Public bucket.** The [Janelia MANC page](https://www.janelia.org/manc-connectome) links the public bucket
  `flyem-manc-exports`, which lists anonymously. Under `v1.0/` there are `manc-v1.0-neuron-properties.feather` (17 MB),
  `manc-traced-adjacencies-v1.0/traced-connections.csv` (75 MB) and `traced-neurons.csv`.
- **Newer synapse table.** `manc-seg-v1p2` holds `manc-v1.2-synapse-partners-minconf-0.0.feather` (1.9 GB).
- **neuPrint.** It serves `manc:v1.0`, `v1.2.1` and `v1.2.3` ([dataset list](https://neuprint.janelia.org/api/dbmeta/datasets)).
- **Mirror.** BANC's bucket also mirrors MANC v1.2.1 and MaleCNS v0.9 tables under unified column names
  (`compiled_data/manc_121/`: meta 1.45 MB, edgelist 87 MB).

## 4. Has anyone simulated this, and what is public

| Project | What it does | Status |
|---|---|---|
| [gianlucamazza/flymsg](https://github.com/gianlucamazza/flymsg) (MIT) | Whole-CNS Shiu-style LIF on MaleCNS v1.0, validation battery, 3D replay, same model on FAFB | Claims 29 of 29 checks passed and agreement with Shiu's brian2 code at r >= 0.998 on FAFB. Self-reported. `flymsg fetch` needs no account |
| [TheMrRaGe/flybrain](https://github.com/TheMrRaGe/flybrain) | LIF on 162,517 neurons and 6.1 M connections. Steering, habituation, olfactory conditioning | Scripts plus a FINDINGS.md log |
| [fsantibanezleal/CAOS_FlyCNS](https://github.com/fsantibanezleal/CAOS_FlyCNS) | Compiles MaleCNS to a simulation graph (SHA-256-locked tables), Shiu LIF plus graded optic lobe (NumPy, PyTorch, WebGPU) | Active |
| [IONOFIELD/FLYCNS](https://github.com/IONOFIELD/FLYCNS) | Brian2 whole-CNS model scored against published physiology, with null models | Needs a neuPrint token |
| [Ryans-sS/malecns-embodied-fly](https://github.com/Ryans-sS/malecns-embodied-fly) | MaleCNS LIF brain driving a NeuroMechFly body (flygym 2.1.0) | The README says the **walking pattern generator is a model**, not the VNC |
| [fruitflydev/therealfly](https://github.com/fruitflydev/therealfly) | MaleCNS LIF cord to flybody joint torques, no hand-written controller | Pre-registered stage 1 **failed** (below) |
| [Pugliese et al.](https://www.biorxiv.org/content/10.1101/2025.09.12.675944.full.pdf), code [smpuglie/Pugliese_cpg_2025](https://github.com/smpuglie/Pugliese_cpg_2025) | Firing-rate model of the MANC and FANC nerve cords (4,604 neurons for the front legs). DNg100 drives a 3-neuron CPG. An LIF check gave "similar" spiking | No body. Left-right leg coupling did not emerge |
| Eon Systems, March 2026 | FlyWire brain plus NeuroMechFly | Gait said to come from NeuroMechFly's built-in controller ([The Decoder](https://the-decoder.com/startup-claims-first-full-brain-emulation-of-a-fruit-fly-in-a-simulated-body/)). Code not released |

- **The therealfly failure.** With DNa01 and DNa02 driven at 150 Hz, 35 to 44 % of VNC neurons fire and the whole CNS
  runs at 8.4 to 10 M spikes/s. The cord oscillates globally at 45 Hz with left and right legs in phase, and a
  degree-preserving scrambled cord shows the same peak. Its stage 0 usefully maps the 708 VNC motor neurons
  (381 leg, 328 with muscle names) to joints.
- **BANC.** Per the [awesome-fly list](https://github.com/cobanov/awesome-fly), bioreservoir ran both MaleCNS and BANC
  and reports BANC failing a lateral check. This is a secondary summary.
- **flygym.** The latest PyPI release is 2.1.0 ([PyPI](https://pypi.org/pypi/flygym/json)). Details of what 2.x
  changes are **UNVERIFIED**.

## 5. Cost relative to FlyWire

| | FlyWire (this repo) | MaleCNS traced | BANC v888 |
|---|---|---|---|
| neurons | 138,639 | 165,122 (1.19x) | about 150,952 proofread |
| edges, any count | 15,091,983 | 25,563,197 (1.69x) | 11.5 M (v2), 13.5 M (v3) |
| edges, weight >= 5 | 2,700,513 | 6,235,682 (2.31x) | not applicable (filtered by synapse size) |
| synapses | 54.5 M | 124.0 M (2.28x) | **UNVERIFIED** |

The FlyWire row is counted from `data/2025_Connectivity_783.parquet`. The MaleCNS row is counted from the weights
file.

- **My estimate.** If the cord stays as sparse as the brain, the cost is 1.2x (neuron updates) to 1.7x (spike
  delivery). That puts the male CNS at roughly 1.2 to 2x real time on four cores, given the 2 to 3x real time quoted for
  FlyWire in [05_existing_software.md](05_existing_software.md). Not measured.
- **The ignited case breaks this.** The therealfly run implies about 9 M spikes/s times about 38 out-edges per
  neuron (6.24 M / 165 k), roughly 340 M synaptic events per model second. That would lose the sparse-activity
  advantage.

## 6. Recommendation

Move to the male CNS, but stage it.

1. **Now: build the male-CNS substrate.**
   - Download the annotations, neurotransmitters and traced-only weights (about 566 MB).
   - Build signed CSR arrays: acetylcholine is +1; GABA, glutamate and histamine are -1; the rest are 0 (the flymsg
     convention).
   - Rescale `w_syn` for the 1.81x synapse density (flymsg's 0.192 is the starting point).
   - Use `flywireType` to port the sensory sets and the readout sets (DNp01, DNa02, MN9 as `CB0701`).
2. **Keep the stand-in as the gait layer.** No leaky integrate-and-fire cord has yet produced a walking rhythm (stage 1
   of therealfly). Use real VNC motor neurons and real descending-to-motor pathways only as readout, and map them to
   joints via the 708-motor-neuron table. Keep the hand-written pattern generator for the rhythm itself.
3. **Fallback if the full male CNS ignites: graft (b).** Keep the FlyWire brain. Join descending neurons by type
   name (456 of 473 types) and run only the male VNC (about 22.7 k neurons per therealfly) as the cord. Edge count for
   that subset is **UNVERIFIED** and needs one pass over the 508 MB weights file.
4. **BANC only if something is missing from MaleCNS.** It is female (the same sex as FlyWire), but its connectivity
   file counts synapses differently, and it needs a second set of calibrations.

Concrete files for step 1:
`body-annotations-male-cns-v1.0-minconf-0.5.feather`,
`body-neurotransmitters-male-cns-v1.0.feather`,
`connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather`.

## What I could not reach

- `www.nature.com` and `www.biorxiv.org`: blocked in WebFetch; `curl` to bioRxiv returned HTTP 429.
- `europepmc.org`: Cloudflare challenge (HTTP 403).
- `github.com` HTML: HTTP 403 (raw.githubusercontent.com works).
- `huggingface.co` and `arxiv.org`: HTTP 403 according to the proxy log.
