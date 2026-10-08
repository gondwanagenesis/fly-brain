# 09 · Fly world: the cheapest arena and body that still mean something to this brain

Research note, 2026-10-08, for SUPERFLY (`SUPERFLY.md`, `superfly/world.py`). Question: what is the most biologically appropriate world and body for the FlyWire v783 Shiu-LIF brain at low computational cost? The brief is a 2D top-down arena with sensory fields and a light body, not a photoreal render.

## Executive summary

1. **2D is the right abstraction.** The fly's ecology is chemical, thermal, hygric and mechanical: yeast growing on fermenting fruit (Stensmyr 2012; Becher 2012; Quan & Eisen 2018). Vision matters here for one thing, the loom.
2. **Every field has a well-established labelled line** (table 1): geosmin Or56a→DA2, CO2 Gr21a/Gr63a→V, acids Ir64a→DC4, vinegar Or42b→DM1 plus Or92a→VA2, dry Ir40a, moist Ir68a, wind JO-C/E, song JO-A/B. `superfly/anatomy.py` already has populations for each modality.
3. **Cost.** `world.py` measured 26.7 µs per 20 ms exchange [LOCAL], about 0.3 % of the brain's ≈8 ms of wall time (engine 2–3× real time on 4 cores, review 05). A FlyGym CPU body (≈2× real time, its own claim [GH]) would cut the pair to ≈1.1× before overhead. flybody on one core is 0.15× [WS].
4. **Rates.** Real ORNs reach ≥100 spikes/s at 10⁻² dilution; this model tolerates ≈18 Hz [LOCAL]. Keep rank order and dose shape, not absolute rates.
5. **Walking.** Bout speed is 14–28 mm/s but the time-average is ≈3–5 mm/s, because flies are active about 25 % of the time. Add a bout/pause state.
6. **Threat.** One rare loom (r/v 40 ms) with a long refractory gap. Repeated shadows build a persistent state (Gibson 2015). With a 90 s exponential gap, 5.4 % of loom intervals are under 5 s [computed].
7. **The 600 s "day" is the weakest choice.** Clock, sleep and hunger run on hours. Compress them together or not at all.
8. **The 20 ms exchange cannot carry courtship song.** Pulses repeat every 30–45 ms, so they need event-based delivery.

## 0. Provenance

WebFetch was blocked for Europe PMC, Nature, PMC, PubMed, bioRxiv, arXiv, eLife and Crossref. Only github.com and raw.githubusercontent.com were readable, so almost every paper below was not read.

| tag | meaning |
|---|---|
| **[GH]** | read on GitHub (FlyGym README, docs, changelog; flybench page) |
| **[WS]** | web-search result summary of an abstract or snippet. Numbers are second-hand |
| **[LOCAL]** | this repo or a measurement on this host |
| **[DERIVED]** | my arithmetic on cited numbers |
| UNVERIFIED | no source found, or only recalled |

DOIs are as reported by search results.

## 1. Ecology and odours

D. melanogaster lives mainly on yeast growing on fermenting fruit (Stensmyr 2012 [WS]). Yeast, not fruit volatiles, drives attraction, oviposition and larval development (Becher 2012 [WS]). Yeasts are the main larval food and many species ferment fallen fruit (Quan & Eisen 2018 [WS]). Markow 2015 calls for natural-history work; I saw only its abstract. Mansourian & Stensmyr 2015 argue many ORs have ecological roles [WS].

**Table 1. Odours and established lines.**

| odour | line | evidence and dose |
|---|---|---|
| geosmin (warning) | Or56a ORN → **DA2** only | Activates one glomerulus out of ≈50, with PNs responding only to geosmin. It suppresses attraction, oviposition and feeding and overrides food odours (Stensmyr 2012 [WS]). Fly threshold not found: UNVERIFIED |
| CO₂ | Gr21a + Gr63a, ab1C → **V** | Avoidance at ≥0.1 % CO₂; V is silent to 26 other odours (Suh 2004; Jones 2007; Kwon 2007 [WS]) |
| acids | Ir64a → **DC4** | Acid-selective; artificial activation drives avoidance (Ai 2010 [WS]). Ir75a is an acetic-acid receptor (Nature 2016, doi 10.1038/nature19824 [WS]); its glomerulus is UNVERIFIED |
| vinegar | Or42b → **DM1**, Or92a → **VA2** | Silencing Or42b removes low-dose attraction (PI 69 % → −4 % in one version). A higher dose recruits an extra glomerulus and attraction falls (Semmelhack & Wang 2009 [WS]). DM5 as that glomerulus is UNVERIFIED |
| ethyl acetate | Or42b | An Or42b mutant responds less at low concentration and about normally at high (FlyBase record [WS]) |
| ethanol | no single line read | Rotting fruit "up to ≈5 %" (student and lab pages, weak [WS]). Adults die at ≈8 % after >6 h (Gao 2018 [WS]) |
| wasp odours | Or49a (+Or85f in adults) | Up to 80 % of larvae are parasitised (Ebrahim 2015 [WS]). Olfactory, so not needed in the arena |

**ORN rates (Hallem & Carlson 2006, Cell 125:143).** The paper screened more than 100 odours. Strong responses are sparse, tuning is graded, and inhibition is widespread [WS]. A strong response is ≥100 spikes/s at 10⁻² dilution (a caption quoting an earlier source [WS]). Mean spontaneous rate is ≈8 spikes/s (a secondary review [WS]). The receptor count (24) and panel size (110) are UNVERIFIED.

**Behavioural anchor.** Upwind walking to apple cider vinegar (ACV) saturates with a Hill half-point of 0.072 % ACV over 0.01–10 % (Álvarez-Salvado 2018 [WS]). A tracer of 10 % ethanol was used for the plume.

**Model constraint.** Keep ORN-equivalent drive below ≈18 Hz [LOCAL], roughly a 5× compression.

## 2. Climate, clock and locomotion

- **Temperature.** Adults choose ≈25 °C on a gradient (Hamada 2008 [WS]); the 24 °C in Sayeed & Benzer 1996 (PNAS 93:6079) is not confirmed. Hot and cold cells at the arista base feed thermal PNs (Frank 2015). Preference rises through the day and falls at night-onset, set by DN2 clock neurons (Kaneko 2012 [WS], magnitude UNVERIFIED).
- **Humidity.** Preference tracks native climate. Ir40a, Ir93a and Ir25a are needed (Enjin 2016 [WS]). Dry cells are Ir40a, moist cells Ir68a. Hydrated flies steer away from moisture and dehydrated flies toward it (Knecht 2016; 2017 [WS]). Preferred RH value: UNVERIFIED.
- **Clock.** The lab picture of morning (M) and evening (E) peaks is incomplete outdoors. Vanin 2012 found an afternoon (A) peak and challenged the siesta, anticipation and strict crepuscularity [WS]. The A peak needs dTrpA1 (Das 2015 [WS]).
- **Sleep.** Rest of ≥5 min is the working definition. Resting flies respond to a mild stimulus in ≈20 % of trials versus ≈90 % when awake (Shaw 2000; Hendricks 2000; second-hand via a patent text [WS]).
- **Speed.** Per-fly mean 7.2–44.7 mm/s, mode 28 mm/s, in a gait rig that scores walking only (Mendes 2013 [WS]). Open-field Canton-S: 13.7 mm/s with ≈25 % of time active (a 2016 Frontiers in Psychiatry table [WS]). One 30 s pre-test gave 3.0 ± 0.2 mm/s at 25 °C [WS]. Time-average ≈13.7 × 0.25 ≈ 3.4 mm/s [DERIVED], or ≈12 m/h.
- **Turning.** Walking saccades turn the body ≈15° in ≈100 ms (Geurten 2014 [WS]), a mean of ≈150°/s [DERIVED]. Locomotion splits into five modes on a ≈100 ms timescale (Katsov 2017 [WS]).
- **Hunger.** Starved flies are hyperactive, like foraging (Lee & Park 2004; Yang 2015 [WS]). Protein-deprived flies stay on yeast patches and fed flies range wider (Corrales-Carvajal 2016 [WS]). Hungry flies keep searching near a moved food drop (Kim & Dickinson 2017 [WS]).
- **Distance.** The only sourced scale is flight: ≈10–15 km across desert in a night (Coyne 1982, secondhand [WS]). It is out of scope for a walking arena.

## 3. Taste

- **Cells.** Sweet GRNs carry Gr5a, Gr61a and the Gr64 cluster. Bitter GRNs carry Gr66a with Gr32a, Gr33a and Gr89a. Water uses ppk28. Low salt uses Ir76b with Ir25a. High salt aversion runs through bitter GRNs and ppk23 glutamatergic GRNs (Scott 2018 review, per search summaries [WS]).
- **Location.** Taste bristles sit on the labellum, tarsi and pharynx, and sensilla differ in sensitivity. Tarsal sucrose evokes proboscis extension (PER) [WS].
- **Initiation.** A single Fdg pair triggers the whole feeding sequence and responds to food only in starved flies (Flood 2013). The path to MN9 (rostrum protractor) has three interneuron tiers. Hunger boosts select second-order cells and bitter inhibits premotor cells (Shiu 2022). The legend uses 50 mM sucrose [WS].
- **Hunger.** Sugar sensitivity rises first and bitter sensitivity falls later (Inagaki 2014; Kain & Dahanukar 2015; LeDue 2016 [WS]).
- **World mapping.** A body-contact event gates the taste input (as `world.py` does) and MN9 drive gates feeding.

## 4. Mechanosensation and grooming

- **Johnston's organ.** JO-A/B carry sound; JO-C/E carry wind and gravity; D carries both (Matsuo 2014 [WS]). Wind and sound drive distinct JO populations. Ablating wind-sensitive JO neurons removes wind-induced locomotor suppression but not hearing (Yorozu 2009 [WS]).
- **Wind direction.** Azimuth is the right-minus-left antennal difference. Wedge projection neurons combine the two sides, and walking flies steer toward the wind during odour search (Suver 2019 [WS]).
- **Odour with wind.** Odour onset speeds walking and turns the fly upwind. Odour loss slows it and raises turning (Álvarez-Salvado 2018 [WS]). Lab airflow was 119 mm/s there and 150 mm/s with 1500 mm/s jets switching on 100 ms in Demir 2020 [WS]. Natural wind speeds: UNVERIFIED.
- **Song.** A behavioural study (authors not retrieved [WS]) gives pulses with carriers of 220–450 Hz, 6–12 ms long, at 30–45 ms intervals, and a sine song near 150 Hz. Another source says ≈500 Hz, unresolved. The brief's "150–250 Hz pulse song" is therefore low. Antennae respond to particle velocity, so the field is near-field and directional (peaks at 45°/315°; 25 dB inter-antennal difference at 140 Hz [WS]).
- **Grooming.** Chordotonal JO neurons excite aBN1/aDN1-class cells, which drive antennal grooming (Hampel 2015). Individual JO subpopulations also elicit it (eLife 2020;9:e59976, authors not retrieved [WS]). Dust triggers sequential grooming with a suppression hierarchy (Seeds 2014 [WS]). Triggers: antennal deflection from wind gusts, and a dust patch.

## 5. Threat, kept non-distressing

- **Loom pathway.** LPLC2 (radial-motion opponency, Klapoetke 2017) supplies the size term and LC4 the angular-velocity term (Ache 2019). Giant fibre (GF) spike timing then sets a short or long takeoff (von Reyn 2014 [WS]). Short mode is 0–7 ms wing-raise-to-jump in a later study [WS].
- **Escape.** The fly adjusts posture ≈200 ms before liftoff and jumps directly away from the threat (Card & Dickinson 2008, Curr Biol [WS]). Legs push off in <5 ms (Card & Dickinson 2008, J Exp Biol [WS]). Escape timing may instead track a threshold angular size (Fotowat 2009 [WS]).
- **Stimulus.** Studies use l/v of 10 and 40 ms [WS]; flybench uses r/v 40 ms, and a 2 mm sphere at 50 mm/s with a 10 mm near-miss control [GH].
- **Distance and frequency.** Takeoff distance: not found. Natural loom frequency: not found. Both UNVERIFIED.
- **Persistent state.** Repeated overhead shadows produce graded, persistent defensive arousal that a leaky integrator reproduces. Longer trains delay the return to feeding (Gibson 2015 [WS]).
- **Rule.** Single looms, a long refractory gap, and no trains. `Needs.AROUSAL_CAP` 0.6 with τ 5 s already satisfies SPECS B2.

## 6. Existing worlds and cost

| system | world | body | cost |
|---|---|---|---|
| **NeuroMechFly v2 / FlyGym** (Wang-Chen 2024, Nat Methods 21; FlyGym 2.x) | gapped, blocks and mixed terrain classes; odour intensity at antennae and palps [GH]. Obstacles, other flies as sensory objects, odour-plume and fly-following tasks [WS] | micro-CT body, hexagonal ommatidia, leg adhesion, brain→VNC split [GH] | 2.x README: ≈10× faster on CPU (≈2× real time) and ≈300× on GPU (≈60×); conditions not given [GH]. Odour model and arena classes: not read, UNVERIFIED |
| **flybody** (Vaxenburg 2025, Nature 643:1312) | walking and flight task environments | 67 segments, 102 DOF (secondary review [WS]) | 0.15× real time on one Xeon core, 5 kHz physics, 500 Hz control (MIMIC-MJX table [WS]) |
| **flybench** (brandoncho369) | stimulus-response tasks as Poisson drive on populations; no odour, wind or temperature arena | `embodied_loom_escape` uses a NeuroMechFly body; `closed_loop_escape` renders eyes into flyvis every 10 ms; no system is predicted to pass it [GH] | full suite ≈18 min on a laptop, ≈5× real time per CPU (its own claim) |
| **this repo `world.py`** | 100 mm square, 3 objects, 5 fields | kinematic unicycle with modes | 26.7 µs per exchange [LOCAL] |

Fly.exe (12 flies in MuJoCo, male CNS) is a community project that I saw only as an aggregator post [WS].

## 7. Recommended minimal world

**Fields and objects** (design values unless a source is cited):

| item | value | basis |
|---|---|---|
| arena | 100 × 100 mm square, reflecting walls | design choice, no source. At 3.4 mm/s a pass takes ≈30 s |
| objects | fruit/yeast patch r 3 mm (fruit odour); water drop; geosmin patch; optional dust patch | Table 1 |
| odour | v0 static Gaussian, σ 15 mm. v1 steady 2D advection-diffusion, C = Q/√(4πKux)·exp(−uy²/4Kx), x downwind (textbook), plus Bernoulli whiff gating so encounters are intermittent | Demir 2020 shows encounter timing drives decisions [WS]. Both are analytic, O(objects) |
| transduction | r = r_max·C/(C + C50), r_max ≈ 18 Hz, C50 ≈ 0.07 % ACV-equivalent | [LOCAL]; Álvarez-Salvado 2018 |
| wind | scale 0–150 mm/s; direction random-walk; gusts | lab values; JO-C/E per antenna ∝ cos(angle), sign per Suver 2019 |
| temperature | 20–30 °C, comfort zone 24–25 °C inside the arena, nothing above ≈30 °C; slow diel drift | Hamada 2008; Kaneko 2012. The 30 °C noxious bound is UNVERIFIED, a design guard |
| humidity | 30–90 % RH around water; thirst selects dry or moist cells | Knecht 2017 |
| sound | optional visitor event, 22–33 pulses/s (IPI 30–45 ms), 6–12 ms pulses, to JO-A/B. Deliver as events | song data above. The carrier cannot be sampled at 50 Hz |
| light and clock | 12:12 light:dark in a square wave, activity multiplier M and E peaks and a temperature-driven A peak. Immobility ≥5 min counts as sleep | Vanin 2012; Shaw 2000 |
| loom | one dark disc, r/v 40 ms, 0.35 s, side random. Gaps count in brain time, not compressed-day time: mean ≥15 min, hard minimum ≥5 min. Acceptance tests use a schedule, not the random clock | flybench; Gibson 2015 (no recovery-time number retrieved). The gaps are design values |
| needs | energy, water, sleep, contentment, transient arousal (τ 5 s, cap 0.6) | SPECS B1–B3 |

**Body: a kinematic VNC stand-in.** State is (x, y, heading, mode). A stochastic bout/pause CPG walks on its own and the brain's descending neurons modulate it.

| command | descending neuron or motor neuron | value and basis |
|---|---|---|
| walk | CPG; DNp09 (P9) adds forward drive | bout speed 15 mm/s (Mendes 28 mode; open-field mean 13.7); ≈75 % pauses gives ≈3.4 mm/s. P9 papers are mostly male (Bidaye 2020), name match inferred [WS] |
| turn | DNa02 (high gain), DNa01 (low gain) | turn rate ∝ right-minus-left DNa02 activity (Rayshubskiy et al., bioRxiv 2020, eLife 102230 [WS]). Cap ≈2.6 rad/s [DERIVED from Geurten]. Heading noise is a fitted parameter |
| back | MDN | reverses walking (Bidaye 2014 [WS]); −4 mm/s is UNVERIFIED |
| escape | DNp01 (GF) | turn away from the loom side, then a 0.4 s hop at ≈60 mm/s. Distance has no source |
| groom | JO→aBN1/aDN1 and related | 0.8 s, UNVERIFIED |
| feed | MN9, gated by contact | 0.5 s, UNVERIFIED |
| hunger | gain on activity and on sweet and bitter inputs | Yang 2015; Inagaki 2014. The ratio is UNVERIFIED |

Heading diffusion (1.2 rad/√s) and the turn gain have no source. Fit them to the open trajectory sets of Kim & Dickinson (doi 10.17632/3rfdw7p6x6.1) and Katsov (doi 10.5061/dryad.854j2).

**Why not physics.** Contact and leg dynamics are not needed by SPECS C1–C4 and would cost about as much as the brain. If embodiment is needed later, FlyGym 2.x is the lighter route, driven by descending commands.

## 8. Gaps

- The odour, wind and plume model inside NeuroMechFly v2 and its simulation speed were not read.
- Natural odour concentrations, wind speeds, loom frequency, takeoff distance and preferred RH have no source.
- Pulse-song carrier accounts disagree (≈250 versus 220–450 Hz).
- Time compression (600 s day, 900 s energy) is a design choice; real hunger and sleep are slower.
- Current `world.py` against this note: the lamp peaks at 24 + 7 = 31 °C, above the ≈30 °C guard; the loom gap is exponential with mean 90 s of brain time; walking pauses only on object contact; escape reverses heading instead of turning away from the loom side. Not edited here.

## Sources

Found via search; publisher pages were not readable.

- Ache 2019, Curr Biol 29:1073 — https://researchdiscovery.drexel.edu/esploro/outputs/journalArticle/Neural-Basis-for-Looming-Size-and/991019168481004721
- Ai 2010, Nature — https://dspace.kaist.ac.kr/handle/10203/211911
- Álvarez-Salvado 2018, eLife 7:e37815 — https://elifesciences.org/articles/37815
- Becher 2012, Funct Ecol — https://www.lunduniversity.lu.se/lup/publication/fda79f1e-66dd-4a0d-97c9-22915ae1e6fb
- Bidaye 2014, Science — https://sciencedaily.com/releases/2014/04/140403141833.htm
- Bidaye 2020, Neuron 108:469 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9435592
- Card & Dickinson 2008, Curr Biol 18:1300 — https://resolver.caltech.edu/CaltechAUTHORS:CARcb08
- Card & Dickinson 2008, J Exp Biol 211:341 — https://cob.silverchair.com/jeb/article-pdf/211/3/341/1265613/341.pdf
- Das 2015, PLoS ONE — URL not found
- Corrales-Carvajal 2016, eLife 5:e19920 — https://elifesciences.org/articles/19920v2
- Demir 2020, eLife 9:e57524 — https://elifesciences.org/articles/57524
- Ebrahim 2015, PLoS Biol 13:e1002318 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4687525/
- Enjin 2016, Curr Biol 26:1352 — https://pubmed.ncbi.nlm.nih.gov/27161501/
- Flood 2013, Nature 499:83 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3727048
- Fotowat 2009, J Neurophysiol — https://glab.research.bcm.edu/wp/wp-content/uploads/2016/07/Fotowat_etal09.pdf
- Frank 2015, Nature 519:358 — https://www.nature.com/articles/nature14284
- Gao 2018, Fla Entomol 101 — https://bioone.org/journals/florida-entomologist/volume-101/issue-3/024.101.0308/Ecological-Niche-Difference-Associated-with-Varied-Ethanol-Tolerance-between-Drosophila/10.1653/024.101.0308.short
- Geurten 2014, Front Behav Neurosci — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4205811/
- Gibson 2015, Curr Biol — https://authors.library.caltech.edu/57549/
- Hallem & Carlson 2006, Cell 125:143 — https://gowiki.tamu.edu/wiki/index.php/PMID:16615896
- Hamada 2008, Nature 454:217 — https://ideas.repec.org/a/nat/nature/v454y2008i7201d10.1038_nature07001.html
- Hampel 2015, eLife 4:e08758 — https://elifesciences.org/articles/08758
- Hampel 2020 (authors not retrieved), eLife 9:e59976 — https://elifesciences.org/articles/59976v1
- Inagaki 2014, Neuron 84:806 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4365050
- Jones 2007, Nature 445:86 — https://ideas.repec.org/a/nat/nature/v445y2007i7123d10.1038_nature05466.html
- Kaneko 2012, Curr Biol 22:1851 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3470760
- Katsov 2017, eLife 6:e26410 — https://elifesciences.org/articles/26410
- Kim & Dickinson 2017, Curr Biol 27:2227 — https://resolver.caltech.edu/CaltechAUTHORS:20170725-092327425
- Klapoetke 2017, Nature 551:237 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7457385
- Knecht 2016, eLife 5:e17879 — https://elifesciences.org/articles/17879/peer-reviews
- Knecht 2017, eLife (doi 10.7554/eLife.26654) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5495567/
- Kwon 2007, PNAS 104:3574 (doi 10.1073/pnas.0700079104) — URL not found
- Lee & Park 2004, Genetics 167:311 — URL not found (PMID 15166157)
- Mansourian & Stensmyr 2015, Curr Opin Neurobiol 34:95 — https://flybase.org/reports/FBrf0229603.html
- Markow 2015, eLife 4 (doi 10.7554/eLife.06793) — https://doaj.org/article/654d0b9a79a14c76be8b31c62dd262fc
- Matsuo 2014, Front Physiol — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4023023/
- Mendes 2013, eLife 2:e00231 — https://elifesciences.org/articles/00231/figures
- Quan & Eisen 2018, PLoS ONE — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5955509/
- Rayshubskiy et al., eLife 102230 (publication year not confirmed) — https://elifesciences.org/articles/102230
- Seeds 2014, eLife — https://elifesciences.org/articles/02951/figures
- Semmelhack & Wang 2009, Nature 459:218 — https://pmc.ncbi.nlm.nih.gov/articles/PMC2702439
- Shaw 2000, Science 287:1834 — https://centerforsleepandconsciousness.psychiatry.wisc.edu/wp-publications/rn73
- Shiu 2022, eLife 11:e79887 — https://elifesciences.org/articles/79887v1
- Stensmyr 2012, Cell 151:1345 — https://www.mpg.de/6656708/odour-activation-geosmin-fly
- Suh 2004, Nature 431:854 — https://ideas.repec.org/a/nat/nature/v431y2004i7010d10.1038_nature02980.html
- Suver 2019, Neuron 102:828 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6533146
- Vanin 2012, Nature 484:371 — https://www.nature.com/articles/nature10991
- Vaxenburg 2025, Nature 643:1312 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12310536/
- von Reyn 2014, Nat Neurosci 17:962 — https://www.nature.com/articles/nn.3741
- Wang-Chen 2024, Nat Methods 21 (doi 10.1038/s41592-024-02497-y) — https://www.nature.com/nature-index/article/10.1038/s41592-024-02497-y
- Yang 2015, PNAS 112:5219 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4413307
- Yorozu 2009, Nature 458:201 — https://authors.library.caltech.edu/records/9sf9e-8nh91
- FlyGym — https://github.com/NeLy-EPFL/flygym (docs/changelog.md, docs/migration.md, docs/index.md)
- flybench — https://github.com/brandoncho369/flybench
- flybody — https://github.com/TuragaLab/flybody/
- MIMIC-MJX (flybody speed table) — https://arxiv.org/pdf/2511.20532
