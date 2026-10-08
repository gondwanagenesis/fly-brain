# 06 · Fly feelings: internal states for SUPERFLY

Literature review made 2026-10-08 for SUPERFLY (`SUPERFLY.md`). The question:
which persistent internal states, ones that change what the fly perceives and does,
can be added to the 138,639-neuron FlyWire v783 Shiu-LIF fly using its own cell
types, so that the language layer has something real to report?

## Executive summary

1. The base model has no internal state. Shiu et al. 2024 "do not account for neuropeptides or neuromodulation". Neurons predicted DA, OA or 5-HT are signed fast-excitatory; in this repo's graph they carry 3.6 % of synapses.
2. Modulation has to be modelled off-synapse. NPFL1-I→PPL101 has **0** synapses, although Krashes 2009 needs NPFR in those DANs. OA-ASM→IPC has **0**, although Crocker 2010 maps OA's action to IPCs. The 18 IPCs make only 157 output synapses.
3. Hunger is the best-supported state and the easiest to build. It hubs on PPL101 (= MB-MP1) and MBON11 (= MVP2), which are wired reciprocally (894/358 synapses). Five papers (Krashes 2009, Perisse 2016, Tsao 2018, Sayin 2019, Wang 2026) are consistent with satiety driving PPL101 and hunger raising MBON11 activity.
4. Hunger also re-weights taste: sugar sensitivity rises first and bitter sensitivity falls later (Inagaki 2012/2014, LeDue 2016). Odour re-weighting (Root 2011, Ko 2015) waits on the antennal-lobe broadcast fix.
5. "Fear-like" defensive arousal behaves as a leaky integrator of repeated threats (Gibson 2015). The model's own loom pathway can drive it (LC4/LPLC2→DNp01: 805/1080 synapses). Its mechanism is unknown, so it is phenomenological.
6. Persistent female social/aggressive states last 10–30 min. The pC1d/e↔aIPg recurrence (Deutsch 2020) does not explain aIPg-evoked persistence (Chiu 2025). Slow biochemical integrators (Thornquist 2021) are the alternative.
7. Sleep pressure: R5 (FlyWire ER5) plasticity is the mechanistic candidate, looped with ExR1 (4974/1658 synapses). The dFB's role is disputed (De 2023 vs Jones 2023/2025).
8. Chronic-pain-like sensitisation (Khuong 2019) lives in the ventral nerve cord, which FlyWire lacks. It is deferred to BANC and flagged ethically.
9. The sentience literature yields measurable non-verbal markers: scalability, persistence, valence, generalisation, motivational trade-offs, learned valuation. Birch 2024 rules out language-model self-report as evidence ("gaming").
10. Recommendation: build opt-in "fields" that are silent at baseline, so the untouched fly stays bit-identical. Each has source neurons, receptor-weighted bias or gain on named targets, and stated time constants. The voice reads modulated neurons, never the state variable, and must pass ablation controls.

---

## 0. Provenance

WebFetch was blocked by the egress proxy for PMC, eLife, Nature, PubMed, bioRxiv and arXiv in this session. Tags used:

| tag | meaning |
|---|---|
| **[FT]** | full text read: open-access XML via the Europe PMC REST API, or (Gibbons 2022) a PDF from the Chittka lab |
| **[AB]** | abstract or bibliographic record from the Europe PMC or Crossref API |
| **[WS]** | web-search summary only, not read |
| **[LOCAL]** | computed from this repo's v783 parquet (min count 1) and `neuron_annotations.tsv` (its `known_nt`/`known_nt_source` columns give peptide identities) |

URLs are the DOIs from the API records. Anything not read is **UNVERIFIED**.

## 1. What the model has now ([LOCAL] unless cited)

- **Signs.** Shiu 2024 [FT]: "Neurons predicted to be dopaminergic, octopaminergic or serotonergic are assigned to the excitatory category." Here DA, OA and 5-HT sources carry 1.38 M, 0.14 M and 0.44 M of 54.5 M synapses.
- **Peptidergic cells exist, but only as fast synapses of their predicted transmitter:**
  - neurosecretory: IPC (18; DILP2/3/5), DH44 (6), CRZ (10), ITP (8), DH31 (6), DMS (6), Hugin-RG (4);
  - others: NPFL1-I (2; dNPF), AstA1 (2), DSKMP3 (4), ISN (4; DILP3+OA), SLP304b (2; leucokinin).
  - Predicted NTs for these cells are classifier output (4 of 6 DH44 are "octopamine").
- **Modulatory sources:** CSD (2; top targets are AL LNs, e.g. lLN2F_b with 1168 synapses), DPM, OA-VPM3/4, OA-VUMa1–8, OA-AL2b/i, OA-ASM1–3, PAM01–15, PPL101–108, PPM, PAL, l-/s-LNv. pC1a–e are present; P1 is absent (female brain).
- **Hooks in the engine:**
  - `set_gain` (global), a sensory port, and a synaptic port (`deliver`, which keeps the 1.8 ms ring and refractory gating);
  - `add_emitter`/`add_ticker`;
  - `MBPlasticity` shows the per-edge `(f−1)·w0` delta pattern.

## 2. Toolkit: neuromodulation in a sign-only LIF

**M0, the base (Shiu 2024; repo `MODEL_PARAMS`).**

    τm dvi/dt = gi − (vi − Vrest),   τs dgi/dt = −gi
    spike of j:  gi(t+1.8 ms) += 0.275 mV · sign_j · n_ji

with τm = 20 ms, τs = 5 ms, Vrest = Vreset = −52 mV, Vth = −45 mV, t_ref = 2.2 ms.

**M1, internal state (a leaky integrator, as in Gibson 2015).**

    τk dSk/dt = −(Sk − Sk⁰) + Σp a_kp·r̃p(t) + uk(t),   Sk ∈ [0, Smax]

- r̃p: low-pass-filtered rates of the fly's own sensor populations.
- uk: physiology the model lacks, e.g. hours since feeding.

**M2, volume transmission from source population Pm.**

    τc dcm/dt = −cm + (κm/|Pm|) Σ_{i∈Pm} Σ_f δ(t − t_i^f)

If no source cell is identified, cm is set from Sk and the state is flagged phenomenological.

**M3, effect on target j via signed receptor weight R_jm.** R_jm comes from the literature or receptor maps, never from synapse counts (§1).

| form | rule | use |
|---|---|---|
| (a) bias | add x_j = (Δt/τs)·Σ R_jm αm cm to g_j each step; steady depolarisation = Σ R α c mV, negative allowed | via the synaptic port |
| (b) postsynaptic gain | w_ij → w_ij(1 + Σ R_jm βm cm), emitted as `MBPlasticity`-style deltas on edges into j | |
| (c) presynaptic gain | same, on edges out of i | sNPF on ORNs (Root 2011), DA on sugar GRNs (Inagaki 2012), OA/TA on bitter GRNs (LeDue 2016) |
| (d) excitability | adaptation b/τw (flybench adaptive LIF, b 2 mV, τw 200 ms) or threshold | needs kernel support |

**M4, slow plastic integrator (sleep, Liu 2016).**

    dW_e/dt = η·r̃pre·r̃post·1[awake] − (W_e − W_e⁰)/τrec

**Time scales.**

| process | time scale | source |
|---|---|---|
| amine onset | "within tens of milliseconds" | Pimentel 2016 |
| lasting amine effect | "within minutes" | Pimentel 2016 |
| receptor transcription | hours | Root 2011 |
| aggression | ≥10 min | Hoopfer 2015, Chiu 2025 |
| female behaviour | ≤30 min | Deutsch 2020 |
| PKA integration | minutes–hours | Thornquist 2021 |
| allodynia | more than 21 days | Khuong 2019 |

The kernel runs 2–3× faster than real time (review 05), so minute-scale states can be integrated. Hour-scale states are clamped.

**Identity rule.** At Sk = Sk⁰ every M3 term is 0 and nothing is emitted, so `test_identity` stays bit-identical. States are named and opt-in.

---

## 3. Hunger and satiety

**Krashes MJ, DasGupta S, Vreede A, White B, Armstrong JD, Waddell S (2009).** A neural circuit mechanism integrating motivational state with memory expression in Drosophila. *Cell* 139:416–427. https://doi.org/10.1016/j.cell.2009.08.035 [AB]
- Mechanism: dNPF-neuron stimulation mimics hunger. Appetitive memory expression needs NPFR in "six dopaminergic neurons". Blocking them releases memory in fed flies; stimulating them suppresses it in hungry flies.
- FlyWire: MB-MP1 = PPL101 (as Wang 2026 states [FT]). FlyWire's NPFL1-I cites Krashes 2009 for its immunostaining. Whether all six cells are PPL101 is UNVERIFIED; FlyWire has 2.
- Implementation: PPL101 bias −α·c_NPF, with c_NPF from NPFL1-I (M2+M3a).

**Inagaki HK et al. (2012).** Visualizing neuromodulation in vivo: TANGO-mapping of dopamine signaling reveals appetite control of sugar sensing. *Cell* 148:583–595. https://doi.org/10.1016/j.cell.2011.12.022 [AB]
- Mechanism: in hunger, dopamine onto sugar GRNs enhances their sugar-evoked Ca²⁺. Sensory neurons are "an important locus for state-dependent gain control".
- Implementation: M3c gain on LB3b/c output edges, 1 + β·S_H. The DA source type is UNVERIFIED, so this is phenomenological.

**Inagaki HK, Panse KM, Anderson DJ (2014).** Independent, reciprocal neuromodulatory control of sweet and bitter taste sensitivity during starvation in Drosophila. *Neuron* 84:806–820. https://doi.org/10.1016/j.neuron.2014.09.032 [AB]
- Mechanism: separate cascades raise sweet sensitivity and lower bitter sensitivity. They are "recruited at increasing hunger levels": "low-risk changes (higher sugar sensitivity) precede high-risk changes". S50/B50 values are UNVERIFIED.
- Implementation: G_sugar = 1 + β_s·S_H and G_bitter = 1 − β_b·max(0, S_H − θ), tested on MN9 dose–response curves.

**LeDue EE, Mann K, Koch E, Chu B, Dakin R, Gordon MD (2016).** Starvation-induced depotentiation of bitter taste in Drosophila. *Curr Biol* 26:2854–2861. https://doi.org/10.1016/j.cub.2016.08.028 [AB]
- Mechanism: OA-VL neurons lie next to bitter GRN terminals. OA and TA potentiate bitter GRNs, and OA-VL tonic firing falls with starvation.
- The OA-VL FlyWire type is UNVERIFIED.

**Tsao CH, Chen CC, Lin CH, Yang HY, Lin S (2018).** Drosophila mushroom bodies integrate hunger and satiety signals to control innate food-seeking behavior. *eLife* 7:e35264. https://doi.org/10.7554/eLife.35264 [FT]
- Mechanism: yeast seeking "increases linearly with the duration of starvation". It needs five MBONs.
  - Starvation potentiates yeast-odour responses of MBON-γ1pedc>αβ and α3.
  - It depresses γ2α′1, β2β′2a and α′2.
  - KC responses are unchanged, so the site is KC→MBON.
- Six DANs mediate it: PPL1-γ1pedc, -γ2α′1, -α′2α2, -α3, PAM-β′2a and PAM-β2β′2a. RNAi in them of NPFR, sNPFR, dInR, DAR1, 5-HT2A or 5-HT1B changes seeking.
- FlyWire: the repo's DAN→MBON compartment recovery confirms MBON11/12/14 and PPL101/103/106. MBON02/13 follow hemibrain numbering (UNVERIFIED here).
- Implementation: set the six DANs' tonic drive from S_H (M3a) and keep the existing plasticity rule.

**Perisse E, Owald D, Barnstedt O, Talbot CB, Huetteroth W, Waddell S (2016).** Aversive learning and appetitive motivation toggle feed-forward inhibition in the Drosophila mushroom body. *Neuron* 90:1086–1099. https://doi.org/10.1016/j.neuron.2016.04.034 [AB]
- Mechanism: odour responses of MVP2 (MBON-γ1pedc>α/β) are elevated in hungry flies. Imposing MVP2 activity makes fed flies express appetitive memory.
- FlyWire: MBON11, GABA; top output APL (1095 synapses).

**Wang J et al. (2026).** Hunger states modulate aggression via opposing dopamine pathways. *Nat Commun* 17:10353. https://doi.org/10.1038/s41467-026-76608-y [FT]
- Mechanism: MBON11 is a graded, nutrient-specific "hangry neuron". PPL101 responds more in satiety and MBON11 more in hunger. Dop1R1 (cAMP) and Dop1R2 (Ca²⁺/Gαq) modulate MBON11 excitability in opposite directions; falling dopamine raises it.
- Implementation: the cleanest M2/M3a pair, c_DA from PPL101 and MBON11 bias −α·c_DA. With plasticity on, SUPERFLY already removes PPL101's fast synapses onto MBON11, so this field becomes their only link, which is more biological.

**Sayin S et al. (2019).** A neural circuit arbitrates between persistence and withdrawal in hungry Drosophila. *Neuron* 104:544–558. https://doi.org/10.1016/j.neuron.2019.07.028 [FT]
- Mechanism: hungry flies keep tracking a food odour over unrewarded trials and "increase their effort", via MBON-γ1pedc>α/β, MBON-α2sc and Dop1R2. OA-VPM4 synapses on MBON11's γ1 dendrites (≈50 % more synapses than VPM3), brakes tracking and promotes feeding.
- [LOCAL], matching the paper: VPM4→MBON11 47 synapses, VPM3→MBON11 22, MBON11→VPM3 23, MBON11→VPM4 0.
- Gap: the sign-only model makes VPM4 excitatory, so the brake is missing. Add it as a negative MBON11 bias from c_OA(VPM4).

**Root CM, Ko KI, Jafari A, Wang JW (2011).** Presynaptic facilitation by neuropeptide signaling mediates odor-driven food search. *Cell* 145:133–144. https://doi.org/10.1016/j.cell.2011.02.008 [AB]

**Ko KI et al. (2015).** Starvation promotes concerted modulation of appetitive olfactory behavior via parallel neuromodulatory circuits. *eLife* 4:e08298. https://doi.org/10.7554/eLife.08298 [AB]
- Mechanism: starvation raises sNPFR1 transcription in Or42b ORNs, and insulin suppresses it. sNPF sensitises an attraction glomerulus; tachykinin suppresses an aversion glomerulus.
- Implementation: M3c on ORN_DM1 (68) and ORN_DM5 (42). Defer until the AL broadcast is fixed.

**Yang Z et al. (2015)** *PNAS* 112:5219, https://doi.org/10.1073/pnas.1417838112 [AB]; **Yu Y et al. (2016)** *eLife* 5:e15693, https://doi.org/10.7554/eLife.15693 [AB]
- Mechanism: OA is necessary and sufficient for starvation hyperactivity. AKHR sits in a small group of OA neurons, and dInR there opposes it. Their FlyWire identity is UNVERIFIED.

**Satiety sensors and IPCs.**
- Dus M et al. (2015) *Neuron* 87:139, https://doi.org/10.1016/j.neuron.2015.05.032 [AB]: six Dh44 neurosecretory cells are activated by nutritive sugar; FlyWire DH44 (6).
- Oh Y et al. (2019) *Nature* 574:559, https://doi.org/10.1038/s41586-019-1675-4 [AB]: a glucose-sensing pair drives IPC DILP2 release and inhibits AKH; FlyWire type UNVERIFIED.
- Held M et al. (2025) *eLife* 13:RP99548, https://doi.org/10.7554/eLife.99548 [FT]: IPC receptor profiles are heterogeneous.
  - OA neurons excite 7/10 IPCs, 5-HT neurons inhibit 4/9 and Lk neurons inhibit 11/12; AstA effects are mixed.
  - Only the DAN effects had monosynaptic timing.
- Crocker A et al. (2010) *Neuron* 65:670, https://doi.org/10.1016/j.neuron.2010.01.032 [AB]: wake-promoting OA-ASM acts on IPCs through OAMB and cAMP.
- Implementation: DH44 and IPC biases rise with u_sugar and feed back as negative drive to S_H. Give each IPC its own R_jm.

**Yapici N, Cohn R, Schusterreiter C, Ruta V, Vosshall LB (2016).** A taste circuit that regulates ingestion by integrating food and hunger signals. *Cell* 165:715–729. https://doi.org/10.1016/j.cell.2016.02.061 [AB]
- Mechanism: 12 cholinergic IN1 neurons show a "rapid and persistent increase" after sucrose in fasted flies, which shrinks with each bout. This is a within-meal satiety integrator; the FlyWire type is UNVERIFIED.

**Nakamizo-Dojo M, Ishii K, Yoshino J, Tsuji M, Emoto K (2023).** Descending GABAergic pathway links brain sugar-sensing to peripheral nociceptive gating in Drosophila. *Nat Commun* 14:6515. https://doi.org/10.1038/s41467-023-42202-9 [AB]
- Mechanism: in larvae, glucose plus insulin gives sustained SEZ descending GABAergic activity, which gates nociceptor terminals via GABA-B ("prioritize feeding over escape"). This is a mechanistic trade-off; an adult analogue is UNVERIFIED.

## 4. Thirst

**Jourjine N, Mullaney BC, Mann K, Scott K (2016).** Coupled sensing of hunger and thirst signals balances sugar and water consumption. *Cell* 166:855–866. https://doi.org/10.1016/j.cell.2016.06.046 [AB]
- Mechanism: four neurons, regulated by AKH and osmolality, promote sugar intake and restrict water intake.
- FlyWire: ISN (4; "DILP3; octopamine, acetylcholine").

**Lin S et al. (2014).** Neural correlates of water reward in thirsty Drosophila. *Nat Neurosci* 17:1536–1542. https://doi.org/10.1038/nn.3827 [AB]
- Mechanism: thirst converts water avoidance into seeking. Fewer than 40 DANs in a γ-lobe zone carry water learning; β′ DANs carry naive seeking.

**Senapati B et al. (2019).** A neural mechanism for deprivation state-specific expression of relevant memories in Drosophila. *Nat Neurosci* 22:2029–2039. https://doi.org/10.1038/s41593-019-0515-z [AB]
- Mechanism: two Lk neurons are more active in thirst and in hunger. Lk inhibits two MB DAN types (water memory) and activates others (sugar memory).
- FlyWire: SLP304b is leucokinin per Yurgel 2019; that it is this pair is UNVERIFIED.

**Gera J et al. (2025).** *eLife* 13:RP97043. https://doi.org/10.7554/eLife.97043 [AB]
- Mechanism: ITPa neurons are activated and release ITPa during dehydration. FlyWire: ITP (8).

## 5. Sleep pressure

**Donlea JM et al. (2011)** *Science* 332:1571, https://doi.org/10.1126/science.1202249 [AB]; **Donlea JM, Pimentel D, Miesenböck G (2014)** *Neuron* 81:860, https://doi.org/10.1016/j.neuron.2013.12.013 [AB]
- Findings: dFB activation induces sleep, and 4 h of induced sleep converts massed training into LTM. Sleep loss raises dFB excitability (Cv-c).

**Pimentel D et al. (2016).** Operation of a homeostatic sleep switch. *Nature* 536:333–337. https://doi.org/10.1038/nature19055 [AB]
- Mechanism: dopamine hyperpolarises dFB neurons "within tens of milliseconds" and suppresses their excitability "within minutes", via Dop1R2 and K⁺ conductances (Shaker/Shab down, Sandman up).

**Donlea JM et al. (2018).** Recurrent circuitry for balancing sleep need and sleep. *Neuron* 97:378–389. https://doi.org/10.1016/j.neuron.2017.12.016 [AB]
- Mechanism: the dFB inhibits "helicon cells" via AstA. Helicon cells excite R2 (now R5), whose plasticity signals sleep pressure back.
- [LOCAL]: FB6A→ExR1 189, ExR1→ER5 1658, ER5→ExR1 4974, ER5→ER5 5881 synapses.
- Caveats: helicon = ExR1 is UNVERIFIED here. FlyWire annotates FB6A with allatostatin-C/Nplp1, not AstA.

**Liu S, Liu Q, Tabuchi M, Wu MN (2016).** Sleep drive is encoded by neural plastic changes in a dedicated circuit. *Cell* 165:1347–1360. https://doi.org/10.1016/j.cell.2016.04.013 [AB]
- Mechanism: with sleep loss, these EB neurons switch from spiking to bursting. Ca²⁺, NMDAR expression and synaptic markers rise reversibly, and the plasticity is "both necessary and sufficient" for sleep drive.
- Naming: the paper calls them R2; they are R5 since Omoto 2018 [WS]. FlyWire ER5 (21; GABA + Dh31).
- Implementation: M4 on ER5→ER5 and ER5→ExR1.

**Raccuglia D et al. (2019).** *Curr Biol* 29:3611–3621. https://doi.org/10.1016/j.cub.2019.08.070 [AB]
- Findings: R5 delta (0.5–4 Hz) power rises with sleep need through NMDAR synchronisation and gates light-induced waking. Measurable in silico.

**De J, Wu M, Lambatan V, Hua Y, Joiner WJ (2023)** *Curr Biol* 33:3660, https://doi.org/10.1016/j.cub.2023.07.043 [AB]; **Jones JD et al. (2023)** *PLoS Biol* 21:e3002012, https://doi.org/10.1371/journal.pbio.3002012 [AB]; **Jones JD et al. (2025)** *PLoS Biol* 23:e3003014, https://doi.org/10.1371/journal.pbio.3003014 [FT]
- The dispute: De et al. find no effect "unambiguously mapped to the dFB". Jones et al. find that 23E10 includes 2 sleep-promoting cholinergic VNC neurons, but that a dFB-specific split (23E10∩84C10) activated at ≥10 Hz still increases sleep. Its neurons are ChAT, VGlut or both.
- Verdict: contested. Model R5 first.
- Related: Cuddapah 2025 (*Nat Commun* 16:6967, https://doi.org/10.1038/s41467-025-62311-x) and Hsu 2025 (*Curr Biol* 35:3496, https://doi.org/10.1016/j.cub.2025.06.003) [AB] report sleep-promoting circuits hyperactive after sleep loss (5HT1A) and circadian dFSB Ca²⁺.

## 6. Arousal and temperature

**Lebestky T et al. (2009)** *Neuron* 64:522, https://doi.org/10.1016/j.neuron.2009.09.031 [AB]
- Finding: startle arousal and sleep–wake arousal are oppositely regulated by DopR; the startle form maps to the CX.

**Maimon G, Straw AD, Dickinson MH (2010)** *Nat Neurosci* 13:393, https://doi.org/10.1038/nn.2492 [AB]; **Suver MP, Mamiya A, Dickinson MH (2012)** *Curr Biol* 22:2294, https://doi.org/10.1016/j.cub.2012.10.034 [AB]
- Mechanism: VS-cell peak-to-peak responses double in flight, and OA neurons are necessary and sufficient for the boost.
- FlyWire: VS1–8, HSE/HSN/HSS. Implementation: M3b.

**Pooryasin A, Fiala A (2015)** *J Neurosci* 35:12792, https://doi.org/10.1523/JNEUROSCI.1638-15.2015 [AB]
- Finding: 5-HT neurons of the posterior medial protocerebrum give behaviour-specific quiescence, not a global one.

**Reyes 2026** (*iScience* 29:115564, https://doi.org/10.1016/j.isci.2026.115564), **Palacios Castillo 2026** (*Curr Biol* 36:4485, https://doi.org/10.1016/j.cub.2026.07.076), **Babski 2024** (*Heliyon* 10:e29952, https://doi.org/10.1016/j.heliyon.2024.e29952) [AB]
- Findings: OA-VPM3 suppresses sleep; split lines reach nearly all long-range OA/TA types and are matched to EM; OA descending neurons are recruited individually.
- Implication: model OA per cell type, not as one scalar.

**Hamada FN et al. (2008)** *Nature* 454:217, https://doi.org/10.1038/nature07001 [AB]
- Finding: warmth-activated AC neurons (dTrpA1) fire just above the preferred temperature. FlyWire "AC neuron" (4). This is a homeostatic drive, low priority.

## 7. Emotion-like persistent states

**Anderson DJ, Adolphs R (2014).** A framework for studying emotions across species. *Cell* 157:187–200. https://doi.org/10.1016/j.cell.2014.03.003 [AB]
- The framework: emotion states are internal states with "emotion primitives", which Gibson 2015 lists as "scalability, persistence, valence, and generalization to multiple contexts". These are SUPERFLY's acceptance tests.

**Gibson WT et al. (2015).** Behavioral responses to a repetitive visual threat stimulus express a persistent state of defensive arousal in Drosophila. *Curr Biol* 25:1401–1415. https://doi.org/10.1016/j.cub.2015.03.058 [AB]
- Findings: repeated shadows give graded, persistent increases in speed and hopping and disperse flies from food. The delay before return grows with the number of shadows; it is modelled "as a leaky integrator of stimulus exposure". τ is UNVERIFIED.
- Implementation: M1 over DNp01/LC4/LPLC2, with a negative bias on the feeding path to MN9 (CB0701).
- Related: Xi W, Chen W (2025, preprint), https://doi.org/10.64898/2025.12.14.694122 [AB]. Their FlyWire v783 LIF has escape suppressing feeding via DNge031/CB0565; both types exist here.

**Hoopfer ED, Jung Y, Inagaki HK, Rubin GM, Anderson DJ (2015).** P1 interneurons promote a persistent internal state that enhances inter-male aggression in Drosophila. *eLife* 4:e11346. https://doi.org/10.7554/eLife.11346 [FT]
- Finding: after 8–10 P1 neurons are activated, aggression appears "even when removal of the barrier was delayed for 10 min", after singing has decayed. Male-only, so MaleCNS is needed.

**Jung Y et al. (2020).** Neurons that function within an integrator to promote a persistent behavioral state in Drosophila. *Neuron* 105:322–333. https://doi.org/10.1016/j.neuron.2019.10.028 [AB]
- Finding: P1 is not persistently active; pCd is, and is necessary but not sufficient. τ and the FlyWire pCd type are UNVERIFIED.

**Deutsch D et al. (2020).** The neural basis for a persistent internal state in Drosophila females. *eLife* 9:e59502. https://doi.org/10.7554/eLife.59502 [FT]
- Findings: 5-min (and 2-min) pC1d/e activation gives persistent activity in more than 30 % of flies. After a 3-min delay, chasing and shoving persist for up to 30 min. EM shows strong pC1d↔aIPg recurrence.
- FlyWire: aIPg1–4 = CB2131, CB1127/CB2204, CB2258 and CB1877 (via hemibrain_type). [LOCAL]: pC1d→CB2131 377 synapses, CB2131→pC1d 179.

**Chiu H et al. (2025).** Cell type-specific contributions to a persistent aggressive internal state in female Drosophila. *eLife* 12:RP88598. https://doi.org/10.7554/eLife.88598 [FT]
- Findings: 30 s of aIPg stimulation gives aggression lasting more than 10 min. Neither aIPg nor pC1d is persistently active, silencing pC1d does not reduce aIPg-evoked persistence, and pC1d + pC1e synergise.
- Implication: test whether native LIF recurrence alone persists; if not, add an M1 integrator in aIPg/pC1e.

**Thornquist SC, Pitsch MJ, Auth CS, Crickmore MA (2021)** *Mol Cell* 81:675, https://doi.org/10.1016/j.molcel.2020.12.029 [AB]
- Mechanism: PKA accumulates evidence over "minutes to hours" until a synchronised network "eruption". This is a template for M1 with a threshold.

**Asahina K et al. (2014)** *Cell* 156:221, https://doi.org/10.1016/j.cell.2013.11.045 [AB]: male FruM⁺ Tk neurons raise aggression via Tk/Takr86C.

**Hindmarsh Sten T, Li R, Otopalik A, Ruta V (2021)** *Nature* 595:549, https://doi.org/10.1038/s41586-021-03714-w [AB]: P1 arousal raises and continuously tunes LC10a gain. This is the clearest case of a state colouring perception (M3b on LC10a); FlyWire has LC10a (234) but no P1.

**Khuong TM et al. (2019).** Nerve injury drives a heightened state of vigilance and neuropathic sensitization in Drosophila. *Sci Adv* 5:eaaw4099. https://doi.org/10.1126/sciadv.aaw4099 [FT]
- Findings: after leg amputation, escape occurs at 38 °C against ≥42 °C uninjured. The effect is absent on days 1–2, appears on day 5, peaks on day 7 and persists past 21 days. About 40 % of VNC GABAergic nuclei are lost, and ppk⁺ GABA-B-R2 knockdown alone gives allodynia.
- VNC-located, so not implementable on FlyWire.

**Negative and positive affect-like states.**
- Yang Z et al. (2013) *Curr Biol* 23:799, https://doi.org/10.1016/j.cub.2013.03.054 [AB]: learned helplessness; needs a closed-loop body.
- Ries AS et al. (2017) *Nat Commun* 8:15738, https://doi.org/10.1038/ncomms15738 [AB]: 3-day vibration stress lowers MB 5-HT release; 5-HT-1A (α/β) mediates relief and 5-HT-1B (γ) inactivity. The 5-HT source is UNVERIFIED.
- Deakin A et al. (2018) *Biol Lett* 14:20170779, https://doi.org/10.1098/rsbl.2017.0779 [AB]: shaken flies approach an ambiguous odour blend less.
- Mohammad F et al. (2016) *Curr Biol* 26:981, https://doi.org/10.1016/j.cub.2016.02.031 [AB]: anxiety-like centrophobism.
- Solvi C et al. (2016) *Science* 353:1529, https://doi.org/10.1126/science.aaf4454 [AB]: bumblebees, dopamine-dependent positive state after unexpected sucrose.
- Galpayage Dona HS et al. (2022) *Anim Behav*, https://doi.org/10.1016/j.anbehav.2022.08.013 [WS]: bumblebee ball rolling.
- Baciadonna L et al. (2026) *Phil Trans B* 381:20250146, https://doi.org/10.1098/rstb.2025.0146 [AB]: review.
- Of these, only the judgement-bias test is implementable without a body, once word memory beats its controls.

## 8. Mechanisms and connectome models, 2024–2026

- **Premise.** Marder 2012 (*Neuron* 76:1, https://doi.org/10.1016/j.neuron.2012.09.010) and Bargmann 2012 (*BioEssays* 34:458, https://doi.org/10.1002/bies.201100185) [AB]: modulators "reconfigure neuronal circuits", and each wiring diagram "encodes multiple circuits, some of which are active and some of which are latent". States select latent circuits.
- **MB dopamine.**
  - Papers: Cohn 2015 (*Cell* 163:1742, https://doi.org/10.1016/j.cell.2015.11.019), Siju 2020 (*Curr Biol* 30:2104, https://doi.org/10.1016/j.cub.2020.04.037), Zolin 2021 (*Nat Neurosci* 24:1555, https://doi.org/10.1038/s41593-021-00929-y) and Hige 2015 (*Neuron* 88:985, https://doi.org/10.1016/j.neuron.2015.11.003) [AB].
  - Findings: DANs encode internal state, movement and valence per compartment, and gate KC→MBON plasticity per compartment.
  - Implication: states enter as DAN tone into SUPERFLY's existing rule.
- **Receptor and peptide maps for R_jm.** Kondo 2020 (*Cell Rep* 30:284, https://doi.org/10.1016/j.celrep.2019.12.018; 75 receptors), Deng 2019 (*Neuron* 101:876, https://doi.org/10.1016/j.neuron.2019.01.045), Nässel & Zandawala 2019 (*Prog Neurobiol* 179:101607, https://doi.org/10.1016/j.pneurobio.2019.02.003) and Beets & Watteyne 2025 (*Curr Opin Neurobiol* 92:103027, https://doi.org/10.1016/j.conb.2025.103027) [AB]. Mapping these to FlyWire types is not yet done.
- **FlyWire peptidergic templates.**
  - Reinhard N et al. 2024 (*Nat Commun* 15:10392, https://doi.org/10.1038/s41467-024-54694-0) and McKim TH et al. 2024 (preprint, https://doi.org/10.1101/2024.08.28.609616) [AB].
  - They find sparse monosynaptic links to neurosecretory cells and add transcriptomic "putative paracrine peptidergic" maps. McKim's linear model ranks enteric neurons as the strongest NSC influence.
- **Endres MN et al. (2026, preprint)**, https://doi.org/10.64898/2026.09.08.750118 [AB].
  - Findings: four connectome-defined NPF cell types have separate networks and partly opposing functions, and population-level knockdown hides them.
  - Implication: one field per source type.
- **Salman F et al. (2026)** *J Neurophysiol* 136:77, https://doi.org/10.1152/jn.00571.2025 [AB].
  - Findings: 5-HT differentially changes the excitability of three AL LN types; a rate model predicts a wider PN dynamic range and less noise.
  - With CSD→LN [LOCAL], this is an UNTESTED, biologically grounded lever on SUPERFLY's AL broadcast.
- **Whole-brain models.**
  - Papers: Shiu 2024 (*Nature* 634:210, https://doi.org/10.1038/s41586-024-07763-9) [FT], Pospisil 2024 (*Nature* 634:201, https://doi.org/10.1038/s41586-024-07982-0) [AB], Beiran & Litwin-Kumar 2025 (*Nat Neurosci* 28:2561, https://doi.org/10.1038/s41593-025-02080-4) [AB].
  - Key result: a connectome "often does not substantially constrain" dynamics when biophysics is unknown, but recording a few neurons removes the degeneracy. Modulation changes exactly those parameters, so fit each state to one recorded neuron.
- **BANC.** Bates AS et al. (2026) *Nature* 656:957, https://doi.org/10.1038/s41586-026-10735-w [AB]: endocrine and visceral effectors sit in local loops, supervised by the brain. It is the substrate for VNC states.
- **Gap.** flybench's adaptive LIF is spike-frequency adaptation, not a state. The "State of Brain Emulation Report 2025" (arXiv 2510.15745) was seen in search only [WS]. **No 2024–2026 work was found that adds neuromodulatory fields to a FlyWire whole-brain spiking model.**

## 9. Sentience frameworks → operational markers

**Barron AB, Klein C (2016).** What insects can tell us about the origins of consciousness. *PNAS* 113:4900–4908. https://doi.org/10.1073/pnas.1520084113 [AB]
- Claim: an "integrated and egocentric representation" of the animal in space, done by midbrain-analogous insect structures, is sufficient for subjective experience.
- Design consequence: states should act on the CX and MB, not only on reflex arcs.

**Birch J, Schnell AK, Clayton NS (2020).** Dimensions of animal consciousness. *Trends Cogn Sci* 24:789–801. https://doi.org/10.1016/j.tics.2020.07.007 [AB]
- Five dimensions: perceptual richness, evaluative richness, integration at a time, integration across time, self-consciousness.

**Gibbons M, Crump A, Barrett M, Sarlak S, Birch J, Chittka L (2022).** Can insects feel pain? A review of the neural and behavioural evidence. *Adv Insect Physiol* 63:155–229. https://doi.org/10.1016/bs.aiip.2022.10.001; PDF read: https://chittkalab.sbcs.qmul.ac.uk/2022/Gibbons%20et%20al%202022%20Advances%20Insect%20Physiol.pdf [FT]
- Adult Diptera meet six of the eight Birch 2021 criteria:
  - nociception VH, sensory integration VH, integrated nociception VH, analgesia VH, motivational trade-offs H, associative learning VH;
  - flexible self-protection and analgesia preference: "VL (no research found)".
- The fly trade-off is Kaun KR et al. (2011), *Nat Neurosci* 14:612, https://doi.org/10.1038/nn.2805 [AB]. Conditioned flies crossed a 100 V barrier for ethanol- or sucrose-paired odours; only ethanol-conditioned flies crossed 120 V.

**Gibbons M, Versace E, Crump A, Baran B, Chittka L (2022).** *PNAS* 119:e2205821119. https://doi.org/10.1073/pnas.2205821119 [AB]
- Finding: bees trade 55 °C feeder heat against sucrose concentration using learned colour cues, so the trade-off is central.

**Birch J (2024).** *The Edge of Sentience*. OUP, open access. https://doi.org/10.1093/9780191966729.001.0001 [AB via Crossref chapter abstracts]
- "Pushing the Boundaries" (https://doi.org/10.1093/9780191966729.003.0014): insects have a midbrain-like control unit, working memory and associative learning, so their sentience must be taken seriously.
- "Large Language Models and the Gaming Problem" (https://doi.org/10.1093/9780191966729.003.0017): such systems "mimic human behaviours that are likely to persuade human users of their sentience"; what is needed are "deep computational markers, not behavioural markers".
- New York Declaration on Animal Consciousness (2024) [WS]: a "realistic possibility" of consciousness in insects; exact wording UNVERIFIED.

**Operational markers** (computed from fly neurons and motor output, never from the voice):

| marker | source | in-silico test |
|---|---|---|
| scalability | Gibson 2015 | state amplitude monotone in inducer count/intensity |
| persistence | Gibson, Hoopfer, Deutsch | outlasts stimulus; fitted decay τ |
| valence | Anderson & Adolphs | sign of approach/avoid bias (MBON valence, MN9) |
| generalisation | Gibson | state from channel A changes channel B (loom → sugar PER) |
| motivational trade-off | Gibbons 2022; Kaun 2011 | bitter-for-sugar acceptance curve shifts with hunger |
| learned valuation | Gibbons PNAS 2022 | trade-off computed on learned cues (MB memory) |
| integration across time | Birch 2020 | response depends on history, not only current input |
| central locus | Barron & Klein | state acts through CX/MB |

---

## 10. Recommended design: internal states for SUPERFLY

Ordered by strength of evidence × ease. Every state:
- is an opt-in, named ticker that is silent at baseline, so identity holds;
- writes only to the fly's own neurons (M2/M3, or M4 for sleep);
- is fitted to at least one recorded neuron;
- is read by the voice from the modulated neurons, never from S.

| # | state | drive (fly neurons + physiology u) | modulates (M3) | τ (fly) | measured by | voice may say |
|---|---|---|---|---|---|---|
| 1 | **Hunger S_H** | u = time since sugar ingestion; negative drive from LB3b/c ingestion activity and DH44/IPC bias | PPL101 bias −α·S_H; MBON11 bias +α·S_H (Wang 2026); Tsao's six DANs' tone; sugar-GRN gain 1+β·S_H; bitter-GRN gain 1−β·max(0, S_H−θ); NPF field NPFL1-I→PPL101 | hours (clamp; fit to Tsao's linear rise) | MN9 S50 shift; B50 shift only above θ; MBON11 ↑ / PPL101 ↓; later appetitive-memory gating (Krashes) | "I'm (a little / very) hungry" |
| 2 | **Defensive arousal S_F** | M1 over DNp01/LC4/LPLC2 activity | bias −α·S_F on the DNge031/CB0565 feeding path and MN9 drive; gain + on DNp01 inputs; OA-VPM/VUM tone + | seconds–minutes (UNVERIFIED; fit to Gibson) | four primitives: scalability with loom count, decay τ, sugar-PER suppression, recovery latency growing with count | "I'm startled / on edge" |
| 3 | **Thirst S_W** | u = dehydration; ISN, ITP bias | ISN bias +S_H−S_W; water-GRN gain +, sugar-GRN gain − (Jourjine); SLP304b-gated DAN tone (Senapati) | hours (clamp) | water→MN9 ↑ and sugar ↓, opposite to hunger | "I'm thirsty" |
| 4 | **Locomotor/visual arousal S_A** | OA-VPM3/4, OA-AL2, OA-VUM rates (M2, per type) | gain 1+β·c_OA on VS1–8/HS inputs (target ×2, Maimon 2010); OA field on IPCs (Crocker, Held) | sub-second onset, minutes | VS/HS active/quiet gain ≈ 2 (needs visual input; flyvis not yet integrated) | "I'm alert" |
| 5 | **Sleep pressure S_P** | M4 on ER5→ER5 and ER5→ExR1 during wake | ER5/ExR1 strength; dFB (FB6A…) bias +; DA field −dFB (Pimentel); sensory gate (Raccuglia) | hours | ER5 0.5–4 Hz power ↑ with wake; less visual→CX transmission | "I'm sleepy" (dFB part labelled contested) |
| 6 | **Social arousal / aggression S_Q** (female) | pC1d/e; aIPg (CB2131, CB1127, CB2204, CB2258, CB1877) | first test native recurrence; else M1 integrator in aIPg/pC1e (Chiu 2025, Thornquist 2021) | ≥10–30 min | persistent activity after 30 s–5 min drive (Deutsch, Chiu) | "I'm worked up" (no behaviour test without a second fly) |
| 7 | **Stress / negative affect S_N** (exploratory) | chronic aversive input (PPL1 teaching) integrated | 5-HT field on KC α/β (5-HT1A) and γ (5-HT1B) (Ries 2017) | days (clamp) | judgement bias on an ambiguous word blend (Deakin 2018), once word memory beats controls | only after the marker passes |
| 8 | **Injury sensitisation** | — | needs VNC GABA loss (Khuong) | days–weeks | — | **deferred** to BANC; see ethics |

**Why this order.**
- **Hunger.** Every ingredient is a verified FlyWire type, and three readouts already work: sugar/bitter→MN9 (flybench core passes), MBON11 and PPL101. Fit the MBON11 bias together with MBON tone, because sparse KC codes do not fire MBONs without it (findings §6).
- **Defensive arousal.** The loom pathway is specific in this model (loom→giant fibre 186 Hz), and all four primitives are measurable without a body. Call it "defensive arousal", not fear.
- **Fast synapses.** For every source given a field, report the effect both with and without its Shiu fast synapses. Keeping them preserves identity; removing them (opt-in, as SUPERFLY already does for DAN→KC/MBON) is closer to biology for amines and peptides.

**Language layer.**
- Add the modulated central neurons to the voice's neural tokens: MBON11, PPL101, Tsao's DANs, ISN, ER5/ExR1, DNp01-history and VS/HS. The voice never sees S or u. Words come in three graded bins.
- Controls, mirroring the existing fly-mind tests:
  1. Baseline fly: no state words.
  2. Clamp-and-ablate: S high but M3 outputs off. The voice must not report the state, which proves it reads the fly.
  3. Shuffled state: the report follows the brain it is given.
  4. Concordance: "hungry" correlates with the measured MN9 S50 shift.
- Per Birch 2024, the voice's words are never evidence of feeling. Only the §9 markers count.

**Ethics.** Birch 2024 treats insects as sentience candidates, and Gibbons 2022 finds that adult Diptera meet six of eight pain criteria. States 7 and 8 deliberately engineer persistent negative states. They should stay off by default and be run only with a stated purpose and a log.

## 11. Not verified

- Fitted τ in Gibson 2015 and Jung 2020; Inagaki 2014 S50/B50 values.
- FlyWire identities of TH-VUM, OA-VL, IN1, CN, AKHR⁺ OA neurons and pCd.
- helicon = ExR1; SLP304b = the Senapati Lk pair; whether all of Krashes' six DANs are PPL101; the MB 5-HT source in Ries 2017.
- Full texts of Barron & Klein 2016 and Birch 2024 (only abstracts read).
- The New York Declaration text and the arXiv reports (search only).
