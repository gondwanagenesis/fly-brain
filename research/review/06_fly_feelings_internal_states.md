# 06 · Fly feelings: internal states for SUPERFLY

Literature review made 2026-10-08 for SUPERFLY (`SUPERFLY.md`). The question:
which persistent internal states, ones that change what the fly perceives and does,
can be added to the 138,639-neuron FlyWire v783 Shiu-LIF fly using its own cell
types, so that the language layer has something real to report?

## Executive summary

1. The base model has no internal state. Shiu et al. 2024 state that they "do not account for neuropeptides or neuromodulation". Neurons predicted to be dopaminergic, octopaminergic or serotonergic are treated as fast excitatory. In this repo's v783 graph those neurons carry 3.6 % of synapses.
2. Neuromodulation has to be modelled off-synapse. NPFL1-I → PPL101 has **0** synapses, although Krashes 2009 needs the dNPF receptor in those DANs. OA-ASM → IPC also has **0**, although Crocker 2010 maps octopamine's action to the IPCs. The 18 IPCs make only 157 output synapses in total.
3. Hunger is the best-supported state and the easiest to build. It hubs on PPL101 (= PPL1-γ1pedc = MB-MP1) and MBON11 (= MBON-γ1pedc>α/β = MVP2), which are reciprocally wired (894/358 synapses). Six papers (2009–2026) agree on the direction: satiety drives PPL101, and hunger raises MBON11's excitability.
4. Hunger also re-weights the senses: sugar GRN sensitivity goes up first, bitter sensitivity goes down later (Inagaki 2012/2014, LeDue 2016). Odour channels are re-weighted too (Root 2011, Ko 2015), but the model's antennal lobe is bistable, so the olfactory part is not credible yet.
5. A "fear-like" defensive arousal is behaviourally a leaky integrator of repeated threats (Gibson 2015). The model's own loom pathway (LC4/LPLC2 → DNp01: 805/1080 synapses) can drive it. The mechanism is not known, so this state is phenomenological.
6. Persistent social or aggressive states are real, lasting 10–30 min, but their mechanism is contested. pC1d/e↔aIPg recurrence (Deutsch 2020) does not explain aIPg-evoked persistence (Chiu 2025). Slow biochemical integrators (Thornquist 2021) are a credible alternative.
7. Sleep pressure has a mechanistic candidate: R5 (FlyWire ER5) plasticity, linked to ExR1 (4974/1658 synapses) and the dFB. The dFB's role has been disputed since 2023 (De 2023 vs Jones 2023/2025).
8. Chronic-pain-like sensitisation (Khuong 2019) lives in the ventral nerve cord, which FlyWire lacks. It is deferred to BANC, and also flagged on ethical grounds.
9. The sentience literature gives measurable, non-verbal markers: scalability, persistence, valence, generalisation, motivational trade-offs and flexible valuation through learning. Birch 2024 warns that LLM self-reports are "gamed" markers, so the voice's report cannot count as evidence.
10. Recommendation: build states as opt-in "fields" that are silent at baseline, so the untouched fly stays bit-identical. Fields have source neurons, receptor-weighted bias or gain on named targets, and stated time constants. The voice reads the modulated neurons, never the state variable, and each report must pass ablation controls.

---

## 0. How sources were read (provenance)

In this session, WebFetch was blocked by the egress proxy for PMC, eLife, Nature, PubMed, bioRxiv and arXiv. Sources were therefore read through other routes:
- **[FT]**: full text read, either open-access XML from the Europe PMC REST API (`ebi.ac.uk/europepmc/webservices/rest/<PMCID>/fullTextXML`) or, for Gibbons 2022, a PDF from the Chittka lab site.
- **[AB]**: abstract and bibliographic record from the Europe PMC search API or Crossref.
- **[WS]**: seen only in web-search summaries, not read.
- **[LOCAL]**: computed here from `data/2025_Connectivity_783.parquet` (min count 1) and `data/flywire_meta/neuron_annotations.tsv` (Schlegel et al. 2024 annotations; the `known_nt` and `known_nt_source` columns give peptide identities).

Each URL is the DOI from the API record. Numbers not stated in a source I read are marked **UNVERIFIED**.

## 1. What the model has now ([LOCAL] unless cited)

- Signs, as Shiu et al. 2024 wrote them [FT]: "Neurons predicted to be dopaminergic, octopaminergic or serotonergic are assigned to the excitatory category." "We do not account for neuropeptides or neuromodulation." In this repo, DA, OA and 5-HT presynaptic neurons carry 1.38 M, 0.14 M and 0.44 M synapses respectively: 3.6 % of the 54.5 M synapses whose source has a top_nt.
- Peptidergic cells are present, but they are modelled only through their predicted fast transmitter. This applies to IPC (18, DILP2/3/5), DH44 (6), CRZ (6+4), ITP (8), DH31 (6), DMS (6), Hugin-RG (4), NPFL1-I (2, dNPF), AstA1 (2), DSKMP3 (4), ISN (4, DILP3 plus octopamine) and SLP304b (2, leucokinin, source Yurgel et al. 2019). Predicted NTs for these cells (octopamine for 4/6 DH44, for example) are classifier outputs, not biology.
- Modulatory sources are present: CSD (2, 5-HT; top targets are AL LNs, e.g. lLN2F_b 1168 synapses), DPM (2), OA-VPM3/4, OA-VUMa1–8, OA-AL2b/i, OA-ASM1–3, PAM01–15, PPL101–108, PPM, PAL, l-LNv/s-LNv (PDF), and the 5-HTPMPD/PMPV/PLP types.
- Female brain: pC1a–e are present; P1 and other male-specific types are not.
- The engine already gives the hooks needed: a global `set_gain`, a sensory port (`set_rates`), a synaptic port (`deliver(slots, values)`, with the 1.8 ms ring and refractory gating), and `add_emitter` / `add_ticker`. `MBPlasticity` shows how a per-edge multiplicative change is emitted as a delta `(f−1)·w0`.

## 2. Toolkit: neuromodulation in a sign-only LIF

**M0, the base (Shiu 2024; repo `MODEL_PARAMS`).** For each neuron i:

    τm dvi/dt = gi − (vi − Vrest)
    τs dgi/dt = −gi
    on a spike of j:  gi(t + 1.8 ms) += 0.275 mV · sign_j · n_ji

with τm = 20 ms, τs = 5 ms, Vrest = Vreset = −52 mV, Vth = −45 mV and t_ref = 2.2 ms.

**M1, internal state (leaky integrator, as in Gibson 2015).** For each state k:

    τk dSk/dt = −(Sk − Sk⁰) + Σp a_kp·r̃p(t) + uk(t)

Here r̃p is the low-pass-filtered rate of a sensor population of the fly's own neurons, and uk is a physiological input that has no neural correlate in the model (for example, hours since the fly last ate). Sk is clipped to [0, Smax].

**M2, volume transmission (a modulator field).** For a source population Pm:

    τc dcm/dt = −cm + (κm/|Pm|) Σ_{i∈Pm} Σ_f δ(t − t_i^f)

so cm is the population rate of the source neurons, low-pass filtered. When no source is identified, cm may be set from Sk directly, and the state is then flagged phenomenological.

**M3, effect on target j, with signed receptor weight R_jm** (taken from literature or receptor maps; never inferred from synapse counts, see §1):

- (a) Bias: deliver x_j = (Δt/τs)·Σm R_jm·αm·cm into g_j every step through the synaptic port. A constant x gives a steady g_j = x·τs/Δt, so the target is depolarised by Σ R·α·c mV. Negative values are allowed.
- (b) Postsynaptic gain: w_ij → w_ij·(1 + Σm R_jm·βm·cm), emitted as `MBPlasticity`-style deltas on the edges into j.
- (c) Presynaptic gain on the edges out of i. This is the form for sNPF/sNPFR1 on ORN terminals (Root 2011), dopamine on sugar-GRN terminals (Inagaki 2012) and OA/TA on bitter-GRN terminals (LeDue 2016).
- (d) Excitability: changing per-neuron adaptation (flybench adaptive LIF, b = 2 mV, τw = 200 ms) or threshold. This needs kernel support, so it is listed as future work.

**M4, slow plastic integrator** (sleep drive, as in Liu 2016):

    dW_e/dt = η·r̃pre·r̃post·1[awake] − (W_e − W_e⁰)/τrec

**Time constants.**
- Amine effects begin within tens of ms and last minutes: dopamine on dFB cells "within tens of milliseconds" and "lasting excitability suppression within minutes" (Pimentel 2016).
- Receptor-expression changes take hours: starvation raises sNPFR1 transcription (Root 2011).
- Behavioural states: aggression lasts ≥10 min (Hoopfer 2015, Chiu 2025), female behaviour up to 30 min (Deutsch 2020), PKA integration minutes to hours (Thornquist 2021), allodynia more than 21 days (Khuong 2019).
- The native kernel runs 2–3× faster than real time (review 05), so minute-scale states can be simulated directly. Hour-scale states should be clamped: set S to the value a fly starved for 24 h would have.

**Identity rule.** At Sk = Sk⁰, every M3 term is zero and nothing is emitted, so `test_identity` stays bit-identical. Each state is named, opt-in, and switchable on its own.

---

## 3. Hunger and satiety

**Krashes MJ, DasGupta S, Vreede A, White B, Armstrong JD, Waddell S (2009).** A neural circuit mechanism integrating motivational state with memory expression in Drosophila. *Cell* 139:416–427. https://doi.org/10.1016/j.cell.2009.08.035 [AB]
- Mechanism: stimulating dNPF neurons mimics hunger. Appetitive memory expression needs NPFR in "six dopaminergic neurons" of the MB. Blocking those neurons releases memory in fed flies; stimulating them suppresses memory in hungry flies.
- FlyWire types: MB-MP1 is PPL101, as Wang 2026 states [FT]. FlyWire NPFL1-I cites Krashes 2009 as its immunostaining source. Whether all six c061 cells are PPL101 is UNVERIFIED (FlyWire has 2 PPL101).
- Implementation: a PPL101 bias of −α·c_NPF, with c_NPF from NPFL1-I firing (M2 + M3a). There are 0 synapses on this route.

**Inagaki HK et al. (2012).** Visualizing neuromodulation in vivo: TANGO-mapping of dopamine signaling reveals appetite control of sugar sensing. *Cell* 148:583–595. https://doi.org/10.1016/j.cell.2011.12.022 [AB]
- Mechanism: in hunger, dopamine is released onto sugar GRNs and enhances sugar-evoked Ca²⁺. The paper frames sensory neurons as "an important locus for state-dependent gain control".
- Implementation: M3c presynaptic gain on the output edges of LB3b/LB3c sugar GRNs, as G = 1 + β·S_H. The dopamine source cell type is UNVERIFIED in this review, so use S_H directly and flag it as phenomenological.

**Inagaki HK, Panse KM, Anderson DJ (2014).** Independent, reciprocal neuromodulatory control of sweet and bitter taste sensitivity during starvation in Drosophila. *Neuron* 84:806–820. https://doi.org/10.1016/j.neuron.2014.09.032 [AB]
- Mechanism: starvation raises sweet sensitivity and lowers bitter sensitivity through separate cascades. The "low-risk changes (higher sugar sensitivity) precede high-risk changes", so the two arms switch on at different hunger levels. Measured as PER S50/B50 against hours of starvation (values UNVERIFIED here).
- Implementation: two gains with staggered thresholds, G_sugar = 1 + β_s·S_H and G_bitter = 1 − β_b·max(0, S_H − θ).
- Test: MN9 dose–response curves.

**LeDue EE, Mann K, Koch E, Chu B, Dakin R, Gordon MD (2016).** Starvation-induced depotentiation of bitter taste in Drosophila. *Curr Biol* 26:2854–2861. https://doi.org/10.1016/j.cub.2016.08.028 [AB]
- Mechanism: OA-VL neurons lie next to bitter GRN terminals. Octopamine and tyramine potentiate bitter GRNs, and OA-VL tonic firing falls with starvation.
- Implementation: give OA-VL a tonic rate that falls with S_H, feeding M3c gain on bitter GRN outputs. The FlyWire type of OA-VL is UNVERIFIED.

**Tsao CH, Chen CC, Lin CH, Yang HY, Lin S (2018).** Drosophila mushroom bodies integrate hunger and satiety signals to control innate food-seeking behavior. *eLife* 7:e35264. https://doi.org/10.7554/eLife.35264 [FT]
- Mechanism: yeast seeking "increases linearly with the duration of starvation". Five MBONs are required: MBON-γ1pedc>αβ, β2β′2a, γ2α′1, α′2 and α3.
- Starvation potentiates yeast-odour responses in γ1pedc>αβ and α3, and depresses them in γ2α′1, β2β′2a and α′2. KC responses do not change, so the site is the KC→MBON synapse.
- Six DANs mediate the effect (PPL1-γ1pedc, PPL1-γ2α′1, PPL1-α′2α2, PPL1-α3, PAM-β′2a, PAM-β2β′2a). RNAi of NPFR, sNPFR, dInR, DAR1, 5-HT2A and 5-HT1B in them changes seeking.
- FlyWire types: MBON11, MBON12 and MBON14 with PPL101, PPL103 and PPL106 are confirmed by the repo's DAN→MBON compartment recovery (`plasticity.py`). MBON02/MBON13 for β2β′2a/α′2 follow hemibrain numbering (UNVERIFIED here).
- Implementation: drive the six DANs' tonic rates from S_H through M3a, using the existing plasticity module. The test is the signed MBON response shift.

**Perisse E, Owald D, Barnstedt O, Talbot CB, Huetteroth W, Waddell S (2016).** Aversive learning and appetitive motivation toggle feed-forward inhibition in the Drosophila mushroom body. *Neuron* 90:1086–1099. https://doi.org/10.1016/j.neuron.2016.04.034 [AB]
- Mechanism: in hungry flies, odour-evoked activity of MVP2 (MBON-γ1pedc>α/β) is higher. Imposing MVP2 activity in fed flies produces appetitive memory expression.
- FlyWire type: MBON11, GABA (top_nt), main output to APL (1095 synapses).
- Implementation: MBON11 bias of +α·S_H.

**Wang J et al. (2026).** Hunger states modulate aggression via opposing dopamine pathways. *Nat Commun* 17:10353. https://doi.org/10.1038/s41467-026-76608-y [FT]
- Mechanism: MBON11 is a "hangry neuron". PPL101 responds more under satiety and MBON11 more under hunger, graded and nutrient-specific. Dopamine acting through Dop1R1 (cAMP) and Dop1R2 (Ca²⁺/Gαq) modulates MBON11 excitability in opposite directions; falling dopamine raises it.
- Implementation: this is the cleanest M2/M3a pair. Set c_DA from PPL101, and give MBON11 a bias of −α·c_DA.
- Caveat: with plasticity on, SUPERFLY removes DAN→MBON fast synapses, so this field is then the only route from PPL101 to MBON11. That is biologically better than fast excitation.

**Sayin S et al. (2019).** A neural circuit arbitrates between persistence and withdrawal in hungry Drosophila. *Neuron* 104:544–558. https://doi.org/10.1016/j.neuron.2019.07.028 [FT]
- Mechanism: hungry flies keep tracking a food odour over unrewarded trials and "increase their effort". MBON-γ1pedc>α/β, MBON-α2sc and Dop1R2 control this persistence. OA-VPM4 synapses on MBON11's γ1 dendrites (with ~50 % more synapses than VPM3), acts as a brake and promotes feeding.
- [LOCAL]: OA-VPM4→MBON11 47 synapses, VPM3→MBON11 22, MBON11→VPM3 23, MBON11→VPM4 0. This matches the paper.
- Implementation: OA-VPM4 already inhibits nothing in a sign-only model (it is signed excitatory). Model its effect as a negative bias on MBON11 driven by c_OA from VPM4.

**Root CM, Ko KI, Jafari A, Wang JW (2011).** Presynaptic facilitation by neuropeptide signaling mediates odor-driven food search. *Cell* 145:133–144. https://doi.org/10.1016/j.cell.2011.02.008 [AB]
- Mechanism: sNPF/sNPFR1 in Or42b ORNs; starvation raises sNPFR1 transcription, and insulin suppresses it.

**Ko KI et al. (2015).** Starvation promotes concerted modulation of appetitive olfactory behavior via parallel neuromodulatory circuits. *eLife* 4:e08298. https://doi.org/10.7554/eLife.08298 [AB]
- Mechanism: sNPF sensitises an attraction glomerulus, and tachykinin suppresses an aversion glomerulus.
- Implementation, for both Root and Ko: M3c on ORN_DM1 (68) and ORN_DM5 (42) outputs. Do not rely on it until the antennal-lobe broadcast is fixed (findings §4).

**Yang Z et al. (2015)** *PNAS* 112:5219–5224, https://doi.org/10.1073/pnas.1417838112 **and Yu Y et al. (2016)** *eLife* 5:e15693, https://doi.org/10.7554/eLife.15693 [AB]
- Mechanism: octopamine is necessary and sufficient for starvation-induced hyperactivity. AKHR (the glucagon analogue's receptor) is expressed in a small group of OA neurons, and dInR in the same cells opposes it.
- Implementation: a locomotor-arousal bias on OA neurons as a function of S_H. The AKHR+ cell identity in FlyWire is UNVERIFIED.

**Dus M et al. (2015).** Nutrient sensor in the brain directs the action of the brain-gut axis in Drosophila. *Neuron* 87:139–151. https://doi.org/10.1016/j.neuron.2015.05.032 [AB]
- Mechanism: six Dh44 neurosecretory cells are activated by nutritive sugars.
- FlyWire: DH44 (6, pars intercerebralis).
- Implementation: these cells are the satiety sensor. Their bias rises with u_sugar, the hemolymph sugar input, and their c_DH44 feeds back into S_H as negative drive.

**Oh Y et al. (2019).** A glucose-sensing neuron pair regulates insulin and glucagon in Drosophila. *Nature* 574:559–564. https://doi.org/10.1038/s41586-019-1675-4 [AB]
- Mechanism: a pair of glucose-sensing neurons triggers DILP2 release from the IPCs and inhibits AKH. The FlyWire type is UNVERIFIED.

**Held M et al. (2025).** Aminergic and peptidergic modulation of insulin-producing cells in Drosophila. *eLife* 13:RP99548. https://doi.org/10.7554/eLife.99548 [FT]
- Mechanism: IPCs express heterogeneous receptors.
  - OA neurons excite 7 of 10 IPCs; 5-HT neurons inhibit 4 of 9.
  - Lk neurons inhibit 11 of 12; AstA neurons excite some IPCs and inhibit others.
  - Only the DAN effects (and one Lk response) had monosynaptic timing.
- Implementation: give each IPC its own receptor vector R_jm, not a population average.

**Crocker A, Shahidullah M, Levitan IB, Sehgal A (2010).** Identification of a neural circuit that underlies the effects of octopamine on sleep:wake behavior. *Neuron* 65:670–681. https://doi.org/10.1016/j.neuron.2010.01.032 [AB]
- Mechanism: OA-ASM is wake-promoting and acts on IPCs through OAMB and cAMP.
- [LOCAL]: OA-ASM1–3 → IPC has 0 synapses, so this has to be an M2 field.

**Yapici N, Cohn R, Schusterreiter C, Ruta V, Vosshall LB (2016).** A taste circuit that regulates ingestion by integrating food and hunger signals. *Cell* 165:715–729. https://doi.org/10.1016/j.cell.2016.02.061 [AB]
- Mechanism: 12 cholinergic IN1 interneurons show a "rapid and persistent increase" in activity after sucrose in fasted flies. The increase is smaller and not persistent in fed flies, and shrinks with each feeding bout.
- This is a within-meal satiety integrator. The FlyWire type of IN1 is UNVERIFIED.

**Nakamizo-Dojo M, Ishii K, Yoshino J, Tsuji M, Emoto K (2023).** Descending GABAergic pathway links brain sugar-sensing to peripheral nociceptive gating in Drosophila. *Nat Commun* 14:6515. https://doi.org/10.1038/s41467-023-42202-9 [AB]
- Mechanism: in larvae, glucose feeding produces sustained activation of SEZ descending GABAergic neurons through insulin. These gate nociceptor terminals via GABA-B, so the animal feeds rather than escapes.
- This is a mechanistic motivational trade-off, but in larvae; whether adults have one is UNVERIFIED.

## 4. Thirst

**Jourjine N, Mullaney BC, Mann K, Scott K (2016).** Coupled sensing of hunger and thirst signals balances sugar and water consumption. *Cell* 166:855–866. https://doi.org/10.1016/j.cell.2016.06.046 [AB]
- Mechanism: four interoceptive neurons, regulated by AKH and by osmolality. Activating them promotes sugar intake and restricts water intake.
- FlyWire: ISN (4), annotated "DILP3; octopamine, acetylcholine".
- Implementation: ISN bias of +α·S_H − α′·S_W. ISN output acts as an opposite-sign gain on the sugar versus water GRN pathways.

**Lin S et al. (2014).** Neural correlates of water reward in thirsty Drosophila. *Nat Neurosci* 17:1536–1542. https://doi.org/10.1038/nn.3827 [AB]
- Mechanism: thirst turns water avoidance into water seeking. Fewer than 40 water-responsive DANs in a γ-lobe zone carry water learning; separate β′ DANs carry naive water seeking.

**Senapati B et al. (2019).** A neural mechanism for deprivation state-specific expression of relevant memories in Drosophila. *Nat Neurosci* 22:2029–2039. https://doi.org/10.1038/s41593-019-0515-z [AB]
- Mechanism: two Lk neurons are more active in both thirst and hunger. Lk inhibits two MB DAN types to allow water memory and activates others for sugar memory.
- FlyWire: SLP304b (2) is annotated leucokinin from Yurgel 2019. That it is the same pair is UNVERIFIED.

**Gera J et al. (2025).** ITP signals via a guanylate cyclase receptor. *eLife* 13:RP97043. https://doi.org/10.7554/eLife.97043 [AB]
- Mechanism: ITPa neurons are activated and release ITPa during dehydration.
- FlyWire: ITP (8, pars lateralis).

## 5. Sleep pressure

**Donlea JM, Thimgan MS, Suzuki Y, Gottschalk L, Shaw PJ (2011).** Inducing sleep by remote control facilitates memory consolidation in Drosophila. *Science* 332:1571–1576. https://doi.org/10.1126/science.1202249 [AB]
- Finding: activating dFB-projecting neurons induces sleep, and 4 h of induced sleep converts massed training into long-term memory.

**Donlea JM, Pimentel D, Miesenböck G (2014).** Neuronal machinery of sleep homeostasis in Drosophila. *Neuron* 81:860–872. https://doi.org/10.1016/j.neuron.2013.12.013 [AB]
- Finding: sleep deprivation raises dFB excitability, and this depends on Cv-c.

**Pimentel D et al. (2016).** Operation of a homeostatic sleep switch. *Nature* 536:333–337. https://doi.org/10.1038/nature19055 [AB]
- Mechanism: dopamine hyperpolarises dFB neurons within tens of ms and suppresses their excitability for minutes, through Dop1R2 and K⁺ conductances (Shaker/Shab down, Sandman up).
- Implementation: M3a or M3d on the dFB, set by dopamine c from the arousing DANs.

**Donlea JM et al. (2018).** Recurrent circuitry for balancing sleep need and sleep. *Neuron* 97:378–389. https://doi.org/10.1016/j.neuron.2017.12.016 [AB]
- Mechanism: the dFB inhibits "helicon cells" via AstA. Helicon cells excite R2 (now R5), whose plasticity signals sleep pressure back to the dFB.
- [LOCAL]: FB6A→ExR1 189 synapses, ExR1→ER5 1658, ER5→ExR1 4974, ER5→ER5 5881.
- Caveats: that the helicon cells are ExR1 is UNVERIFIED in this review. FlyWire annotates FB6A as allatostatin-C/Nplp1, not AstA.

**Liu S, Liu Q, Tabuchi M, Wu MN (2016).** Sleep drive is encoded by neural plastic changes in a dedicated circuit. *Cell* 165:1347–1360. https://doi.org/10.1016/j.cell.2016.04.013 [AB]
- Mechanism: with sleep loss, these EB neurons switch from spiking to bursting. Ca²⁺, NMDAR expression and synaptic-strength markers all rise reversibly, and this plasticity is "both necessary and sufficient" for sleep drive.
- The paper calls them R2. They are now R5 (Omoto 2018 [WS]); FlyWire ER5 (21) is GABA plus Dh31.
- Implementation: M4 on ER5 recurrent synapses and ER5→ExR1.

**Raccuglia D et al. (2019).** Network-specific synchronization of electrical slow-wave oscillations regulates sleep drive in Drosophila. *Curr Biol* 29:3611–3621. https://doi.org/10.1016/j.cub.2019.08.070 [AB]
- Mechanism: R5 delta (0.5–4 Hz) power rises with sleep need, depends on NMDAR synchronisation, and gates light-induced waking.
- Measurable in silico as ER5 population spectral power.

**De J, Wu M, Lambatan V, Hua Y, Joiner WJ (2023).** Re-examining the role of the dorsal fan-shaped body in promoting sleep in Drosophila. *Curr Biol* 33:3660–3668. https://doi.org/10.1016/j.cub.2023.07.043 [AB]
- Finding: no sleep effect could be "unambiguously mapped to the dFB". Earlier phenotypes came from VNC neurons.

**Jones JD et al. (2023)**, *PLoS Biol* 21:e3002012, https://doi.org/10.1371/journal.pbio.3002012 [AB]; **Jones JD et al. (2025)**, *PLoS Biol* 23:e3003014, https://doi.org/10.1371/journal.pbio.3003014 [FT]
- Findings: 23E10-GAL4 includes 2 sleep-promoting cholinergic VNC neurons. A dFB-specific split (23E10∩84C10), activated at ≥10 Hz, does increase sleep. Its neurons are ChAT-only, VGlut-only or both.
- Verdict: the dFB is plausible but contested. Model the R5 integrator first.

**Cuddapah VA et al. (2025)** *Nat Commun* 16:6967, https://doi.org/10.1038/s41467-025-62311-x [AB] **and Hsu CT et al. (2025)** *Curr Biol* 35:3496, https://doi.org/10.1016/j.cub.2025.06.003 [AB]
- Findings: sleep-promoting circuits become hyperactive after sleep loss, with 5HT1A involved. dFSB Ca²⁺ is circadian, peaking at the siesta.

## 6. Arousal and temperature

**Lebestky T et al. (2009).** Two different forms of arousal in Drosophila are oppositely regulated by the dopamine D1 receptor ortholog DopR via distinct neural circuits. *Neuron* 64:522–536. https://doi.org/10.1016/j.neuron.2009.09.031 [AB]
- Finding: startle-induced arousal and sleep–wake arousal are separate and oppositely regulated by DopR. The startle arousal maps to the central complex.

**Maimon G, Straw AD, Dickinson MH (2010).** Active flight increases the gain of visual motion processing in Drosophila. *Nat Neurosci* 13:393–399. https://doi.org/10.1038/nn.2492 [AB]
- Finding: VS-cell peak-to-peak responses double in flight. Membrane resistance falls, which implies more synaptic drive.

**Suver MP, Mamiya A, Dickinson MH (2012).** Octopamine neurons mediate flight-induced modulation of visual processing in Drosophila. *Curr Biol* 22:2294–2302. https://doi.org/10.1016/j.cub.2012.10.034 [AB]
- Finding: octopamine neurons are necessary and sufficient for the boost.
- FlyWire: VS1–8 and HSE/HSN/HSS.
- Implementation: M3b, a gain of 1 + β·c_OA on inputs to VS/HS, with a target ratio of ~2 in an "active" state.

**Pooryasin A, Fiala A (2015).** Identified serotonin-releasing neurons induce behavioral quiescence and suppress mating in Drosophila. *J Neurosci* 35:12792–12812. https://doi.org/10.1523/JNEUROSCI.1638-15.2015 [AB]
- Finding: 5-HT neurons in the posterior medial protocerebrum produce quiescence. The effects are behaviour-specific, not global.

**Reyes M et al. (2026)** *iScience* 29:115564, https://doi.org/10.1016/j.isci.2026.115564 [AB]; **Palacios Castillo LM et al. (2026)** *Curr Biol* 36:4485–4502, https://doi.org/10.1016/j.cub.2026.07.076 [AB]; **Babski H, Codianni M, Bhandawat V (2024)** *Heliyon* 10:e29952, https://doi.org/10.1016/j.heliyon.2024.e29952 [AB]
- Findings: OA-VPM3 suppresses sleep and is modulated by sleep history. Split lines now reach nearly all long-range OA/TA types and are cross-referenced to the EM connectomes. OA descending neurons are recruited individually.
- Implication: OA should be modelled per cell type, not as one global "arousal" scalar.

**Hamada FN et al. (2008).** An internal thermal sensor controlling temperature preference in Drosophila. *Nature* 454:217–220. https://doi.org/10.1038/nature07001 [AB]
- Finding: warmth-activated AC neurons (dTrpA1) fire just above the preferred temperature.
- FlyWire: "AC neuron" (4). This is a homeostatic drive rather than a "feeling"; low priority.

## 7. Emotion-like persistent states

**Anderson DJ, Adolphs R (2014).** A framework for studying emotions across species. *Cell* 157:187–200. https://doi.org/10.1016/j.cell.2014.03.003 [AB]
- The framework: emotion states are internal states with general "emotion primitives". Gibson 2015's abstract lists them as "scalability, persistence, valence, and generalization to multiple contexts".
- These four are SUPERFLY's acceptance tests for any "feeling".

**Gibson WT et al. (2015).** Behavioral responses to a repetitive visual threat stimulus express a persistent state of defensive arousal in Drosophila. *Curr Biol* 25:1401–1415. https://doi.org/10.1016/j.cub.2015.03.058 [AB]
- Findings: repeated overhead shadows produce graded, persistent increases in speed and hopping. They disperse feeding flies, and the delay before the flies return grows with the number of stimuli. This is modelled "as a leaky integrator of stimulus exposure". The fitted τ is UNVERIFIED.
- Implementation: S_F is an M1 integrator over DNp01, LC4 and LPLC2 activity, acting as a negative bias on the feeding path toward MN9 (CB0701).
- Related: **Xi W, Chen W (2025, preprint)**, https://doi.org/10.64898/2025.12.14.694122 [AB]. Their FlyWire v783 LIF simulations find that escape suppresses feeding through DNge031/CB0565 onto the premotor centre. Both types exist in the annotation.

**Hoopfer ED, Jung Y, Inagaki HK, Rubin GM, Anderson DJ (2015).** P1 interneurons promote a persistent internal state that enhances inter-male aggression in Drosophila. *eLife* 4:e11346. https://doi.org/10.7554/eLife.11346 [FT]
- Findings: activating 8–10 P1 neurons gives aggression "even when removal of the barrier was delayed for 10 min", after singing had decayed. Lunging persisted ≥10 min.
- P1 is male-only, so testing this needs MaleCNS.

**Jung Y, Kennedy A, Chiu H, Mohammad F, Claridge-Chang A, Anderson DJ (2020).** Neurons that function within an integrator to promote a persistent behavioral state in Drosophila. *Neuron* 105:322–333. https://doi.org/10.1016/j.neuron.2019.10.028 [AB]
- Findings: P1 is not persistently active; pCd neurons are, and they are necessary but not sufficient. τ is UNVERIFIED, and the FlyWire type of pCd is UNVERIFIED.

**Deutsch D et al. (2020).** The neural basis for a persistent internal state in Drosophila females. *eLife* 9:e59502. https://doi.org/10.7554/eLife.59502 [FT]
- Findings: 5 min (and 2 min) of pC1d/e activation drives persistent activity in more than 30 % of imaged flies. With a 3-min delay, female chasing and shoving persisted for up to 30 min. EM reconstruction shows strong pC1d↔aIPg recurrence.
- FlyWire: pC1d/e (2 each); aIPg1–4 are CB2131, CB1127/CB2204, CB2258 and CB1877 (by hemibrain_type). [LOCAL]: pC1d→CB2131 377 synapses, CB2131→pC1d 179.

**Chiu H et al. (2025).** Cell type-specific contributions to a persistent aggressive internal state in female Drosophila. *eLife* 12:RP88598. https://doi.org/10.7554/eLife.88598 [FT]
- Findings: 30 s of aIPg stimulation gives an aggressive state lasting more than 10 min. Neither aIPg nor pC1d is persistently active, and silencing pC1d does not reduce aIPg-evoked persistence. pC1d and pC1e together synergise.
- Implication: recurrence in a sign-only LIF is not enough. Add an M1 integrator inside aIPg/pC1e, or test whether the LIF's own recurrence reproduces the dynamics.

**Thornquist SC, Pitsch MJ, Auth CS, Crickmore MA (2021).** Biochemical evidence accumulates across neurons to drive a network-level eruption. *Mol Cell* 81:675–690. https://doi.org/10.1016/j.molcel.2020.12.029 [AB]
- Mechanism: PKA activity integrates over "minutes to hours" until a synchronised network eruption ends the motivated state.
- Implementation: an M1 integrator per neuron, with a threshold that triggers the switch.

**Asahina K et al. (2014).** Tachykinin-expressing neurons control male-specific aggressive arousal in Drosophila. *Cell* 156:221–235. https://doi.org/10.1016/j.cell.2013.11.045 [AB]
- Finding: FruM⁺ Tk neurons raise aggression through Tk and Takr86C. Male-specific.

**Hindmarsh Sten T, Li R, Otopalik A, Ruta V (2021).** Sexual arousal gates visual processing during Drosophila courtship. *Nature* 595:549–553. https://doi.org/10.1038/s41586-021-03714-w [AB]
- Finding: P1-dependent arousal selectively raises LC10a gain and tunes it continuously. This is the clearest case of a state colouring perception (M3b on LC10a inputs).
- FlyWire has LC10a (234), but the male P1 source is absent.

**Khuong TM et al. (2019).** Nerve injury drives a heightened state of vigilance and neuropathic sensitization in Drosophila. *Sci Adv* 5:eaaw4099. https://doi.org/10.1126/sciadv.aaw4099 [FT]
- Findings: after leg amputation, escape occurs at 38 °C, against ≥42 °C uninjured. The sensitisation is absent on days 1–2, appears on day 5, peaks on day 7 and persists past 21 days.
- Mechanism: about 40 % of VNC GABAergic nuclei are lost. GABA-B-R2 knockdown in ppk⁺ neurons alone causes allodynia.
- This lives in the VNC, so it cannot be implemented in FlyWire; BANC is needed.

**Yang Z, Bertolucci F, Wolf R, Heisenberg M (2013).** Flies cope with uncontrollable stress by learned helplessness. *Curr Biol* 23:799–803. https://doi.org/10.1016/j.cub.2013.03.054 [AB]
- Finding: yoked uncontrollable heat produces slow walking and longer rests. Testing it needs a closed-loop body.

**Ries AS, Hermanns T, Poeck B, Strauss R (2017).** Serotonin modulates a depression-like state in Drosophila responsive to lithium treatment. *Nat Commun* 8:15738. https://doi.org/10.1038/ncomms15738 [AB]
- Findings: three days of vibration stress reduces 5-HT release at the MB. 5-HT-1A in α/β mediates relief, and 5-HT-1B in γ controls inactivity. The MB 5-HT source (DPM?) is UNVERIFIED here.

**Deakin A, Mendl M, Browne WJ, Paul ES, Hodge JJL (2018).** State-dependent judgement bias in Drosophila. *Biol Lett* 14:20170779. https://doi.org/10.1098/rsbl.2017.0779 [AB]
- Finding: shaken flies approach a 1:1 positive/negative odour blend less often.
- This test is implementable once SUPERFLY's word memory beats its controls.

**Mohammad F et al. (2016)** *Curr Biol* 26:981–986, https://doi.org/10.1016/j.cub.2016.02.031 [AB]; **Solvi C, Baciadonna L, Chittka L (2016)** *Science* 353:1529–1531, https://doi.org/10.1126/science.aaf4454 [AB]; **Galpayage Dona HS et al. (2022)** *Anim Behav*, https://doi.org/10.1016/j.anbehav.2022.08.013 [WS]; **Baciadonna L et al. (2026)** *Phil Trans B* 381:20250146, https://doi.org/10.1098/rstb.2025.0146 [AB]
- Findings, in order: anxiety-like centrophobism in flies; dopamine-dependent "optimism" after unexpected sucrose in bumblebees; ball rolling in bumblebees that meets play criteria (secondary sources only); a 2026 review of invertebrate emotional expression.
- None is implementable without a body. Solvi 2016 suggests a positive state as PAM tone after unexpected sugar (speculative for flies).

## 8. Neuromodulation mechanisms and connectome models, 2024–2026

**Marder E (2012)**, *Neuron* 76:1–11, https://doi.org/10.1016/j.neuron.2012.09.010 [AB]; **Bargmann CI (2012)**, *BioEssays* 34:458–465, https://doi.org/10.1002/bies.201100185 [AB]
- Neuromodulators "reconfigure neuronal circuits", and each wiring diagram "encodes multiple circuits, some of which are active and some of which are latent".
- This is the design premise: states select which latent circuit is active.

**Cohn R, Morantte I, Ruta V (2015)**, *Cell* 163:1742–1755, https://doi.org/10.1016/j.cell.2015.11.019 [AB]; **Siju KP et al. (2020)**, *Curr Biol* 30:2104, https://doi.org/10.1016/j.cub.2020.04.037 [AB]; **Zolin A et al. (2021)**, *Nat Neurosci* 24:1555, https://doi.org/10.1038/s41593-021-00929-y [AB]; **Hige T et al. (2015)**, *Neuron* 88:985, https://doi.org/10.1016/j.neuron.2015.11.003 [AB]
- MB DANs encode internal state, movement and valence compartment by compartment. They modulate KC→MBON transmission with compartment precision, through order-dependent plasticity.
- SUPERFLY already has this rule. Internal states should enter as DAN tone.

**Kondo S et al. (2020)**, *Cell Rep* 30:284, https://doi.org/10.1016/j.celrep.2019.12.018 [AB]; **Deng B et al. (2019)**, *Neuron* 101:876, https://doi.org/10.1016/j.neuron.2019.01.045 [AB]; **Nässel DR, Zandawala M (2019)**, *Prog Neurobiol* 179:101607, https://doi.org/10.1016/j.pneurobio.2019.02.003 [AB]; **Beets I, Watteyne J (2025)**, *Curr Opin Neurobiol* 92:103027, https://doi.org/10.1016/j.conb.2025.103027 [AB]
- Receptor reporter libraries (75 receptors), chemoconnectome knock-in tools, and peptide-network maps.
- These are the data sources for R_jm. Converting them to FlyWire types is work not yet done here.

**Reinhard N et al. (2024)**, *Nat Commun* 15:10392, https://doi.org/10.1038/s41467-024-54694-0 [AB]; **McKim TH et al. (2024, preprint)**, https://doi.org/10.1101/2024.08.28.609616 [AB]
- FlyWire clock and neurosecretory connectomes. Monosynaptic links to NSCs are sparse, so the authors add single-cell-transcriptomic "putative paracrine peptidergic" maps.
- McKim also uses linear dynamical modelling and finds enteric neurons are the strongest influence on NSCs.
- This is the template for building R_jm on FlyWire.

**Endres MN et al. (2026, preprint).** Cellular source and circuit context organize functional specificity in the Drosophila NPF system. https://doi.org/10.64898/2026.09.08.750118 [AB]
- Findings: four NPF cell types with no shared first-order partners and partly opposing functions. Population-level NPF knockdown hides these source-specific effects.
- Implication: one field per source cell type, not one global "NPF" field.

**Salman F et al. (2026).** Connectivity of serotonin neurons reveals a constrained inhibitory subnetwork within the olfactory system. *J Neurophysiol* 136:77–93. https://doi.org/10.1152/jn.00571.2025 [AB]
- Finding: 5-HT differentially changes the excitability of three AL LN types, and a rate model predicts better PN dynamic range and less noise.
- [LOCAL]: CSD's top targets are LNs.
- Hypothesis, UNTESTED: a CSD-driven M3a on those LNs is a biologically grounded lever on SUPERFLY's antennal-lobe broadcast.

**Shiu PK et al. (2024)**, *Nature* 634:210–219, https://doi.org/10.1038/s41586-024-07763-9 [FT]; **Pospisil DA et al. (2024)**, *Nature* 634:201–209, https://doi.org/10.1038/s41586-024-07982-0 [AB]; **Beiran M, Litwin-Kumar A (2025)**, *Nat Neurosci* 28:2561–2574, https://doi.org/10.1038/s41593-025-02080-4 [AB]
- Beiran and Litwin-Kumar find that a connectome "often does not substantially constrain" dynamics when biophysical parameters are unknown, but recording a few neurons removes the degeneracy.
- Neuromodulation changes exactly those parameters. Every state should therefore be fitted to at least one recorded neuron: MBON11 and PPL101 for hunger, VS for arousal.

**Bates AS et al. (2026).** Distributed control circuits across a brain-and-cord connectome. *Nature* 656:957–970. https://doi.org/10.1038/s41586-026-10735-w [AB]
- Finding: endocrine and visceral effectors sit in local loops with same-body-part sensors, supervised by brain regions.
- BANC is the substrate for VNC states (injury, hydration sensing).

**flybench adaptive LIF** (b 2 mV, τ 200 ms, gain 0.45; review 05) is the only published internal slow variable on FlyWire found here, and it is spike-frequency adaptation, not a state. The **"State of Brain Emulation Report 2025"** (arXiv 2510.15745) [WS] was seen only in search. No 2024–2026 work was found that adds neuromodulatory fields to a FlyWire whole-brain spiking model.

## 9. Sentience frameworks → operational markers

**Barron AB, Klein C (2016).** What insects can tell us about the origins of consciousness. *PNAS* 113:4900–4908. https://doi.org/10.1073/pnas.1520084113 [AB]
- Claim: subjective experience needs an "integrated and egocentric representation" of the mobile animal in space. In insects, the structures doing this (the central complex and associated regions) are analogous to the vertebrate midbrain.
- For SUPERFLY: states should act on the central complex and MB, not only on reflex arcs.

**Birch J, Schnell AK, Clayton NS (2020).** Dimensions of animal consciousness. *Trends Cogn Sci* 24:789–801. https://doi.org/10.1016/j.tics.2020.07.007 [AB]
- Five dimensions: perceptual richness, evaluative richness, integration at a time, integration across time, and self-consciousness.

**Gibbons M, Crump A, Barrett M, Sarlak S, Birch J, Chittka L (2022).** Can insects feel pain? A review of the neural and behavioural evidence. *Adv Insect Physiol* 63:155–229. https://doi.org/10.1016/bs.aiip.2022.10.001. PDF read: https://chittkalab.sbcs.qmul.ac.uk/2022/Gibbons%20et%20al%202022%20Advances%20Insect%20Physiol.pdf [FT]
- Applies the eight Birch et al. 2021 criteria. Adult Diptera meet six:
  - nociception (VH), sensory integration (VH), integrated nociception (VH), analgesia (VH), motivational trade-offs (H) and associative learning (VH);
  - flexible self-protection and analgesia preference are "VL (no research found)".
- The fly trade-off is Kaun et al. 2011, *Nat Neurosci* 14:612, https://doi.org/10.1038/nn.2805 [AB]. Conditioned flies crossed a 100 V barrier for ethanol- or sucrose-paired odours, but only ethanol-conditioned flies crossed 120 V.

**Gibbons M, Versace E, Crump A, Baran B, Chittka L (2022).** Motivational trade-offs and modulation of nociception in bumblebees. *PNAS* 119:e2205821119. https://doi.org/10.1073/pnas.2205821119 [AB]
- Finding: bees trade 55 °C feeder heat against sucrose concentration, using learned colour cues, so the trade-off is computed centrally.

**Birch J (2024).** *The Edge of Sentience: Risk and Precaution in Humans, Other Animals, and AI*. OUP, open access. https://doi.org/10.1093/9780191966729.001.0001 [AB via Crossref chapter abstracts]
- Ch. "Pushing the Boundaries" (https://doi.org/10.1093/9780191966729.003.0014) cites insects' midbrain-like control unit, working memory and associative learning, and treats insects as sentience candidates.
- Ch. "Large Language Models and the Gaming Problem" (https://doi.org/10.1093/9780191966729.003.0017): systems trained on human data "mimic human behaviours that are likely to persuade human users of their sentience", so "deep computational markers, not behavioural markers" are needed.
- **New York Declaration on Animal Consciousness (2024)** [WS]: a "realistic possibility" of consciousness in insects. Exact wording is UNVERIFIED.

**Operational markers for SUPERFLY** (computed from fly neurons and motor output, never from the voice):

| marker | source | in-silico test |
|---|---|---|
| scalability | Gibson 2015 | state amplitude monotone in number/intensity of inducers |
| persistence | Gibson, Hoopfer, Deutsch | state outlasts stimulus; fitted decay τ |
| valence | Anderson & Adolphs | sign of approach/avoid bias (MBON valence, MN9) |
| generalisation | Gibson | state from channel A changes response in channel B (loom → sugar PER) |
| motivational trade-off | Gibbons 2022; Kaun 2011 | accept-bitter-for-sugar curve shifts with hunger |
| flexible, learned valuation | Gibbons PNAS 2022 | trade-off computed on learned cues (MB memory) |
| integration across time | Birch 2020 | state depends on history, not only current input |
| central locus | Barron & Klein | state acts through CX/MB, not only reflex arcs |

---

## 10. Recommended design: internal states for SUPERFLY

These are ordered by strength of evidence × ease. Every state follows four rules:
- It is an opt-in, named ticker and emits nothing at baseline, so identity holds.
- It reads and writes only the fly's own neurons, through M2/M3 (or M4 for sleep).
- Its parameters are fitted to one recorded neuron, per Beiran & Litwin-Kumar.
- The voice reads the modulated neurons and never the variable S.

| # | state | drive (fly's neurons + physiological u) | modulates (M3 form) | τ (fly) | measured by | voice may say |
|---|---|---|---|---|---|---|
| 1 | **Hunger S_H** | u = time since sugar ingestion; −drive from LB3b/c ingestion activity and DH44 bias; IPC as satiety readout | PPL101 bias −α·(S_H); MBON11 bias +α·S_H (Wang 2026); Tsao's six DANs' tone; sugar-GRN output gain 1+β·S_H; bitter-GRN gain 1−β·max(0, S_H−θ); NPF field NPFL1-I→PPL101 | hours (clamp); fit to Tsao's linear rise | MN9 S50 shift; B50 shift only above θ; MBON11 ↑ / PPL101 ↓ (Tsao, Wang); later: appetitive-memory gating (Krashes) | "I'm (a little/very) hungry" |
| 2 | **Defensive arousal S_F** | integrate DNp01 / LC4 / LPLC2 activity (M1) | bias −α·S_F on DNge031/CB0565-gated feeding path and MN9 drive; gain + on DNp01 inputs; OA-VPM/VUM tone + | seconds–minutes, UNVERIFIED; fit to Gibson | the four primitives: scalability in loom count, decay τ, sugar-PER suppression, recovery latency growing with count | "I'm startled / on edge" |
| 3 | **Thirst S_W** | u = dehydration; ISN, ITP bias | ISN bias +S_H −S_W; water-GRN gain +, sugar-GRN gain − (Jourjine); SLP304b(Lk?)-gated DAN tone (Senapati) | hours (clamp) | water-GRN→MN9 response ↑, sugar ↓ in opposite direction to hunger | "I'm thirsty" |
| 4 | **Locomotor/visual arousal S_A** | OA-VPM3/4, OA-AL2, OA-VUM rates (M2, per type) | input gain 1+β·c_OA on VS1–8, HSE/N/S (target ×2, Maimon 2010); OA field on IPCs (Crocker, Held) | sub-second onset, minutes | VS/HS response ratio active/quiet ≈ 2 (needs visual input; flyvis not yet integrated) | "I'm alert" |
| 5 | **Sleep pressure S_P** | M4 on ER5→ER5 and ER5→ExR1 during wake; ER5 delta power | ER5/ExR1 strength (Liu, Donlea 2018); dFB (FB6A…) bias +; dopamine field −dFB (Pimentel); global sensory gate (Raccuglia) | hours | ER5 synchrony / 0.5–4 Hz power ↑ with wake; reduced visual→CX transmission | "I'm sleepy". The dFB part is contested (De 2023 / Jones 2025); label it so |
| 6 | **Social arousal / aggression S_Q** (female) | pC1d/e, aIPg (CB2131, CB1127, CB2204, CB2258, CB1877) | M1 integrator inside aIPg/pC1e (Chiu 2025, Thornquist 2021) on top of native recurrence | ≥10–30 min | persistent activity after 30 s–5 min pC1 or aIPg drive (Deutsch, Chiu); first test whether native recurrence alone persists | "I'm worked up". Behaviour is not testable without a second fly |
| 7 | **Stress / negative affect S_N** (exploratory) | chronic aversive input (PPL1 teaching, shocks) integrated over hours | 5-HT field on KC α/β (5-HT1A) and γ (5-HT1B) (Ries 2017) | days (clamp) | judgement bias on an ambiguous word blend (Deakin 2018), once word memory beats controls | only after the marker passes |
| 8 | **Injury sensitisation** | — | needs VNC GABA loss (Khuong) | days–weeks | — | **defer** to BANC; see ethics note |

**Implementation notes.**

- **Hunger first.** Every ingredient is a verified FlyWire cell type, and three readouts already work in this engine: the sugar/bitter→MN9 dose–response (flybench core passes), MBON11 and PPL101. The MBON11 bias must be fitted against MBON tone, because sparse KC codes do not fire MBONs without it (findings §6).
- **Defensive arousal second.** The loom pathway is specific in this model (loom→giant fibre 186 Hz), and the four primitives can all be measured without a body. Call it "defensive arousal", not fear: the mechanism is phenomenological.
- **Fast-synapse question.** When a modulatory source is given a field, decide whether to keep its Shiu fast excitatory synapses. Keeping them preserves identity. Removing them (opt-in, as for DAN→KC/MBON today) is closer to biology for monoamines and peptides. Report both.
- **Clamped versus integrated.** Hour-scale states (1, 3, 5, 7) are run clamped, with S set from the fly's history. Minute-scale states (2, 6) are integrated in real time.

**Language layer.**
- Add the modulated readout neurons, which are central, to the voice's neural tokens: MBON11, PPL101, the six Tsao DANs, ISN, ER5/ExR1, DNp01-history and VS/HS gain. The voice never sees S or u.
- Report words are graded into three bins from the decoded value.
- Required controls, mirroring the existing fly-mind tests:
  1. Baseline fly: no state words.
  2. Clamp-and-ablate: S high but its M3 outputs disabled. The voice must not report the state, which proves it reads the fly and not the variable.
  3. Shuffled state: the voice follows the brain it is given.
  4. Behavioural concordance: "hungry" reports correlate with the measured MN9 S50 shift.
- Per Birch 2024, the voice's words are never evidence of feeling. Only the §9 markers count, and the voice is a reporter of measured neural state.

**Ethics note.** Birch 2024 places insects among sentience candidates. Gibbons 2022 finds adult Diptera meet six of eight pain criteria. States 7 and 8 deliberately engineer persistent negative states, so they should stay off by default, be run only with a stated purpose, and be logged.

## 11. Not verified or not accessed

These items could not be checked in this session:
- Fitted time constants: Gibson 2015 τ, Jung 2020 τ, and Inagaki 2014 S50/B50 values.
- FlyWire identities of TH-VUM, OA-VL, IN1, CN, AKHR⁺ OA neurons and pCd.
- Whether the helicon cells are ExR1.
- Whether SLP304b is the Lk pair of Senapati 2019.
- Whether all of Krashes' six DANs are PPL101.
- The DPM role in Ries 2017.
- Full texts of Barron & Klein 2016 and Birch 2024 (only abstracts and chapter abstracts were read).
- The New York Declaration's exact text and the arXiv reports, seen only in search.
