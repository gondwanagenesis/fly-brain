# 07 — Fly memory and insect intelligence: what SUPERFLY can implement

Literature review, 2026-10-08, for SUPERFLY (`SUPERFLY.md`; `research/superfly_findings.md` §4, §6).

**Sources.** WebFetch was blocked for publisher hosts that day. Full texts came from the Europe PMC REST and NCBI BioC
APIs; preprint abstracts from the bioRxiv and arXiv APIs. Tags:
**[FT]** full text searched this session · **[AB]** abstract only · **[SS]** search-engine summary only (weaker) · **[MD]** metadata only.
Links are DOI or PMC resolvers for records retrieved this session. **UNVERIFIED** marks claims not checked against a source.
*Derived* marks my own arithmetic.

---

## Executive summary

1. A fly has about 15 compartmental memories, each with its own write rate, decay, capacity and update rule (Aso & Rubin 2016). SUPERFLY uses one rule for all of them.
2. Real plasticity is large: one 1-s odour + DAN pairing cuts the MBON-γ1pedc CS+ response by ~80 % (spikes) and ~90 % (EPSC charge) for ≥40 min (Hige 2015). So the ~1 Hz effect means evoked MBON responses are tiny, not that learning is weak.
3. Timing: depression for DAN 0 to +0.5 s after KC onset, potentiation for DAN 1.2 s before, ~nothing at 6 s (Handler 2019). DAN alone potentiates (Cohn 2015) and causes forgetting (Berry 2012).
4. LTM needs spaced training (15-min ITI), protein synthesis and αβ/vertical-lobe output, and is gated by oscillating DANs. Massed training gives ARM; ARM and LTM are exclusive (Tully 1994; Isabel 2004; Plaçais 2012).
5. MBON→DAN feedback is everywhere (all five PPL1 and 99 of 150 PAM DANs counted by Li 2020). It implements extinction, reconsolidation, safety and second-order learning, the last through SMP108. Every cell type involved is in FlyWire v783.
6. Likely causes of SUPERFLY's failure: (a) APL is non-spiking in the fly but a spiking LIF cell in the model; (b) KC→MBON drive is not normalised. Eschbach's connectome model sets KC→MBON weights to 1/(N_KC·f).
7. More KCs is a real, measured upgrade: KC expansion improved odour discrimination (Ahmed 2023). Bees have ~170,000 KCs per MB; FlyWire has 5,177 in total. Keep ~6 claws per KC and renormalise drive per MBON.
8. Central-complex memories (place learning, seconds-long working memory, FB pattern memory, R→EPG heading maps that remap in minutes) are a second route to a persistent self.
9. Sleep consolidates: 4 h of induced sleep made massed training into LTM (Donlea 2011); DANs are reactivated after training (Dag 2019). The dFB's role was contested in 2023.
10. Order of work: fix the MB code → compartment rules, loops, consolidation and replay → grow the MB with word and visual input → CX plasticity → bee-style tests.

---

## Part 1 — Mushroom-body memory

### 1.1 The synapse: rule, magnitude, timing

**Hige T, Aso Y, Modi MN, Rubin GM, Turner GC (2015). Heterosynaptic plasticity underlies aversive olfactory learning in *Drosophila*. Neuron 88:985–998.** https://doi.org/10.1016/j.neuron.2015.11.003 · https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068 [FT]

- **Mechanism.** Odour + DAN pairing gives odour-specific LTD at KC→MBON synapses. It does not need MBON spikes, it depends on order (backward pairing had no effect), and it stays within one compartment.
- **Numbers.**
  - One pairing (1-s OCT, four 1-ms light pulses at 2 Hz) cut MBON-γ1pedc odour-evoked spikes (0–1.4 s window, spontaneous rate subtracted) from 118 ± 8.3 to 24 ± 7.4 for the CS+. The CS− went from 110 to 83.
  - EPSC charge fell 90 ± 3.7 %. Lasted ≥40 min.
  - CS− depression of 20–27 % tracked KC overlap (30–33 % shared KCs).
  - The same protocol left MBON-α2sc unchanged.
- **FlyWire.** MBON11 (γ1pedc>α/β), PPL101, MBON18 (α2sc), PPL105.
- **Implementation.**
  - Allow one-trial depression to f ≈ 0.1–0.2 in fast compartments; the current floor is f ≥ 0.3.
  - Null model for controls: ΔR(CS−) ≈ overlap · ΔR(CS+). At SUPERFLY's word overlap of ~0.08, a control should move by <10 % of the paired effect. More than that means the change is driven by network state.

**Hige T, Aso Y, Rubin GM, Turner GC (2015). Plasticity-driven individualization of olfactory coding in mushroom body output neurons. Nature.** https://doi.org/10.1038/nature15396 · PMC4860018 [FT]
- MBON tuning differs between flies but is nearly identical across one fly's two hemispheres, and this needs *rutabaga*. A fly's MBON tuning is a record of its own history, which gives a test for "remembering being itself" (design section).

**Cohn R, Morantte I, Ruta V (2015). Cell.** https://doi.org/10.1016/j.cell.2015.11.019 · PMC4732734 [FT]
- DAN activation alone *potentiates* KC→MBON (γ4); KC+DAN pairing depresses it. Both can be produced, reversibly, in one preparation.
- DAN activity follows motor state: flailing goes with high γ2/γ3 and low γ4/γ5 activity.

**Handler A, Graham TGW, Cohn R, Morantte I, Siliciano AF, Zeng J, Li Y, Ruta V (2019). Distinct dopamine receptor pathways underlie the temporal sensitivity of associative learning. Cell 178:60–75.** https://doi.org/10.1016/j.cell.2019.05.040 · PMC9012144 [FT]
- **Mechanism.** DopR1→cAMP gives depression (forward); DopR2→Gαq→ER Ca²⁺ gives potentiation (backward). Single trials write and reverse associations, reliably across 50 trials.
- **Numbers.** γ4 plasticity was significant at ISI (DAN onset − KC onset) of −1.2 s (potentiation) and 0 s and +0.5 s (depression). At 6 s it was minimal.
- **Implementation.** SUPERFLY's τ_kc = τ_da = 1 s sits inside this window. Aso & Rubin's behavioural windows are tens of seconds because their odours lasted 10–60 s.

**Davidson AM, Kaushik S, Hige T (2023). eNeuro.** https://doi.org/10.1523/eneuro.0275-23.2023 [AB]
- The LTD is presynaptic and confined to the compartment. Strongly driven boutons depress while weakly driven ones potentiate, so a single coincidence rule is a simplification.

### 1.2 Parallel compartments, parallel rules

**Aso Y, Rubin GM (2016). Dopaminergic neurons write and update memories with cell-type-specific rules. eLife 5:e16135.** https://doi.org/10.7554/eLife.16135 · PMC4987137 [FT]
- **Method.** Optogenetic activation of single DAN types in place of the US.
- **Numbers.**
  - PPL1-γ1pedc: robust after one round; retained at 10 min, largely gone by 24 h; no 4-day memory even after spaced training. Keeps only the latest association ("effective memory capacity of one").
  - PPL1-α3: barely detectable after one pairing, but 1- and 4-day memory after 10× spaced training. Same for PPL1-γ2α′1 + PPL1-α′2α2.
  - PAM-α1: keeps two associations; stable.
  - DAN within 30 s after the onset of a 10-s odour writes aversive memory; DAN 20–60 s *before* the odour writes appetitive memory.
  - Unreinforced re-exposure, or DAN without odour, reduces memory in a compartment-specific way.
  - γ1pedc + α1 together give a valence that flips over time (fast vs slow decay).
- **FlyWire** (type-to-compartment names as printed by Li 2020). PPL101 (γ1pedc), PPL106 (α3), PPL103 (γ2α′1), PPL105 (α′2α2), PAM11 (α1); MBON11, MBON14, MBON12, MBON18/13, MBON07.

### 1.3 Memory phases and consolidation

- **Tully T, Préat T, Boynton SC, Del Vecchio M (1994). Cell.** https://doi.org/10.1016/0092-8674(94)90398-0 [AB]. One-day memory has two independent parts.
  - ARM: decays within 4 days, cold-shock resistant, cycloheximide-insensitive, *radish*-dependent.
  - LTM: no appreciable decay over 7 days, needs protein synthesis.
- **Isabel G, Pascual A, Préat T (2004). Science.** https://doi.org/10.1126/science.1094932 [AB]. LTM formation erases ARM. Without vertical lobes, "the more these flies are trained, the less they remember".
- **Plaçais P-Y et al. (2012). Nat Neurosci.** https://doi.org/10.1038/nn.3055 [AB]. Blocking three pairs of MB-projecting DANs after training enhanced ARM; blocking them during spaced-training ITIs prevented LTM. Two pairs oscillate: massed training weakens the oscillations, LTM formation strengthens them.
- Pascual A, Préat T (2001), Science, https://doi.org/10.1126/science.1064200 [MD]: the vertical-lobe requirement for LTM, as cited by later papers.
- **Cervantes-Sandoval I, Martin-Peña A, Berry JA, Davis RL (2013). J Neurosci.** https://doi.org/10.1523/jneurosci.0451-13.2013 · PMC3733538 [FT]. Blocking γ output cut appetitive STM/ITM (15 min–3 h) by ~30–50 % and spared LTM (9–24 h). Blocking α/β cut STM/ITM ~30 % and abolished LTM. Retrieval migrates from γ/α′β′ to α/β.
- **Berry JA, Cervantes-Sandoval I, Nicholas EP, Davis RL (2012). Neuron.** https://doi.org/10.1016/j.neuron.2012.04.007 · PMC4083655 [FT]. Ongoing activity in MP1 (PPL1-γ1pedc) and MV1 (PPL1-γ2α′1) after learning sets the forgetting rate. dDA1/DopR1 is needed for acquisition, DAMB/DopR2 for forgetting.
- **Jacob PF, Waddell S (2020). Neuron 106:977–991.** https://doi.org/10.1016/j.neuron.2020.03.013 · PMC7302427 [FT].
  - Spaced training (5–10 trials, 15-min ITI, at the ERK/MAPK peak) writes an aversive CS+ memory and a slower, more persistent, protein-synthesis-dependent "safety" memory for the CS−.
  - CS+ memory: depressed responses in MBON-α2sc and MBON-α3 (PPL1). Safety memory: depressed CS− responses in MBON-β′2mp and the β′1 tuft of MBON-γ3β′1 (PAM-β′2mp, PAM-β′1, PAM-γ3).
  - FlyWire: MBON18, MBON14, MBON03, MBON09; PAM05/06, PAM13/14, PAM12. **Note:** this repo's annotation has no separate MBON08 (γ3) type.
- **Ichinose T, Aso Y, Yamagata N, Abe A, Rubin GM, Tanimoto H (2015). eLife 4:e10719.** https://doi.org/10.7554/eLife.10719 · PMC4643015 [AB]. MBON-α1 (MBON07) synapses onto PAM-α1 (PAM11); both are needed to acquire and consolidate appetitive LTM. Consolidation comes "at the cost of memory specificity".

### 1.4 MBON→DAN loops: extinction, reconsolidation, prediction error, second-order learning

- **Li F, Lindsey JW, Marin EC, et al. (2020). The connectome of the adult *Drosophila* mushroom body provides insights into function. eLife.** https://doi.org/10.7554/elife.62576 · PMC7909955 [FT].
  - All five PPL1 DANs and 99/150 PAM DANs get direct MBON input; 19/20 typical and 12/14 atypical MBON types contact DANs.
  - 22/34 MBON types contact FB tangential neurons.
  - ~8 % of KCs are mainly visual; 129 uniglomerular PNs give 63.6 % of KC input.
- **Takemura S-y et al. (2017). eLife.** https://doi.org/10.7554/elife.26975 · PMC5550281 [FT].
  - MBON-α3: 948 presynaptic KCs, 12,770–13,129 synapses (~13.5 per KC).
  - DAN>KC synapses are ~10 % as many as KC>MBON; only 6 % of KC>MBON synapses have a DAN terminal within 300 nm, so dopamine acts by volume transmission.
  - KC>DAN, KC>KC and KC>APL synapses are abundant.
- **Cervantes-Sandoval I, Phan A, Chakraborty M, Davis RL (2017). eLife.** https://doi.org/10.7554/elife.23789 [AB]. KC→DAN cholinergic input is needed for learning and keeps DANs chronically active. In SUPERFLY a word alone can therefore partly teach itself; control for it.
- **Felsenberg J, Barnstedt O, Cognigni P, Lin S, Waddell S (2017). Nature 544:240–244.** https://doi.org/10.1038/nature21716 · PMC5392358 [FT].
  - CS+ re-exposure without reward extinguishes reward memory. The path is V2 MBONs (α2sc, α′1, α2p3p, α′3ap, α′3m) driving aversive PPL1 DANs in neighbouring zones.
  - CS− re-exposure instead reconsolidates it, via MBON-γ2α′1 and PPL1-γ2α′1.
- **Felsenberg J, et al. (2018). Cell 175:709–722.** https://doi.org/10.1016/j.cell.2018.08.021 · PMC6198041 [FT].
  - Omitting the punishment is learned as reward through PAM-γ5 (PAM01). The old and new memories coexist and are integrated in M4β′/M6 (MBON03, MBON01).
  - Those MBONs had been potentiated by aversive learning, through release of feedforward inhibition from MVP2 (MBON11).
  - Two CS+ re-exposures at 15-min ITI, or five at 1-min ITI, were needed.
- **Yamada D, Bushey D, Li F, et al. (2023). eLife 12:e79042.** https://doi.org/10.7554/elife.79042 · PMC9937650 [FT].
  - Stable α1 memory teaches fast compartments through the cholinergic interneuron SMP108 (2 cells in FlyWire). SMP108 is disinhibited when the glutamatergic MBON-α1's CS response is depressed.
  - SMP108 drives PAM-γ5, γ4, β′2a, β′2m and β′2p, and is required for second-order conditioning.
  - α1 memory still instructed second-order learning one day later; the second-order memory itself decayed to chance by 24 h.
  - **Testable in SUPERFLY today** (Shiu signs treat glutamate as inhibitory): after word + PAM11 pairing, SMP108's response to that word should rise.
- **Eschbach C, Fushiki A, Winding M, et al. (2020). Nat Neurosci 23:544–555.** https://doi.org/10.1038/s41593-020-0607-9 · PMC7145459 [FT].
  - Larval connectome: 61 of 102 pre-DAN partner types relay MBON feedback; many DANs get >50 % of their input through feedback.
  - Connectome-constrained model:
    ```
    dr_i/dt  = -r_i + [ Σ_j W_ij r_j + b_i + I_i(t) ]_+
    dw/dt    = -r̄_k d_i + r_k d̄_i        anti-Hebbian timing rule (bars = low-pass)
    τ_W dW/dt = w - W ;   W_KC(0) = 1/(N_KC · f)
    ```
  - EM weights are scaled to Σ_j W_ij² = 1.5. Rates are reset between CS presentations, so memory has to be synaptic. Feedback motifs improved extinction, second-order and context performance.
  - This is the closest published analogue of SUPERFLY's rule, and the 1/(N_KC·f) normalisation is the missing piece.

### 1.5 The Kenyon-cell code and its regulation

- **Sparseness.**
  - Honegger KS, Campbell RA, Turner GC (2011), J Neurosci, https://doi.org/10.1523/jneurosci.1099-11.2011 · PMC3180869 [FT]: ~5 % of KCs per odour per imaging plane; 6 % single-cell; 9 % for mixtures; ~20 % may be active on a single trial.
  - Turner, Bazhenov, Laurent (2008), J Neurophysiol, https://doi.org/10.1152/jn.01283.2007 [SS]: KC spontaneous rate 0.1 ± 0.4 spikes/s.
  - Gruntman & Turner (2013), Nat Neurosci, https://doi.org/10.1038/nn.3547 · PMC3908930 [FT]: KCs need several active claws to spike.
- **Lin AC, Bygrave AM, de Calignon A, Lee T, Miesenböck G (2014). Nat Neurosci.** https://doi.org/10.1038/nn.3660 · PMC4000970 [FT]. Blocking the APL–KC loop reduces sparseness, raises inter-odour correlation, and abolishes learned discrimination of *similar* odours.
- **Amin H, Apostolopoulou AA, Suárez-Grimalt R, Vrontou E, Lin AC (2020). eLife.** https://doi.org/10.7554/elife.56954 [AB]. APL is **non-spiking**. Its inhibition is spatially localised, and each KC inhibits itself via APL more than it inhibits other KCs.
- **Apostolopoulou AA, Lin AC (2020). PNAS.** https://doi.org/10.1073/pnas.1921294117 [AB]. Four days of APL activation are compensated (more KC excitation, less APL activation); APL block is barely compensated.
- **Abdelrahman NY, Vasilaki E, Lin AC (2021). PNAS.** https://doi.org/10.1073/pnas.2102158118 · PMC8670477 [FT]. At coding level 0.1, realistic variability between KCs in input number, weight and threshold degrades memory. Activity-dependent compensation of weights, inhibition or thresholds that equalises KC mean activity rescues it, and the predicted correlations appear in the hemibrain.
- **MacKenzie AJ, Beatty L, Ulibarri JM, Azhar D, Amematsro P, Butts AR, Caron SJC (2025; v2 2026-08-21). Connectivity biases generate a learning hierarchy in the *Drosophila* mushroom body. bioRxiv.** https://doi.org/10.1101/2025.10.29.684686 [AB].
  - Some PN types connect up to 15-fold more often than others.
  - Odours activating >20 % of KCs were learned robustly; those under 10 % poorly. The response criterion is UNVERIFIED.
  - VL1 activates KCs broadly but supports no learning.
  - SUPERFLY's 5–9 % word codes may sit at the hard-to-learn end.
- **Hafez OA, Escribano B, Ziegler RL, Hirtz JJ, Niebur E, Pielage J (2023). eLife.** https://doi.org/10.7554/elife.77578 · PMC10069864 [FT]. MBON-α3 ex vivo: spontaneous 12.1 Hz, Vm −56.7 mV, τm ≈ 16 ms, spike-frequency adapting. An EM-based compartmental model shows the cell is compact and integrates stochastic KC input.

### 1.6 MB learning models with usable equations

- **Bennett JEM, Philippides A, Nowotny T (2021). Nat Commun.** https://doi.org/10.1038/s41467-021-22592-4 · PMC8105414 [FT]. DANs compute prediction errors from MBON feedback, and plasticity ignores the MBON rate. Their Eq. 6 is `Δw± = η·k·(λ − d∓)`: baseline KC-driven potentiation λ balanced by opposite-valence dopamine. It is homeostatic by construction.
- **Jürgensen A-M, Sakagiannis P, Schleyer M, Gerber B, Nawrot MP (2024). iScience 27:108640.** https://doi.org/10.1016/j.isci.2023.108640 · PMC10824792 [FT]. Spiking larval model with APL and MBON→DAN feedback.
  - Eligibility e_i is set to 1 at each KC spike and decays. Each DAN spike applies `Δw_i = −α·e_i`.
  - Homeostasis: `Δw_i = h(w_init − w_i)` per MBON spike, with α and h co-scaled.
  - Reward-evoked DAN rates were 33–39 Hz.
- **Betkiewicz R, Lindner B, Nawrot MP (2020). eNeuro.** https://doi.org/10.1523/eneuro.0305-18.2020 · PMC7294456 [FT]. KC adaptation `τ_A dI_A/dt = −I_A + τ_A ΔI_A Σδ(t−t_s)`, with ΔI_A = 0.132 nA and τ_A = 389 ms, and 12 PN inputs per KC. Adaptation gives sparseness in time and lateral inhibition gives sparseness across the population.
- **Others.**
  - Springer & Nawrot (2021), eNeuro, https://doi.org/10.1523/eneuro.0549-20.2021 [FT]: 2,000 KCs, 5–15 PNs each, top 5 % kept, KC→MBON weights start at 0.01; reproduces extinction.
  - Jiang & Litwin-Kumar (2021), PLoS Comput Biol, https://doi.org/10.1371/journal.pcbi.1009205 [AB]: heterogeneous DAN tuning still gives coherent learning, with prediction error as a population mode.
  - Li H, Yu L, Yu Q, Zang Y (2025), arXiv:2510.21315 [AB]: lateral inhibition plus spike-frequency adaptation together are best for odour discrimination under noise.
- **Shiu PK, et al. (2024). Nature.** https://doi.org/10.1038/s41586-024-07763-9 · PMC11446845 [FT]. SUPERFLY's L0: V_rest −52 mV, V_th −45 mV, τ_m 20 ms, τ_syn 5 ms, delay 1.8 ms, refractory 2.2 ms, W_syn 0.275 mV. DA, OA and 5-HT neurons are excitatory and "will be modelled less well".
- I found no peer-reviewed whole-brain FlyWire LIF with dopamine plasticity. Two GitHub projects (`TemurTurayev/fly-neuromod`, `snuri00/fruit-fly-connectome`) claim one: UNVERIFIED.

---

## Part 2 — Other fly memories: central complex and sleep

### 2.1 Central complex

- **Ofstad TA, Zuker CS, Reiser MB (2011). Visual place learning in *Drosophila melanogaster*. Nature.** https://doi.org/10.1038/nature10131 · PMC3169673 [FT]. In a 36 °C arena, flies learn over 10 trials of 5 min to find a cool tile by the visual panorama. Silencing ellipsoid-body ring-neuron lines (R15B07, R28D01) impaired learning; MB lines did not.
- **Neuser K, Triphan T, Mronz M, Poeck B, Strauss R (2008). Nature 453:1244–1247.** https://doi.org/10.1038/nature07003 [SS]. Working memory for a vanished target lasts "several seconds" (≥4 s per the press release). It needs GABAergic ring neurons, with S6KII/*ignorant* in a subset of them.
- **Kuntz S, Poeck B, Strauss R (2017). Curr Biol 27:613–623.** https://doi.org/10.1016/j.cub.2016.12.056 [SS]. The trace needs NO/cGMP and H₂S signalling at ring-neuron presynapses. Earlier work placed Ignorant in R3 and/or R4d.
- **Liu G, Seiler H, Wen A, Zars T, Ito K, Wolf R, Heisenberg M, Liu L (2006). Nature 439:551–556.** https://doi.org/10.1038/nature04381 [SS]. Short-term memory for "elevation" sits in FB stratum F5 and for "contour orientation" in F1.
- **Fisher YE, Lu J, D'Alessandro I, Wilson RI (2019). Nature.** https://doi.org/10.1038/s41586-019-1772-4 · PMC7753972 [FT]. Visual cues inhibit EPG neurons through GABAergic R neurons. The inhibition map reorganises "over minutes", consistent with LTD of co-active R→EPG synapses.
- **Kim SS, Hermundstad AM, Romani S, Abbott LF, Jayaraman V (2019). Nature.** https://doi.org/10.1038/s41586-019-1767-1 · PMC8115876 [FT]. Model rule: R and EPG co-active → depression; EPG alone → potentiation. Several maps can be stored if plasticity is presynaptically gated. The resulting individual "offset" is stable.
- **Dan C, Hulse BK, Kappagantula R, Jayaraman V, Hermundstad AM (2024). Neuron 112:2581–2599.** https://doi.org/10.1016/j.neuron.2024.04.036 [AB]. HD neurons are needed to update heading preferences under thermal reinforcement. The circuit locus of the goal memory is not in the abstract (UNVERIFIED).
- **Flores-Valle A, Honnef R, Seelig JD (2025; v2 2026-07-19 retitled "Neural drift during rest drives walking direction and memory consolidation in *Drosophila*"). bioRxiv.** https://doi.org/10.1101/2025.03.20.644317 [AB]. FB activity drifts at rest, offset 180° from walking; downstream neurons undo the offset, so rest reactivates walking directions. Learning changes the drift. The circuit is supported by connectome modelling. This is the best evidence for CX replay.
- **Supporting.**
  - Seelig & Jayaraman (2015), Nature, https://doi.org/10.1038/nature14446 [MD]: the EPG bump.
  - Hulse et al. (2021), eLife, https://doi.org/10.7554/elife.66039 [MD]: CX connectome.
  - Pisokas, Heinze, Webb (2020), eLife, https://doi.org/10.7554/elife.53985 [AB], and Kakaria & de Bivort (2017), Front Behav Neurosci, https://doi.org/10.3389/fnbeh.2017.00008 [MD]: connectome ring-attractor models.
  - Buchanan, Kain, de Bivort (2015), PNAS, https://doi.org/10.1073/pnas.1500804112 [AB]: lifelong, non-heritable left/right biases controlled by CX columnar neurons.
- **FlyWire types** (repo annotation): EPG 47, PEN_a 20, PEN_b 22, ER4d 24, hDeltaK 31, EL 17, FB6A 4.

### 2.2 Sleep and consolidation

- **Donlea JM, Thimgan MS, Suzuki Y, Gottschalk L, Shaw PJ (2011). Science.** https://doi.org/10.1126/science.1202249 · PMC4064462 [FT]. Four hours of dFB-induced sleep (104y>TrpA1) after massed courtship training produced LTM; dFB activation without sleep did not. See also Donlea, Pimentel, Miesenböck (2014), Neuron, https://doi.org/10.1016/j.neuron.2013.12.013 [MD].
- **Haynes PR, Christmann BL, Griffith LC (2015). eLife.** https://doi.org/10.7554/elife.03868 · PMC4305081 [FT]. DPM neurons, required for consolidation, promote sleep with GABA onto wake-promoting α′/β′ KCs.
- **Dag U, Lei Z, Le JQ, Wong A, Bushey D, Keleman K (2019). eLife.** https://doi.org/10.7554/elife.42786 · PMC6428568 [FT]. vFB neurons induce post-training sleep and reactivate DAN-aSP13. These DANs are needed for STM via DopR1 in γ KCs, and again in a post-training window for LTM; the experiments silence them 5–7 h after training. Which PAM type aSP13 is: UNVERIFIED.
- **Bushey D, Tononi G, Cirelli C (2011). Science.** https://doi.org/10.1126/science.1202839 [AB]. Synapses grow in wake and shrink only with sleep.
- Chouhan NS, Griffith LC, Haynes P, Sehgal A (2021), Nature 589:582–585, https://doi.org/10.1038/s41586-020-2997-y [SS]: hunger decides whether consolidation needs sleep.
- **Caveat.** Jones JD … Dissel S (2023), PLoS Biol, https://doi.org/10.1371/journal.pbio.3002012 [SS], and De J, Wu M, Lambatan V, Hua Y, Joiner WJ (2023), Curr Biol, https://doi.org/10.1016/j.cub.2023.07.043 [SS], map 23E10 sleep promotion to VNC neurons. The dFB "switch" is contested.

---

## Part 3 — What makes insects more intelligent

### 3.1 Bee cognition (candidate SUPERFLY benchmarks)

- **Giurfa M, Zhang S, Jenett A, Menzel R, Srinivasan MV (2001). Nature.** https://doi.org/10.1038/35073582 [AB]. Delayed matching- and non-matching-to-sample, with transfer to new stimuli and across modalities: the concepts of "sameness" and "difference".
- **Avarguès-Weber A, Dyer AG, Combe M, Giurfa M (2012). PNAS.** https://doi.org/10.1073/pnas.1202576109 [AB]. Two concepts learned at once (spatial relation + difference) and transferred.
- **Howard SR, Avarguès-Weber A, Garcia JE, Greentree AD, Dyer AG (2018). Science.** https://doi.org/10.1126/science.aar4975 [AB]. Trained on "less than" with 1–6 elements, bees place zero at the low end.
- **Alem S, Perry CJ, Zhu X, Loukola OJ, Ingraham T, Søvik E, Chittka L (2016). PLoS Biol.** https://doi.org/10.1371/journal.pbio.1002564 · PMC5049772 [FT]. String pulling.
  - Untrained bees: 0/291 in test 1, 2/135 in test 2. Stepwise training: 23/40. Observers: 15/25.
  - Transmission chains: 25/47, 17/29 and 12/28, against 0 in control colonies. The mechanism is associative.
- **Bridges AD, Royka A, Wilson T, Lockwood C, Richter J, Juusola M, Chittka L (2024). Nature.** https://doi.org/10.1038/s41586-024-07126-4 · PMC10954542 [FT]. A third of naive observers learned a two-step puzzle box that is unrewarded at step one.
- **Galpayage Dona HS, Solvi C, Kowalewska A, Mäkelä K, MaBouDi H, Chittka L (2022). Do bumble bees play? Anim Behav 194:239–251.** https://doi.org/10.1016/j.anbehav.2022.08.013 [SS]. Ball rolling is intrinsically rewarding and meets the criteria for play.
- **Sequence and timing.**
  - Kheradmand B, Richardson-Ramos I, Chan S, Nelson C, Nieh JC (2025), Insects 16:358, https://doi.org/10.3390/insects16040358 [AB]: alternation sequence, 64 % correct in the second half (320 visits, 20 bees).
  - Davidson A, Nanda I, Ong A, Chittka L, Versace E (2025), Biol Lett, https://doi.org/10.1098/rsbl.2025.0440 [MD]: duration discrimination.
  - Loukola et al. (2017), Science, https://doi.org/10.1126/science.aag2360 [MD]: improving on observed ball rolling.
- **MB models.** Cope AJ et al. (2018), PLoS Comput Biol, https://doi.org/10.1371/journal.pcbi.1006435 [AB]: an MB model with the protocerebral tract learns sameness and difference without top-down control. Peng & Chittka (2017), Curr Biol, https://doi.org/10.1016/j.cub.2016.10.054 [MD].

### 3.2 Mushroom-body size and Kenyon-cell number

- **Honeybee.** ~**170,000 KCs per mushroom body** (~340,000 per brain): Witthöft 1967, with Mobbs 1982, seen only via secondary citations [SS].
  - Nicolaidou AR, El Jundi B, Rössler W, Groh C (2026), J Comp Neurol, https://doi.org/10.1002/cne.70169 [AB], reportedly gives 368,000 per brain (28,000 class II). That number is not in the abstract: UNVERIFIED.
  - **Fly** (this repo's v783 annotation): 5,177 KCs in total. KCg-m 2,189, KCab 1,643, KCa′b′ 916, KCg-d 295, KCab-p 128. That is ~2,600 per hemisphere, ~65× fewer than one bee MB (*derived*).
- **Farris SM, Schulmeister S (2011). Proc R Soc B.** https://doi.org/10.1098/rspb.2010.2161 [AB]. Large, visually innervated MBs arose with parasitoidism, ~90 Myr before sociality. The proposed driver is spatial and associative learning.
- **Couto A et al. (2023), Nat Commun, https://doi.org/10.1038/s41467-023-39618-8 [SS]; Young FJ et al. (2024), iScience, https://doi.org/10.1016/j.isci.2024.108949 [SS].**
  - *Heliconius* MBs are ~4× larger than relatives', mainly through visual input, and this goes with better long-term visual memory.
  - Colour memory held at 8 days with recall at 13; recall correlated with calyx synapse number.
- **Elkahlah NA, Rogow JA, Ahmed M, Clowney EJ (2020). eLife.** https://doi.org/10.7554/elife.52278 · PMC7028369 [FT]. PN boutons scale linearly with KC number, roughly doubling at the largest expansions. Claws per KC stay constant.
- **Ahmed M et al. (2023). Input density tunes Kenyon cell sensory responses in the *Drosophila* mushroom body. Curr Biol.** https://doi.org/10.1016/j.cub.2023.05.064 · PMC10529417 [FT]. **The key causal result.**
  - From 500 to 4,000 KCs, KCs keep ~6 claws and their selectivity.
  - Discrimination is maintained with fewer KCs and *augmented* with more.
  - Raising claws by 50 % (Tao knockdown) raised the share of KCs responding to every odour from <20 % to 40 % and worsened discrimination.
  - For grafted KCs: keep ~6 PN inputs each; scale PN boutons with KC number.

### 3.3 Expansion coding and continual learning

- **Litwin-Kumar A, Harris KD, Axel R, Sompolinsky H, Abbott LF (2017). Neuron 93:1153–1164.** https://doi.org/10.1016/j.neuron.2017.01.030 [AB]. KC representation dimension peaks at the anatomical in-degree; sparse wiring can beat dense. The exact optimal K: UNVERIFIED.
- **Babadi B, Sompolinsky H (2014). Neuron.** https://doi.org/10.1016/j.neuron.2014.07.035 [AB]. Random expansion amplifies noise, more so when sparser; synapses that encode stimulus clusters fix it.
- **Dasgupta S, Stevens CF, Navlakha S (2017). Science 358:793–796.** https://doi.org/10.1126/science.aam9868 [AB; numbers from the preprint figure caption via SS]. Fly hash: 50 PNs → 2,000 KCs (sparse binary random), with APL keeping the top 5 %. PN inputs per model KC: UNVERIFIED.
- **Shen Y, Dasgupta S, Navlakha S (2023). Reducing catastrophic forgetting with associative learning: a lesson from fruit flies. *Neural Computation* 35(11):1797–1819.** https://doi.org/10.1162/neco_a_01615 [AB]. **Correction to the brief: Neural Computation, not PNAS** (preprint arXiv:2107.07617). Sparse expansion plus "partial freezing" (only active KCs' synapses to the target MBON change) provably forgets less than a perceptron.
- **Xie M, Muscinelli SP, Harris KD, Litwin-Kumar A (2023). eLife.** https://doi.org/10.7554/eLife.82914 [SS]. Smooth tasks favour much denser codes than classical theory; sparseness should match the task.
- Caron SJ, Ruta V, Abbott LF, Axel R (2013), Nature, https://doi.org/10.1038/nature12063 [MD]: random PN→KC convergence. MacKenzie 2025 adds biases at the level of cell types.

---

## Part 4 — Practical guidance for SUPERFLY's failure mode

| quantity | value | source |
|---|---|---|
| KC coding level | ~5 % (6 % single-cell); models 5–10 %; >20 % for robust learning | Honegger 2011; Springer 2021; Abdelrahman 2021; MacKenzie 2025 |
| KC spontaneous | 0.1 ± 0.4 spikes/s | Turner 2008 [SS] |
| claws per KC | ~6 | Ahmed 2023 |
| KC adaptation | ΔI_A 0.132 nA, τ_A 389 ms | Betkiewicz 2020 |
| MBON-α3 baseline | 12.1 Hz (ex vivo) | Hafez 2023 |
| MBON-α3 input | 948 KCs × ~13.5 synapses | Takemura 2017 |
| one-trial LTD | −80 % spikes, −90 % EPSC; CS− −20–27 % | Hige 2015 |
| window | potentiation at ISI −1.2 s; depression at 0 and +0.5 s; ~none at 6 s | Handler 2019 |

**Derived from Shiu parameters.**
- One synapse (0.275 mV, τ_m 20 ms, τ_syn 5 ms) gives a PSP peak of 0.275·(5/15)·(e^{−0.46} − e^{−1.85}) ≈ **0.043 mV**.
- The steady-state depolarisation is W_syn·τ_syn·R. Holding MBON-α3 at threshold (+7 mV) from KC drive alone needs R ≈ 5,100 synaptic events/s. That is ~377 KC spikes/s, or **~8 Hz sustained in each of 5 % of its 948 KCs**.
- Real KCs fire a few spikes per odour. So silent MBONs cannot respond to sparse codes, which is SUPERFLY's "0/96 fire". A 12.1 Hz baseline is real, so `tone` is justified, but it should be set per MBON.

**Steps, in order.**

1. **Graded APL graft** (opt-in, named; APL is non-spiking, Amin 2020). Replace the two LIF APLs' chemical synapses with:
   ```
   a_c(t) = lowpass_{τ≈10 ms}( Σ_{k∈c} n_k(t) )        per compartment / calyx region (localised, Amin 2020)
   I_k   -= g_APL · a_{c(k)}
   ```
   Tune g_APL to a word coding level of 5–10 % (up to ~20 % per MacKenzie). This is the "top-k" step of Dasgupta and Springer, done biophysically.
2. **KC-only spike-frequency adaptation** with Betkiewicz's values, instead of adaptation across the whole brain.
3. **Homeostatic KC thresholds** in a "development" phase, then frozen (Abdelrahman 2021; Apostolopoulou & Lin 2020). Generic form; their exact update was not reproduced:
   ```
   θ_k ← θ_k + η_θ ( p̄_k − p* )
   ```
4. **Normalise KC→MBON drive per MBON** (Eschbach: W = 1/(N_KC·f)). Multiply every KC weight onto MBON m by
   ```
   s_m = N_nat,m·f_nat / (N_nat,m·f_nat + N_graft,m·f_graft)
   ```
   Then 20,000 extra KCs add capacity and signal-to-noise, not raw drive. This breaks the "bigger MB → broadcast" coupling in findings §6.
5. **Depress deeply but locally.** Let f fall to 0.1–0.2 only in the compartments whose DANs fired (the A-matrix already does this), and keep unpaired MBONs fixed.
6. **Read out the way physiologists do.**
   - Compare CS+ with CS− in the same MBON (Hige's 118→24 vs 110→83).
   - Use 1-s test pulses and the first 0.3–1 s, starting from a quiet state.
   - Use the overlap null model, and report a valence-weighted index (`valence` already exists).
7. **Replace the passive `tau_recover`** with dopamine-alone potentiation (Cohn; Berry) and per-MBON-spike homeostasis `Δw = h(w_init − w)` (Jürgensen).
8. **Add a "word alone, no DAN" arm** to measure self-teaching through KC→DAN input (Cervantes-Sandoval 2017).
9. **Cell-class hygiene.** In this repo's annotation, `top_nt` is "dopamine" for **5,172 of 5,177 KCs**. KC output synapses are cholinergic (Barnstedt O et al. 2016, Neuron, https://doi.org/10.1016/j.neuron.2016.02.015 [MD]). `mb.DAN` is selected by `cell_class` (checked in `superfly/anatomy.py`), so it is safe. Never select dopaminergic synapses by NT prediction.

---

## Recommended design: a multi-timescale fly memory for SUPERFLY

This is a SUPERFLY design proposal combining the cited rules; it is not a published model. For each edge e = (k, m) in compartment c:

w_e = w0_e · (1 + u_e + s_e + a_e), where u is labile, s is consolidated LTM, and a is ARM.

```
eKC_k : KC eligibility, τ_e ≈ 1 s (Handler; Jürgensen)      d_c = Σ_d A[c,d] n_d, d̄_c low-pass τ ≈ 1 s

du_e/dt = η_c [ −eKC_k·d_c + β·n_k·d̄_c ]                     timing rule (Eschbach form)
          − ρ_c · d_c · (1 − eKC_k/eKC_max) · u_e             DAN alone: forget, toward 0 (Cohn; Berry)
          − u_e/τu_c                                          compartment decay
          − h·u_e  per spike of MBON m                         homeostasis (Jürgensen)

consolidation, only while gate G_c(t) = 1 (spaced ITI or offline phase):
ds_e/dt = κ_c · P_c(t) · (u_e − s_e)
dP_c/dt = −P_c/τ_P + Σ_i δ(t − t_i − Δ_ITI)                   "protein-synthesis" signal that needs spacing (15-min ITI)
da_e/dt = κ_a·M_c(t)·(u_e − a_e) − a_e/τ_a − λ·|ds_e/dt|      ARM from massed trials M; erased as LTM forms (Isabel)
```

| class | DAN → MBON (FlyWire) | write | decay | overwrite | → s? |
|---|---|---|---|---|---|
| fast aversive | PPL101→MBON11 (γ1pedc), PPL103→MBON12 (γ2α′1) | one trial, f ≈ 0.15 | ~1 day | capacity 1 | no |
| fast appetitive | PAM01/PAM15→MBON01 (γ5β′2a), PAM08→MBON21 | one trial | hours | partial | via loops |
| slow aversive | PPL106→MBON14 (α3), PPL105→MBON18 (α2sc) | ~10 spaced trials | days | retains | spaced only |
| slow appetitive | PAM11→MBON07 (α1) | brief | ≥1 day | retains ≥2 | yes; teaches via SMP108 |
| safety | PAM05/06→MBON03, PAM12/13/14→MBON09 | spaced, ordered | longest | — | yes |

Ratios come from Aso & Rubin, Hige, Cervantes-Sandoval and Jacob & Waddell; absolute values must be tuned.

**Native loops to keep** (no grafting needed; DANs keep their fast *inputs*):
- MBON07→PAM11: appetitive LTM.
- MBON07 ⊣ SMP108 → PAM01/08/02/06/05: second-order conditioning.
- V2 MBONs (e.g. MBON18) → PPL1: appetitive extinction.
- MBON03/01 → PAM01: aversive extinction.
- MBON12 → PPL103: reconsolidation.

**Offline phase ("sleep").**
- Run a quiet engine. Replay recently tagged KC ensembles together with their DANs (Dag 2019), open G_c so u → s, and scale unconsolidated u by 1 − δ (Bushey 2011).
- Add CX drift replay (Flores-Valle 2025).
- This is how massed episodes become LTM (Donlea 2011).

**"Remembering being itself."**
- Persist (s, a, θ_k, R→EPG maps) as the fly's own state file.
- Test it as Hige (Nature 2015) did. Two SUPERFLY individuals with the same connectome but different histories should show MBON tuning that correlates more within a fly across sessions than between flies. Heading offsets (Kim 2019) and turn biases (Buchanan 2015) should persist.

---

## Recommended path to more intelligence (ranked by realism and expected gain)

1. **Make the MB a working memory substrate** (Part 4). Realism high: every step is a measured fly property. Gain: a prerequisite for everything below.
2. **Compartment rules, MBON→DAN loops, multi-timescale consolidation.** Realism high. Gain large: extinction, reconsolidation, safety, second-order and context learning (feedback improved Eschbach's model).
3. **Grow the MB as development does:** more KCs, ~6 claws each, PN boutons scaled, drive renormalised, PN-type biases (MacKenzie). Realism high to medium: 25,000 KCs is well under a bee's ~170,000. Gain: discrimination and capacity (Ahmed; Litwin-Kumar; Shen).
4. **Visual KC input beside the word lobe.** About 8 % of fly KCs are visual; bees and *Heliconius* expanded exactly this. Realism medium-high. Gain: cross-modal association and DMTS transfer (Cope 2018).
5. **CX plasticity:** R→EPG remapping (Fisher; Kim), goal-heading memory (Dan), ring-neuron short-term traces (Neuser; Kuntz), rest drift and replay (Flores-Valle). Realism high. Gain: place learning, working memory, a spatial self.
6. **Offline consolidation.** Realism medium (the dFB is contested). Gain: retention from few episodes.
7. **Bee tasks as the intelligence yardstick**, alongside flybench: DMTS/DNMTS with words, "less than", alternation sequences, duration. A fly-sized MB may fail some, and that is informative.
8. **Social learning via the language model.** Realism low; keep the voice an observer (SUPERFLY rule 3).

**Avoid:** global Hebbian rules; dopamine as fast excitation; more claws per KC; graft drive left unnormalised; trusting NT predictions for KCs.
