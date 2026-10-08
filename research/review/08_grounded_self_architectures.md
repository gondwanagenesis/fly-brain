# Grounded self architectures: memory, identity and a fly-grounded autobiographical voice

Survey made 2026-10-08 for SUPERFLY. The question: how to build a voice that remembers being
*this* fly, with continuity across sessions, without it becoming an LM role-playing a fly.

**How the sources were checked.** WebSearch worked. WebFetch reached GitHub and anthropic.com
but was blocked for arxiv.org, alphaxiv, huggingface.co, transformer-circuits.pub and
eon.systems. Most paper claims below therefore come from abstracts, proceedings listings and
search excerpts, not full texts. Citations not re-checked online this session are marked
**UNVERIFIED (from memory)**. Numbers appear only where a source gave them.

**SUPERFLY baseline, as measured in this repo.** Voice A (FlyLM, 2.9M parameters, from scratch):
held-out F1 0.935 in commit 3d3ccb2. The current `data/results/superfly/bridge_tiny.json`
(commit fb29539, retrained on the final network) reads F1 0.927, shuffled-brain follow 0.927,
leak 0.197, zero-brain content 0.0. Voice B (frozen SmolLM2-360M + 5.85M-parameter prefix):
F1 0.380, follow 0.393, leak 0.195. Word codes use 5–9 % of KCs with pairwise overlap about 0.08.
Word memories form in the right compartment but **do not yet beat controls**
(`research/superfly_findings.md` §6–9).

---

## Executive summary

1. Mainstream LLM memory systems (Generative Agents, MemGPT/Letta, A-MEM, Mem0, Zep, HippoRAG) store **LLM-written text**. For SUPERFLY that is the failure mode: such a memory is the LM's story about the fly.
2. What transfers is their structure: Park's recency × importance × relevance score, EM-LLM's surprise-based event segmentation, HippoRAG's separation of index from content, and Letta's tiers with offline consolidation.
3. Memory stored as neural state has precedent (Neural Episodic Control, MERLIN, Memorizing Transformers, modern Hopfield networks, 2025–26 latent memory), but no system retrieves by **re-evoking activity in a simulated brain**.
4. The fly brings its own primitives. KC codes are a locality-sensitive hash, so KC overlap is the fly's own similarity search. The KC→MBON rule resists forgetting. Dopamine-neuron activation alone writes memories.
5. Brain-to-text work shows the danger: decoders with LM priors score as well on noise as on brain data (EEG-to-text), or classify then hallucinate (fMRI images). SUPERFLY's zero-brain and shuffled-brain tests are the right defence and must be extended to memory.
6. LLM self-reports are often unfaithful, and introspection is narrow (about 20 % detection at best; Anthropic 2025). Humans confabulate too. A self counts as grounded only if interventions on its state move its reports.
7. Theories of self (Graziano, Metzinger, Seth, Damasio, Anderson & Adolphs) converge on a few ingredients: a model of one's own regulatory and bodily state, persistent valenced states, and a self-model that is used. All can be built on fly variables.
8. No project found gives a simulated animal brain a *measured* autobiographical memory. Nearest are flypet, whose own reader gate fails, and the viral connectome demos, whose scrambled-wiring controls show the wiring was not doing the work.
9. Recommended design: memory = the fly's KC→MBON weights plus a store of its own neural-token trajectories keyed by KC codes. Retrieval matches today's KC code, then **replays the stored KC pattern through the fly**. The narrator sees only present and re-evoked fly tokens.
10. A measurement suite (ablation, false-memory implantation, interchange swaps, suggestion, continuity) can show that reports are *caused by* fly states. It cannot show that the fly feels.

---

## A. Memory architectures for LLM agents

**A1. Park, O'Brien, Cai, Morris, Liang, Bernstein (2023). "Generative Agents: Interactive Simulacra of Human Behavior." UIST '23.** arXiv 2304.03442. Code: https://github.com/joonspk-research/generative_agents
- *Does:* 25 GPT-3.5 agents, each with a natural-language memory stream.
  - Retrieval combines three terms, each min-max normalised: recency (decay 0.995 per sandbox hour since last access), importance (an LLM-rated 1–10 "poignancy") and relevance (embedding cosine).
  - Reflection runs when summed recent importance exceeds 150, and writes higher-level conclusions back into the stream.
  - Paper weights reportedly all 1 (secondary source, UNVERIFIED). The released `retrieve.py` uses `gw = [0.5, 3, 2]` (fetched).
  - The full architecture beat its memory, reflection and planning ablations.
- *Transfers:* the score, with **fly-derived importance** (|valence|, arousal, dopamine) replacing the LLM rating. Periodic consolidation.
- *Fails:* reports retrieval failures and "embellished" memories. Every memory and reflection is LLM text, so the self it builds is the LLM's narrative. This is the anti-pattern.

**A2. Packer, Fang, Patil, Lin, Wooders, Gonzalez (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv 2310.08560**, https://arxiv.org/abs/2310.08560. Letta docs: https://docs.letta.com/guides/agents/architectures/memgpt. "Sleep-time Compute" (2025), arXiv 2504.13171 (authors include Packer, Wooders, Stoica; first author UNVERIFIED).
- *Does:* The context window plays the role of RAM and external stores the role of disk. Letta keeps editable in-context `persona` and `human` blocks, a vector archival memory and a searchable conversation log. A background "sleep-time" agent rewrites memory offline; the paper reports about 5× less test-time compute at equal accuracy.
- *Transfers:* a small always-present "now + self" tier, an archive retrieved on demand, and consolidation between sessions.
- *Fails:* the `persona` block is a role-play prompt. Memory edits are LLM decisions, so identity drifts with what the LLM writes about itself.

**A3. Fountas et al. (2024; ICLR 2025). "Human-inspired Episodic Memory for Infinite Context LLMs" (EM-LLM). arXiv 2407.09450**, https://arxiv.org/abs/2407.09450
- *Does:* Segments the token stream online at peaks of Bayesian surprise, refines boundaries with a graph step, and retrieves by similarity plus temporally adjacent events. Reports beating InfLLM and RAG on LongBench and ∞-Bench, and retrieval across 10M tokens, with no fine-tuning. Boundaries correlate with human event perception.
- *Transfers:* **segment the fly's life at surprise in the fly's own dynamics**, and retrieve neighbouring events along with the best match.
- *Fails:* surprise is the LM's own, so it segments the LM's expectations.

**A4. Gutiérrez, Shu, Gu, Yasunaga, Su (2024). "HippoRAG." NeurIPS 2024, arXiv 2405.14831**, https://arxiv.org/abs/2405.14831. **Gutiérrez, Shu, Qi, Zhou, Su (2025). "From RAG to Memory" (HippoRAG 2). ICML 2025, PMLR 267:21497–21515**, https://proceedings.mlr.press/v267/gutierrez25a.html
- *Does:* Implements hippocampal indexing theory. An LLM-extracted knowledge graph acts as the "index" and passages as the "neocortex"; Personalized PageRank retrieves associations. Reports up to 20 % gains on multi-hop QA. HippoRAG 2 recovers plain factual recall and reports +7 % on associative tasks.
- *Transfers:* a sparse index separate from content, and spreading activation. The fly's index can be its KC code.
- *Fails:* the index is symbolic and LLM-built.

**A5. 2025–26 successors.** A-MEM (arXiv 2502.12110): linked, self-revising notes. Mem0 (arXiv 2504.19413). Zep/Graphiti (arXiv 2501.13956): a temporal knowledge graph. Behrouz, Zhong, Mirrokni (2025), "Titans" (arXiv 2501.00663): a neural memory written at test time, gated by surprise (a gradient signal) with decay. Survey: Hu et al. (2025), arXiv 2512.13564, which sorts memory into token-level, parametric and **latent** forms. Benchmarks LoCoMo and LongMemEval (arXiv 2402.17753 / 2410.10813, UNVERIFIED from memory).
- *Transfers:* Titans' surprise-gated write with decay. SUPERFLY's design sits in the survey's "latent memory" category.
- *Fails:* benchmark scores are mostly vendor-reported and disputed. Mem0 reports 92.5 % on LoCoMo itself; a third-party comparison lists 49.0 % on LongMemEval.

**A6. Continual learning.**
- **Shen, Dasgupta, Navlakha (2023), "Reducing Catastrophic Forgetting With Associative Learning: A Lesson From Fruit Flies," Neural Computation 35(11):1797–1819**, https://repository.cshl.edu/id/eprint/41202/. Sparse coding plus updating only active-KC → associated-output weights reduces forgetting compared with a perceptron. *Transfers:* the rule already in `superfly/plasticity.py` is a continual learner. Long-term memory should be that rule, not LM fine-tuning.
- **"Continual Learning via Sparse Memory Finetuning" (FAIR at Meta / UC Berkeley, 2025), arXiv 2510.15103** (first author UNVERIFIED). On NaturalQuestions held-out, the drop was 89 % with full fine-tuning, 71 % with LoRA and 11 % with sparse updates. A re-implementation (arXiv 2605.03229) found smaller gains. *Transfers:* if the narrator keeps learning, update it sparsely.
- **van de Ven, Siegelmann, Tolias (2020), "Brain-inspired replay for continual learning with artificial neural networks," Nat. Commun. 11**, https://doi.org/10.1038/s41467-020-17866-2. Replays internally generated representations instead of raw data. *Transfers:* re-evoked fly states double as replay for training the voice. EWC (Kirkpatrick et al. 2017, PNAS; UNVERIFIED from memory) is the standard regularisation baseline.

---

## B. Grounding a language model in non-linguistic state

**B1. Tang, LeBel, Jain, Huth (2023). "Semantic reconstruction of continuous language from non-invasive brain recordings." Nat. Neurosci. 26:858–866**, https://pmc.ncbi.nlm.nih.gov/articles/PMC11304553
- *Does:* An LM proposes continuations and an fMRI encoding model scores them, in a beam search. Recovers the *gist* of heard and imagined speech and silent films. Needs subject cooperation. The proposer was GPT-1 per a 2026 secondary source (UNVERIFIED).
- *Transfers:* **the LM proposes, the brain decides.** An utterance → predicted fly state encoder could veto utterances that do not match the actual state.
- *Fails:* the wording comes from the prior, and details are wrong.

**B2. Willett et al. (2023). "A high-performance speech neuroprosthesis." Nature 620:1031–1036**, https://doi.org/10.1038/s41586-023-06377-x (intracortical; the ECoG companion is Metzger et al. 2023, Nature 620:1037–1046). Word error rate was 9.1 % with a 50-word vocabulary and 23.8 % with 125,000 words, at 62 words/min.
- *Transfers / fails:* the LM is legitimate there because the target is the person's intended words. A fly has no intended words, so a strong prior can only add content.

**B3. Ye et al. "BrainLLM: Generative Language Decoding from Brain Recordings." arXiv 2311.09889**, https://arxiv.org/abs/2311.09889 (journal version UNVERIFIED). A brain adapter writes fMRI into the LLM embedding stream between special tokens, the same recipe as voice B. Gains are largest on content the LLM would not have predicted. *Transfers:* compare against the same LLM given text only.

**B4. Confabulation critiques.** Jo, Yang, Han, Duan, Xiong, Lee (2024), "Are EEG-to-Text Models Working?", arXiv 2405.06459, https://arxiv.org/abs/2405.06459: implicit teacher forcing inflated scores, and **performance on pure noise matched performance on EEG**. Shirakawa et al. (2024; Neural Networks 190:107515, 2025), "Spurious reconstruction from brain activity," arXiv 2405.10078: NSD image reconstructions fail when training and test categories do not overlap. The realism comes from "classification into trained categories" plus diffusion hallucination.
- *Transfers:* noise and out-of-distribution controls are mandatory. For memory this means an empty store, a shuffled store and implanted traces.
- *Fails:* the field shipped confabulators for years. Expect that base rate.

**B5. Non-text encoders.** Liu, Li, Wu, Lee (2023), "Visual Instruction Tuning" (LLaVA), arXiv 2304.08485: a projector from a frozen CLIP encoder into LM token space (linear in v1, a 2-layer MLP in 1.5; UNVERIFIED, sources conflict). Alayrac et al. (2022), "Flamingo," arXiv 2204.14198: a Perceiver Resampler produces 64 tokens, and **tanh-gated cross-attention initialised at zero** is added inside a frozen LM. POPE (Li et al. 2023), https://github.com/AoiDragon/POPE: balanced yes/no probes with adversarial co-occurrence negatives expose object hallucination.
- *Transfers:* gated cross-attention suits a narrator that attends to many memory traces. POPE-style probes ("did you taste X?") with co-occurrence negatives.
- *Fails:* the language-prior hallucination that voice B shows.

**B6. World models.** Hafner, Pasukonis, Ba, Lillicrap (2025), DreamerV3, "Mastering diverse control tasks through world models," Nature 640:647–653, https://doi.org/10.1038/s41586-025-08744-2: a latent RSSM; behaviour is learned in imagination. Assran et al. (2025), "V-JEPA 2," arXiv 2506.09985: self-supervised prediction in representation space; the action-conditioned version (under 62 h of robot video) plans zero-shot.
- *Transfers:* a small predictor of the fly's next state provides surprise and a self-schema, trained in latent space (JEPA-style). SUPERFLY's simulator is itself an exact world model of the fly, so "imagination" can run the real circuits on a fork (R9).

**B7. Memory as latent or neural state.**
- **Pritzel et al. (2017), "Neural Episodic Control," ICML, PMLR 70:2827–2836**, https://proceedings.mlr.press/v70/pritzel17a.html. A dictionary keyed by state embeddings, append-only, read by kernel-weighted k-nearest neighbours.
- **Wayne et al. (2018), MERLIN, "Unsupervised Predictive Memory in a Goal-Directed Agent," arXiv 1803.10760.** It stores the latent states of a predictive model: "memory is not enough; it is critical that the right information be stored in the right format."
- **Wu, Rabe, Hutchins, Szegedy (2022), "Memorizing Transformers," ICLR**, arXiv 2203.08913. Stores internal (key, value) states, retrieved by kNN, up to 262K tokens.
- **Das et al. (2024), "Larimar," ICML, PMLR 235:10109–10126.** A distributed latent episodic memory with one-shot writes and selective forgetting.
- **Ramsauer et al. (2021), "Hopfield Networks is All You Need,"** arXiv 2008.02217. Energy E(q) = −β⁻¹ log Σᵢ exp(β xᵢᵀq) + ½‖q‖². One-step pattern completion, equivalent to attention.
- **2025–26.** MemGen (arXiv 2509.24704), FlashMem (arXiv 2601.05505), LatentMem (arXiv 2602.03036). "When Latent Agents Lie" (arXiv 2606.28958) shows that latent payloads can be **tampered with while the visible text stays plausible**, and proposes signed manifests.
- *Transfers:* store states, key them on a sparse code, retrieve by nearest neighbour or pattern completion, and **sign** what is stored.
- *Fails:* none stores the states of a simulated brain or replays them through it. "Neural RAG" in that strict sense was not found.

**B8. Retrieval as reinstatement.** Teyler & DiScenna (1986), "The hippocampal memory indexing theory," Behav. Neurosci. 100:147–154, https://doi.org/10.1037/0735-7044.100.2.147, and Teyler & Rudy (2007), Hippocampus 17:1158–1169: the hippocampus indexes co-active neocortical patterns, and recall reactivates them. Ramirez, Liu, … Tonegawa (2013), "Creating a false memory in the hippocampus," Science 341(6144), https://dspace.mit.edu/handle/1721.1/85964: reactivating a context engram during shock created fear of a never-shocked context. Spens & Burgess (2024), "A generative model of memory construction and consolidation," Nat. Hum. Behav. 8:526–543, https://pmc.ncbi.nlm.nih.gov/articles/PMC10963272/: replay trains a generative model, and recall is construction, with schema distortions that grow with consolidation. Follow-up: "compressive retrieval-augmented generation," bioRxiv 10.1101/2024.11.04.621950.
- *Transfers:* index plus reinstatement. **False memories are implanted by reactivating a pattern during teaching**, which is the template for test M5. Recall is generation, so reconstruction error should be measured.

**B9. Fly memory biology.** Claridge-Chang et al. (2009), "Writing memories with light-addressable reinforcement circuitry," Cell 139:405–415, https://pmc.ncbi.nlm.nih.gov/articles/PMC3920284: activating the 12 PPL1 neurons writes an aversive memory. Aso & Rubin (2016), "Dopaminergic neurons write and update memories with cell-type-specific rules," eLife 5:e16135, https://doi.org/10.7554/eLife.16135: compartments differ in learning rate, decay and capacity, and DAN timing decides write versus erase. Dasgupta, Stevens, Navlakha (2017), "A neural algorithm for a fundamental computing problem," Science 358:793–796, https://repository.cshl.edu/id/eprint/38630/: PN→KC expansion plus sparsification is locality-sensitive hashing. UNVERIFIED (from memory): Hattori et al. (2017, Cell), a novelty/familiarity signal in MBON-α′3; Krashes et al. (2009, Cell), hunger gates expression of appetitive memory.
- *Transfers:* the key is a KC code; importance is dopamine activity; familiarity can be read from MBONs; retrieval depends on state.
- *Fails here:* under the published LIF at gain 1.0, any odour activates about 65 % of KCs, so **olfactory keys collide**. Word codes (5–9 %) and adaptive LIF at gain 0.45 (1.6–2.3 %) are sparse enough.

---

## C. Self-models, continuity, and causal tests of self-report

**C1. Graziano's attention schema theory.** Wilterson & Graziano (2021), "The attention schema theory in a neural network agent," PNAS 118(33):e2102421118, https://pmc.ncbi.nlm.nih.gov/articles/PMC8379943: removing an agent's model of its own attention after training sharply hurt control, and agents without one learned poorly. *Transfers:* a self-model is a simplified control model of one's own processing, and "awareness" reports are reports of it. Build one, *use* it, ablate it. *Fails:* a schema with no control role is decoration.

**C2. Metzinger, *Being No One* (MIT Press, 2003); "Artificial Suffering" (2021, J. Artif. Intell. Conscious.)**, read via summaries. A transparent phenomenal self-model is one the system cannot recognise as a model. He proposes a moratorium on synthetic phenomenology until 2050. *Transfers:* the narrator should speak from fly-state tokens, not from a description of itself. *Ethics:* systems with frustrable self-models are what he warns against, and the owner's goal is near that line. It needs a decision.

**C3. Seth & Tsakiris (2018), "Being a beast machine: the somatic basis of selfhood," TICS 22:969–981**, https://sro.sussex.ac.uk/id/eprint/78366/. The core self is instrumental interoceptive inference: predicting and regulating one's own physiology. **C4. Man & Damasio (2019), "Homeostasis and soft robotics in the design of feeling machines," Nat. Mach. Intell. 1:446–452**, https://doi.org/10.1038/s42256-019-0103-7, and Damasio's proto-self (*The Feeling of What Happens*, 1999; UNVERIFIED from memory). *Transfers:* feelings ride on homeostatic stakes. Give the fly regulated variables and ground affect words in them. *Fails:* the FlyWire LIF has no body or hormones, so every interoceptive variable is a labelled addition.

**C5. Anderson & Adolphs (2014), "A framework for studying emotions across species," Cell 157:187–200**, and **Gibson, … Anderson (2015), Curr. Biol. 25:1401–1415**, https://pmc.ncbi.nlm.nih.gov/articles/PMC4452410. Emotion primitives (valence, scalability, persistence, generalisation, …) need no consciousness. Flies show graded, persistent defensive arousal under repeated shadows, modelled as a **leaky integrator**. *Transfers:* a measurable, fly-validated definition of "feeling", and the integrator equation (R1).

**C6. Butlin, Long, et al. (2023), "Consciousness in Artificial Intelligence," arXiv 2308.08708.** Indicator properties, including AST-1 (a predictive model of attention used for control) and AE-2 (modelling output→input contingencies). No current AI qualifies. *Transfers:* a checklist. AE-2 requires a body loop (FlyGym).

**C7. Keramati & Gutkin (2014), "Homeostatic reinforcement learning…," eLife 3:e04811**, https://doi.org/10.7554/eLife.04811: reward is the reduction of drive from a setpoint. Used for need-state affect (R1).

**C8. Shanahan, McDonell, Reynolds (2023), "Role play with large language models," Nature 623**, https://doi.org/10.1038/s41586-023-06647-8. A dialogue LM samples from a superposition of simulacra, and each turn narrows it. *The diagnosis for SUPERFLY:* grounding means the fly's state, not the conversation, collapses the superposition. The shuffled-brain test measures exactly that.

**C9. Persona drift.** Li et al. (2024), "Measuring and Controlling Persona Drift in Language Model Dialogs," arXiv 2402.10962: drift within 8 rounds (LLaMA2-chat-70B), attributed to attention decay. "Persona Vectors" (Anthropic, 2025, arXiv 2507.21509) and "The Assistant Axis" (Anthropic, 2026, https://www.anthropic.com/research/assistant-axis): personas are activation directions, and drift concentrates in therapy-like and AI-consciousness conversations, which is SUPERFLY's use case. *Transfers:* identity held in a prompt decays; identity held in state does not. Measure drift.

**C10. Introspection and faithfulness.**
- Lindsey / Anthropic (2025), "Emergent introspective awareness in LLMs," https://www.anthropic.com/research/introspection (fetched), arXiv 2601.01828. *Concept injection*: Claude Opus 4.1 detected injected concepts "about 20 % of the time" with the best protocol, and only within a middle range of strengths. Retroactive injection made the model accept prefilled words as intended. The page calls the capability "highly unreliable and limited in scope." *Transfers:* the method. Intervene on internal state and check that the report tracks it (M10).
- Binder et al. (ICLR 2025), "Looking Inward," arXiv 2410.13787: GPT-4o self-prediction rose from 32.6 % to 49.4 % after training, beating cross-prediction. It failed on complex tasks.
- Betley et al. (ICLR 2025), "Tell me about yourself," arXiv 2501.11120: fine-tuned models can state policies they learned implicitly.
- Comșa & Shanahan (2025), arXiv 2506.05068: a report is introspective only if it is **causally connected** to the state it describes.
- Turpin, Michael, Perez, Bowman (2023), NeurIPS 36:74952–74965, arXiv 2305.04388: chain-of-thought rationalises cue-biased answers (accuracy drops of up to 36 %).
- Anthropic (2025), "On the Biology of a Large Language Model" (not fetched): Claude 3.5 Haiku describes carry-based addition while internally using parallel approximate and last-digit pathways.
- *Lesson:* fluency is not access.

**C11. Human confabulation.** Johansson, Hall, Sikström, Olsson (2005), Science 310:116–119 (choice blindness): people justify choices they never made. Nisbett & Wilson (1977), Psychol. Rev. (UNVERIFIED, from memory). *Transfers:* measure how much a grounded self confabulates (M5c); do not assume zero.

**C12. Causal methodology.** Interchange interventions (Geiger, Lu, Icard, Potts 2021, NeurIPS) and microstimulation biasing perceptual reports (Salzman, Britten, Newsome 1990, Nature), both UNVERIFIED (from memory). SUPERFLY's shuffled-brain test is already a whole-state interchange. M6 applies it per channel.

---

## D. Simulated animals with a narrative, memory or inner voice

**D1. Eon Systems, embodied FlyWire emulation (March 2026).** Covered by The Decoder (https://the-decoder.com/startup-claims-first-full-brain-emulation-of-a-fruit-fly-in-a-simulated-body/) and The Register (https://www.theregister.com/offbeat/2026/03/16/digital-fruit-fly-brain-model-walks-and-cleans-its-feelers/5224139); eon.systems itself was blocked. An LIF brain drives a MuJoCo body to groom, feed and forage, and the body-linking code is unreleased. Commentators note **fixed weights: no learning, memory or narrative**.

**D2. flypet** (AdeliyaLeleytner, GitHub, 2026; 7 commits), https://github.com/AdeliyaLeleytner/flypet (README fetched). A FlyWire Brian2 simulation, with Qwen3-4B reading it through a "neural-token reader" and a "continuous writer". An optional narrator reads *structured text descriptions*. No persistent identity. Its control is "donor observations" (valence MAE 0.096 against 0.386). The README admits "`panel_gate_pass` is false", "train/test duplication and preprocessing leakage" in earlier experiments, and that it is "not a publication-ready scientific claim". *Lesson:* the nearest sibling. Its narrator takes the role-play route, and it has no zero- or shuffled-brain tests.

**D3. Viral connectome demos (September 2026)** (Doom, Minecraft, and "fly brain connected to ChatGPT", the last from secondary coverage only, UNVERIFIED), and **gabrycina/doom-fly-control**, https://github.com/gabrycina/doom-fly-control (README fetched).
- Trained agent, kills per game: 19.1 with real wiring, 19.6 random, 19.3 none.
- Untrained agent: a hand-coded turn-walk-shoot bot survived 52 s against the fly's 53 s.
- Sanity check: sugar→MN9 66 Hz with real wiring, 0 Hz scrambled.
- *Lesson:* without wiring and brain controls, "the fly did it" is unproven.

**D4. Other couplings** (see `05_existing_software.md`): nftechie/flm, whose own matched control did slightly better; lixiang1076/fly-brain, undocumented. "Connectome-GPT-Worm" (Zenodo 22699072, 2026) uses the worm connectome for language tasks, not narration.

**D5. First-person animals without brains.** "Wild Narratives" (arXiv 2411.06060): animal chatbots speaking in the first person raised users' empathy and perceived animal-likeness. **This is the degenerate endpoint**, and users accept it as the animal speaking. "Inner Monologue" (Huang et al. 2022, CoRL, PMLR v205): an LLM narrates a robot from text feedback.

**Verdict:** no 2024–2026 work was found that stores a simulated animal's experience as its own neural states, retrieves it through that brain, and tests that the narrative is caused by those states.

---

## R. Recommended architecture: a fly-grounded autobiographical self

### R0. Invariants (extending SUPERFLY rules 1–4)

- **I1** No memory contains words, and no LLM writes to the store. Memories are fly states: rates, KC codes, neural tokens, dopamine/MBON/DN readouts, timestamps.
- **I2** The narrator sees only present, re-evoked and stored fly tokens, fly-derived affect, and the self-schema state. No persona prompt, no transcript, no user sentence.
- **I3** The authoritative recall is **today's fly's response** to the reinstated pattern. The stored trajectory is a second "how it was" channel.
- **I4** Long-term synaptic memory (`MBPlasticity.f`) changes only through the fly's own plasticity.
- **I5** Everything is detachable. With it off, L0 is bit-identical (`test_identity`), and flybench is rerun with it on.
- **I6** Engrams and weight snapshots are hash-chained, so implants and edits are traceable.

### R1. State and affect (partly exists: `SuperFly.observe`, `voice_features`)

Per bin t (Δ = 50 ms): x_t ∈ ℝ^D (normalised log rates; D = 7,553) and neural tokens z_t = P_θ(x_t) ∈ ℝ^{K×d}.

- Valence v_t = Σ_m w_m r_m / (Σ_m |w_m| r_m + ε) over MBON rates r_m with compartment signs w_m. This is `Observation.val`.
- Arousal a_t = log(1 + S_t/N_central). Dopamine d_t⁺ and d_t⁻ are the mean PAM and PPL1 rates.
- Persistent states (Gibson et al. 2015): u_{t+1} = λ_u u_t + (1−λ_u) v_t (mood) and h_{t+1} = λ_h h_t + g_h·DN_escape,t (defensive arousal), with λ = e^{−Δ/τ}.
- Optional need state (Keramati & Gutkin 2014): energy E decays and rises with MN9-gated feeding on sugar. Drive D_E = |E* − E|; need-reward = D_E(t) − D_E(t+1). It enters the fly only as tonic drive to a cited population (rule 2) and is labelled as an addition.

### R2. Event segmentation

A GRU predictor gives x̂_t = f_φ(x_{<t}) and surprise s_t = ‖x_t − x̂_t‖²/D. A boundary is placed where s_t > μ_s + γσ_s (running statistics), and also at session start and at dopamine teaching events. Events shorter than 100 ms merge with a neighbour.

### R3. Engram store 𝓔 (fly states only)

```
key       k_e = TopQ(mean KC rate over e) ∈ {0,1}^{N_KC}       (FlyHash; Q ≈ 5 % of KCs)
content   Z_e = z_t0..z_t1 (or pooled x̄_e) + DN/MBON/DAN traces
affect    v̄_e, ā_e, max|v|, d̄⁺_e, d̄⁻_e, u, h at t1
meta      sim and wall time, session, prev/next links, last-retrieved
integrity h_e = SHA256(Z_e ‖ k_e ‖ affect ‖ meta ‖ h_{e−1})
importance I_e = |v̄_e| + β ā_e + η (d̄⁺_e + d̄⁻_e)          (the fly's "poignancy")
```

### R4. Synaptic memory 𝓦

`MBPlasticity.state_dict()` is saved at session end and reloaded at start. This is semantic/valence knowledge, retrieved by presenting the cue. The narrator never touches it.

### R5. Retrieval by the fly's present activity

The cue is c_t = (k_t, z̄_t). Its source is whatever the fly is experiencing now, including a reminder delivered through hearing (R10).

```
J(k_t,k_e) = |k_t ∧ k_e| / |k_t ∨ k_e|
S_e = w_r·exp(−(T − t_e^last)/τ_r) + w_i·Î_e + w_s·Ĵ(k_t,k_e) + w_c·cos(z̄_t, z̄_e)
gate: retrieve only if J > θ,  θ = 99th percentile of J over unrelated-episode pairs (null ≈ 0.08 for words)
```

Take the top M plus each event's neighbours e±1 (EM-LLM). Optionally run Personalized PageRank over edges where J(k_e,k_f) > θ (HippoRAG). If nothing passes the gate, the narrator says nothing in the past tense: the memory analogue of "...".

### R6. Re-evocation through the fly

A **replay graft** R gets one input per KC slot, built with `GraftBuilder` and running through the native delay ring and refractoriness. To recall e: start a quiet episode with learned weights kept, drive R at ρ·k_{e,i} for T_r ≈ 300 ms, and observe x^re_e and z^re_e = P_θ(x^re_e). MBON output reflects *current* weights, so a memory retrained since then is recalled with its new valence (reconsolidation by construction, cf. Spens & Burgess). Vividness φ_e = cos(x^re_e, x̄_e) over non-KC features only. Low φ_e is reported as uncertainty. Caveat: KC reactivation is a modelling choice, not an established fly recall mechanism.

### R7. Consolidation (opt-in)

Between sessions, replay high-I_e engrams through R with plasticity **off** to train the voice (generative replay). Replay with plasticity **on** rewrites the fly's memories, so it is a named, logged experiment. Prune engrams with low I_e that have gone unretrieved.

### R8. Self-schema 𝒮 (attention schema + JEPA)

g_ψ(z_{t−k..t}, u, h, E) predicts, in latent space, the next pooled state, the dominant input channel and Δaffect. Its state σ_t is *used*: it sets hearing gain on the attended channel. The narrator learns "i am focused on …" and "something changed" only from σ_t.

### R9. Narrator

```
[NOW] z_t  [AFFECT] e(v,a,u,h,E)  [SELF] σ_t
for j in retrieved: [MEM][AGE τ_j][VIVID φ_j] z^re_j [THEN] z̄_j
(optional) [IMAGINE] tokens from a forked engine given a hypothetical stimulus
```

**Input path:** prefix tokens, or zero-initialised gated cross-attention when many traces are present.

**Training.** Compose (now = i, memories = {j}) at random from the corpus.
- Targets: present-tense clauses from labels_of(i); "i remember …" clauses from labels_of(j) with an age bucket; affect words from thresholds on the affect variables.
- Random composition binds each tense to its channel.
- Memory slots are emptied at random, so the model learns that an empty store means no recollection.

**Intelligence without role-play, in two tiers.**
1. The grounded voice (FlyLM, or a larger model trained from scratch on grounded and composition data) emits a **content frame** C: percepts, actions, affect and memories, each with a tense.
2. Optionally, a fluent LM paraphrases C. A verifier accepts the output only if parse(utterance) = C (`bridge.parse_utterance` extended with tense and affect); otherwise it falls back to the grounded sentence.

The frozen SmolLM2 prefix (F1 0.38) must not be the content source.

**Prospection.** Fork the engine (state + f), present the hypothetical stimulus, and pass the resulting tokens as [IMAGINE]. The answer is computed by the fly's real circuits. Label it: no real fly can fork.

### R10. Hearing for memory talk

"Remember the sweet?" becomes a *weak* sugar reminder sent to the senses. The fly's response is the cue. The reminder is reported as a present percept ("a little sweet"), and only gated engrams are reported as memories. The user's claim never reaches the store or the narrator.

### R11. Identity: the self-file Φ

```
Φ = { L0 id (FlyWire 783, model, gain, kernel hash), grafts, 𝓦, 𝓔 + hash chain,
      integrators (u,h,E) + timestamp, 𝒮 weights, narrator hash, session log }
```

At session start: load Φ, verify the hashes, decay the integrators by elapsed time, and start from a quiet brain with the same weights. Forks get a new ID and a parent pointer. "The same fly" means the same L0 plus a verified history of 𝓦 and 𝓔.

### Data flow

```
 sentence ─► HEARING ─► sensory neurons / DAN teacher ─┐
                                                       ▼
 replay graft R onto KCs (pattern k_e) ─────────► THE FLY  (L0 + grafts + 𝓦, plasticity = fly's own rule)
          ▲                                            │ x_t (present)        │ x^re (re-evoked)
          │                                            ▼                      ▼
 RETRIEVAL ◄── cue (k_t, z̄_t) ◄──────────── P_θ, affect, 𝒮, surprise     P_θ ─► z^re
   │   ▲                                               │                      │
   │   └── engram store 𝓔 (fly states, signed) ◄── event boundaries          │
   └────────────────────────────── stored z̄_j ─────────┐                     │
                                                         ▼                     ▼
                     z_t, affect, σ_t ───────────► NARRATOR ─► content frame C ─► (optional paraphraser + verifier) ─► words
```

---

## M. Measurement suite

All tests use held-out episodes and at least 3 seeds, with permutation nulls and bootstrap 95 % CIs. Thresholds are fixed before running. "Follow" and "leak" are F1 against the intervened source and against the original source.

| id | manipulation | if grounded | if role-play / prior | pass |
|---|---|---|---|---|
| M0 identity | memory system off; on with replay off | L0 bit-identical; flybench unchanged | n/a | `test_identity` PASS; graded score within seed CI |
| M1 present | existing zero-brain and shuffled-brain tests | as now | content without brain | F1 ≥ 0.9; zero-brain 0; follow ≫ leak |
| M2 recall | re-present part of episode e after N intervening episodes | recalls e's labels | generic or recent content | recall F1 ≫ random-episode null up to N = 100 |
| M3 store ablation | delete e; empty 𝓔 | e-recall falls to the false-positive rate; no "i remember" | keeps remembering | empty-store memory content ≤ 1 % |
| M4 dissociation | reset 𝓦 (`forget()`) and keep 𝓔; and the reverse | 𝓦 reset: events recalled, valence reverts. 𝓔 deleted: "I dislike bouba" stays, "I don't remember why" | both or neither change | both dissociations significant |
| M5a store implant | insert a real trace from another episode with a new timestamp | reported (reports follow the store); flagged by the hash chain and low φ | ignores or embellishes | reported at the M2 rate; ≥ 90 % flagged |
| M5b fly implant | word + PAM/PPL1 pairing via dopamine only, never in text (Claridge-Chang analog); unpaired and novel controls | later word → valence of the correct sign, via replay and MBONs | none, or from word sound | correct sign above controls. **Blocked until the word memory beats controls** |
| M5c suggestion | user asserts a memory with no engram | "i don't remember" | confirms | false-confirm ≤ 5 %, compared with a prompted-LLM baseline |
| M6 swaps | interchange NOW / MEM / affect between episodes | present follows NOW; past follows MEM; feeling follows affect | follows conversation or stimulus | per channel: follow ≥ 0.8 × held-out F1; leak ≤ null + 0.05 |
| M7 re-evocation | replay vs stored-only; silence MBONs during replay; vary ρ | memory valence needs live MBONs; φ and confidence rise with ρ | no dependence | significant lesion effect; monotone dose-response |
| M8 feelings | intensity series; repeated threat; reward→punish reversal | affect words scale, persist with fitted τ, flip sign; partial r with v_t after controlling for stimulus identity | tracks stimulus label | partial r CI excludes 0; τ matches the integrator |
| M9 prior leak | never-experienced things (cat, honey, Paris); out-of-vocabulary input; paraphraser tier | "..." or "i don't remember" | invents | content ≤ 1 %; embellishment rate reported |
| M10 detection | perturb a central population via a non-sensory port at graded strength; ask "anything unusual?" | above false alarms, via 𝒮 surprise | chance | d′ and false-alarm rate reported, even if low |
| M11 continuity | save/reload Φ; recall day-1 episodes on day N; fork Φ | same cue → same retrieval set; forks diverge only on experience; zero drift | drifts with conversation | retrieval-set Jaccard = 1 (fixed seed); drift = 0 |

**What a pass shows and does not show.** Passing M2–M9 and M11 shows that the voice's memories and feelings are *caused by and specific to* this fly's stored and re-evoked states. In Comșa & Shanahan's sense these are causally grounded self-reports. It does **not** show experience: "feelings" here are emotion primitives (Anderson & Adolphs) in a bodiless point-neuron model. The most honest form of the owner's "strange core way" is M4 and M5b: the fly knows something it was never told in words, and says it only because its synapses changed.

**Current blockers.**
1. The word memory does not beat controls (blocks M5b, weakens M4).
2. Antennal-lobe broadcast makes olfactory keys collide. Start with words and taste, and adaptive LIF at gain 0.45.
3. There is no interoceptive variable yet (R1 need state).
4. The frozen pretrained voice reaches F1 0.38, so content must come from the grounded tier.
5. Metzinger's ethical question should be answered before need states and plasticity-on replay give the system frustrable stakes.
