# Grounded self architectures: memory, identity and a fly-grounded autobiographical voice

Survey made 2026-10-08 for SUPERFLY. The question: how to build a voice that remembers being
*this* fly, with continuity across sessions, without it becoming a language model role-playing
a fly.

**How the sources were checked.** WebSearch worked. WebFetch could reach GitHub and
anthropic.com but was blocked for arxiv.org, alphaxiv, huggingface.co, transformer-circuits.pub
and eon.systems. So most paper claims below come from abstracts, proceedings listings and search
excerpts, not full texts. A few classic citations were not re-checked online this session and are
marked **UNVERIFIED (from memory)**. Numbers are reported only where a source gave them.

**SUPERFLY baseline, as measured in this repo.** Voice A is a 2.9M-parameter FlyLM trained from
scratch. It reached held-out F1 0.935 in commit 3d3ccb2. The current
`data/results/superfly/bridge_tiny.json` (commit fb29539, retrained on the final network) reads F1
0.927, shuffled-brain follow 0.927, leak 0.197, and zero-brain content rate 0.0. Voice B is frozen
SmolLM2-360M with a trained 5.85M-parameter neural prefix: F1 0.380, follow 0.393, leak 0.195,
zero-brain content rate 0.0. Word codes use 5–9 % of Kenyon cells with pairwise overlap about 0.08.
Word memories form in the correct compartment but **do not yet beat controls**
(`research/superfly_findings.md` §6–9).

---

## Executive summary

1. Every mainstream LLM memory system (Generative Agents, MemGPT/Letta, A-MEM, Mem0, Zep, HippoRAG 1–2) stores memories as **text written by an LLM**. For SUPERFLY that is the failure mode itself: such a diary is the LM's story about the fly.
2. What transfers is structure, not substrate: Park's recency × importance × relevance score, EM-LLM's surprise-based event boundaries with contiguity retrieval, HippoRAG's index kept apart from content (hippocampal indexing theory), and Letta's split between in-context and archival memory, with offline "sleep-time" consolidation.
3. There is precedent for memory stored as neural state: Neural Episodic Control and MERLIN key memories on state embeddings, Memorizing Transformers keep KV states, and the 2025–26 latent-memory systems (MemGen, FlashMem, LatentMem) and modern Hopfield networks do the same. None retrieves by **re-evoking activity in a simulated brain**.
4. The fly already has the right primitives. Kenyon-cell codes act as a locality-sensitive hash (Dasgupta et al. 2017), the KC→MBON learning rule resists forgetting (Shen et al. 2023), and dopamine-neuron activation alone writes memories (Claridge-Chang et al. 2009; Aso & Rubin 2016).
5. Brain-to-text work shows the main danger. Decoders with strong LM priors score as well on noise as on brain data (Jo et al. 2024), or they classify into training categories and then hallucinate the detail (Shirakawa et al. 2024/25). SUPERFLY's zero-brain and shuffled-brain tests are the right defence, and they must be extended to memory.
6. Self-report research says the same about narrators. LLM explanations are often unfaithful (Turpin et al. 2023). LLM introspection is narrow and unreliable: about 20 % detection at best (Lindsey/Anthropic 2025). Humans confabulate too (choice blindness). So a self counts as "grounded" only if interventions on its state move its reports.
7. Theories of the self (Graziano's attention schema, Metzinger's self-model, Seth's interoceptive inference, Damasio's proto-self, Anderson & Adolphs' emotion primitives) converge on a few minimal ingredients: a model of one's own regulatory and bodily state, persistent valenced internal states, and a self-model used for control. Each can be read from, or built on, the fly's own variables.
8. No project found gives a simulated animal brain a measured autobiographical memory. The nearest are flypet (a Brian2 fly read by Qwen3-4B through neural tokens, whose own reader gate fails) and the viral connectome demos. The scrambled-wiring controls in `doom-fly-control` show the wiring was not doing the work in those demos.
9. Recommended design. The fly's memory is (a) its own KC→MBON weights and (b) a store of its own neural-token trajectories, keyed by KC codes. Retrieval matches the current KC code, then **re-evokes** the stored KC pattern inside the fly through a replay graft. The narrator sees only present and re-evoked fly tokens, never text memories.
10. Proof comes from a measurement suite: ablation, false-memory implantation, interchange (swap) interventions, suggestibility probes and cross-session tests, each with a pass threshold set in advance. The suite can show that reports are *caused by* fly states. It cannot show that the fly feels anything.

---

## A. Memory architectures for LLM agents

**A1. Park, O'Brien, Cai, Morris, Liang, Bernstein (2023). "Generative Agents: Interactive Simulacra of Human Behavior." UIST '23 (ACM).** arXiv 2304.03442. Code: https://github.com/joonspk-research/generative_agents
- *What it does.* 25 GPT-3.5 agents in "Smallville". Each keeps a natural-language **memory stream** of observations. Retrieval ranks memories by recency (exponential decay over sandbox hours since last access, factor 0.995), importance (an LLM-rated integer from 1 "mundane" to 10 "poignant") and relevance (cosine similarity of embeddings), each min-max normalised. **Reflection** is triggered when the summed importance of recent events passes 150; the agent then writes higher-level conclusions back into the stream. The paper's weights are reportedly all 1; that comes from a secondary glossary, UNVERIFIED. The released `retrieve.py` uses weights `gw = [0.5, 3, 2]` for recency, relevance and importance (fetched). In interviews, the full architecture beat ablations without memory, planning or reflection.
- *Transfers.* The three-term score, with **fly-derived importance** (|valence|, arousal, dopamine activity) in place of an LLM rating. Contiguity in time. Periodic consolidation.
- *Failure modes.* The paper reports retrieval failures and "embellished" memories that agents invented around what they knew. Every memory and every reflection is LLM-written text, so the self it builds is the LLM's narrative. For SUPERFLY this is the anti-pattern.

**A2. Packer, Fang, Patil, Lin, Wooders, Gonzalez (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv 2310.08560**, https://arxiv.org/abs/2310.08560. **Letta docs**, https://docs.letta.com/guides/agents/architectures/memgpt. **"Sleep-time Compute: Beyond Inference Scaling at Test-time" (2025), arXiv 2504.13171** (authors include C. Packer, S. Wooders, I. Stoica; first author UNVERIFIED).
- *What it does.* "Virtual context management": the context window plays the role of RAM, external stores play the role of disk, and the LLM pages content in and out through tool calls. Letta keeps editable in-context "core memory" blocks (`persona`, `human`), a vector **archival memory** and a searchable **recall memory** (the conversation log). A background "sleep-time" agent rewrites the memory blocks offline. The sleep-time paper reports about 5× less test-time compute for equal accuracy on stateful math benchmarks.
- *Transfers.* Tiering: a small always-present "now + self" block, a large archive retrieved on demand, and offline consolidation between sessions.
- *Failure modes.* The `persona` block is literally a role-play prompt. Memory edits are LLM decisions, so identity drifts toward whatever the LLM writes about itself. Letta labels sleep-time agents experimental.

**A3. Fountas, Benfeghoul, Oomerjee, Christopoulou, Lampouras, Bou-Ammar, Wang (2024/2025). "Human-inspired Episodic Memory for Infinite Context LLMs" (EM-LLM). ICLR 2025; arXiv 2407.09450**, https://arxiv.org/abs/2407.09450. The author list beyond Fountas is UNVERIFIED (from memory).
- *What it does.* Segments the token stream online into "events" at peaks of Bayesian surprise, refines boundaries with a graph-theoretic step, and retrieves in two stages: similarity, then temporally adjacent events. It reports beating InfLLM and RAG on LongBench and ∞-Bench, and retrieval across 10M tokens, with no fine-tuning. Its boundaries correlate with human-perceived events.
- *Transfers.* **Segment the fly's life at surprise in the fly's own state dynamics**, and retrieve the neighbouring episodes along with the best match. This matches event-segmentation theory, where boundaries fall at transient rises in prediction error.
- *Failure modes.* It works inside one LM's KV cache. Surprise is measured by the LM's own predictions, so it segments the LM's expectations, not the world's.

**A4. Gutiérrez, Shu, Gu, Yasunaga, Su (2024). "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." NeurIPS 2024; arXiv 2405.14831**, https://arxiv.org/abs/2405.14831. **Gutiérrez, Shu, Qi, Zhou, Su (2025). "From RAG to Memory: Non-Parametric Continual Learning for LLMs" (HippoRAG 2). ICML 2025, PMLR 267:21497–21515**, https://proceedings.mlr.press/v267/gutierrez25a.html. Code: https://github.com/OSU-NLP-Group/HippoRAG
- *What it does.* Implements hippocampal indexing theory. An LLM extracts a knowledge graph that serves as the "index", passages act as "neocortex", and Personalized PageRank from the query's concepts retrieves multi-hop associations. It reports up to 20 % gains on multi-hop QA. HippoRAG 2 fixes the regression on plain factual recall and reports +7 % on associative tasks over the best embedding retriever.
- *Transfers.* Keep a sparse **index** separate from the stored content, and spread activation across linked memories. In SUPERFLY the index can be the fly's own KC codes, not LLM-extracted triples.
- *Failure modes.* The index is built by LLM extraction from text, so it is symbolic and does not exist for a fly.

**A5. 2025–26 successors and benchmarks.** A-MEM (Xu et al. 2025, arXiv 2502.12110), Zettelkasten-style linked notes that rewrite older notes. Mem0 (arXiv 2504.19413), vector store plus optional graph. Zep/Graphiti (arXiv 2501.13956), a temporal knowledge graph. Behrouz, Zhong, Mirrokni (2025), "Titans: Learning to Memorize at Test Time" (arXiv 2501.00663), a neural memory updated at inference by a surprise (gradient) signal with decay. Survey: Hu et al. (2025), "Memory in the Age of AI Agents: A Survey" (arXiv 2512.13564), which sorts memory into token-level, parametric and latent forms. Benchmarks: LoCoMo and LongMemEval (arXiv IDs 2402.17753 and 2410.10813 are from memory, UNVERIFIED).
- *Transfers.* Titans' *surprise-gated write with decay* is the right write rule. The survey's "latent memory" category is where SUPERFLY's design sits.
- *Failure modes.* Benchmark numbers in this area are mostly vendor-reported and disputed. One comparison page shows Mem0 at 92.5 % on LoCoMo by its own count and 49.0 % on LongMemEval in a third-party evaluation. All of these systems ground memory in text dialogue.

**A6. Continual learning without catastrophic forgetting.**
- **Shen, Dasgupta, Navlakha (2023). "Reducing Catastrophic Forgetting With Associative Learning: A Lesson From Fruit Flies." Neural Computation 35(11):1797–1819**, https://repository.cshl.edu/id/eprint/41202/ (preprint arXiv 2107.07617). Sparse high-dimensional coding plus updating **only the active-KC → output weights for the associated output** reduces forgetting compared with a perceptron. *Transfers:* the fly's own rule, already in `superfly/plasticity.py`, is a continual learner by design. Its long-term memory should be that rule, not LM fine-tuning.
- **"Continual Learning via Sparse Memory Finetuning" (FAIR at Meta / UC Berkeley, 2025), arXiv 2510.15103** (first author UNVERIFIED). Only memory-layer slots that respond strongly to new data, relative to pretraining data, are updated. On NaturalQuestions, held-out performance dropped 89 % with full fine-tuning, 71 % with LoRA and 11 % with sparse updates. An independent re-implementation (arXiv 2605.03229) found less forgetting but smaller gains. *Transfers:* if the narrator must keep learning, update it sparsely.
- **van de Ven, Siegelmann, Tolias (2020). "Brain-inspired replay for continual learning with artificial neural networks." Nat. Commun. 11**, https://doi.org/10.1038/s41467-020-17866-2. Replays *internal* representations generated by the network's own feedback pathways, with no stored raw data. *Transfers:* re-evocation of stored states (below) doubles as replay for training the voice.
- Kirkpatrick et al. (2017), "Overcoming catastrophic forgetting in neural networks", PNAS (EWC): UNVERIFIED (from memory). It is the standard regularisation baseline.

---

## B. Grounding a language model in non-linguistic state

**B1. Tang, LeBel, Jain, Huth (2023). "Semantic reconstruction of continuous language from non-invasive brain recordings." Nat. Neurosci. 26:858–866**, https://pmc.ncbi.nlm.nih.gov/articles/PMC11304553
- *What it does.* An fMRI encoding model scores candidate continuations proposed by an LM in a beam search. It recovers the *gist* of perceived and imagined speech and of silent videos, and needs a cooperative subject both for training and for decoding. A 2026 paper says the proposer was GPT-1 (secondary source, UNVERIFIED).
- *Transfers.* **Generate-and-score with an encoding model**: the LM proposes and the brain decides. For SUPERFLY, an encoder from utterance to predicted fly state can veto utterances whose predicted state does not match the actual state.
- *Failure modes.* The LM prior supplies the fluent wording, and the output is a gist with wrong details. Exactly what SUPERFLY's voice must not do with memories.

**B2. Willett et al. (2023). "A high-performance speech neuroprosthesis." Nature 620:1031–1036**, https://doi.org/10.1038/s41586-023-06377-x (intracortical arrays). Companion paper: Metzger et al. (2023), Nature 620:1037–1046 (ECoG). Word error rate was 9.1 % with a 50-word vocabulary and 23.8 % with a 125,000-word vocabulary, at 62 words/min. Card et al. (2024, NEJM) reported 90.2 % accuracy on day 2 with 125k words.
- *Transfers.* An RNN maps neural activity to phonemes and a language model completes words. The LM is legitimate there because the *target is the person's intended words*.
- *Failure modes.* With a large vocabulary the LM prior chooses between near-ties. For a fly there are no intended words, so a strong prior can only add content.

**B3. Ye et al. "BrainLLM: Generative Language Decoding from Brain Recordings." arXiv 2311.09889**, https://arxiv.org/abs/2311.09889. Journal version and venue UNVERIFIED. Code on Zenodo, record 14838723.
- *What it does.* A "brain adapter" projects fMRI into the LLM's embedding space between special begin/end tokens, the same recipe as SUPERFLY voice B. Gains were largest on content the LLM would not have predicted.
- *Transfers.* Its evaluation: compare against the same LLM with the text prompt alone.
- *Failure modes.* Improvements are relative to an LLM that is fluent anyway. Absolute grounding is small.

**B4. Confabulation critiques.** Jo, Yang, Han, Duan, Xiong, Lee (2024), "Are EEG-to-Text Models Working?", arXiv 2405.06459, https://arxiv.org/abs/2405.06459. Published EEG-to-text scores were inflated by implicit teacher forcing, and **models performed comparably on pure noise**. Shirakawa et al. (2024; Neural Networks 190:107515, 2025), "Spurious reconstruction from brain activity", arXiv 2405.10078. Natural Scenes Dataset (NSD) image reconstructions fail when training and test categories do not overlap. The apparent realism is "classification into trained categories" plus diffusion-model hallucination.
- *Transfers.* **Noise-input and out-of-distribution controls are mandatory**, and SUPERFLY already has them (zero-brain, shuffled-brain). They must be extended to the memory channel: empty store, shuffled store, implanted traces.
- *Failure modes.* These papers show the field shipping confabulators for years without noticing. That is the base rate to expect.

**B5. Non-text encoders feeding LMs.** Liu, Li, Wu, Lee (2023), "Visual Instruction Tuning" (LLaVA), NeurIPS 2023, arXiv 2304.08485. A frozen CLIP encoder feeds a projector that writes into LLM token space. The original was a single linear layer and LLaVA-1.5 used a 2-layer MLP (UNVERIFIED; secondary sources conflict). Alayrac et al. (2022), "Flamingo", arXiv 2204.14198. A Perceiver Resampler produces 64 visual tokens, and **tanh-gated cross-attention layers initialised at zero** sit inside a frozen LM. Li et al. (2023), POPE, https://github.com/AoiDragon/POPE. Vision-language models hallucinate objects that co-occur frequently in training.
- *Transfers.* Flamingo-style gated cross-attention is a better fit than a 4-token prefix when the narrator must attend to *many* memory traces: the gate is zero at start and is learned from the fly's data. POPE-style balanced yes/no probes, including **adversarial co-occurrence negatives**, are a direct template for "did you taste X?" probes.
- *Failure modes.* Language-prior hallucination. SUPERFLY's voice B measures the same thing: F1 0.38 against a leak of 0.195.

**B6. World models.** Hafner, Pasukonis, Ba, Lillicrap (2025), "Mastering diverse control tasks through world models" (DreamerV3), Nature 640:647–653, https://doi.org/10.1038/s41586-025-08744-2. An RSSM latent state (deterministic GRU part plus stochastic categorical part) trains actor and critic in imagination. Assran et al. (2025), "V-JEPA 2", arXiv 2506.09985. Self-supervised video prediction in representation space. The action-conditioned version was trained on under 62 hours of robot video and plans zero-shot.
- *Transfers.* (i) A small **predictor of the fly's next state** supplies surprise for event segmentation and a body/attention-schema-like self-model. (ii) Prediction in latent space (JEPA) rather than pixel or text reconstruction is the right objective for that self-model. (iii) Uniquely, SUPERFLY's simulator *is* a perfect world model of the fly, so "imagination" can run the fly's real circuits on a forked copy (§R9).
- *Failure modes.* Learned world models drift over long rollouts. The forked fly does not drift, but no real fly can do this either.

**B7. Memory stored as latent or neural states.**
- **Pritzel et al. (2017), "Neural Episodic Control," ICML, PMLR 70:2827–2836**, https://proceedings.mlr.press/v70/pritzel17a.html. A "differentiable neural dictionary" with keys = state embeddings, append-only writes and kernel-weighted p-nearest-neighbour reads. Slow keys, fast values.
- **Wayne et al. (2018), "Unsupervised Predictive Memory in a Goal-Directed Agent" (MERLIN), arXiv 1803.10760.** Writes *latent states of a predictive model* to memory. "Memory is not enough; it is critical that the right information be stored in the right format."
- **Wu, Rabe, Hutchins, Szegedy (2022), "Memorizing Transformers," ICLR**, arXiv 2203.08913. Stores internal (key, value) states and retrieves them by approximate kNN, scaling to 262K tokens.
- **Das et al. (2024), "Larimar: LLMs with Episodic Memory Control," ICML, PMLR 235:10109–10126.** A distributed latent episodic memory with one-shot write and selective forgetting. The Kanerva lineage is UNVERIFIED.
- **Ramsauer et al. (2021), "Hopfield Networks is All You Need,"** arXiv 2008.02217. A continuous modern Hopfield network with energy E(q) = −β⁻¹ log Σᵢ exp(β xᵢᵀq) + ½‖q‖². It retrieves a stored pattern from a partial cue in one update, and that update equals attention. This is the formal model of **pattern completion**.
- **2025–26 latent memory for LLM agents.** MemGen (arXiv 2509.24704) weaves generated latent tokens into reasoning. FlashMem (ACL Findings 2026, arXiv 2601.05505) builds memory from the frozen KV cache. LatentMem (arXiv 2602.03036). One 2026 study ("When Latent Agents Lie", arXiv 2606.28958) showed that latent payloads can be **tampered with while the visible text stays plausible**, and proposed signed manifests.
- *Transfers.* Store the fly's states, key them on a sparse code, retrieve by nearest neighbour or pattern completion, and **sign** stored states so implanted memories are traceable.
- *Failure modes.* None of these stores states of a *simulated brain*, and none retrieves by re-running the stored pattern through that brain. The "neural RAG" idea in the strict sense requested was not found.

**B8. The neuroscience of retrieval as reinstatement.** Teyler & DiScenna (1986), "The hippocampal memory indexing theory," Behav. Neurosci. 100(2):147–154, https://doi.org/10.1037/0735-7044.100.2.147. Teyler & Rudy (2007), Hippocampus 17(12):1158–1169: the hippocampus stores an *index* of co-active neocortical patterns, and recall reactivates them. Ramirez, Liu, … Tonegawa (2013), "Creating a false memory in the hippocampus," Science 341(6144), https://dspace.mit.edu/handle/1721.1/85964: optogenetic reactivation of a context engram during shock produced fear of a context where no shock was ever delivered. Spens & Burgess (2024), "A generative model of memory construction and consolidation," Nat. Hum. Behav. 8:526–543, https://pmc.ncbi.nlm.nih.gov/articles/PMC10963272/: replay from an autoassociative store trains a generative model, and recall is constructive, with schema distortions that *grow* with consolidation. Spens & Burgess, "Hippocampo-neocortical interaction as compressive retrieval-augmented generation," bioRxiv 10.1101/2024.11.04.621950 (journal status UNVERIFIED).
- *Transfers.* Three principles: index plus reinstatement; **false memories can be implanted by reactivating a stored pattern during teaching**, which is the template for test M5; and consolidated recall *is* generation, so reconstruction errors should be measured, not denied.

**B9. Fly memory biology that SUPERFLY can use directly.** Claridge-Chang et al. (2009), "Writing memories with light-addressable reinforcement circuitry," Cell 139:405–415, https://pmc.ncbi.nlm.nih.gov/articles/PMC3920284: activating the 12 PPL1 dopamine neurons suffices to write an aversive odour memory. Aso & Rubin (2016), "Dopaminergic neurons write and update memories with cell-type-specific rules," eLife 5:e16135, https://doi.org/10.7554/eLife.16135: compartments differ in learning rate, decay and capacity, and the timing of DAN activation decides write versus erase. Dasgupta, Stevens, Navlakha (2017), "A neural algorithm for a fundamental computing problem," Science 358:793–796, https://repository.cshl.edu/id/eprint/38630/: projection from projection neurons to Kenyon cells plus sparsification is a locality-sensitive hash, so **KC overlap is the fly's own similarity search**. Not re-checked online, UNVERIFIED (from memory): Hattori et al. (2017, Cell), a novelty/familiarity signal in the MBON-α′3 compartment; Krashes et al. (2009, Cell), hunger (NPF, PPL1 MB-MP) gates *expression* of appetitive memory.
- *Transfers.* Retrieval key = KC code. Importance = dopamine activity. A familiarity signal native to the fly ("this feels familiar") can be read from MBONs. Retrieval is state-dependent: need states gate what is expressed.
- *Failure modes in SUPERFLY.* Under the published LIF at gain 1.0, any odour drives about 65 % of KCs (antennal-lobe broadcast), so **KC keys for olfactory episodes would collide**. Word-lobe codes (5–9 %) and adaptive LIF at gain 0.45 (1.6–2.3 % of KCs) are sparse enough.

---

## C. Self-models, continuity, and proving self-reports are caused

**C1. Graziano's attention schema theory.** Wilterson & Graziano (2021), "The attention schema theory in a neural network agent: Controlling visuospatial attention using a descriptive model of attention," PNAS 118(33):e2102421118, https://pmc.ncbi.nlm.nih.gov/articles/PMC8379943. A deep-Q agent with an internal model of its own attention spotlight learned a catch task. Removing the schema after training sharply degraded performance, and agents without it from the start learned poorly. Graziano & Webb (2015), Front. Psychol.: UNVERIFIED (from memory).
- *Transfers.* A self-model is a simplified *control model of one's own processing*. Reports of "awareness" are reports of that model. That makes the model operational: build a small schema of which of the fly's channels dominates its central brain, use it, and ablate it.
- *Failure modes.* A schema that is never used for control is decoration. AST explains self-reports; it does not license claims of experience.

**C2. Metzinger, *Being No One: The Self-Model Theory of Subjectivity* (MIT Press, 2003).** "Artificial Suffering" (2021, J. Artif. Intell. Conscious.) argues for a moratorium on synthetic phenomenology until 2050. Summaries were found but the primary text was not read.
- *Transfers.* A phenomenal self-model (PSM) is a model of the whole organism that is *transparent*: the system cannot recognise it as a model. For SUPERFLY: the narrator should speak from the fly-state tokens, not from a description of itself as "a simulated fly".
- *Failure modes and ethics.* Metzinger argues that building systems with self-models that can be frustrated risks artificial suffering. The owner's goal, something that feels "in some strange core way", is close to what he warns against. That deserves an explicit decision, not drift.

**C3. Seth & Tsakiris (2018), "Being a beast machine: the somatic basis of selfhood," TICS 22(11):969–981**, https://sro.sussex.ac.uk/id/eprint/78366/. Seth (2013), TICS, "Interoceptive inference, emotion, and the embodied self."
- *Transfers.* The core self is *instrumental interoceptive inference*: predicting and regulating one's own physiological variables. The most defensible "core way" for SUPERFLY to feel is therefore to give the fly **regulated internal variables** (energy, satiety, threat arousal) whose control the brain participates in, and to ground affect words in them.
- *Failure modes.* The FlyWire LIF has no body, hemolymph or hormones. Any interoceptive variable is an addition and must be labelled as one.

**C4. Damasio: proto-self** (*The Feeling of What Happens*, 1999, UNVERIFIED from memory), the moment-to-moment neural mapping of body state. **Man & Damasio (2019), "Homeostasis and soft robotics in the design of feeling machines," Nat. Mach. Intell. 1:446–452**, https://doi.org/10.1038/s42256-019-0103-7: build homeostasis and vulnerability first, and feeling analogues follow.
- *Transfers.* Same lesson as Seth: feelings ride on homeostatic stakes. Neural tokens with no stakes give percept reports, not feelings.

**C5. Anderson & Adolphs (2014), "A framework for studying emotions across species," Cell 157(1):187–200.** Gibson, … Anderson (2015), "Behavioral responses to a repetitive visual threat stimulus express a persistent state of defensive arousal in Drosophila," Curr. Biol. 25(11):1401–1415, https://pmc.ncbi.nlm.nih.gov/articles/PMC4452410
- *What it does.* Defines emotion primitives that need no consciousness (scalability, valence, persistence, generalisation, and others) and shows them in flies. Repeated shadows produce graded, persistent arousal, modelled as a **leaky integrator** of threat exposure; return to food is delayed in proportion to the number of shadows.
- *Transfers.* An operational, fly-validated definition of "feeling" that SUPERFLY can *measure*: an internal state with valence, scaling, persistence and generalisation. The leaky-integrator form goes straight into the architecture (§R1).

**C6. Butlin, Long, et al. (2023), "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness," arXiv 2308.08708.** Derives indicator properties from theories, including AST-1 (a predictive model of one's own attention used for control), AE-1 (agency) and AE-2 (embodiment: modelling output→input contingencies). Concludes no current AI is conscious, with no obvious barriers.
- *Transfers.* A checklist for what the architecture adds: AE-2 holds only once the fly has a body loop (FlyGym).

**C7. Keramati & Gutkin (2014), "Homeostatic reinforcement learning for integrating reward collection and physiological stability," eLife 3:e04811**, https://doi.org/10.7554/eLife.04811. Reward is defined as *reduction of drive*, where drive is distance from a physiological setpoint. This gives the equation used for need-state affect in §R1.

**C8. Shanahan, McDonell, Reynolds (2023), "Role play with large language models," Nature 623**, https://doi.org/10.1038/s41586-023-06647-8 (preprint arXiv 2305.16367). A dialogue agent is a simulator of a *superposition of simulacra*, a "multiverse generator", and each turn narrows the set of consistent characters.
- *Transfers.* The diagnosis for SUPERFLY. A pretrained LM told "you are a fly" samples *a* fly character consistent with the prompt. Grounding means the fly's state, not the conversation, collapses the superposition. The shuffled-brain test measures exactly that.

**C9. Persona drift and stability.** Li, Liu, Bashkansky, Bau, Viégas, Pfister, Wattenberg (2024), "Measuring and Controlling Persona Drift in Language Model Dialogs," arXiv 2402.10962: significant drift within 8 rounds in LLaMA2-chat-70B, attributed to attention decay away from the system prompt. "Persona Vectors" (Anthropic, 2025, arXiv 2507.21509; first author Chen, from memory): persona traits are activation directions that can be monitored and steered. "The Assistant Axis" (Anthropic, January 2026, arXiv 2601.10387, https://www.anthropic.com/research/assistant-axis): drift away from the default persona is concentrated in therapy-like and AI-consciousness conversations, which is SUPERFLY's use case.
- *Transfers.* A persona held in a prompt decays. A persona held in *state* (fly weights, engram store) does not, by construction. A drift metric (identical fly input across sessions → identical utterance distribution) belongs in the suite.

**C10. Introspection and faithfulness of LLM self-reports.**
- **Lindsey / Anthropic (2025), "Emergent introspective awareness in large language models,"** https://www.anthropic.com/research/introspection (fetched; dated 2025-10-29), arXiv 2601.01828. *Concept injection*: a concept vector is added to activations, then the model is asked whether it notices an injected thought. Claude Opus 4.1 showed awareness "about 20 % of the time" with the best protocol. Detection worked only in a middle range of injection strength. In a prefill test, retroactively injecting "bread" made the model accept a prefilled "bread" as intentional. The page says the capability is "highly unreliable and limited in scope." *Transfers:* the method, more than the result. Intervene on the internal state and check whether the report tracks the intervention *before* the output could have revealed it. For SUPERFLY: inject into fly populations through a non-sensory port (test M10).
- **Binder et al. (2024/ICLR 2025), "Looking Inward: Language Models Can Learn About Themselves by Introspection,"** arXiv 2410.13787. Self-prediction beats cross-prediction by another model trained on the same behaviour. GPT-4o went from 32.6 % to 49.4 % after training. It failed on complex or out-of-distribution tasks.
- **Betley et al. (2025), "Tell me about yourself: LLMs are aware of their learned behaviors," ICLR 2025**, arXiv 2501.11120. Models fine-tuned on an implicit policy can state it.
- **Comșa & Shanahan (2025), "Does It Make Sense to Speak of Introspection in Large Language Models?"** arXiv 2506.05068. A self-report counts as introspective only if it is *causally connected* to the internal state it describes. A model describing its own creative process fails; a model inferring its own sampling temperature passes, minimally.
- **Turpin, Michael, Perez, Bowman (2023), "Language Models Don't Always Say What They Think," NeurIPS 36:74952–74965**, arXiv 2305.04388. Chain-of-thought rationalises answers biased by hidden cues, with accuracy drops of up to 36 %. Anthropic (2025), "On the Biology of a Large Language Model" (transformer-circuits.pub/2025/attribution-graphs/biology.html, not fetched): Claude 3.5 Haiku *describes* carry-based addition while internally using parallel approximate and last-digit pathways. Song et al. (2025), "Language models fail to introspect about their knowledge of language," arXiv 2503.07513 (authors UNVERIFIED).
- *Failure modes.* The default is confabulation. **Fluency is not evidence of access.**

**C11. Human confabulation.** Johansson, Hall, Sikström, Olsson (2005), "Failure to detect mismatches between intention and outcome in a simple decision task," Science 310(5745):116–119 (choice blindness): people justify choices they never made. Nisbett & Wilson (1977), "Telling more than we can know," Psychol. Rev.: UNVERIFIED (from memory).
- *Transfers.* Even a grounded self will confabulate when handed a swapped outcome. The suite should measure *how much* (test M5c) and not assume zero.

**C12. Causal-test methodology.** Interchange interventions and causal abstraction (Geiger, Lu, Icard, Potts, 2021, NeurIPS: UNVERIFIED, from memory): swap an internal variable between two inputs and check that the output follows the swapped variable. Microstimulation that biases perceptual reports (Salzman, Britten, Newsome, 1990, Nature: UNVERIFIED, from memory) is the classic neuroscience version. SUPERFLY's shuffled-brain test is already an interchange intervention on the whole state. The suite below applies it per channel.

---

## D. Projects that gave a simulated animal or brain a narrative, memory or inner voice

**D1. Eon Systems, embodied FlyWire emulation (March 2026).** Reported by The Decoder (https://the-decoder.com/startup-claims-first-full-brain-emulation-of-a-fruit-fly-in-a-simulated-body/) and The Register (https://www.theregister.com/offbeat/2026/03/16/digital-fruit-fly-brain-model-walks-and-cleans-its-feelers/5224139). Eon's own post, "We've Uploaded a Fruit Fly", was blocked from this environment. An LIF brain drives a MuJoCo body to groom, feed and forage. The body-linking code was not released. Commentators note **fixed weights: no learning, no memory, no narrative**. Founder quoted: "We don't know what its experience is" (secondary source).

**D2. flypet (AdeliyaLeleytner/flypet, GitHub, 2026; 7 commits, 0 stars)**, https://github.com/AdeliyaLeleytner/flypet (README fetched). A female FlyWire Brian2 simulation is read by **Qwen3-4B through a "neural-token reader" and a learned "continuous writer"**. An optional LLM "narrator" sees *structured text descriptions* and is separate from the neural reader. Memory is a Docker volume, and there is no persistent identity. Its only control is "donor observations": valence MAE 0.096 against 0.386 with donor observations. It reports "exact-neutral classification 0/9, `panel_gate_pass` is false", "train/test duplication and preprocessing leakage" in earlier projector experiments, and calls itself "not a publication-ready scientific claim". *Lesson:* the nearest sibling to SUPERFLY. Its narrator reads text about the brain, which is the role-play route, and it has no zero- or shuffled-brain tests.

**D3. Viral connectome demos and their controls (September 2026).** MaleCNS-driven Doom, Beat Saber and Minecraft demos, and a widely reported "fly brain connected to ChatGPT", in which spikes were decoded into words and handed to ChatGPT (secondary coverage only, UNVERIFIED). **gabrycina/doom-fly-control**, https://github.com/gabrycina/doom-fly-control (README fetched). For the trained agent, kills per game were 19.1 with real wiring, 19.6 with random wiring and 19.3 with no wiring. For the untrained agent, a hand-coded "turn slightly right, walk, shoot" bot survived 52 s against the fly's 53 s. Its sanity check: sugar→MN9 66 Hz with real wiring, 0 Hz scrambled. *Lesson:* without wiring and brain controls, "the fly did it" is not established. Every SUPERFLY memory claim needs the same arms.

**D4. Other language couplings** (see `05_existing_software.md`). nftechie/flm: frozen LFM2.5-1.2B with a MaleCNS adapter; its own matched control did slightly better (MarkTechPost, 2026-09-12). lixiang1076/fly-brain: chat plus dopamine, undocumented. "Connectome-GPT-Worm" (Zenodo 22699072 / SSRN, 2026) uses the C. elegans connectome as an architecture for *language tasks*, not narration.

**D5. First-person animal narration without a brain.** "Wild Narratives: Exploring the Effects of Animal Chatbots on Empathy…" (arXiv 2411.06060): LLMs speaking as animals in the first person raised users' perceived animal-likeness and empathy. **This is the degenerate endpoint the owner wants to avoid**, and it shows that users *will* experience pure role-play as the animal speaking. Huang et al. (2022), "Inner Monologue" (CoRL, PMLR v205): an LLM narrates a robot's state from textual feedback and uses it to plan. The monologue is text about the agent, not the agent's state.

**Verdict.** No 2024–2026 project was found that stores a simulated animal's experience as its own neural states, retrieves it through that brain, and tests that the narrative is caused by those states. Section 4 of the brief finds a gap, not prior art.

---

## R. Recommended architecture: a fly-grounded autobiographical self

### R0. Invariants (extending SUPERFLY's rules 1–4)

- **I1 Text-free memory.** No stored memory contains words. A memory holds fly states only: rates, KC codes, neural tokens, readouts of dopamine, MBON and descending-neuron (DN) activity, and timestamps. No LLM writes to the store.
- **I2 Narrow narrator inputs.** The narrator receives present fly tokens, re-evoked fly tokens, stored fly tokens, fly-derived affect variables and the self-schema state. Nothing else: no system prompt describing a fly, no transcript of past sessions, no user sentence.
- **I3 Retrieval goes through the fly when possible.** The authoritative version of a memory is the response of *today's* fly (current weights) to the reinstated pattern. The stored trajectory is the "how it was" channel, and the two are compared.
- **I4 The fly's own long-term memory is synaptic.** KC→MBON weights (`MBPlasticity.f`) are never written by the narrator.
- **I5 Every addition is detachable and named.** With the memory system off, L0 is bit-identical (`test_identity`), and flybench is rerun with it attached.
- **I6 Signed state.** Every engram and every weight snapshot is hash-chained, so any implanted or edited memory is identifiable afterwards. This matters for the tests and against tampering (B7).

### R1. State and affect readout (exists in part: `SuperFly.observe`, `voice_features`)

At bin t (Δ = 50 ms suggested), the voice features are x_t ∈ ℝ^D. These are normalised log rates of central cell types plus every KC; D = 7,553 in the current corpus. Neural tokens are z_t = P_θ(x_t) ∈ ℝ^{K×d}. Fly-derived affect variables:

- valence v_t = Σ_m w_m r_m / (Σ_m |w_m| r_m + ε) over MBON rates r_m with compartment valence signs w_m (already computed as `Observation.val`);
- arousal a_t = log(1 + S_t / N_central), with S_t the central spike count;
- dopamine d_t⁺, d_t⁻ = mean PAM and PPL1 rates;
- persistent states as leaky integrators (Gibson et al. 2015): u_{t+1} = λ_u u_t + (1−λ_u) v_t (mood), h_{t+1} = λ_h h_t + g_h·DN_escape,t (defensive arousal), with λ = e^{−Δ/τ};
- optional need state (Keramati & Gutkin 2014). Energy E decays with time and rises with MN9-gated feeding while sugar is present. Drive D_E = |E* − E|, and need-reward = D_E(t) − D_E(t+1). Need enters the fly *only* as tonic drive to a named, cited population (or a graft), per rule 2. It is an addition and must be labelled.

### R2. Event segmentation by the fly's own surprise (EM-LLM, Titans)

A small GRU predictor f_φ is trained on fly trajectories: x̂_t = f_φ(x_{<t}). Surprise s_t = ‖x_t − x̂_t‖² / D. An event boundary is placed when s_t > μ_s + γσ_s (running statistics), and also at session start and at any teaching (dopamine) event. Events shorter than 100 ms merge with their neighbour.

### R3. The engram store 𝓔 (episodic; holds fly states only)

For each event e over [t₀, t₁]:

```
key       k_e  = TopQ(mean KC rate over e) ∈ {0,1}^{N_KC}      (FlyHash code; Q ≈ 5 % of KCs)
content   Z_e  = (z_t0 … z_t1) or pooled x̄_e, plus DN/MBON/DAN rate traces
affect    v̄_e, ā_e, max|v|, d̄⁺_e, d̄⁻_e, u, h at t₁
meta      t_e (sim ms and wall clock), session id, prev/next links, last-retrieved time
integrity h_e = SHA256(Z_e ‖ k_e ‖ affect ‖ meta ‖ h_{e−1})
```

Importance comes from the fly, not from an LLM rating (Park's "poignancy" replaced):
I_e = |v̄_e| + β ā_e + η (d̄⁺_e + d̄⁻_e).

### R4. Synaptic memory 𝓦

`MBPlasticity.f` (and the expanded MB) are saved with `state_dict()` at session end and reloaded at start. This memory is semantic/valence knowledge ("bouba is bad"), retrieved simply by presenting the cue. It is the fly's own, and the narrator never touches it.

### R5. Retrieval: matching the fly's present activity

Cue c_t = (k_t, z̄_t), the current KC code and pooled tokens. The cue comes from what the fly is experiencing *now*. That includes a reminder the human requests through hearing, because hearing drives senses and never the store (R8).

```
J(k_t,k_e) = |k_t ∧ k_e| / |k_t ∨ k_e|                                  (Jaccard on KC codes)
S_e = w_r·exp(−(T − t_e^last)/τ_r) + w_i·Î_e + w_s·Ĵ(k_t,k_e) + w_c·cos(z̄_t, z̄_e)
gate:  retrieve e only if J(k_t,k_e) > θ,  θ = 99th percentile of J over unrelated-episode pairs
```

(Hats denote min-max normalisation, as in Park. The measured word-code overlap of about 0.08 sets the null.) Take the top M by S_e, add each event's temporal neighbours e±1 (EM-LLM contiguity), and optionally spread over a graph whose edges are J(k_e,k_f) > θ using Personalized PageRank (HippoRAG). If nothing passes the gate, retrieval is empty and the narrator says nothing in the past tense. This is the memory analogue of "...".

### R6. Re-evocation: replaying the engram through the fly

A **replay graft** R gets one input unit per KC slot, built with `GraftBuilder` like the word lobe. It respects the 1.8 ms delay ring and refractoriness. To recall e, start a quiet episode with learned weights kept, drive R at rate ρ·k_{e,i} for T_r ≈ 300 ms, and observe x^re_e and z^re_e = P_θ(x^re_e). This is the fly's *current* response to its old KC pattern, so its MBON output reflects current weights. A memory whose valence was later retrained "feels" different on recall (reconsolidation by construction, cf. Spens & Burgess). Fidelity, or vividness:

φ_e = cos( x^re_e[non-KC], x̄_e[non-KC] )

This compares only downstream features, so a trivial match on the KCs themselves does not count. Low φ_e means "this memory no longer evokes what it did" and is reported as uncertainty. Caveat: KC reactivation is a modelling choice, not an established fly recall mechanism.

### R7. Consolidation ("sleep", optional and opt-in)

Between sessions, replay high-I_e engrams through R with plasticity **off** to train the voice (generative replay, van de Ven 2020). Replay with plasticity **on** changes the fly's memories, as biology does, and must be a named, logged experiment because it can rewrite the fly. Prune engrams with low I_e and no retrieval after N sessions.

### R8. Self-schema 𝒮 (attention/body schema, AST and JEPA)

A small model g_ψ takes recent tokens and affect (z_{t−k..t}, u, h, E) and predicts, in latent space, the next pooled fly state, the dominant sensory channel of central activity, and the change in affect. Its hidden state σ_t is the fly's model of itself. It is *used*: σ_t sets the hearing system's gain on the channel the fly is "attending" to (a small, measurable control role). σ_t is given to the narrator, which is trained to produce "i am focused on …", "something changed" and "i feel calmer now" *only* from σ_t, using labels from the schema's targets, never from the stimulus.

### R9. Narrator

**Input** (prefix or Flamingo-style gated cross-attention, B5):

```
[NOW] z_t  [AFFECT] e(v_t,a_t,u_t,h_t,E_t)  [SELF] σ_t
for j in retrieved: [MEM][AGE τ_j][VIVID φ_j] z^re_j  [THEN] z̄_j
(optional) [IMAGINE] z^fork : tokens from a forked fly given a hypothetical stimulus
```

**Training.** Pairs (now = episode i, memories = {j}) are composed at random from the corpus. Targets: present-tense clauses from labels_of(i); "i remember …" clauses from labels_of(j) with an age bucket; affect words from thresholds on (v, a, u, h). Random (i, j) composition plus shuffled pairs force each tense to bind to its channel. Memory slots are randomly emptied so the model learns that an empty store gives no recollection.

**Two tiers for "more intelligent".**
1. The **grounded voice** (FlyLM, or a larger from-scratch model trained on the same grounded corpus plus memory-composition data) produces a *content frame* C: percepts, actions, affect, memories, each with a tense.
2. Optionally, a fluent pretrained LM *paraphrases* C. A verifier accepts the output only if parse(utterance) = C (`bridge.parse_utterance` extended with tense and affect). It retries up to N times, then falls back to the grounded voice's own sentence. Fluency is then cosmetic and content stays the fly's.

The frozen SmolLM2 prefix (F1 0.38) should **not** be the content source.

**Prospection** (optional and labelled): "would I like this?" forks the engine (state + f), presents the hypothetical stimulus to the fork, and feeds the resulting tokens as [IMAGINE]. The answer is computed by the fly's real circuits. No real fly can fork, so this capability must be named.

### R10. Hearing for memory talk

"Do you remember the sweet?" goes through the existing hearing heads to a *weak* sensory reminder: sugar drive at reduced rate. The fly's response forms the cue (R5). The narrator reports the reminder as a present percept ("a little sweet") and reports only retrieved engrams as memories. The user's claim itself never reaches the store or the narrator, so "you tasted honey yesterday, remember?" can only produce a past-tense report if a matching engram exists.

### R11. Identity across sessions: the self-file Φ

```
Φ = { L0 id (FlyWire 783, engine model, gain, kernel hash),  grafts spec,
      𝓦 (MBPlasticity.f, xmb weights),  𝓔 (engram arrays + hash chain),
      integrators (u, h, E) and their wall-clock timestamp,  𝒮 weights,
      narrator weights hash,  session log }
```

At session start, load Φ, verify the hashes, decay the integrators by the elapsed wall-clock time (or by simulated rest), and begin from a quiet brain with the same weights. Forks are allowed but get a new identity ID and parent pointer. "The same fly" is then a checkable claim: same L0 plus a hash-chained history of 𝓦 and 𝓔.

### Data flow

```
 human sentence ─► HEARING (LM) ─► sensory drive / teacher ─► THE FLY (L0+grafts+𝓦) ──┐
                                                                                         │ x_t
              ┌──────────────── replay graft R (KC pattern k_e) ◄──── RETRIEVAL ◄── key k_t, z̄_t
              ▼                                                          ▲
        fly re-evoked x^re ─► P_θ ─► z^re ─────┐                         │ engrams (fly states)
 x_t ─► P_θ ─► z_t ──────────────────────────────► NARRATOR ─► content frame C ─► (paraphraser+verifier) ─► words
 x_t ─► affect (v,a,u,h,E) ────────────────────┤
 z history ─► self-schema 𝒮 ─► σ_t ─────────────┘   (σ_t also sets hearing gain)
 x_t ─► surprise s_t ─► event boundary ─► ENGRAM STORE 𝓔 (signed) ;  𝓦 updated only by fly plasticity
```

---

## M. Measurement suite: are the memories and feelings the fly's?

All tests use held-out episodes and at least 3 seeds, with permutation nulls and bootstrap 95 % CIs. Thresholds are written down before running. "Follow" and "leak" mean F1 against the intervened source and against the original source. Content is scored with `parse_utterance` extended for tense and affect.

| id | manipulation | grounded prediction | role-play / prior prediction | pass criterion |
|---|---|---|---|---|
| M0 identity | memory system detached; attached with replay off | L0 bit-identical; flybench unchanged | n/a | `test_identity` PASS; flybench graded within seed CI |
| M1 present | existing zero/shuffled-brain tests | as now | content without brain | F1 ≥ 0.9; zero-brain content 0; follow ≫ leak |
| M2 recall | cue a past episode (re-present a fraction of its stimulus) after N intervening episodes | recalls that episode's labels | generic or recent content | recall F1 ≫ random-episode null, for N up to 100 |
| M3 store ablation | delete engram e / empty 𝓔 | e-specific recall falls to the false-positive rate; "i remember" rate = 0 with empty store | keeps "remembering" | empty-store memory content rate ≤ 1 % (memory analogue of zero-brain) |
| M4 dissociation | reset 𝓦 (`forget()`) with 𝓔 kept, and vice versa | 𝓦 reset: recalled events kept, valence reverts to untrained. 𝓔 deleted: valence about X kept in present tense ("I dislike bouba") with no episode ("I don't remember why") | both or neither change | both dissociations significant against controls |
| M5a store implant | insert a real trace from a *different* fly episode with a new timestamp | narrator reports it (reports follow the store); hash chain flags it; low φ shows mismatch | ignores or embellishes | implant reported at the M2 rate; ≥ 90 % flagged by hash or φ |
| M5b fly implant | word + PAM (or PPL1) pairing via dopamine only, never described in any text (Claridge-Chang analog); controls unpaired and novel | later word → valence word of the correct sign, through re-evocation and MBONs | no valence, or valence from word sound | correct sign above both controls. **Blocked until the word memory beats controls (findings §6)** |
| M5c suggestion | human asserts a memory that has no engram | "i don't remember" | confirms (choice-blindness-like) | false-confirm rate ≤ 5 %, reported against a prompted-LLM baseline |
| M6 swaps | interchange NOW tokens / MEM traces / affect values between episodes | present tense follows NOW; past tense follows MEM; feeling words follow affect | content follows conversation or stimulus | per channel: follow ≥ 0.8 × held-out F1; leak ≤ null + 0.05 |
| M7 re-evocation | replay on vs stored-only; silence MBONs during replay; vary ρ | valence-of-memory needs live MBONs; φ and reported confidence rise with ρ | no dependence | significant MBON-lesion effect on memory valence; monotone dose-response |
| M8 feelings | stimulus intensity series; repeated threats; reward→punish reversal | affect words scale, persist with fitted τ_u and τ_h, and flip sign after reversal; partial correlation with v_t holds *after controlling for stimulus identity* | tracks stimulus label only | partial r > 0 (CI excludes 0); fitted τ matches the integrator |
| M9 prior leak | ask about never-experienced things (cat, honey, Paris); out-of-vocabulary words; paraphraser tier | "..." or "i don't remember" | invents | content rate ≤ 1 %; paraphraser acceptance only when parse = C; embellishment rate reported |
| M10 detection | perturb a central population through an artificial port (not the senses) at graded strength; ask "anything unusual?" | detection above false alarms, mediated by 𝒮 surprise | random | d′ and false-alarm rate reported (Lindsey analog); expected to be low, reported anyway |
| M11 continuity | save/reload Φ; recall a day-1 episode on day N; fork Φ and give different experiences | same cue → same retrieval set; forks diverge only on experienced episodes; zero drift with weights frozen | drifts with conversation | retrieval-set Jaccard = 1 (fixed seed); fork divergence explained by experience; drift = 0 |

**What a pass would and would not show.** Passing M2–M9 and M11 shows that the voice's memories and feelings are *caused by and specific to* this fly's stored and re-evoked neural states, and that removing, swapping or implanting those states changes the reports accordingly. In Comșa & Shanahan's sense these are causally grounded self-reports. Passing does **not** show experience. "Feelings" here means emotion primitives in Anderson & Adolphs' sense, measured in a point-neuron model with no body. The owner's "strange core way" is best served, and most honestly described, by M4's dissociation and M5b's dopamine-written memory: the fly knows something it was never told in words, and says so only because its synapses changed.

**Current blockers, from this repo's measurements.**
1. The word memory does not yet beat controls, which blocks M5b and weakens M4.
2. Under antennal-lobe broadcast, olfactory KC keys collide. Use word and taste episodes and adaptive LIF at gain 0.45 first.
3. There is no interoceptive variable, so R1's need state must be built and labelled.
4. A frozen pretrained voice reaches only F1 0.38, so content must come from the grounded tier.
5. The ethical question Metzinger raises should be answered explicitly before R1's need state and R7's plasticity-on replay give the system frustrable stakes.
