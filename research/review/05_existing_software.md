# Existing software for simulating and interacting with the whole fly brain

Survey made 2026-10-08 for SUPERFLY. Sources are the projects' own READMEs
(fetched that day) and press coverage; star counts and dates are as shown on
GitHub's `flywire` topic page that day. Nothing here was run except flybench.

## The answer to "has someone built a better version of this?"

**For the simulator, no.** This repo's native kernel is still the fastest exact
CPU engine found: 2–3× faster than real time on four cores, bit-identical to the
PyTorch reference (now on Linux too). flybench reports its own reference at
5.4× *slower* than real time per CPU; webgpu-fly ~4× slower than real time.

**For the parts around it, yes — and they are adopted, not rebuilt:**

| need | best existing | verdict |
|---|---|---|
| "is it still a fly?" test | **flybench** (brandoncho369) — 36 cited tasks, shuffled-wiring controls, seeds | **adopted** as SUPERFLY's identity gate (`superfly/bench/flybench_adapter.py`) |
| most realistic neuron model on FlyWire | flybench's **adaptive LIF** (b 2 mV, τ 200 ms, gain 0.45): core 1.00, hard 0.681, graded 0.791 (their 2026-09-16 table) | **adopted** as kernel model 9, `lif_adapt` |
| neurotransmitter ground truth | **flyconnectome/drosophila_neurotransmitters** (`gt_data.csv`, 6,107 rows, CC-BY-4.0) | **adopted** for auditing signs (`data/external/nt_gt_data.csv`) |
| body | **FlyGym / NeuroMechFly v2** (NeLy-EPFL), **flybody** (TuragaLab) | to integrate (closed_loop.py already targets FlyGym) |
| vision front end | **flyvis** (TuragaLab, Lappalainen et al. 2024) | to integrate |
| brain + nerve cord | **MaleCNS v1.0** (Janelia), **BANC** | candidate second substrate; most 2026 hobby projects use MaleCNS |
| effective-connectivity / pathway tools | connectome-interpreter (Yin), navis, fafbseg, Codex | use as needed |

## Projects closest to the SUPERFLY goal

| project | what it does | how it relates |
|---|---|---|
| [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) (upstream) | Shiu et al. model on 5 backends + benchmark harness | brain only. Eon's March 2026 embodied demo (walk, groom, feed in NeuroMechFly) was reported as not releasing the body-linking code ([The Decoder](https://the-decoder.com/startup-claims-first-full-brain-emulation-of-a-fruit-fly-in-a-simulated-body/), [The Register](https://www.theregister.com/a/5224139)); company-reported 91–95 % accuracy figures use different metrics |
| [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain) (65★) | 138,639-neuron FlyWire LIF in NeuroMechFly/MuJoCo: vision, smell, taste, grooming, flight; preprint | has a body. Needs an RTX GPU (~30 min per 100k steps). Signs **glutamate as excitatory** (Shiu: inhibitory). Plasticity is a global Hebbian rule, not the fly's. Reports "consciousness proxy" metrics without validation |
| [abgnydn/webgpu-fly](https://github.com/abgnydn/webgpu-fly) | FlyWire brain + MANC cord + flybody in the browser | ~4× slower than real time; brain→cord link is a cell-type name join across two animals; gait is hand-written; README states dynamics are not quantitatively validated |
| [lixiang1076/fly-brain](https://github.com/lixiang1076/fly-brain) (25★) | Brian2 whole brain + natural-language chat + dopamine learning | nearest stated goal. Two commits; the language model and the learning rule are not documented; no embodiment; no validation beyond one sugar→MN9 run |
| [nftechie/flm](https://github.com/nftechie/flm) | frozen LFM2.5-1.2B + MaleCNS adapter (278k params) correcting the LM's logits | the "fly" is an abstract tanh reservoir, not spiking neurons, no transmitter signs; **its own matched direct-input control did slightly better** — the README says it "does not establish an advantage from fly anatomy". The cautionary case for SUPERFLY's fly-first design |
| [migkapa/flyputer](https://github.com/migkapa/flyputer) | local Gemma (Ollama) calling tools over a 2-hop LIF sub-circuit | LLM as operator of a toy sim; self-described toy |
| [brandoncho369/flybench](https://github.com/brandoncho369/flybench) | the benchmark | adopted; see above |

The [awesome-fly](https://github.com/cobanov/awesome-fly) list (685★, updated
2026-10-05) indexes ~90 more: games driven by MaleCNS circuits, desktop flies,
reservoirs, ports to Apple MPS/MLX, CUDA (FastFly), .NET (dotFly), Rust
(Connectome OS). None combines the fly's own plasticity, grafted neurons and a
measured language interface.

## What nobody has done (the SUPERFLY gap)

1. The fly's own learning rule at its own learning site (dopamine-gated KC→MBON,
   compartments from the connectome), on the whole brain.
2. Grafted neurons simulated as first-class cells of the same network.
3. A language model coupled through neural tokens with ablations proving the
   content is the fly's (zeroed / shuffled / silenced-fly controls).
4. An external behavioural benchmark run before and after every addition.

## Shared open problem

flybench's FINDINGS (2026-09-14, -16) record the same antennal-lobe broadcast
SUPERFLY measured independently (`research/superfly_findings.md`): one
glomerulus drives 84 % of PNs; DA1 lifetime sparseness 0.0009 against 0.90
measured; "antennal-lobe lateral inhibition is what all three [olfactory tasks]
are waiting for". No project found has solved it.
