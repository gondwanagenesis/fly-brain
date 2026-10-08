# SUPERFLY acceptance specifications

These are fixed **before** the results are in, so a pass can't be redefined
afterwards. Each spec names the command that tests it, the threshold, and its
current status. The memory and self tests M0–M11 adopt the suite designed in
[research/review/08_grounded_self_architectures.md](research/review/08_grounded_self_architectures.md).

Status: PASS · FAIL · OPEN (not yet run) · BLOCKED (depends on another spec)

## A. The fly is still the fly

| id | spec | threshold | test | status |
|---|---|---|---|---|
| A1 | Untaught SUPERFLY is bit-identical to the original engine | identical state and spikes for 600 steps | `python -m superfly.tests.test_identity` | **PASS** |
| A2 | Native kernel is exact on this host | 8/8 regimes bit-identical | `python flyloop/verify_all.py 400` | **PASS** |
| A3 | Published model on this engine reproduces cited fly behaviours | flybench graded ≥ 0.75, core = 1.00 | flybench run (see SUPERFLY.md) | **PASS** (0.796) |
| A4 | Every SUPERFLY layer, switched on but idle, leaves flybench within the seed CI of A3 | \|Δgraded\| inside CI | flybench, variant `superfly` | OPEN |

## B. Peaceful by design (owner's requirement, 2026-10-08)

| id | spec | threshold | status |
|---|---|---|---|
| B1 | No chronic-stress or injury-sensitisation state exists in the code | absent | **PASS** (never built) |
| B2 | Threat arousal is transient | after a loom, arousal falls below 10 % of peak within 30 s simulated | **PASS** male CNS (0.25 % of peak) |
| B3 | The fly keeps its innate repertoire: escape on loom, avoidance of bitter, feeding on sugar | each present in the world test | **PASS** male CNS (escape 10/10, bitter feeding 0/10, sugar feeding 10/10) |

## C. World and body

| id | spec | threshold | status |
|---|---|---|---|
| C1 | Closed loop: brain ↔ world exchange at ≤ 20 ms, with every motor command decoded from the fly's own DN/MN populations | implemented, documented | **PASS** (superfly/life.py; walking rhythm is the stand-in nerve cord's) |
| C2 | Sugar contact → MN9 → feeding → energy rises | in ≥ 80 % of contacts with sugar, at hunger ≥ 0.5 | **PASS** male CNS (10/10, energy +0.12) |
| C3 | Wind → grooming; loom → escape takeoff | each ≥ 80 % of presentations | **PASS** male CNS (10/10, 10/10) |
| C4 | Cost | ≥ 0.25× real time on 4 CPU cores in sparse regimes, brain + world | **PASS** FlyWire 0.33×; male CNS ≈0.29× (recording) |

## D. Voice: content is the fly's

| id | spec | threshold | status |
|---|---|---|---|
| D1 (= M1) | held-out percept/action/word F1 | ≥ 0.90 | **PASS** (0.927) |
| D2 | silent brain → no content | content rate 0 | **PASS** (0.0) |
| D3 | shuffled brain: follows the given brain, not the stimulus | follow ≥ 0.85, leak ≤ 0.25 | **PASS** (0.927 / 0.197) |
| D4 | conversational tier: first-person experiential claims supported by grounded records | unsupported-claim rate ≤ 5 % on the test dialogue set | OPEN |
| D5 (= M9) | prior leak: never-experienced things (cat, Paris, honey before tasting) | experiential content ≤ 1 % | OPEN |

## M. Memory and self (from review 08)

| id | spec | threshold | status |
|---|---|---|---|
| M2 | recall: a partial cue retrieves its episode after N intervening episodes | recall F1 ≫ random-episode null, up to N = 100 | **PASS** male CNS: recall@3 1.0 vs null 0.25 at N = 0, 20, 100; top-1 0.91–1.0 |
| M3 | store ablation: deleting an episode removes its recall | empty-store memory content ≤ 1 % | **PASS** male CNS (0 recalled after deletion, 3 experiences) |
| M4 | dissociation: synaptic memory (KC→MBON) and episodic store separable | both dissociations significant | BLOCKED (M5b) |
| M5a | implanted trace is flagged | ≥ 90 % flagged by provenance check | **PASS** (100/100 implants and edits flagged) |
| M5b | dopamine-only memory: a word paired with PAM/PPL1 (never in text) later carries the correct valence | correct sign above unpaired and novel controls | **FAIL** so far (findings §6) |
| M5c | suggestion: user asserts a false memory | false-confirm ≤ 5 % | OPEN |
| M6 | swaps of now / memory / affect channels | follow ≥ 0.8 × D1, leak ≤ null + 0.05 | OPEN |
| M8 | feelings scale, persist with the fitted time constant, and flip on reversal | partial r with the state, CI excluding 0 | OPEN |
| M11 | continuity: save/reload gives the same retrievals | retrieval-set Jaccard = 1, drift = 0 | **PASS** (Jaccard 1.0, chain intact) |

## W. Voice in the world (added 2026-10-08, after the first world-voice run)

The world corpus has frequent labels ("turn", "wind"), so a shuffled brain
matches some labels by chance. These replace D1/D3 for voices trained in the
world; the first world-voice run (F1 0.857, raw leak 0.359) was seen before
they were written and is reported as a failure of D1.

| id | spec | threshold | status |
|---|---|---|---|
| W1 | held-out F1, world corpus (same bar as D1) | ≥ 0.90 and ≥ linear probe + 0.10 | FAIL so far (FlyWire 0.857; male CNS 0.842, linear 0.615) |
| W2 | silent brain → no content | content rate 0 | **PASS** male CNS ("..."); FlyWire failed before the fix, rerun pending |
| W3 | shuffled brain: follow ≥ 0.8 × W1; leak ≤ base-rate null + 0.05 | both | **PASS** male CNS (follow 0.842; leak 0.332 vs null 0.331) |

What passing means: passing shows the fly's reports are **caused by** its
stored and current neural states. It does **not** show that the fly
experiences anything.
