# Compute and agent budget log

The owner set a budget of API credit for the overnight build ($180, revised
to $165 remaining on 2026-10-08). This file records where agent work went so
the spend can be checked against the account's own usage page, which is the
authoritative figure (it cannot be read from inside the session).

Policy: research and front-end work go to cheaper models with precise briefs;
the main session does architecture, code that touches the brain, and
verification. Long CPU jobs (corpus recording, training) run in the background
and cost no tokens while they run.

| task | model tier | tokens | output |
|---|---|---|---|
| review 06: fly feelings / internal states | large | 331,420 | research/review/06_fly_feelings_internal_states.md |
| review 08: grounded-self architectures | large | 278,719 | research/review/08_grounded_self_architectures.md |
| review 07: fly memory and intelligence | large | 387,634 | research/review/07_fly_memory_intelligence.md |
| review 09: the fly's world | mid (sonnet) | 296,095 | research/review/09_fly_world.md |
| review 10: full-CNS connectomes (BANC, male CNS) | mid (sonnet) | running | research/review/10_full_cns_connectomes.md |
| Lab front-end (3D brain + modules + world + chat) | mid (sonnet) | running | superfly/lab/index.html |

Earlier agents (reviews 01-05, before this log) are not itemised here.
Local compute: one 4-core container; Qwen2.5-1.5B and SmolLM2-360M run on CPU,
so no GPU or hosted-model cost is incurred by the fly itself.
