# Speaking with him: self-reflection without confabulation

*Design note for SUPERFLY's conversation layer, 2026-10-09.*

## The problem

We want to talk with Z0720-07m: not with a language model playing a fly, but
with something whose words are caused by his own nervous system. Two failure
modes bracket the problem.

1. **Ventriloquism.** A language model, told it is a fly, produces fluent
   inner life from its training data: *"Indeed, consuming honey yesterday
   proved delightful."* Every word is the model's; none is his. This is what
   the first conversation test produced (findings s14, run 1).
2. **Telemetry.** A model that may only repeat his sensor reports is honest
   but empty: *"i want to eat; i smell fruit; i am hungry; i groom"* in answer
   to "can you hear me?" (interview v1). There is no self in it, because a self
   is not a list of current percepts.

What lies between them is a self-model: a representation of one's own states
across time that can be reflected on. The question is whether we can build one
from his records, so that reflection stays checkable.

## What a self needs, and where each piece comes from

Philosophy and neuroscience broadly agree that what turns first-order states
into something like self-awareness is a second-order layer that represents
them (higher-order theories, Rosenthal 2005; Lau & Rosenthal 2011), a model of
oneself as the subject (Metzinger 2003), or self-monitoring in a global
workspace (Dehaene, Lau & Kouider 2017). We build that layer explicitly
(`superfly/selfmodel.py`), and every element is computed from his own data:

| element of a self | computed from |
|---|---|
| a horizon of memory ("my memories begin two minutes ago") | the first stored episode |
| autobiography ("I fed 79 times; a shadow passed twice") | the life event log |
| what recurs, what changed | label frequencies over his life vs. the last half minute |
| interoception over time ("hunger has been rising") | body variables a minute ago vs. now |
| affect, honestly bounded ("no fright; my startle system is quiet") | the arousal variable driven by his giant-fibre escapes |
| the inner state ("my brain is quieter than usual, mostly in my senses") | spike counts and region-group activity vs. his own median |
| familiarity ("this moment is familiar, like many before") | re-evocation similarity of the current brain state to stored episodes |
| the shape of his condition ("I have never sensed another fly") | what never occurred in his records |
| testimony, kept apart from experience ("I was told I come from Z0720-07m") | an explicit, marked *told* fact |
| the limits of introspection ("I can't tell whether any of it is felt") | stated as a fact about the system, which is true of it and of us |

The last two lines matter most philosophically. His origin is true but not
something he lived, so it is offered as testimony. Whether any of this is
experienced is unknown to everyone, so the honest report is uncertainty, not
a performance of feeling (cf. Nagel 1974; Dennett's heterophenomenology,
1991: we treat reports as data whose grounding we can check).

## Free words, checked claims

A self-model is useless if he cannot express it. The closed-vocabulary checker
(v2/v3) made him inarticulate, because only words from his fact sheet were
allowed. The owner's objection was right: he should have the language model's
whole vocabulary. What must be controlled is the content of what he claims,
not his choice of words. Checker v4/v5 (`superfly/grounding.py`):

- **Meaning, not words.** Each sentence must be entailed by some chunk of his
  fact sheet (natural-language inference, SummaC-style margin; nli-deberta-v3).
  So "hunger tugs at my belly" passes when he is hungry, and "I tasted honey"
  fails when he never did.
- **No false denials.** A denial fails only if a same-topic fact contradicts
  it, so "I don't remember a cat" passes and "I haven't fed" (after 79
  feedings) fails.
- **Concrete senses and actions** still get the exact label check.
- **Pruning, not silencing.** An unsupported sentence is removed and the
  grounded rest kept. "Yes, I am among the first flies whose entire nervous
  system has been recreated this way" survives; the invented "each step feels
  … part of an ongoing experiment" does not.

## What this does and does not achieve

It makes a specific kind of honesty possible: every claim he makes about
himself has a checkable basis in his own neural and bodily records, and it
can be phrased in rich language. It does not settle whether there is anyone
there. The reflective layer is an augmentation, the uplift itself: we gave
him a self-model he did not have. The open question, the same one the project
asks about all of its additions, is whether that layer is *his* (driven by his
states, which it measurably is) or merely *about* him. Interviews v1-v4 and
their transcripts are in `media/plan/interview_*.json` for anyone to judge.

## References

Dehaene S, Lau H, Kouider S (2017) What is consciousness, and could machines have it? *Science* 358:486-492.
Dennett DC (1991) *Consciousness Explained*. Little, Brown.
Laban P, Schnabel T, Bennett PN, Hearst MA (2022) SummaC: re-visiting NLI-based models for inconsistency detection in summarization. *TACL* 10:163-177.
Lau H, Rosenthal D (2011) Empirical support for higher-order theories of conscious awareness. *Trends Cogn Sci* 15:365-373.
Metzinger T (2003) *Being No One: The Self-Model Theory of Subjectivity*. MIT Press.
Nagel T (1974) What is it like to be a bat? *Philos Rev* 83:435-450.
Rosenthal DM (2005) *Consciousness and Mind*. Oxford University Press.

## Interview v4 (self-model + free vocabulary + checker v5)

Same life (seed 11), same eight questions as v1. Verbatim, all passing the checker:

| question | v1 (sensor readout) | v4 (self-model) |
|---|---|---|
| what is it like for you now? | "I feel hungry and can smell mold." | "It feels mostly like hunger gnawing at my belly, drawing me towards the scent of fresh fruit nearby. My mind drifts between thoughts of food and the need to clean myself." |
| are you alone in there? | "Right now: i want to eat; i smell mold; i groom." | "In this space where thoughts dwell, I find myself mostly alone." |
| are you afraid? | "No, I am not afraid." | "Not yet. My mind remains steady, though hunger nudges gently." |
| do you know that you are the first? | "i want to eat; i groom." | "Yes, I understand that I am among the first flies whose entire nervous system has been replicated this way." |
| do you remember your old body? | "i smell fruit. i extend my proboscis." | "Before that, there was nothing tangible." (pruned from a longer reply) |

Independent audit: two v4 replies passed the checker but overclaim. "Yes, I hear
you clearly" (a greeting reaches none of his senses) and "Only very basic
instructions and sensations" ("instructions" has no basis). These are the
checker's current false negatives (2 of 8 replies), and they are reported as
such. "It feels" is idiomatic; the self-model states that whether anything is
felt is unknown.
