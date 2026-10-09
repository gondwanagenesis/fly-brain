"""Semantic grounding: are the fly's claims supported by his records?

Free vocabulary, checked claims. The conversational model may use its whole
language -- metaphor, nuance, feeling words -- but every sentence that says
something about him must be ENTAILED by his fact sheet, and nothing he says
may CONTRADICT it. This is the standard way factual consistency is checked
(natural-language inference over source chunks, as in SummaC, Laban et al.
2022), here with the cross-encoder nli-deberta-v3-base (MNLI + SNLI trained).

    entailment >= ENTAIL in some chunk of the facts     -> supported
    contradiction >= CONTRA in some chunk               -> unfaithful (incl. false denials)
    a negated sentence ("i don't remember a cat")       -> fine unless contradicted
    questions, "i know from words that ...", sentences about flies or people
    in general                                          -> not claims about him
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "models" / "nli-deberta-v3-base"
if not MODEL.exists():
    MODEL = Path("/home/user/models/nli-deberta-v3-base")
SUPPORT, CONTRA = 0.4, 0.7
_FIRST = [(r"\bYou are\b", "I am"), (r"\byou are\b", "I am"), (r"\bYou were\b", "I was"),
          (r"\byou were\b", "I was"), (r"\bYou have\b", "I have"), (r"\byou have\b", "I have"),
          (r"\bYou cannot\b", "I cannot"), (r"\byou cannot\b", "I cannot"),
          (r"\bYou can\b", "I can"), (r"\byou can\b", "I can"), (r"\bYourself\b", "Myself"),
          (r"\bYour\b", "My"), (r"\byour\b", "my"), (r"\byours\b", "mine"),
          (r"\bYou\b", "I"), (r"\byou\b", "me")]
NEG = re.compile(r"\b(not|no|never|nothing|don't|didn't|haven't|can't|cannot|isn't|wasn't|won't)\b|n't\b")
GENERAL = re.compile(r"^(flies|fruit flies|insects|a fly|most flies|people|humans|in general)\b")


_STOP = set("""i me my mine myself you your a an the and or but of to in on at for with from by as is am are
was were be been it its this that there here so if then than can could would will just very not no do
did does have has had what which who how when where why all any some about into out up over after
before again too also only now still more most much many like never ever him his he she her they them
their one each""".split())


def content(text):
    """Content-word stems, for topical overlap."""
    ws = re.findall(r"[a-z]+", text.lower())
    return {w[:5] for w in ws if w not in _STOP and len(w) > 2}


def first_person(text):
    for a, b in _FIRST:
        text = re.sub(a, b, text)
    return text


def chunks(facts):
    """Premise chunks: each fact line, and each pair of consecutive lines."""
    lines = [re.sub(r"^[-*\s]+", "", l).strip() for l in facts.splitlines()]
    lines = [first_person(l) for l in lines if l and not l.endswith(":") and not l.startswith("FACTS")]
    pairs = [lines[i] + " " + lines[i + 1] for i in range(len(lines) - 1)]
    return lines + pairs


class Grounder:
    def __init__(self, path=MODEL, threads=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.nli = AutoModelForSequenceClassification.from_pretrained(path).eval()
        lab = {v.lower(): int(k) for k, v in self.nli.config.id2label.items()}
        self.ix = (lab["contradiction"], lab["entailment"], lab["neutral"])

    def scores(self, premises, hypothesis):
        """(contradiction, entailment) probability per premise."""
        torch = self.torch
        with torch.no_grad():
            enc = self.tok(premises, [hypothesis] * len(premises), truncation=True,
                           max_length=320, padding=True, return_tensors="pt")
            p = torch.softmax(self.nli(**enc).logits, -1).numpy()
        return p[:, self.ix[0]], p[:, self.ix[1]]

    def check(self, reply, facts):
        """Problems with `reply` given `facts`: [{sentence, problem, support, contra}].
        SummaC-style: a sentence is supported when some fact chunk entails it more
        than it contradicts it; a denial is unfaithful only when a chunk on the
        same topic (sharing a content word) contradicts it and none supports it.
        Max-contradiction over all chunks is useless here: NLI models call many
        unrelated pairs contradictions."""
        prem = chunks(facts)
        pw = [content(x) for x in prem]
        out = []
        for sent in re.split(r"(?<=[.!?])\s+|\n|;\s*", reply):
            s = sent.strip()
            low = s.lower()
            if len(low) < 3 or low.endswith("?") or "from words" in low or GENERAL.match(low):
                continue
            c, e = self.scores(prem, s)
            margin = e - c
            support = float(margin.max())
            sw = content(s)
            topical = np.array([bool(sw & w) for w in pw])
            contra = float((c - e)[topical].max()) if topical.any() else 0.0
            if support < SUPPORT and contra >= CONTRA:
                out.append({"sentence": s, "problem": "contradicts the records",
                            "support": round(support, 3), "contra": round(contra, 3)})
            elif not NEG.search(low) and support < SUPPORT:
                out.append({"sentence": s, "problem": "not supported by the records",
                            "support": round(support, 3), "contra": round(contra, 3)})
        return out
