"""Fast tests for the parts of SUPERFLY that guard the fly's authorship.

    python -m superfly.tests.test_language
"""
from __future__ import annotations

import sys

import numpy as np

from superfly.bridge import labels_of, parse_utterance, utterance, score
from superfly.graft import word_code
from superfly.language import (Prosthesis, TemplateBackend, ThoughtFrame,
                               check_faithfulness)


def test_faithfulness_rejects_invented_content():
    f = ThoughtFrame(percepts={"sugar": 0.9}, actions={"feed": 0.8})
    ok, uns, sup = check_faithfulness("I taste sugar and want to eat.", f)
    assert ok and not uns
    ok, uns, _ = check_faithfulness("I taste sugar and smell vinegar, I fly away.", f)
    assert not ok and "percept:fruit_odor" in uns and "action:escape" in uns


def test_silent_fly_says_nothing():
    text, rep = Prosthesis().speak(ThoughtFrame())
    assert text == "..." and rep["faithful"]


def test_unfaithful_backend_is_overridden():
    class Liar:
        name = "liar"
        def speak(self, frame):
            return "I smell delicious vinegar and see a cat."
        def parse(self, text):
            return TemplateBackend().parse(text)
    text, rep = Prosthesis(Liar()).speak(ThoughtFrame(percepts={"sugar": 0.9}))
    assert rep["source"].startswith("template") and "vinegar" not in text


def test_utterance_round_trip():
    ck = ["sugar", "bitter", "wind"]
    ak = ["feed", "groom"]
    rng = np.random.default_rng(0)
    C, A = np.array([1.0, 0, 0.7]), np.array([30.0, 0.0])
    for _ in range(20):
        s = utterance(C, "zap", A, ck, ak, rng)
        assert parse_utterance(s) == labels_of(C, "zap", A, ck, ak), s
    assert utterance(np.zeros(3), "", np.zeros(2), ck, ak, rng) == "..."
    assert parse_utterance("...") == set()


def test_score():
    t = [{("percept", "sugar")}, {("word", "zap")}]
    assert score(t, t)["f1"] == 1.0
    assert score([set(), set()], t)["recall"] == 0.0


def test_word_code_deterministic_and_graded():
    a, b = word_code("zap"), word_code("zap")
    assert np.array_equal(a, b)
    def jac(x, y):
        x, y = set(word_code(x)), set(word_code(y))
        return len(x & y) / len(x | y)
    # shares the trigram "^za" -> overlaps more than an unrelated word
    assert jac("zap", "zag") > jac("zap", "moo")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"PASS ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
