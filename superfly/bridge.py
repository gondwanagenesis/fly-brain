"""The neural bridge: train the fly's voice on the fly's own brain.

Two voices, trained on the same corpus (superfly/experiments/make_corpus.py)
and judged by the same fly-mind tests:

  A  FlyLM (superfly/flylm.py) -- a ~2M-parameter transformer trained FROM
     SCRATCH on sentences grounded in this fly's experience. It knows nothing
     else, so it has nothing to confabulate from.
  B  a pretrained small LM (SmolLM2-360M-Instruct), FROZEN. Only a projector
     is trained: fly activity -> K neural tokens written straight into the
     LM's input-embedding stream (the LLaVA recipe, with a brain where the
     image encoder would be). Fluent, but it brings a prior.

THE FLY-MIND TESTS (held-out episodes)
  content F1        concepts/actions/words in the utterance vs the episode
  zero-brain        neural tokens from an all-zero (silent) brain: any content
                    produced is the LM's prior speaking -> should be "..."
  shuffled-brain    tokens from a DIFFERENT episode: does the content follow
                    the brain it was given (follow rate) or the true stimulus
                    (leak rate)? A voice that is the fly's follows the brain.
A linear probe on the same features gives the ceiling: what is decodable at
all from this fly's central brain.

    python -m superfly.bridge tiny        # train + evaluate voice A
    python -m superfly.bridge pretrained  # train + evaluate voice B
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from superfly.engine import ROOT
from superfly.flylm import FlyLM, FlyLMConfig, Vocab, pad_batch, n_params

from superfly.anatomy import SUFFIX                 # noqa: E402
CACHE = ROOT / "data" / f"superfly_cache{SUFFIX}"   # per connectome
OUT = ROOT / "data" / "results" / f"superfly{SUFFIX}"

# ------------------------------------------------------------------ grammar
PERCEPT = {
    "sugar": ["sweet", "i taste sugar", "something sweet"],
    "bitter": ["bitter", "i taste something bitter", "yuck, bitter"],
    "water": ["water", "i taste water"],
    "salt": ["salty", "i taste salt"],
    "umami": ["savory", "i taste something savory"],
    "co2": ["i smell co2"],
    "geosmin": ["i smell mold"],
    "male_pheromone": ["i smell another fly"],
    "fruit_odor": ["i smell fruit"],
    "sound": ["i hear a buzz", "a sound"],
    "wind": ["wind on my antennae", "a breeze"],
    "heat": ["hot", "it is hot"],
    "cold": ["cold", "it is cold"],
    "dry": ["dry air"],
    "humid": ["damp air"],
    "shadow": ["a shadow", "something above me", "a dark shape"],
    # interoceptive states (body -> MBON11/PPL101/ISN/ITP; superfly/life.py)
    "hungry": ["i am hungry"],
    "thirsty": ["i am thirsty"],
}
ACT = {
    "feed": ["i want to eat", "i extend my proboscis"],
    "groom": ["i groom", "i clean my antennae"],
    "walk_forward": ["i walk forward"],
    "turn": ["i turn"],
    "walk_backward": ["i back away"],
    "escape": ["i fly away"],
}
ACT_THR = 5.0      # Hz of the action's descending population

# reverse map for scoring generated text
_PHRASES = sorted([(p, ("percept", k)) for k, ps in PERCEPT.items() for p in ps]
                  + [(p, ("action", k)) for k, ps in ACT.items() for p in ps],
                  key=lambda x: -len(x[0]))


def parse_utterance(text):
    t = " " + text.lower() + " "
    found = set()
    for p, lab in _PHRASES:
        if p in t:
            found.add(lab)
            t = t.replace(p, " ")
    for w in re.findall(r"'([a-z]+)'", text.lower()):
        found.add(("word", w))
    return found


def labels_of(C, W, A, concept_keys, action_keys):
    lab = set()
    for k, v in zip(concept_keys, C):
        if v > 0:
            lab.add(("percept", str(k)))
    for k, v in zip(action_keys, A):
        if v >= ACT_THR:
            lab.add(("action", str(k)))
    if W:
        lab.add(("word", str(W)))
    return lab


def utterance(C, W, A, concept_keys, action_keys, rng):
    bits = []
    for k, v in zip(concept_keys, C):
        if v > 0:
            p = str(rng.choice(PERCEPT[str(k)]))
            if v < 0.5 and rng.random() < 0.5:
                p = "a little " + p if not p.startswith("i ") else p + " a little"
            bits.append(p)
    if W:
        bits.append(f"i hear '{W}'")
    for k, v in zip(action_keys, A):
        if v >= ACT_THR:
            bits.append(str(rng.choice(ACT[str(k)])))
    if not bits:
        return "..."
    return ". ".join(bits) + "."


# ------------------------------------------------------------------ hearing
HEAR_TEMPLATES = ["here is some {s}", "{s} for you", "can you taste the {s}?",
                  "there is {s} nearby", "feel the {s}", "a bit of {s}",
                  "lots of {s}!", "careful, {s}", "try this {s}",
                  "here is some {s} for you", "do you like {s}?", "i brought {s}",
                  "a strong {s}", "some {s} is coming", "watch out, {s}",
                  "the {s} is here", "a little {s}", "so much {s}"]
FILLER = ["", "", "", "fly, ", "hey, ", "look, ", "ok, "]
PRAISE = ["good fly", "well done", "yes, good", "nice job"]
SCOLD = ["bad fly", "no!", "stop that", "don't"]


def human_sentence(rng, concept_keys, words_pool):
    """A random human utterance and the fly-side targets it should produce."""
    from superfly.language import CONCEPT
    target = np.zeros(len(concept_keys), np.float32)
    parts, reward, word = [], 0.0, ""
    hearable = [k for k in concept_keys if k in CONCEPT]   # you can't talk a fly hungry
    for k in rng.choice(hearable, rng.choice([0, 1, 1, 1, 2]), replace=False):
        syn = str(rng.choice(CONCEPT[str(k)].words))
        t = str(rng.choice(HEAR_TEMPLATES))
        parts.append(str(rng.choice(FILLER)) + t.format(s=syn))
        target[concept_keys.index(str(k))] = 1.0          # presence; strength
        # comes from the adverbs at run time (language._intensity)
    if rng.random() < 0.3:
        word = str(rng.choice(words_pool))
        parts.append(str(rng.choice(["say '{w}'", "listen: '{w}'", "the word '{w}'"])).format(w=word))
    r = rng.random()
    if r < 0.15:
        parts.append(str(rng.choice(PRAISE))); reward = 1.0
    elif r < 0.3:
        parts.append(str(rng.choice(SCOLD))); reward = -1.0
    if not parts:
        parts = [str(rng.choice(["hello", "hi fly", "how are you?", "what do you sense?"]))]
    joiner = str(rng.choice([", ", " and ", ". "]))
    return joiner.join(parts), target, word, reward


# ------------------------------------------------------------------ data
def load(path=None, seed=0):
    d = np.load(path or CACHE / "corpus.npz", allow_pickle=False)
    X = np.log1p(d["X"].astype(np.float32))
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_te = max(1, n // 5)
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-3
    keep = X[tr].std(0) > 0                       # features that ever vary
    Xn = ((X - mu) / sd)[:, keep]
    return dict(X=Xn.astype(np.float32), C=d["C"], W=d["W"], A=d["A"],
                ck=[str(x) for x in d["concept_keys"]],
                ak=[str(x) for x in d["action_keys"]],
                tr=tr, te=te, mu=mu, sd=sd, keep=keep,
                model=str(d["model"]), gain=float(d["gain"]))


def all_sentences(D):
    rng = np.random.default_rng(1)
    out = []
    for i in range(len(D["X"])):
        for _ in range(3):
            out.append(utterance(D["C"][i], D["W"][i], D["A"][i], D["ck"], D["ak"], rng))
    return out


# ------------------------------------------------------------------ scoring
def score(pred_sets, true_sets):
    tp = sum(len(p & t) for p, t in zip(pred_sets, true_sets))
    fp = sum(len(p - t) for p, t in zip(pred_sets, true_sets))
    fn = sum(len(t - p) for p, t in zip(pred_sets, true_sets))
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9))


def fly_mind_tests(speak, D, n_shuffle=1):
    """speak(feature_rows) -> list[str]. Returns the fly-mind report."""
    te = D["te"]
    X = D["X"]
    truth = [labels_of(D["C"][i], D["W"][i], D["A"][i], D["ck"], D["ak"]) for i in te]
    out = speak(X[te])
    pred = [parse_utterance(s) for s in out]
    rep = {"held_out": score(pred, truth)}
    # zero brain: the silent fly, normalised (all rates 0 -> (log1p(0)-mu)/sd)
    zero = ((np.zeros_like(D["mu"]) - D["mu"]) / D["sd"])[D["keep"]].astype(np.float32)
    zout = speak(np.repeat(zero[None], 8, 0))
    rep["zero_brain_outputs"] = sorted(set(zout))
    rep["zero_brain_content_rate"] = float(np.mean([len(parse_utterance(s)) > 0 for s in zout]))
    # shuffled brain: does content follow the features or leak the truth?
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(te))
    sout = speak(X[te][perm])
    spred = [parse_utterance(s) for s in sout]
    follow = score(spred, [truth[j] for j in perm])
    leak = score(spred, truth)
    rep["shuffled_follow_f1"] = follow["f1"]
    rep["shuffled_leak_f1"] = leak["f1"]
    rep["examples"] = [{"truth": sorted(map(list, t)), "said": s}
                       for t, s in list(zip(truth, out))[:12]]
    return rep


def linear_probe(D):
    """Linear baseline: per-label logistic regression on the same features,
    weight-decayed, each label's threshold tuned on the training split."""
    X = torch.as_tensor(D["X"])
    tr, te = D["tr"], D["te"]
    labs = sorted({l for i in range(len(X)) for l in
                   labels_of(D["C"][i], D["W"][i], D["A"][i], D["ck"], D["ak"])})
    Y = np.zeros((len(X), len(labs)), np.float32)
    li = {l: j for j, l in enumerate(labs)}
    for i in range(len(X)):
        for l in labels_of(D["C"][i], D["W"][i], D["A"][i], D["ck"], D["ak"]):
            Y[i, li[l]] = 1
    Y = torch.as_tensor(Y)
    lin = nn.Linear(X.shape[1], len(labs))
    opt = torch.optim.AdamW(lin.parameters(), 1e-3, weight_decay=0.1)
    for _ in range(600):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(lin(X[tr]), Y[tr]).backward()
        opt.step()
    with torch.no_grad():
        ptr = torch.sigmoid(lin(X[tr])).numpy()
        pte = torch.sigmoid(lin(X[te])).numpy()
    thr = np.full(len(labs), 0.5)
    for j in range(len(labs)):                 # per-label F1-optimal threshold
        best = -1
        for t in np.linspace(0.05, 0.95, 19):
            p_, y_ = ptr[:, j] > t, Y[tr, j].numpy() > 0
            f = 2 * (p_ & y_).sum() / max(p_.sum() + y_.sum(), 1)
            if f > best:
                best, thr[j] = f, t
    P = pte > thr
    pred = [{labs[j] for j in np.flatnonzero(p)} for p in P]
    truth = [{labs[j] for j in np.flatnonzero(y)} for y in Y[te].numpy()]
    return score(pred, truth)


# ------------------------------------------------------------------ voice A
def train_tiny(D, steps=4000, bs=32, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    words_pool = sorted({str(w) for w in D["W"] if w})
    rng0 = np.random.default_rng(seed + 1)
    hear_txt = [human_sentence(rng0, D["ck"], words_pool)[0] for _ in range(20000)]
    vocab = Vocab(all_sentences(D) + hear_txt + ["..."],
                  extra=[f"'{w}'" for w in words_pool])
    cfg = FlyLMConfig(vocab=len(vocab), n_feat=D["X"].shape[1],
                      n_concepts=len(D["ck"]), ctx=48)
    m = FlyLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.05)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(D["X"])
    tr = D["tr"]
    t0 = time.perf_counter()
    m.train()
    for s in range(steps):
        b = rng.choice(tr, bs)
        txt = [utterance(D["C"][i], D["W"][i], D["A"][i], D["ck"], D["ak"], rng) for i in b]
        ids = pad_batch([vocab.encode(t) for t in txt])
        logits = m.speech_logits(X[b], ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1), ignore_index=0)
        # hearing: human sentence -> the fly's senses and the teacher
        hs = [human_sentence(rng, D["ck"], words_pool) for _ in range(bs)]
        hids = pad_batch([vocab.encode(h[0], bos=False, eos=False) for h in hs])
        hc, _, hr = m.hear(hids)
        tc = torch.as_tensor(np.stack([h[1] for h in hs]))
        tr_ = torch.as_tensor([h[3] for h in hs], dtype=torch.float32)
        loss = loss + 2.0 * F.binary_cross_entropy(hc, tc) + F.mse_loss(hr, tr_)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s % 500 == 0 or s == steps - 1:
            print(f"  step {s:5d} loss {loss.item():.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    m.eval()

    def speak(rows):
        with torch.no_grad():
            return m.speak(torch.as_tensor(rows), vocab)
    return m, vocab, speak


# ------------------------------------------------------------------ voice B
class NeuralPrefix(nn.Module):
    def __init__(self, n_feat, d_model, k=4, hidden=512):
        super().__init__()
        self.k, self.d = k, d_model
        self.net = nn.Sequential(nn.LayerNorm(n_feat), nn.Linear(n_feat, hidden),
                                 nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, k * d_model))
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        return self.net(x).view(x.size(0), self.k, self.d) * self.scale


def train_pretrained(D, model_dir="/home/user/models/SmolLM2-360M-Instruct",
                     steps=400, bs=8, lr=1e-3, seed=0, k=4):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_dir)
    lm = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    emb = lm.get_input_embeddings()
    d_model = emb.weight.shape[1]
    proj = NeuralPrefix(D["X"].shape[1], d_model, k=k)
    # the scale of real token embeddings, so neural tokens land in-distribution
    proj.scale.data.fill_(float(emb.weight.std()))
    opt = torch.optim.AdamW(proj.parameters(), lr, weight_decay=0.01)
    prompt = tok("The fly says:", return_tensors="pt").input_ids[0]
    eos = tok.eos_token_id
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(D["X"])
    tr = D["tr"]
    print(f"  frozen LM {sum(p.numel() for p in lm.parameters()) / 1e6:.0f}M params; "
          f"trainable projector {n_params(proj) / 1e6:.2f}M", flush=True)
    t0 = time.perf_counter()
    for s in range(steps):
        b = rng.choice(tr, bs)
        txt = [utterance(D["C"][i], D["W"][i], D["A"][i], D["ck"], D["ak"], rng) for i in b]
        tgt = [tok(" " + t, add_special_tokens=False).input_ids + [eos] for t in txt]
        L = max(len(t) for t in tgt)
        ids = torch.full((bs, len(prompt) + L), eos, dtype=torch.long)
        lab = torch.full_like(ids, -100)
        for i, t in enumerate(tgt):
            ids[i, :len(prompt)] = prompt
            ids[i, len(prompt):len(prompt) + len(t)] = torch.as_tensor(t)
            lab[i, len(prompt):len(prompt) + len(t)] = torch.as_tensor(t)
        nt = proj(X[b])
        inp = torch.cat([nt, emb(ids)], 1)
        labels = torch.cat([torch.full((bs, k), -100, dtype=torch.long), lab], 1)
        out = lm(inputs_embeds=inp, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            print(f"  step {s:4d} loss {out.loss.item():.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)

    @torch.no_grad()
    def speak(rows, max_new=24):
        res = []
        for r in rows:
            nt = proj(torch.as_tensor(r)[None])
            inp = torch.cat([nt, emb(prompt[None])], 1)
            g = lm.generate(inputs_embeds=inp, max_new_tokens=max_new, do_sample=False,
                            eos_token_id=eos, pad_token_id=eos)
            res.append(tok.decode(g[0], skip_special_tokens=True).strip())
        return res
    return proj, (lm, tok), speak


def main(which="tiny", corpus=None, tag=""):
    D = load(corpus)
    print(f"corpus: {len(D['X'])} episodes, {D['X'].shape[1]} varying features, "
          f"fly model {D['model']} gain {D['gain']}", flush=True)
    probe = linear_probe(D)
    print(f"linear baseline: F1 {probe['f1']:.3f}", flush=True)
    if which == "tiny":
        m, vocab, speak = train_tiny(D)
        info = {"voice": "FlyLM from scratch", "params": n_params(m)}
        torch.save({"state": m.state_dict(), "cfg": m.c.__dict__, "vocab": vocab.state_dict(),
                    "mu": D["mu"], "sd": D["sd"], "keep": D["keep"], "ck": D["ck"]},
                   CACHE / f"flylm_tiny{tag}.pt")
    else:
        proj, _, speak = train_pretrained(D)
        info = {"voice": "SmolLM2-360M frozen + neural prefix",
                "trainable_params": n_params(proj)}
        torch.save({"proj": proj.state_dict(), "mu": D["mu"], "sd": D["sd"],
                    "keep": D["keep"]}, CACHE / "flylm_prefix_smollm2.pt")
    rep = fly_mind_tests(speak, D)
    rep.update(info, linear_probe=probe, fly_model=D["model"], fly_gain=D["gain"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"bridge_{which}{tag}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=1, default=str))
    for ex in rep["examples"][:8]:
        print("   truth:", ex["truth"], "\n   said :", ex["said"])
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "tiny", a[1] if len(a) > 1 else None, a[2] if len(a) > 2 else "")
