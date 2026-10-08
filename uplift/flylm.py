"""FlyLM: a small language model wired into the fly's neurons, both ways.

Not an LLM reading a summary. The fly's population activity enters the
model's own embedding stream, and the model's hidden states drive the fly's
own sensory neurons.

    FLY -> LANGUAGE  (speech)
        rates of the fly's central-brain cell types  (no sensory neurons:
        a percept has to reach the brain to count)
          -> NeuralProjector (MLP)  -> K "neural tokens" in embedding space
          -> prepended to the token stream of a causal transformer
          -> the transformer writes the fly's utterance

    LANGUAGE -> FLY  (hearing)
        the same transformer reads the human's sentence
          -> pooled hidden state -> SensoryProjector (MLP)
          -> drive for each fly-perceivable concept, plus a word-sense code
          -> UpliftEngine sensory / word-sense ports

Why a SMALL model trained from scratch: everything a model this size knows
about the world, it learned from sentences grounded in this fly's own
simulated experience. It has no prior about honey, cats or Paris to
confabulate from, so content can only come through the neural tokens. The
ablations in uplift/metrics.py measure exactly that (zeroed, shuffled and
silenced-fly prefixes). The same NeuralProjector can be attached to a
pretrained small model later; the ablations then say how much of its fluency
is the fly and how much is its prior.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, UNK, SEP = "<pad>", "<bos>", "<eos>", "<unk>", "<sep>"


class Vocab:
    def __init__(self, sentences, extra=()):
        words = set()
        for s in sentences:
            words.update(tokenize(s))
        words.update(extra)
        self.itos = [PAD, BOS, EOS, UNK, SEP] + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s, bos=True, eos=True):
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in tokenize(s)]
        return ([self.stoi[BOS]] if bos else []) + ids + ([self.stoi[EOS]] if eos else [])

    def decode(self, ids):
        out = []
        for i in ids:
            w = self.itos[int(i)]
            if w == EOS:
                break
            if w in (PAD, BOS, SEP):
                continue
            out.append(w)
        s = " ".join(out)
        s = re.sub(r" ([.,!?'])", r"\1", s).replace(" ' ", "'")
        return s

    def state_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_state(cls, d):
        v = cls.__new__(cls)
        v.itos = list(d["itos"])
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


def tokenize(s):
    return re.findall(r"[a-z0-9_]+|\.\.\.|[.,!?']", s.lower())


@dataclass
class FlyLMConfig:
    vocab: int
    n_feat: int                 # fly features (cell-type rates)
    n_concepts: int             # sensory concepts the hearing head drives
    n_wordsense: int = 32       # word-sense channels
    d: int = 128
    layers: int = 4
    heads: int = 4
    k_neural: int = 4           # neural tokens per utterance
    ctx: int = 48
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, c: FlyLMConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.attn = nn.MultiheadAttention(c.d, c.heads, dropout=c.dropout,
                                          batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(c.d, 4 * c.d), nn.GELU(),
                                 nn.Linear(4 * c.d, c.d), nn.Dropout(c.dropout))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class FlyLM(nn.Module):
    def __init__(self, c: FlyLMConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx + c.k_neural, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight                  # tied
        # FLY -> LANGUAGE: features -> K neural tokens
        self.neural = nn.Sequential(
            nn.LayerNorm(c.n_feat), nn.Linear(c.n_feat, 256), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(256, c.k_neural * c.d))
        self.neural_tag = nn.Parameter(torch.zeros(1, c.k_neural, c.d))
        # LANGUAGE -> FLY: pooled hidden state -> concept drive, word code
        self.hear_concepts = nn.Linear(c.d, c.n_concepts)
        self.hear_words = nn.Linear(c.d, c.n_wordsense)
        self.hear_reward = nn.Linear(c.d, 1)

    # ---------------------------------------------------------- shared trunk
    def _run(self, x):
        T = x.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        x = x + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            x = b(x, mask)
        return self.ln(x)

    def neural_tokens(self, feats):
        B = feats.size(0)
        return self.neural(feats).view(B, self.c.k_neural, self.c.d) + self.neural_tag

    # ---------------------------------------------------------- speech
    def speech_logits(self, feats, ids, neural=None):
        """ids: (B, T) starting with BOS. Returns logits for ids[:, 1:]."""
        nt = self.neural_tokens(feats) if neural is None else neural
        x = torch.cat([nt, self.tok(ids)], 1)
        h = self._run(x)
        return self.head(h[:, self.c.k_neural:-1])

    @torch.no_grad()
    def speak(self, feats, vocab, max_len=24, neural=None, greedy=True, temp=0.8):
        self.eval()
        if feats.dim() == 1:
            feats = feats[None]
        nt = self.neural_tokens(feats) if neural is None else neural
        ids = torch.full((feats.size(0), 1), vocab.stoi[BOS], dtype=torch.long)
        for _ in range(max_len):
            x = torch.cat([nt, self.tok(ids)], 1)
            logits = self.head(self._run(x)[:, -1])
            nxt = (logits.argmax(-1) if greedy else
                   torch.multinomial(F.softmax(logits / temp, -1), 1)[:, 0])
            ids = torch.cat([ids, nxt[:, None]], 1)
            if (nxt == vocab.stoi[EOS]).all():
                break
        return [vocab.decode(r[1:]) for r in ids]

    # ---------------------------------------------------------- hearing
    def hear(self, ids):
        """ids: (B, T) of the human's sentence. Mean-pooled hidden state."""
        h = self._run(self.tok(ids))
        m = (ids != 0).float()[..., None]
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return (torch.sigmoid(self.hear_concepts(pooled)),
                self.hear_words(pooled),
                torch.tanh(self.hear_reward(pooled))[:, 0])


def pad_batch(seqs, pad=0):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s)
    return out


def n_params(m):
    return sum(p.numel() for p in m.parameters())
