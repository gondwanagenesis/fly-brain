"""SUPERFLY -- an uplifted fruit fly that is still the fruit fly.

The native 138,639-neuron FlyWire brain (Shiu et al. 2024 LIF, run by the
bit-exact native kernel in flyloop/) is the core and is never edited in
place. Everything SUPERFLY adds -- the fly's own dopamine learning, grafted
neurons, a trained interface and a small language model wired into its
neurons -- attaches through audited ports, and every addition is measured
for how much of the behaviour is still the fly's.

    L0  the fly        superfly.engine.SuperflyEngine
    L1  its learning   superfly.plasticity.MBPlasticity
    L2  its interface  superfly.fly.SuperFly (senses in, readouts out)
    L3  its grafts     superfly.graft
    L4  its voice      superfly.flylm (neural tokens), superfly.language
"""
