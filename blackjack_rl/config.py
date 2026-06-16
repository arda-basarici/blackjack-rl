"""ExperimentConfig — the knobs for a single training run.

Holds only what we know we need (algorithm, exploration rate, episode count, seed, which
state features are active). Code reads from here rather than hardcoding constants, so a new
knob is a one-line addition. Minimal now, architected to grow. See DESIGN.md D9.
"""
