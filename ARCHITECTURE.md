# Blackjack RL — Architecture & Implementation Decisions

A running log of *implementation* decisions made during the build — the **how**, distinct
from `DESIGN.md`, which holds the **what/why** (D1–D10) settled before the code. Living
document: append as choices are made or revised. Same convention as the Phase 2 projects.

Implementation decisions are numbered **A1, A2, …** here (the design-level decisions are
D1–D10 in `DESIGN.md`; "A#" does not refer to Problem A).

## Problem A vs Problem B (quick reference)

The project is built in two staged problems (full definition in `DESIGN.md` §3):

- **Problem A — per-round play.** Learn how to play a single hand (hit / stand / double /
  split) against the dealer's upcard. **No counting** — a single round has nothing to count.
  Has a **clean ground truth**, the proven-optimal basic-strategy table, so it is auditable
  cell by cell. Built first.
- **Problem B — counting & betting.** Add tracking the count and sizing the bet across hands.
  The state grows, and there is **no clean table to diff against** — the optimum becomes an
  EV-vs-risk-of-ruin tradeoff. Built after A is complete.

## Design shape (as built so far)

One job per module; data flows one direction:

```
config.ExperimentConfig
        │
        ▼
env.rollout ──── wraps a policy as a Strategy, plays one hand through the Phase 2 engine,
  (uses state.py)  returns Episode(steps=[(state_key, action), ...], reward)
        │
        ▼
training/  (Stage 2)  ──►  agents/  (Q-table now, net later; each is a Strategy)
        │                          │
        ▼                          ▼
persistence.save_run         evaluation/  (policy_diff, metrics)  ──►  analysis/
 (record.json, never
  overwritten)
```

The Phase 2 engine is an installed dependency; `env.py` is the **only** module that touches
its internals (the `HandSimulator` / `HandResult`). Everything downstream sees only `Episode`.

## Decisions

### A1 — The env is episode-capture, not control-flipping (implements D7)
`HandSimulator.play_hand()` is atomic, so rather than build a `reset()/step()` env we wrap the
acting policy in a `_Recorder` (a `Strategy`) that logs `(encode_state(state), action)` as the
engine queries it, play one hand, and read the terminal reward off the `HandResult`. No engine
change. Monte Carlo updates after whole episodes anyway, so episode-granularity is the right
fit. For Problem A each rollout deals a **fresh shoe** (independent, counting-free hands) at a
**flat bet of 1**, so the payout *is* the per-unit reward. Reproducibility comes from seeding
the global RNG once per run (the engine shuffles with `random`), never per hand.

### A2 — One state contract, enforced by construction (implements D2)
`state.encode_state` is the single definition of "what is a state" — `(player_value,
player_is_soft, dealer_upcard)` for A. The env records trajectories through the *same* function
the agent will use to look up values, so the two cannot silently disagree. *Forward note:* this
key does **not** distinguish a pair (e.g. 8+8) from a non-pair of equal value (10+6) — fine for
no-split A, but `encode_state` must grow a pair/split feature when Stage 3 adds splits.

### A3 — `ExperimentConfig` is frozen and minimal (implements D9)
Only the knobs in play now: `num_episodes`, `epsilon` (fixed), `seed`. Frozen (immutable,
hashable, serializes cleanly). Deliberate omissions — discount γ (no intermediate reward),
state-feature flags (arrive with B), algorithm/ruleset (provenance, not hyperparameters) — are
documented in the module so the restraint is visible.

### A4 — Persistence: generic I/O now, record assembly later (implements D8)
Stage 1 ships only the reusable primitives `git_hash()` and `save_run()`; assembling the
project-specific record (config + metrics + Q-table + visit counts, with state-key flattening)
lives with the trainer that produces those artifacts (Stage 2). Two improvements over the
pathfinding version: `save_run` **auto-stamps** provenance (`run_id`, `timestamp`, `git_hash`)
so a run can't be saved without it, and it is **collision-safe** (appends `-1`, `-2`, …), so
"never overwrite" is guaranteed rather than merely likely.

### A5 — Honest testing boundary for the env (implements DESIGN §7)
The env's house edge is *not* unit-tested against 0.45%: over a test-sized sample the standard
error exceeds the 0.45% signal, so a tight assert would flake. Instead the suite checks what it
can deterministically — episode structure, reproducibility under a seed, and a one-sided smoke
bound (`mean reward > -0.10`) that catches gross failures (random play, sign flips). The precise
0.45% validation is an *evaluation*-time check (Stage 2), not a unit test. The test refuses to
claim what it can't reliably measure.

### A6 — Cross-project dependency needs static-resolution paths (tooling)
The Phase 2 engine and this project are installed **editable**, which modern setuptools backs
with an import hook that static analysers (Pylance/Pyright, mypy) can't follow — imports resolve
at runtime but show as unresolved. Fixed with explicit source paths: `pyrightconfig.json` in the
project and `python.analysis.extraPaths` in the repo-root `.vscode/settings.json` (used when the
whole repo is the workspace, since Pyright reads only the root config).

## Module map (current)

- `blackjack_rl/state.py` — `encode_state`, `StateKey`
- `blackjack_rl/env.py` — `Episode`, `rollout`, `rollout_many`, `problem_a_config`, `_Recorder`
- `blackjack_rl/config.py` — `ExperimentConfig`
- `blackjack_rl/persistence.py` — `git_hash`, `save_run`
- `blackjack_rl/agents/`, `training/`, `evaluation/` — Stage 2+
