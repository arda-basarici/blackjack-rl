# Blackjack RL — Architecture & Implementation Decisions

A running log of *implementation* decisions made during the build — the **how**, distinct
from `DESIGN.md`, which holds the **what/why** (D1–D10) settled before the code. Living
document: append as choices are made or revised. Same convention as the Phase 2 projects.

## Design shape (intended)

One job per module; data flows one direction:

```
config        →  env (wraps Phase 2 play_hand)  →  training/  →  agents/ (Q-table / net)
                                                       │
                                              persistence (save run + visit counts)
                                                       │
                          agents/ (as Strategy)  →  evaluation/ (policy_diff, metrics)  →  analysis/
```

The Phase 2 engine is an installed dependency; we add nothing to it but a `pyproject.toml`.
Both learned policies expose the engine's `Strategy.decide(state)` contract, so evaluation is
model-agnostic.

## Decisions

(none yet — first entries land with Stage 1)
