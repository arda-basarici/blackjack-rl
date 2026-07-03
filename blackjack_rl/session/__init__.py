"""Problem B — counting & betting over sessions (DESIGN's Phase 3; ARCHITECTURE A14–A19).

A *session* is many hands from one persisting, depleting shoe against a running, finite
bankroll: the agent sizes its bet from the count, the shoe depth, and the bankroll, on the
log-growth (Kelly) objective. This subpackage holds the session-specific code and reuses what
sits below it:
- core/ — the engine boundary, schedules, run persistence,
- dqn/  — the QNetwork and TD update the bettor trains with.

``SessionConfig`` lives in ``env.py``, not ``core.config``: it is the env/MDP config (bankroll,
ruin, bet spread, horizon), so it lives with the env exactly as ``problem_a_config`` lives in
``core.env``; ``core.config`` holds *training* hyperparameters, a different concern.
"""
