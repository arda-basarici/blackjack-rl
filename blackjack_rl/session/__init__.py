"""Step 3 — Problem B: counting & betting over sessions. See DESIGN.md §3, §8 stage 6, D11–D17.

A *session* is many hands from one depleting shoe against a running, finite bankroll. The agent
both **plays** (count-aware, EV) and **bets** (Kelly / log-growth, ruin-aware). This subpackage
holds the B-specific code and reuses what's below it:
- core/  — the engine boundary, config, run persistence,
- dqn/   — QNetwork and the (count-aware) play agent.

Built **bet-first** (B0 → B6). Planned extensions to shared files, made when their stage lands:
- SessionConfig — stubbed here in env.py for now; consider moving to core/config.py at B0
  (it is the *third* config, the documented trigger to revisit config organization).
- dqn/agent.py — optional count features for the count-aware play model, added at B3.
"""
