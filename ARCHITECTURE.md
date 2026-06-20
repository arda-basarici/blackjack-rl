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

### A7 — No checkpoint/resume yet; deterministic rerun is the recovery (deferred)
Training runs in memory and persists only at the end, so a crash mid-run loses the run. This
is an accepted trade at this stage: the Q/N tables are tiny (no OOM risk), the run is
deterministic (same seed -> identical agent, so recovery is simply rerunning), and Stage-2
runs are minutes, not hours. Deferred, not dropped: periodic checkpoint + resume earns its
place once a session runs for hours (much larger episode counts, or the Stage-5 DQN). The
essential detail when we add it: checkpoint and restore the RNG state (`random.getstate()`)
alongside (episode index, Q, N, config), or a resumed run diverges from a same-seed
uninterrupted one and determinism is lost.

### A8 — Exploration is configurable; value-updates can be recency-weighted
Exploration is a pluggable schedule (`schedules.py`: constant / linear / exponential /
harmonic), and the Q-update can use a constant step-size alpha instead of the 1/N sample
average. Both are config-driven and default to the original behaviour (constant epsilon,
sample average), so earlier runs reproduce. Motivation, from the policy-diff investigation:
fixed-epsilon on-policy MC control has an **exploration-vs-bias tradeoff** — low epsilon keeps
the common "stiff" hands clean but never samples the rare soft doubles; high epsilon samples
them but biases the values (Q[hit] is depressed by the noisy on-policy continuation) and
breaks the stiffs. Decaying epsilon is the textbook fix, but only with a recency-weighted
update: a 1/N sample average cannot forget the early high-epsilon returns, so decay *alone*
behaves like a mid-epsilon fixed run (confirmed empirically). The decision logged here is that
the machinery exists and why; the best concrete setting (schedule + alpha) is being decided by
experiment. *Empirical verdict.* Across configs the agent rediscovers ~93-95% of basic strategy at ~1% house edge. decay+alpha (linear 0.3->0, alpha=0.001) is best overall - lowest edge (~0.86% at a 1M-hand eval) and fewest genuine disagreements (12) - but only modestly: it fixes several soft doubles while alpha-noise introduces new close-call flips on stiff hands, so the residual *relocates* rather than disappearing. That residual is largely irreducible within this method + state (coarse state can't separate soft-double contexts, near-tie cells need impractically many samples, constant-alpha adds jitter); closing it needs a richer state (Stage 3) or a different method, not more episodes - confirmed by the learning curve, which shows convergence by ~2-3M episodes. Caveat learned: at 200k eval hands the edge estimates misranked the top configs; only 1M re-evaluation revealed decay+alpha best. Rank on tight estimates, not noisy ones.

### A9 — Saved runs are self-contained, re-loadable, comparable
Each run's record carries the full Q-table and visit counts (D8 / D10), so `load_agent`
rebuilds the trained policy from a record with **no retraining**. This enables re-evaluating a
policy at more hands or other seeds for a tighter CI (`python -m blackjack_rl.evaluate`), and
comparing variants by reading their records (the investigation notebook's experiment ledger).
Re-evaluation recomputes only the *edge*; fidelity (agreement / categories) is a property of
the policy and is read straight from the record.

### A10 — Learning-curve instrumentation: size training, don't guess it
`train` emits a checkpoint every `progress_every` episodes (policy churn = greedy cells
changed since the last checkpoint, min state visit count, states covered, current epsilon),
collected into the run record as `learning_curve`. This makes convergence *visible*: the
policy-churn knee shows how many episodes are actually needed, and confirms edge converges
earlier than rare-cell fidelity — so experiments can stop at the plateau instead of guessing.
Churn uses a deterministic argmax (not the random-tie-break greedy) so ties don't register as
spurious change; with constant-alpha, churn settles to a small plateau rather than zero.

### A11 — Splits extend the code (config-driven state); models retrain, prior runs stay reproducible
Splits are added by *extending*, not rebuilding: `encode_state` gains pair-awareness, the agent
gains the `split` action, and env/trainer handle the branching episode. The state is made
**config-driven** (no-split vs split mode) so earlier no-split runs remain reproducible from
their saved config - backward compatible. Because splits change the state and action space, the
existing trained Q-tables cannot be "taught" incrementally; the policy is **retrained** on the
new space (the no-split runs stay as valid historical results). The one genuinely new design
point is the **branching credit assignment**: each split sub-hand's decisions are credited with
*that* sub-hand's return (the engine stamps this per decision) and the split decision with the
net - not the whole hand's total dumped on every step (D6's "tree, not chain"). Evaluation,
persistence, and the Strategy interface are unchanged.

### A12 — Exploring starts: the capstone that proves the residual is coverage (Problem A)
The investigation's central claim — the residual gap to basic strategy is *coverage* (rare cells are
rarely experienced, so rarely learned) — is tested directly with **Monte-Carlo Exploring Starts**:
every episode begins from a deliberately chosen (state, action) pair (uniform over all 1,090 legal
2-card decision starts), then follows the greedy policy (no epsilon — the start *is* the exploration;
the Sutton & Barto variant). Built **without touching the engine**: `PreparedDeck` subclasses the
engine `Deck` to deal a forced prefix (player-1, dealer-up, player-2) then a random shoe, and
`ForcedFirstAction` wraps a `Strategy` to seed the first action; both are handed to the unmodified
`HandSimulator`. The agent, MC update, encode, evaluate/diff, and persistence are **reused by import**
— the run saves a normal record stamped `method="exploring_starts"`, so it sits in the ledger
alongside the others. The dealer hole stays random (dealer-blackjack episodes are discarded, matching
the natural conditional under which a state is actually played). Greedy follow-on with constant-alpha;
the global RNG is seeded once (reproducible).

*Empirical verdict — the prediction held.* Forcing coverage collapsed genuine disagreements from 30
(decay+alpha+splits) to **9**, and **every survivor is a near-tie** (max ev_gap ~0.06, vs ~0.22
before): the previously under-split rare pairs (3,3 / 4,4 / 9,9 / 2,2-v6) flip to *split*, the
under-explored doubles clear, and the edge moves to basic strategy (per the 1M-hand ledger eval). What
remains is the **representation / near-tie floor** we predicted would survive — genuinely-tied
decisions where "wrong" costs almost nothing. The residual *was* mostly the cost of learning from
experience; the experiment earned the claim rather than asserting it. Two honest limits, both logged
in the notebook: the edge claim rests on the 1M re-eval (not the noisy 200k figure in the record), and
ES forces *2-card* starts, so a few thin deep / post-split states (the `under_visited` cells) aren't
seeded — coverage is near-uniform over decision *starts*, not literally every reachable state.

### A13 — DQN stabilization knobs are flag-gated and default-off (the investigation toolkit)

The Stage-5 investigation added several DQN knobs, all on `DQNConfig` with defaults that reproduce
prior runs exactly, each behind a CLI flag so experiments are one-liners and the default path is
unchanged:

- **`lr_schedule` / `lr_end`** (`--lr-schedule`) — decaying learning rate, reusing the (generalized)
  `schedules` module. The lever that *settles* the high-variance terminal action (the tabular `1/n`
  step, ported to the net). Constant is the default. (CONCEPTS §26.)
- **`target_tau`** (`--target-tau`) — soft/Polyak target instead of hard sync. (CONCEPTS §25.)
- **`swa`** (`--swa`) — Stochastic Weight Averaging (readout). Pairs with *constant* lr. (§25.)
- **`double_after`** (`--double-after`) — curriculum: hit/stand first, then enable `double` (gated in
  selection *and* the bootstrap max). Decontaminates the base; doesn't beat the variance floor.
- **`reward_baseline` / `baseline_c`** (`--reward-baseline`) — dealer control variate on the terminal
  reward (`bust`/`stand`), in `evaluation/dealer_baseline.py`. Unbiased + leak-free, but cancels only
  the dealer-explained variance — a *negative* result for the double, which is draw-dominated.
  (CONCEPTS §28; REPORT_NOTES.)
- **`num_threads`** (`--threads`, 0 = all cores) and **`device`** (`--device cpu|cuda|auto`).
  Multi-thread gives a modest speedup on the tiny net; **GPU does not help Problem A** — the bottleneck
  is the sequential Python env loop, so GPU sits at ~0% (confirmed empirically). Both are kept for
  Problem B's larger nets / a vectorized env.

Also: trained weights now persist (`model.pt` beside `record.json`) so a run is re-loadable for
representation analysis without retraining (`evaluation/embedding.py`), partly lifting A7 for the net.

## Module map (current)

- `blackjack_rl/state.py` — `encode_state`, `StateKey`
- `blackjack_rl/env.py` — `Episode`, `rollout`, `rollout_many`, `problem_a_config`, `_Recorder`
- `blackjack_rl/config.py` — `ExperimentConfig` (epsilon schedule + step_size knobs)
- `blackjack_rl/schedules.py` — `make_epsilon_schedule`, `KINDS`
- `blackjack_rl/util.py` — `format_duration`
- `blackjack_rl/persistence.py` — `git_hash`, `save_run`, `load_record`
- `blackjack_rl/agents/tabular.py` — `TabularAgent` (Q/N, epsilon-greedy, constant-alpha option)
- `blackjack_rl/training/monte_carlo.py` — `train`, `_apply_episode`
- `blackjack_rl/training/exploring_starts.py` — `PreparedDeck`, `ForcedFirstAction`, `start_cards_for`, `enumerate_start_pairs`, `es_rollout`, `train_exploring_starts`, `run_exploring_starts` + CLI (the ES capstone, A12)
- `blackjack_rl/evaluation/metrics.py` — `evaluate_policy`, `GreedyPolicy`, `EdgeResult`
- `blackjack_rl/evaluation/policy_diff.py` — `diff_policy`, `classify`, `CellDiff`, `DiffReport`
- `blackjack_rl/experiment.py` — `run_experiment`, `load_agent`, `RunResult`
- `blackjack_rl/__main__.py` — training CLI · `blackjack_rl/evaluate.py` — re-evaluate CLI
- `analysis/policy_investigation.ipynb` — the policy-diff investigation
