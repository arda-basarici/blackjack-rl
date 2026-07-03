# ARCHITECTURE — Blackjack RL

How the project is built and why that structure — the module graph, the seams, and the
structural decisions, kept as a clean snapshot of the code as it stands. Edited in place; the
build chronology lives in the session log. What was decided and why → [DESIGN.md](DESIGN.md);
the front door → [README.md](README.md).

Structural decisions are anchored **A1, A2, …** where they are made below. (The prefix is
historical and unrelated to "Problem A" — decision-ID prefixes should not collide with domain
vocabulary, a lesson encoded in the documentation standard for future projects.)

*Snapshot of the completed project · last updated 2026-07-03.*

---

## Design shape

One direction of dependency, from the validated engine down through shared foundations to the
three phase packages:

```
        Phase-2 blackjack engine        (installed package — READ-ONLY)
        Deck/shoe · HandSimulator · GameState · BasicStrategy
                 │
                 │   the Strategy / BetPolicy contracts —
                 │   the only boundary anything crosses
                 ▼
        core/ — the shared foundation
          │  state.py         the one state definition (encode_state, StateKey)
          │  env.py           episode capture through the unmodified engine
          │  config.py        frozen experiment knobs, saved with every run
          │  schedules.py     decay curves — epsilon and learning rates alike
          │  persistence.py   provenance-stamped, never-overwritten run records
          │
          ├──► evaluation/    house edge · cell-by-cell policy audit
          │                   (grades both play phases)
          │
          ├──► tabular/       Phase 1 — Monte Carlo control on a Q/N table
          │
          ├──► dqn/           Phase 2 — QNetwork · TD trainer · stabilizer knobs
          │
          └──► session/       Phase 3 — session env (shoe · bankroll · ruin) ·
                              references → Kelly · bettors (flat / Kelly / learned) ·
                              session metrics · cell_eval
                                └─ reuses dqn's QNetwork + td_update
                                   (the one deliberate cross-package edge)
```

The layering rules, stated once:

- **`core` knows nothing above it.** It holds what every phase consumes — and nothing that
  belongs to one phase. The concrete test case: the bet-training config lives in
  `session/train.py`, *not* in `core/config.py`, because it composes a session-layer type and
  the base layer must never import upward (**A18**). Config placement follows the dependency
  arrow, not the "all configs together" instinct.
- **Phase packages never import each other — with one deliberate exception.** `session` imports
  `dqn`'s network and TD update, because reusing the play stack for the bettor *is* a design
  point (the "one network" thread), not a convenience leak. `tabular` and `dqn` share nothing
  but `core` and `evaluation`, which is what makes their comparison honest.
- **Only `core/env.py` and `session/env.py` touch engine internals.** Everything downstream
  sees plain captured data, never the engine's types.
- **Effects live at the shell.** Library code is importable and pure-by-contract; the
  multiprocessing pools, file layouts, figure rendering, and PDF generation live in `scripts/`
  and the two env/persistence edges.

### The life of a run

The dependency graph is the static shape; the second axis is the **artifact flow** — the
provenance loop that makes every number in the frozen reports regenerable from recorded inputs:

```
  config + seed
        │
        ▼
  env capture ──► trainer ──► trained policy
        │
        ▼
  evaluation / cell_eval      the edge · the cell audit · the baseline ladder
        │
        ▼
  runs/<run-id>/              record.json (config · seed · git hash — provenance) ·
        │                     model.pt / Q-table (reload, never retrain) · eval artifacts
        ▼
  analysis_loader.py          records → tidy frames · shared plotters · figure provenance
        │
        ├──► analysis/        the chapter notebooks
        │
        └──► scripts/         the report generators — headline numbers
                  │           asserted against the loaded data at build
                  ▼
          the three frozen PDFs
```

Two properties of this loop are load-bearing. Nothing downstream of `runs/` recomputes —
notebooks and reports *read*, through one loader, so a re-evaluated run updates every consumer
consistently. And the flow crosses git visibility once, deliberately: `runs/` and `logs/` are
git-ignored (every artifact there is regenerable from its recorded inputs), and the one thing
committed code depends on — the 20M-hand edge reference — was promoted out of `runs/` into a
committed artifact (`session/data/`) precisely so that boundary is never crossed implicitly.

---

## Module responsibilities

One line per module; signatures live in the docstrings, rendered to an API reference with
`python scripts/make_docs.py` → `docs/api/` (pdoc; output git-ignored, regenerate on demand).

**`core/` — the shared foundation**

| module | single job |
|---|---|
| `state.py` | the one definition of "a state": `encode_state`, `StateKey`, the read-only `StateLike` protocol |
| `config.py` | frozen experiment configs (tabular + DQN) — only knobs actually chosen, saved with every run |
| `env.py` | episode capture: play one hand through the unmodified engine, return the trajectory + reward |
| `schedules.py` | decay schedules (constant / linear / exponential / harmonic) — used for epsilon *and* learning rates |
| `persistence.py` | `save_run` / `load_record`: provenance auto-stamped, collision-safe, never overwritten |
| `paths.py` | single source for `runs/`, `logs/`, and the committed edge-reference path |
| `util.py` | small shared helpers |

**`evaluation/` — grading (Problem A)**

| module | single job |
|---|---|
| `metrics.py` | house edge through the engine, with sample-size-aware confidence intervals |
| `policy_diff.py` | the cell audit: learned vs basic-strategy action per cell — agree / near-tie / genuine / under-visited |

**`tabular/` — Phase 1** · **`dqn/` — Phase 2**

| module | single job |
|---|---|
| `tabular/agent.py` | the Q/N lookup table: epsilon-greedy play, sample-average or constant-alpha updates |
| `tabular/monte_carlo.py` | Monte Carlo control over captured episodes |
| `tabular/exploring_starts.py` | the coverage capstone: forced (state, action) starts without engine changes |
| `tabular/experiment.py` · `evaluate.py` · `__main__.py` | run orchestration, re-evaluation CLI, training CLI |
| `dqn/agent.py` | `QNetwork` + the network policy (scalar/one-hot encodings, action masking) |
| `dqn/deep_q.py` | the TD trainer: replay sampling, target network, the flag-gated stabilizer knobs |
| `dqn/replay.py` | the experience buffer |
| `dqn/network_diff.py` | fidelity by interrogation: query every cell, reuse the tabular diff unchanged |
| `dqn/embedding.py` | reload trained nets; representation (embedding) analysis |
| `dqn/dealer_baseline.py` | the dealer control variate on terminal rewards |
| `dqn/experiment.py` · `exploring_starts_dqn.py` | run orchestration; the coverage capstone ported to the net |

**`session/` — Phase 3**

| module | single job |
|---|---|
| `env.py` | the session MDP: persistent shoe, bankroll, ruin barrier; `BetPolicy` contract; the growth/ruin regime configs |
| `references.py` | reconstructed ground truth: measured edge-by-count, the analytic Kelly curve, the literature index plays |
| `bet_agent.py` | the bettors: `FlatBet` (floor), `KellyBet` (analytic), `BetAgent` (the learned DQN bettor) |
| `train.py` | `BetTrainConfig` + the online bet-training loop (reuses the DQN stack) |
| `metrics.py` | growth / ruin / drawdown / bankroll distribution — pure functions over played sessions |
| `cell_eval.py` | the generic parallel evaluation engine every betting experiment runs on |
| `persistence.py` | save/reload trained bettors: rebuild spec + full provenance |
| `data/` | the committed 20M-hand edge-by-count reference artifact |

**Top level:** `analysis_loader.py` (the notebooks' and report generators' single data layer) ·
`analysis/` (chapter notebooks per phase) · `scripts/` (training/eval runners, `measure_*` /
`plot_*` experiment pairs, the three report generators, docs/notebook utilities) · `tests/`.

---

## Key abstractions & seams

- **The `Strategy` contract** — the engine boundary. Every policy in the project, learned or
  analytic, is a `Strategy` (play) or a `BetPolicy` (bet); the engine and the graders never
  know what is behind the interface.
- **Captures are plain frozen data.** A played hand is an `Episode`/`CapturedHand`; a played
  session is a `SessionCapture` of `HandRecord`s. They are the only things that cross from the
  effectful envs into the functional core — trainers, metrics, and diffs all consume captures,
  never the engine.
- **One state definition per problem.** `encode_state` (hands) and `encode_bet_state` (bets)
  are each the single source of their encoding; envs record through the same function agents
  look up with, so the two can never silently disagree.
- **One network stack.** `QNetwork` and the TD update in `dqn/deep_q.py` serve both the play
  agent and the bettor — the bettor is the play machinery pointed at a new decision.
- **One schedules module, three consumers.** Tabular exploration decay, DQN learning-rate
  decay, and the bettor's harmonic learning rate all draw from `core/schedules.py`.
- **Pure core, effects at the shell.** Metrics are pure over captures; the edge accumulator
  merges losslessly (below); parallelism, files, and rendering stay in `scripts/` and
  `cell_eval` workers.

---

## Structural stories

### The environment seam — capture, don't control

The engine's hand-playing routine is atomic, so the environment **captures episodes instead of
flipping control** (**A1**): the acting policy is wrapped in a recorder `Strategy`, one hand is
played through the unmodified engine, and the trajectory plus terminal reward are read off the
returned result. No `reset()/step()` machinery, no engine changes; Monte Carlo — and, later,
TD over reconstructed transitions — fit episode granularity naturally. Reproducibility comes
from seeding the global RNG once per run, never per hand.

The recorder logs states through **the same encoding function the agent looks up with**
(**A2**), so the state contract is enforced by construction rather than by discipline. The
encoding grew pair-awareness when splits arrived, behind a config flag, so earlier no-split
runs stay reproducible from their saved configs.

Phase 3 scales the same idea up a level: the session environment is a **capture driver over
whole sessions** (**A14**) — persistent shoe, bankroll bookkeeping, hard ruin barrier — whose
per-hand records *are* the bettor's transitions. One code path serves both training-data
capture and baseline evaluation, which is what makes every rung of the baseline ladder
measurable on identical terms. The bet side gets its own minimal contract: `BetPolicy`
(count, shoe depth, bankroll → wager), deliberately decoupled from the engine's own betting
interface because the bet is decided *before* the deal. Its indexed variant reports the exact
menu index chosen, so the value-based trainer reconstructs exact actions even where the
environment clamps a wager to the remaining bankroll.

### Configuration, persistence, provenance

Configs are **frozen dataclasses holding only the knobs actually in play** (**A3**), with the
deliberate omissions documented where they would be expected — each later investigation added
a knob rather than a refactor, and every default preserves the behavior of earlier runs.

Runs are **never overwritten and carry their provenance by construction** (**A4**): the save
path auto-stamps run id, timestamp, and git hash, and collides safely. Records are
**self-contained and reloadable** (**A9**) — full Q-table and visit counts, or network weights
beside the record — so any policy can be re-evaluated at more hands, or analyzed structurally,
without retraining. The bet side extends the same convention (**A19**): a saved bettor
separates its *rebuild spec* (what `load_bet_agent` needs) from its *training config* (pure
provenance), training checkpoints are addressable (`run_dir@session`), and evaluations
themselves persist as structured, provenance-stamped artifacts with cached baselines — the
expensive analytic rungs are computed once, keyed, and reused.

Checkpoint/resume was **deliberately skipped for the tabular phase** (**A7**): runs were
minutes long and deterministic, so the recovery path was rerunning the seed. The decision was
revisited exactly where its premise broke — network runs persist weights, bet runs checkpoint —
which is the intended shape of a deferred decision.

### The honest testing boundary

The environment's house edge is **not unit-tested against the anchor** (**A5**): at test-sized
samples the standard error exceeds the signal, so a tight assert would flake. The suite checks
what is deterministic — episode structure, seeded reproducibility, a one-sided smoke bound that
catches gross failures — and leaves precise validation to evaluation-time checks sized for it.
The test refuses to claim what it cannot reliably measure.

### Cross-project dependency, statically resolvable

The engine arrives via editable install, which static analyzers cannot follow through import
hooks; explicit source paths in the Pyright config restore full type checking across the
project boundary (**A6**). One analyzer (Pyright, the same engine the editor runs) is the
single type authority.

### The investigation toolkit — knobs that default to yesterday

Every lever the investigations added is **config-driven and default-off**, so the default path
always reproduces prior results and each experiment is a one-line flag:

- **Exploration schedules and recency-weighted updates** for the tabular agent (**A8**) — the
  machinery that let the exploration-vs-bias tradeoff be studied at all. Verdict in one line:
  decaying epsilon works only with a recency-weighted update, and the residual it cannot close
  is coverage — the finding the [policy audit](blackjack-rl-policy-audit.pdf) is built on.
- **Learning-curve instrumentation** (**A10**): periodic checkpoints of policy churn, state
  coverage, and minimum visit counts make convergence *visible*, so training length is sized
  from the churn knee instead of guessed.
- **The DQN stabilizer set** (**A13**): decaying learning rate, soft/Polyak targets, weight
  averaging, a double-then-full curriculum, the dealer control variate, thread/device
  selection — each behind a flag, each preserving the default. Their individual verdicts —
  which stabilizers move the instability and which cannot in principle — are the subject of
  [from table to network](from-table-to-network.pdf). One structural note that outlived the
  experiments: the GPU sits idle on this problem (the sequential Python env loop is the
  bottleneck), so the device knobs exist but the CPU is the honest default.

### Splits extend the code; models retrain

Splits were added **by extension, not rebuild** (**A11**): the state encoding gained
pair-awareness behind the config flag, the agent gained the action, and the env/trainer handle
the branching episode — each split sub-hand's decisions credited with *that* sub-hand's return,
the split decision with the net. Because state and action space changed, policies were
retrained; the no-split runs remain valid, reproducible history. Evaluation, persistence, and
the `Strategy` seam needed no change.

### One experiment, two representations — exploring starts

The coverage capstone was built **without touching the engine** (**A12**): a deck subclass
deals a forced starting prefix, a wrapper forces the first action, and the unmodified engine
plays the rest; agent, update rule, evaluation, and persistence are reused by import, so the
run lands in the same ledger as every other. The same harness was then **ported to the
network**, making the two phases answer the same question with the same instrument — coverage
fixes the table's residual (the tabular finding), while the net, which generalizes into unseen
cells, responds differently (the network finding). Both write-ups cite this harness:
[the policy audit](blackjack-rl-policy-audit.pdf) ·
[from table to network](from-table-to-network.pdf).

### The reference layer — measured once, committed, decoupled

The betting phase's ground truth is *built*, so its construction is held to artifact
discipline (**A15**). The edge-by-count accumulator keeps raw Welford moments with a **pure,
lossless merge**, so the 20M-hand measurement fans out across cores as independent seeded
streams and combines exactly — the parallel runner lives in `scripts/`, the mergeable core in
the library. The resulting reference is **promoted to a committed artifact** in
`session/data/`, because committed code (the Kelly bettor, the figures) must never depend on
git-ignored run output.

### One evaluation engine for every betting experiment

The bet ladder, the sit-out (Wonging) study, and the bankroll sweep all run on **one generic
parallel cell-evaluation engine** (**A16**): a cell is a labelled (config, play, bet) triple;
workers reduce sessions to per-session scalars in-process (only tiny arrays cross process
boundaries), seeds are per-(cell, worker) streams so results are regenerable, and the session —
not the hand — is the unit of replication, because hands within a session are correlated and
would fake tight confidence intervals. The scripts own only cell definitions and rendering;
the tested plumbing exists once.

### The bettor is the play stack, re-aimed

The learned bettor **constructs the play phase's `QNetwork` and trains with the same TD
update** (**A17**) — no second learning stack. Its own surface is deliberately thin: an
isolated bet-state encoder, the bet menu as a constructor parameter, and the bankroll
normalized by a fixed constant rather than the session's start — three small seams that kept
every later experiment (menu variants, bankroll-encoding ablation, coverage retraining) a
config change instead of a redesign.

### One data layer from run records to report pages

Every notebook and all three report generators read runs through **`analysis_loader` — a
single records-to-DataFrame data layer with shared plotters** (**A20**). Figures in the frozen
reports are rendered by the *same* plotting functions the notebooks use (zero duplicated chart
code), every figure carries machine-generated provenance of the runs behind it, and the
betting report *asserts its pinned headline numbers against the loaded data at build time*, so
the PDF cannot silently drift from the artifacts. Loader defaults are fail-safe: special-cased
sweeps (the coverage retraining) are excluded unless explicitly requested, because they share
selection signatures with the canonical runs and would silently hijack "latest per seed"
queries.

---

## Deliberately not done

- **No `reset()/step()` gym-style environment.** Capture suffices for every method used;
  genuine step-wise interactive control is the one thing that would require engine changes.
  Revisit only if a method demands mid-hand intervention.
- **No registry/plugin experiment harness.** Frozen configs + CLI flags + `scripts/` runners
  carried all sixteen sessions of investigation; the heavier harness never earned its cost.
- **No generalized cross-project report tool.** Three self-contained generators share a house
  style by convention; consolidation was deliberately deferred until after the planned
  repo-per-project restructure.
- **No GPU path for this project.** Measured, not assumed: the tiny nets are bottlenecked by
  the sequential Python environment loop. The device/thread knobs remain for a future
  vectorized environment.
- **The factored play+bet orchestration and the monolithic baseline** were cut with the
  project's scope — the structural story ends at the bet head; the rationale lives in
  [DESIGN.md](DESIGN.md) under *Scope cut & future work*.
