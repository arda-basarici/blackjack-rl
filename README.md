# Blackjack RL

**Can a reinforcement-learning agent rediscover decisions that can be proven optimal — and when
it fails, can the failure be explained precisely?**

One validated blackjack engine, one evaluation harness, and one question asked three times with
the ground truth progressively removed: a lookup table learning against a *proven* strategy
table, a neural network learning against the same table, and a learned bettor sizing wagers on
a counted shoe — where no correct table exists and the reference had to be measured and derived.
The arc ends in an inversion that is the project's strongest result: **the learned bettor never
rediscovers Kelly betting, and the project proves *why not*** — the counting edge is real but
sits below the per-hand noise it must be learned from.

> A research/portfolio project on a deliberately solved toy domain. The contribution is the
> **audit** — grading learned policies against provable or reconstructed references, on
> identical terms, with every disagreement attributed to a named cause — not a blackjack bot.

---

## The question, asked three times

| | The agent | The ground truth | The verdict | Full story |
|---|---|---|---|---|
| **1** | tabular Monte Carlo control | exact — the proven-optimal basic-strategy table | rediscovers **~93%** of the table from win/loss alone; forcing coverage of rare states collapses genuine disagreements **30 → 9**, all near-ties — the residual was the cost of experience, not the method | [the policy audit](blackjack-rl-policy-audit.pdf) |
| **2** | a DQN on the same hands | exact — the same table | the table wins on **simplicity and robustness**: the net plateaus below it out of the box and needs a stack of stabilizers to approach what a lookup gets for free — the cost of generalization, measured | [from table to network](from-table-to-network.pdf) |
| **3** | a DQN bettor on a counted shoe | reconstructed — a 20M-hand measured edge curve → the analytic Kelly bettor | **RL converges to flat-minimum betting, never Kelly**; only the analytic bettor beats flat. An oracle control and two falsification experiments pin the wall as *informational* (sub-noise signal), not architectural | [betting against the noise](betting-against-the-noise.pdf) |

Headline numbers, with their baselines and caveats:

- The betting edge is real: measured over **20M hands**, the player crosses break-even at true
  count **≈ +0.76** (the folklore says "+1") — but full Kelly stays under 2% of bankroll, and at
  a 400-unit bankroll even the analytic Kelly bettor is **net-negative** (the table-minimum tax:
  something must be wagered on bad counts; sitting them out flips growth positive).
- The learned bettor's failure is *diagnosed*, not observed: the same network fed a denoised
  reward learns the Kelly ramp immediately, and the "it keyed on wealth, not edge" alternative
  was tested twice (encoding ablation, coverage retraining) and **falsified twice**.
- Every number above is regenerable: runs persist config, seed, and git hash; the betting
  report *asserts its headline numbers against the run data at build time*.

## What it demonstrates

Evaluation design and intellectual honesty as engineering skills: reference-anchored grading,
controlled falsification experiments, negative results reported as findings, uncertainty
respected (axes never collapsed, no ranking on noisy estimates), and reproducibility by
construction.

---

## Run it

Requires the Phase-2 engine checked out as a sibling project (`blackjack-sim`).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ..\..\phase2-data\blackjack-sim   # the validated engine + BasicStrategy
pip install -e ".[dev]"                          # this package (pyproject.toml is the single dependency source)

python -m pytest tests -q                        # the test suite
```

The real entry points:

```powershell
python -m blackjack_rl.tabular                   # train the tabular agent (Phase 1)
python -m blackjack_rl.dqn.experiment            # train the DQN player  (Phase 2)
python scripts\train_bet_agent.py                # train the DQN bettor  (Phase 3)
python scripts\measure_bet_ladder.py             # the analytic baseline ladder (flat / Kelly / over-bet)
python scripts\make_docs.py                      # render the API reference (pdoc → docs/api/)
```

## Layout

| where | what |
|---|---|
| `blackjack_rl/core/` | shared foundation: state encoding, configs, episode capture, schedules, run persistence |
| `blackjack_rl/evaluation/` | grading: house edge, cell-by-cell policy audit |
| `blackjack_rl/tabular/` | Phase 1 — Monte Carlo control on a Q-table |
| `blackjack_rl/dqn/` | Phase 2 — QNetwork + TD trainer + the investigation's stabilizer knobs |
| `blackjack_rl/session/` | Phase 3 — session env, measured references, the bettors |
| `analysis/` | chapter notebooks per phase (data via the shared `analysis_loader`) |
| `scripts/` | training/eval runners, experiment + figure pairs, the three report generators |

## Scope & limits

- One rule set (6-deck, dealer stands soft 17, 3:2), one counting system (Hi-Lo), simulation
  only — nothing here claims to generalize beyond them.
- The betting objective (log-growth / Kelly) is a *chosen* risk preference, stated as such —
  there is no universal "correct" bettor.
- The strongest Phase-3 result is a **negative** one, presented as the finding it is: on this
  signal-to-noise, analytic structure beats end-to-end learning.

## Deeper

[DESIGN.md](DESIGN.md) — the decisions and why · [ARCHITECTURE.md](ARCHITECTURE.md) — the
structure and why that shape · the three reports linked in the table above.
