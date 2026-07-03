# DESIGN — Blackjack RL

What was built and why — the decisions and their reasoning, kept as a clean snapshot of the
design as it stands. Edited in place, not appended to; the chronological journey lives in the
session log and the three reports. How the code is structured → [ARCHITECTURE.md](ARCHITECTURE.md);
the front door → [README.md](README.md).

*Snapshot of the completed project · last updated 2026-07-03.*

---

## Objective

Can a reinforcement-learning agent, given nothing but the outcome of play, rediscover decisions
that can already be proven or derived — and where it cannot, can the failure be explained
precisely?

The project asks that question three times, removing a rung of ground truth each time:

| | Learns | Ground truth | The question | Report |
|---|---|---|---|---|
| **1** | per-hand play, by lookup table | exact — the proven-optimal table | rediscoverable from outcomes alone? | [the policy audit](blackjack-rl-policy-audit.pdf) |
| **2** | per-hand play, by neural network | exact — the same table | what does approximation cost? | [from table to network](from-table-to-network.pdf) |
| **3** | bet sizing over a counted shoe | reconstructed — measured edge curve → Kelly | does learning work below the noise? | [betting against the noise](betting-against-the-noise.pdf) |

Blackjack RL with a tabular agent is a textbook exercise — stated up front. The value is
therefore not the agent; it is the **audit**: every run graded against a reference on identical
terms, every disagreement attributed to a named cause. The success criterion is that **the audit
closes**: each gap between the learned policy and the reference is explained as one of
*insufficient experience*, *representation limits*, *a statistical near-tie*, or *a signal below
the noise* — with the experiment that earned the attribution, not a hand-wave.

The findings live in the three frozen reports linked above, one per phase; this document holds
the decisions behind them.

---

## The instrument

Everything runs on the blackjack engine built and validated in the preceding simulator project
(six decks, dealer stands soft 17, 3:2 blackjack). Its measured basic-strategy house edge of
**≈ 0.45%** is the anchor every agent in every phase is graded against, and its derived
basic-strategy table is the per-hand ground truth.

**The engine's `Strategy` contract is reused for all evaluation, and the engine is never
modified** (**D2**). Every trained policy — lookup table or network, player or bettor — is
wrapped as a `Strategy` and played through the unmodified engine, graded by the same harness on
the same metrics. Two policies are comparable *by construction*, because model internals never
touch the evaluation; and every result inherits the engine's validation rather than re-earning
it.

The alternative — a purpose-built RL environment — would have turned every comparison
against the simulator project's numbers into an apples-to-oranges argument.

The engine is **consumed as an installed package, not vendored** (**D10**): an editable install
from the sibling project, whose only change was packaging metadata. One source of truth, no copy
to drift; the accepted cost is that this project requires its sibling checked out beside it.

---

## The shape of the problem

Blackjack contains two learning problems, split by where ground truth ends.

**Problem A — playing the hand.** Hit, stand, double, split against the dealer's upcard, one
hand at a time. Counting is structurally absent — within a single hand there is nothing to
count — and a proven-optimal table exists, so the problem is fully auditable. Phases 1 and 2
both live here.

**Problem B — counting and betting.** Across hands, the depleting shoe makes the deck
composition informative: the agent tracks the count and sizes its bet. Here there is **no
correct table to diff against** — the optimum is a tradeoff between growth and risk of ruin,
so any objective encodes a chosen risk preference. The mature part of the design is knowing
exactly where clean ground truth ends and saying so. Phase 3 lives here.

The two problems were **built sequentially, never in parallel** (**D3**) — the per-hand work
complete and reported before betting began — so that Problem B's irreducible fuzziness could
never contaminate Problem A's clean audit, and each phase closed as a defensible artifact on
its own.

Formally, each hand is a Markov decision process: the state is what the player may see, the
reward is the settled outcome (+1 win, +1.5 blackjack, −1 loss, 0 push — and nothing before
settlement), an episode is one hand. Plain blackjack is cleanly Markov — player total, softness,
and upcard are all that matter. Counting *breaks* that, because deck composition now matters and
isn't in the naive state; Problem B restores the Markov property by folding the true count and
shoe depth into the state. Most toy projects never state this; being explicit about it is part
of the audit stance.

---

## Phase 1 — learning the provable: the tabular agent

The Problem A state space is a few hundred discrete cells. For that, **a lookup table is the
right tool, and the value is in the audit, not the agent** (**D1**): tabular Monte Carlo control
is exact and fully inspectable, cell by cell. Reaching for a network *because it is impressive*
would be the wrong-tool signal — the network arrives later, as a deliberate experiment, not as
a default.

Training needed an environment, and the engine's hand-playing routine is atomic — it plays a
whole hand internally, consulting the strategy at each decision, and returns the full record.
So the environment is an **episode-capture wrapper, not a control-flipping one** (**D7**): the
learning agent is wrapped as a `Strategy`, the states handed to it are recorded, one hand is
played through the unmodified engine, and the trajectory and reward are read back from the
result. Monte Carlo control updates after complete episodes anyway, so nothing step-by-step is
lost — and the engine stays untouched, which keeps the reuse promise of the instrument.

Two audit-serving disciplines were fixed from day one. **Every run persists its config, seed,
git hash, metrics — and its per-state visit counts — and is never overwritten** (**D8**). The
visit counts are not hygiene: the project's central diagnostic — *did the agent fail to learn
this cell, or was there nothing to learn?* — is unanswerable without knowing how often each
state was actually seen. And **the experiment config started minimal, architected to grow**
(**D9**): only the knobs in play, read from config rather than hardcoded, so each later
investigation added a knob instead of a refactor.

Splitting a pair turns one episode into a tree of sub-hands, which complicates the
credit-assignment story, so **splits were staged in, not dropped** (**D6**): learning ran
without splits first, the whole machinery was validated on the clean single-chain case, and
splits were added before Problem A was called complete. The rationale is isolation of
variables — had splits been present from day one and the agent underperformed, an RL bug and a
split-handling bug would have been indistinguishable. The final action space has no holes.

> **Outcome.** The agent rediscovers ~93% of the basic-strategy table from win/loss alone. The
> residual disagreements were diagnosed, not shrugged at: forcing coverage of every legal
> starting state (Monte Carlo with exploring starts) collapsed the genuine disagreements from 30
> to 9, and every survivor is a near-tie where the two actions differ by almost nothing — the
> residual was mostly the *cost of learning from experience* (rare cells are rarely visited),
> with a floor of honest non-differences beneath it. That distinction — **failed to learn vs
> nothing to learn** — is the audit closing as designed. Full analysis:
> [the policy audit](blackjack-rl-policy-audit.pdf).

---

## Phase 2 — the same truth, approximated: the network

State growth alone never justified a neural network — even Problem B's state stays small and
discrete. Instead, **the DQN was introduced on Problem A as a deliberate literacy experiment,
where ground truth still exists to grade it** (**D4**): does function approximation recover
what can be proven optimal, and what does it cost in samples, variance, and stability to learn
what a lookup holds exactly? The network then carries into Problem B, where a table no longer
cleanly applies and generalization finally has a reason to exist.

The switch is a *double* one, and the design names both honestly: the representation changes
(table → network) **and** the learning rule changes (Monte Carlo's full returns → temporal-
difference bootstrapping). A caveat stated up front: blackjack hands are one-to-four decisions
with terminal-only reward, so over such short horizons TD buys almost nothing — the experiment
is about the *network*, and the write-up must not pretend bootstrapping is the improvement.

> **Outcome.** The network pays for its generalization: out of the box it plateaus well below
> the table, and reaching table-level fidelity takes a stack of stabilizers the table never
> needed. The report's verdict is kept deliberately narrow: *for this small, known, exactly
> tabulatable problem, the table is the simpler and more robust tool* — the DQN's value is
> showing what generalization adds, what it destabilizes, and why representation choices matter,
> before moving to a problem where the table no longer applies. Full analysis:
> [from table to network](from-table-to-network.pdf).

---

## Phase 3 — betting, where the truth runs out

Problem B has no correct table, so the first decision is the objective itself — and it must be
chosen, not discovered. The objective is **expected log-bankroll growth — the Kelly criterion —
on a ruin-aware session** (**D11**): per-hand reward is the log-wealth increment, so the return
over a session is the log of final-over-starting bankroll. Log-growth is a *chosen* risk
preference (expected-value maximization over-bets toward ruin; alternatives like mean-variance
exist), and the design says so plainly. Its virtue is that over-betting punishes itself in
log — growth and ruin-avoidance live in one number. A practical consequence discovered in
training and folded here: for pure growth the per-hand log-objective is *myopic* (each hand's
Kelly bet is optimal regardless of horizon), so the growth regime trains with no discounting
at all.

Under log-growth, playing and betting **separate**: optimal play maximizes per-hand expected
value regardless of bankroll, and optimal betting is the Kelly fraction given the edge at the
current count. So the agent was designed **factored — a play model and a bet model — with a
monolithic end-to-end agent as the comparison baseline** (**D12**): the structure the objective
dictates rather than a convenience (exact only in the small-bet limit; doubles and splits couple
weakly to bankroll — a caveat the write-up states). The factoring is also what made a
*bet-first* build order possible: the bet model, running on fixed basic-strategy play, is the
classic counting bettor and carries most of the edge.

In the end only that bet head was built — the count-aware play head and the monolithic baseline
were cut with the project's scope (see *Scope cut & future work*) — but the separation argument
stands, because it is what justified betting as the self-contained core.

The bet model's **state is the given Hi-Lo true count, the shoe depth, and the bankroll**
(**D13**). The count is *fed*, not discovered — this is what restores the Markov property that
counting broke. Bankroll is in the state because a ruin-aware optimum depends on it; an agent
that cannot see its wealth cannot learn restraint. (Letting the agent *discover* its own
counting statistic from raw deck composition was parked as a stretch goal, never core.)

The session itself is a **finite bankroll with a hard ruin barrier** (**D14**) — the risk
preference made concrete. Ruin is terminal; the bankroll is in min-bet units. This decision's
*expected* headline moved twice as evidence arrived, and the current statement is the honest
one: measurement showed Kelly-proportional sizing essentially never ruins here — only
over-betting ruins — so the learnable skill is **restraint** (a bet ceiling), not a subtle bend
below Kelly. The design realizes this as two named regimes differing only in bankroll:

| Regime | Bankroll | What it isolates |
|---|---|---|
| *growth* | 400 units | spread top ≈ full Kelly at high counts; ruin dormant — the pure betting lever |
| *ruin* | 200 units | the same spread now over-bets; the barrier is reachable — restraint is the thing to learn |

The originally contingent question "does the *encoding* of bankroll matter?" was in fact
triggered and run — see the outcome below.

The bet action is a **discrete spread of unit bets** (**D15**): value-based Q-learning selects
by argmax over a finite action set and bootstraps through a max over it, so a continuous bet
would force a second learning algorithm (actor-critic) into the project for one decision.
Discretizing keeps one method for both heads at ~zero cost — real table bets are discrete units
anyway. The spread was then *derived, not assumed*: sized against the measured edge curve
(arithmetic 1–8, matching Kelly's roughly linear ramp; geometric spacing over-resolves the
bottom and jumps at the top — stated as reasoning, since the alternative was never measured) and
**held constant across every experiment**, so that measured differences always attribute to the
variable under study, never to the action set.

With no table to diff against, evaluation rests on **a baseline ladder and reconstructed
references** (**D17**). The references: an edge-by-count curve measured in the training
environment itself (20M hands; break-even at true count ≈ +0.76, consistent with the folklore
"+1"), the analytic full-Kelly bet curve built from it, and the literature's index-play table.
The ladder, bottom up:

| Rung | Bettor | Role |
|---|---|---|
| 1 | flat bet + basic play | the floor — everything above it is what betting adds |
| 2 | analytic Kelly, on the same discrete spread | the fair comparison |
| 3 | analytic Kelly, continuous | the unreachable ceiling (gap to the discrete rung measured ≈ 0) |
| 4 | the learned bettor | the subject |

The Kelly rung appears in both discrete and continuous forms deliberately, so the
learned-vs-Kelly gap isolates *learned vs analytic* rather than conflating it with *discrete vs
continuous*. Outcomes are always reported on **two axes that are never collapsed** — growth and
ruin (with drawdown and the final-bankroll distribution behind them) — because a bettor can top
the growth table by surviving on luck.

> **Outcome — the finding inverted the design's expectation, and that is the result.** The
> learned bettor never rediscovers Kelly: across regimes, stabilizers, and sample sizes it
> converges to flat-minimum betting, and the intermediate Kelly-shaped policies it passes
> through are noise excursions, not better policies. Only the analytic Kelly bettor beats flat
> betting — and even it is net-negative at modest bankrolls (the table-minimum tax: something
> must be bet on negative-count hands; sitting those out flips growth positive). The wall is
> *informational, not architectural*: a denoised-reward oracle learns the ramp instantly with
> the same network, and the seductive alternative — "it keyed on wealth, not edge" — was tested
> twice (an encoding ablation and a coverage experiment) and falsified twice. The count edge is
> real but sits below the per-hand noise the agent must estimate it from; the analytic route —
> measure the edge once, derive the bet — is simply the right tool. Structure beats end-to-end
> learning on a sub-noise signal. Full analysis:
> [betting against the noise](betting-against-the-noise.pdf).

---

## The evaluation ethic (constant across all three phases)

The phases share one methodology, which is the project's actual through-line:

- **Identical terms.** Every policy is graded through the same engine via the same contract —
  never a bespoke evaluation per model.
- **A reference and a ladder, always.** Against the table where one exists; against
  reconstructed analytic references where it doesn't; with cheap baselines underneath so every
  claimed effect has an attribution.
- **Axes are never collapsed.** Fidelity and edge for hand-play; growth and ruin for betting.
  One number always hides the story (a high-growth bettor with 53% ruin is a survivor-bias
  artifact, not a winner).
- **Rank only on tight estimates.** Edges carry sampling error; ranking close configs on noisy
  short evaluations misranked them more than once. Claims ride on re-evaluations sized for the
  gap being claimed, with uncertainty reported.
- **Diagnose, don't assert.** Every headline attribution was earned by a controlled experiment:
  forced coverage for the tabular residual, ruled-out suspects for the network's instability,
  the oracle and two falsification tests for the bettor's wall.

---

## Scope & non-goals

- **No engine changes, ever.** All work is additive around the validated engine.
- **No network where a table suffices** — except as the explicitly framed experiment of the
  network phase.
- **No clean-optimum claims for the betting problem.** It optimizes a chosen risk preference,
  and the write-ups say so.
- **No deep offline RL.** Offline learning was scoped to tabular before being cut entirely.
- **Depth over breadth.** Rigor on the core questions — baselines, confidence intervals, the
  named-cause audit — rather than semi-related side studies, each of which had to earn its
  slot individually (most didn't; see below).

---

## Scope cut & future work

The project ends at the betting report — a deliberate endpoint, not an abandonment. What was
cut, and where the threads live:

- **The count-aware play head and the monolithic baseline** — the second half of the factored
  design — were cut at the endpoint decision: betting carries most of the edge, the
  factored-vs-monolithic question reprises an arc already run twice (table vs net, analytic vs
  learned), and the bet investigation had already produced the deeper finding. The separation
  rationale survives inside the factored-agent decision; the index-play reference table, built
  as the play head's audit target, remains in the codebase unconsumed. Revisit only if the
  project is ever extended past its report.
- **Offline learning from logged data** (**D5** — tombstone). Dropped mid-project: its lesson
  ("coverage of the behavior data governs what can be learned") had already been shown twice
  elsewhere, and it added no deep-learning content. The thread would belong to an
  RLHF-adjacent study in a later phase, not here.
- **Play-model training policy** (**D16** — tombstone). Natural-play training with contingent
  coverage forcing, designed for the play head; died with it. Its sibling idea — oversampling
  rare high counts for the *bettor* — was analyzed, predicted confirmatory, and deferred; it
  survives as a future-work entry in the betting report.
- **Learned counting** (feed raw deck composition, audit the discovered statistic against
  Hi-Lo) — parked as a stretch before Phase 3 began; never core. A genuine future experiment.
- Smaller deferred items (bet-head multi-step returns, horizon-relative ruin, prioritized
  replay, paired common-random-number evaluation) are catalogued honestly in the betting
  report's future-work section — considered, scoped out on purpose.
