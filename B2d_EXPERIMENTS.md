# B2d Bettor — Experiment Log (session 2026-06-29)

Diagnostic runs for the learned DQN bettor — the value-based "rung 3" above the analytic flat/Kelly
baselines. **Question:** does RL rediscover the count→bet Kelly ramp from raw per-hand log-growth? Early
demos showed the agent collapsing to a **count-independent flat bet**; these runs trace *why* and *which
lever fixes it*.

> **Self-contained record.** This committed file stands alone — every number needed to read a run is in
> its table, and nothing here points to another document. Raw run outputs live under `logs/` (local,
> git-ignored); the distilled tables are here.
>
> **Evidence tiers** (labelled on every run): **exploratory** = single-seed / short / scratch script —
> a lead, not settled; **decisive** = a controlled diagnostic that cleanly answers a yes/no; **committed**
> = a full measured run with confidence intervals (the report-grade baselines).
>
> **Living log** — append every new run here (newest tests at the bottom, before *Findings summary*),
> following the *Appending a run* convention at the end.

### Run index

| # | Test | Tier | Headline |
|---|---|---|---|
| 0 | Initial B2d demo (pre-session) | exploratory | flat / over-bets neutral — motivated this session |
| 1 | Throughput probe (γ=1 baseline) | exploratory | ~2.7 sess/s; γ=1 wanders |
| 2 | γ-sweep | exploratory | γ→stability only; all flat; γ=1 diverges |
| 3 | Oracle diagnostic | **decisive** | PASS — code sound |
| 4 | Real-reward lever sweep | exploratory | **batch size** breaks the flatline |
| 5 | `torch_threads` benchmark | decisive | 1 thread fastest |
| — | Prior committed baselines (B1–B2c) | committed | reference index |

---

## Shared configuration (fixed across every run unless a test varies it)

| Group | Setting | Value |
|---|---|---|
| Regime | `growth_config` | start **400u**, `max_hands` 1000, 6-deck shoe, `BET_SPREAD`=(1..8) |
| Play | strategy | fixed `BasicStrategy` (bettor is the only learner) |
| Net | `QNetwork` hidden | (64, 64) |
| Replay | buffer / warmup / `train_every` | 50,000 / 1,000 / 4 |
| Replay | `batch_size` | **128** (varied in Test 4) |
| Target | sync / tau / double | hard-sync every 1,000 grad steps / 0 / off |
| Optim | Adam `lr` | 1e-3 |
| Explore | `epsilon` | 0.1 constant (varied in Test 4) |
| Repro | seed / `torch_threads` | 0 / 1 |
| Probe | `greedy_bet_curve` | bankroll=400, depth=3.0 decks, counts (−4,−2,0,+2,+4,+6,+8) |

**Kelly target curve** (discrete, at bankroll 400u) — what every run is compared against:

| TC | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|---|
| **bet (u)** | 1 | 1 | 1 | 2 | 5 | 8 | 8 |

---

## Test 0 — Initial B2d demo (pre-session, PRELIMINARY)

The run that started this thread — first `train_bet` smoke after building B2d-2. Exact final curve was
**not logged** (predates the JSON-saving scratch scripts); recorded qualitatively from the run notes.

**Config:** ~2,500 sessions, real reward, `BetTrainConfig` **defaults** (γ=1.0, scale=1, batch=128, ε=0.1).

| Observation | Value |
|---|---|
| `bet@TC0` | settled ~5 (should be the min, 1) — **over-bets neutral** |
| `bet@TC+6` | wandered 1↔8 — **noisy at high counts** |
| Verdict | does **not** rediscover Kelly; flagged signal≪noise + high-count rarity |

---

## Test 1 — Throughput probe (also the first γ=1 baseline)

Single run to measure training speed; doubled as the first real-reward γ=1 read.

**Config:** 1000 sessions, real reward, γ=1.0, scale=1, batch=128, ε=0.1.

| Result | Value |
|---|---|
| Throughput | **~2.7 sessions/sec** (1000 sess in 376s) |
| Bottleneck | ~85% gradient steps (246k grad steps for 1000 sessions) |
| `bet@0` / `bet@6` over training | 6/3 → 7/7 → 8/8 → **5/1** (wandering, final *inverted*) |
| Verdict | γ=1 does **not** learn the ramp; loss tiny and rising |

---

## Test 2 — γ-sweep (real reward, isolate the discount)

4 parallel runs, **only γ varies**. 2000 sessions, real reward, scale=1, batch=128, ε=0.1.

| Run | γ | loss trajectory | final curve (−4…+8) | shape |
|---|---|---|---|---|
| g0 | 0.0 | stable ~4e-4 | `2 2 2 2 2 2 2` | flat |
| g0.9 | 0.9 | stable ~4e-4 | `7 7 7 7 7 7 7` | flat |
| g0.99 | 0.99 | stable ~3e-4 | `4 4 4 4 4 4 4` | flat |
| g1.0 | 1.0 | **diverges 5e-4 → 1.4e-2 (25×)** | `6 6 6 6 6 6 6` | flat (faint *inverted* tilt) |

**Read:** γ governs **stability** (γ=1 diverges — telescoping variance over 1000 hands), but **not** the
ramp — every γ stays count-flat at batch=128. γ is necessary (drop 1.0) but **not sufficient**.

---

## Test 3 — Oracle diagnostic (is the code broken?)

Replaces the realized log-reward with its **deterministic expectation** from the measured edge curve
(`(b/W)·mean_return − ½(b/W)²·variance`), removing all per-hand noise. Scaled to O(1) so a flat result
could only mean a bug.

**Config:** 2000 sessions, **oracle reward**, γ=0, scale=1000, batch=128, ε=0.1. (`scripts/scratch_oracle.py`)

| Checkpoint | curve (−4…+8) |
|---|---|
| 250 | `1 1 1 2 4 7 8` |
| 1000 | `1 1 1 2 4 7 8` |
| 2000 (final) | `1 1 1 2 5 7 8` |
| **Kelly** | `1 1 1 2 5 8 8` |

**Verdict: PASS — code is sound.** Clean, **stable**, monotone ramp matching Kelly (±1 level). The
encode→Q→argmax→transition path works; the current net/buffer/batch *can* represent and learn the ramp.
So the real-reward flatline is a **noise/coverage/scale wall, not a bug.**

---

## Test 4 — Real-reward lever sweep (which lever breaks the flatline?)

6 parallel runs, real reward, mostly one-knob-off a baseline (γ=0, scale=100, batch=128, ε=0.1).
(`scripts/scratch_real_sweep.py`)

| Run | n_sess | γ | scale | batch | ε | final curve (−4…+8) | shape |
|---|---|---|---|---|---|---|---|
| `base` | 2500 | 0 | 100 | 128 | 0.1 | `1 1 1 1 1 1 1` | flat |
| `scale1` | 2500 | 0 | **1** | 128 | 0.1 | `7 7 7 7 7 7 7` | flat |
| `g099` | 2500 | **0.99** | 100 | 128 | 0.1 | `1 1 1 1 1 1 1` | flat |
| `eps03` | 2500 | 0 | 100 | 128 | **0.3** | `3 3 3 3 3 3 3` | flat |
| `batch512` | 2500 | 0 | 100 | **512** | 0.1 | `1 1 1 1 1 1 1` | flat (spiked `+8` mid-run) |
| **`batch2048`** | **1200** | 0 | 100 | **2048** | 0.1 | **`2 1 2 1 5 7 8`** | **RAMP** |
| | | | | | | **Kelly** `1 1 1 2 5 8 8` | |

**Read:** **batch size is the only lever that breaks the flatline.** scale, γ, ε all stayed flat at
batch=128; raising batch to 2048 produced a ramp — upper half near-exact (`+4→5`, `+8→8`; `+6→7` vs 8),
lower half still noisy. Mechanism: it's a **gradient-SNR** problem — bigger batch averages per-hand
reward noise so the ~1e-4 count-signal surfaces in the gradient direction.

**Caveats:** `batch2048` ran only 1200 sessions (confounded: bigger batch *and* less data, yet ramped
anyway — strengthens the claim). Single seed, still wandering between checkpoints. Big batch costs ~4×
per session (1.9s vs 0.5s).

---

## Test 5 — `torch_threads` microbenchmark (can we speed training?)

3000 `td_update` calls on the tiny net (3→64→64→8, batch 128), across thread counts.

| threads | throughput |
|---|---|
| **1** | **988 upd/s** ← fastest |
| 2 | 870 upd/s |
| 4 | 934 upd/s |
| 8 | 810 upd/s |
| 16 | 596 upd/s (40% slower) |

**Read:** multi-threading **hurts** for this tiny net (thread overhead > compute). `torch_threads=1` is
both reproducible and fastest. Wall-clock comes from **parallel independent runs** (1 thread each, ~20
on 22 cores), not threads-per-run.

---

## Prior committed baselines (B1–B2c) — reference index

These are **committed, measured** experiments (not re-run here) that set the analytic ladder the learned
bettor is graded against. Headline numbers are self-contained (full four-axis tables live with their run
artifacts); each run id carries its own provenance (timestamp + git hash).

| Run | What | Headline |
|---|---|---|
| edge-by-count, 20M hands (run `20260627-231428_edge-by-count_seed0`, git `eab466d`) | per-count edge + Kelly curve | break-even **TC +0.76**; TC0 −0.32%±0.05; full-Kelly <2% of bankroll |
| bet-ladder, 20k sess/cell | flat / kelly-disc / kelly-cont / flat-8 × {growth, ruin}, four-axis | Kelly beats flat on growth (−0.048 vs −0.150 ×1e-4); full-Kelly **net-negative even @400u** (table-min tax); discretization ≈ 0; **ruin is pure over-betting** (Kelly 0%, flat-8 20%/53%) |
| Wonging (min-wager sit-out) | back-counting / abstain on −EV shoes | flips growth **negative→positive** both configs; lower drawdown too |
| bankroll sweep | growth vs starting bankroll | bankroll-size & Wonging are **substitute remedies** for the table-min tax; converge at the 8u spread-cap |

---

## Findings summary

1. **Code is sound** (oracle PASS) — the flatline is not a bug.
2. **γ governs stability, not the ramp** — γ=1 diverges (telescoping variance); γ<1 stable but still flat.
   Use γ=0 for growth (Kelly is a myopic optimum); intermediate γ reserved for the ruin regime.
3. **Batch size is the lever that produces the ramp** — it's a gradient-SNR problem; bigger batch
   averages out the per-hand noise so the count signal surfaces. (Scale, ε did not move it alone.)
4. **The ramp IS learnable from real rewards** — not a hopeless 20M-hand wall. Recipe forming:
   **big batch + scaled reward + γ=0 + more data** (+ high-count coverage for the lower-half fine structure).
5. **Speed:** ~2.5–2.7 sess/s; threads don't help; parallelism = independent runs across cores.

**Next candidate run:** `batch=4096, ~5000 sessions, γ=0, scale=100` (≈3–5 hr, background) — does the
ramp stabilize and the lower half resolve? Optionally a 2nd seed in parallel.

---

## Open items & decisions

- **(2026-06-29) `BetTrainConfig` default `gamma` → 0.0** (was 1.0). γ=1 diverges on the bettor
  (telescoping 1000-hand return); γ=0 is the growth-regime optimum (Kelly is the myopic per-hand
  log-optimum). Adjustable per-run.
- **TODO — the ruin regime needs its OWN γ.** The growth recipe (γ=0) does **not** transfer: ruin-
  avoidance is a genuine multi-step effect γ=0 is blind to. Before the ruin eval-ready run, sweep
  γ ≈ {0.9, 0.95, 0.99} × big batch to find the value that captures it without diverging (γ=1 is out).
  Don't guess it. *(Our γ-sweep so far was growth-only.)*
- **TODO — bridge to deliverable:** add `reward_scale` (cosmetic) to the real `BetTrainConfig`; runner
  (`train_bet_agent.py`) with `save_run`+`model.pt`; eval (`eval_bet_agent.py`) four-axis via `cell_eval`;
  figures (bet-vs-count overlay, bankroll trajectories).

---

## Appending a run (convention)

For each new run, add a `## Test N — <name>` section above this one with: (1) a one-line **purpose**,
(2) a **config** line/table noting only what differs from the *Shared configuration*, (3) a **results**
table — the bet-vs-count curve (counts −4…+8) vs the Kelly target, (4) a one-line **read**, and (5)
its **evidence tier** (exploratory / decisive / committed). Add a row to the *Run index* at the top.
Keep the curve format `−4 −2 0 +2 +4 +6 +8` so every run is directly comparable. Raw outputs go under
`logs/`; this file holds the distilled tables, not the logs.
