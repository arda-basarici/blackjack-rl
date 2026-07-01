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
| 4 | Real-reward lever sweep | exploratory | **batch** breaks the flatline (intermittently) |
| 5 | `torch_threads` benchmark | decisive | 1 thread fastest |
| 6 | Batch-threshold ladder (@2500) | exploratory | ramp needs batch ≥~2048 but does **not stabilize** (wanders flat↔ramp) |
| 7 | Long runs (batch 4096/2048 @5000) | exploratory | converges to **flat-1** (≈Flat) — not Kelly |
| 8 | Stability levers @γ=0 (double/Polyak/buffer) | exploratory | double/Polyak **no-op at γ=0**; buffer rules out *forgetting* |
| 9 | **Four-axis performance evals (deliverable)** | **central** | learned bettor **loses to / becomes Flat, never Kelly** (CI-backed) — *at γ=0* |
| 10 | **γ>0: double-DQN, scale, Huber-delta** | exploratory | **revises 4 & 9** — scale NOT cosmetic (Huber); count-dep *forms* at γ>0; high end = coverage |
| 11 | Harmonic (orbit-collapse) + "RL visits Kelly" | exploratory | harmonic collapses orbit (§33); lows lock, high end *visits* ramps but wanders |
| 12 | Longer sweep (2.5× data, double×regime×batch) | exploratory | more data → flat-1; tails visit ramps, don't hold; ⚠ double no-op at γ=0 |
| 13 | **H3 + multi-seed hardening (deliverable)** | **central** | ramps **far worse than flat** (dd 14–18%); RL≈flat CI-backed; **double-ON safety was seed-luck**; γ0.9 too low |
| 14 | **Encoding ablation (wealth vs thin edge)** | **central** | drop/re-encode bankroll → **encoding-invariant, all ≈ flat** → the wall is FUNDAMENTAL (thin edge), not representational |
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

**Caveats / correction (see Test 6):** this `batch2048` final ramp was caught at a *lucky checkpoint* —
the curve **wanders** between flat and ramp across checkpoints, it does not converge. Test 6 (batch 2048
at 2500 sessions) ended **flat**. So batch makes the ramp *appear* but does **not stabilize** it. Also:
only 1200 sessions (confounded with less data), single seed; big batch costs ~4× per session.

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

## Test 6 — Batch-threshold ladder (where does the ramp turn on?)

Clean OFAT: batch ∈ {256, 1024, 2048} at **equal 2500 sessions** (γ0, scale100, seed0), filling the
ladder between `base` (128) and Test-4's `batch2048`. Final curves vs Kelly `1 1 1 2 5 8 8`:

| batch | final curve (−4…+8) | shape |
|---|---|---|
| 128 (`base`) | `1 1 1 1 1 1 1` | flat |
| 256 | `1 1 1 1 1 1 1` | flat |
| 512 | `1 1 1 1 1 1 1` | flat |
| 1024 | `1 1 1 1 1 1 1` | flat |
| 2048 | `1 1 1 1 1 1 1` | flat (FINAL) — but **wandered through ramps mid-run** (`+4:8 +6:7 +8:7` @1560) |

**Read — corrects Test 4.** The ramp is **intermittent, not stable.** ≤1024 stays flat; 2048 *oscillates*
between flat and ramp across checkpoints and happened to end flat here (Test 4's 1200-session run happened
to end *on* a ramp — a lucky checkpoint). So **batch is necessary to make the ramp appear, not sufficient
to hold it.** Stabilising the argmax is the real open problem (→ Test 7; contingency: lr-decay).

---

## Test 7 — Long runs: does more batch + data converge? (DONE)

batch 4096 (seeds 0, 1) + batch 2048, all **5000 sessions**, **plain γ=0 DQN** (no double-DQN, no Polyak —
hard target sync, 50k buffer), scale 100. Final curves — **all identical:**

| run | final curve (−4…+8) |
|---|---|
| b4096_s0 / b4096_s1 / b2048_s0 | `1 1 1 1 1 1 1` |
| Kelly | `1 1 1 2 5 8 8` |

**Read — this resolves the wandering.** With enough data + batch it **stops wandering and converges — to
flat-minimum (flat-1), NOT Kelly.** The wandering in shorter runs was the un-converged transient. *Why:*
at γ=0, betting >1 only pays at the **rare** high counts; at the common low/neutral counts it's −EV
(table-min tax + variance). Averaged over the (low-count-dominated) natural distribution, the minimum wins
almost everywhere and the rare high counts can't pull the argmax up. So the converged policy is **"always
bet the table minimum"** — the opposite of count-aware sizing. (3 runs, 2 seeds, all identical → solid.)

**Convergence trace** (checkpoints; loss falls monotonically *as* the policy collapses to flat-1):

```
b2048_s0 (batch2048)  1250: 1 7 1 3 1 4 8   →  1875: 1 1 1 1 1 1 1 (holds)   loss 0.45→0.19
b4096_s0 (batch4096)  1250: 1 1 1 3 8 6 6   →  3125: 1 1 1 1 1 1 1           loss 0.60→0.25
b4096_s1 (batch4096)  2500: 2 1 1 1 2 4 8   →  4375: 1 1 1 1 1 1 1           loss 0.53→0.25
```

Three things the trace shows:
- **Loss falls the whole time** (0.5–0.6 → 0.19–0.25) *while* the policy collapses to flat-1 — the net is
  genuinely *converging*, and flat-1 is the policy it converges to.
- **The ramp appears transiently** (e.g. `b4096_s0` @1250 `…+4:8 +6:6 +8:6`) — the net briefly "notices" the
  rare high-count edge and bets high there, the *right idea* — but **can't hold it** (`+8` is consistently
  the last to fall). It finds the ramp in flashes, then converges *away* from it to "bet the minimum."
- **Smaller batch converged faster** (b2048 ~1875 vs b4096 ~4375) — noisier gradients reach the same flat-1
  sooner; bigger batch is smoother but slower to the same endpoint.

---

## Test 8 — Stability levers @ γ=0 (the limit-cycle hypothesis)

Tested the "boom-and-bust limit-cycle" fixes (REPORT_NOTES). **Two no-ops, one decisive-negative.**

- **double-DQN & Polyak (`b2048_dbl`, `b2048_polyak`) — NO-OP at γ=0** (see ⚠ in *Open items*). Bit-identical
  to baseline and to each other — they only touch the *bootstrap*, which γ=0 zeroes. Killed unfinished.
- **Bigger buffer (`b2048_bigbuf`, 500k = 10×) — does NOT stabilise; rules out *forgetting*.** Wanders as
  much as baseline. Decisive *within-run* evidence: 500k doesn't evict until ~session 500, so the first
  ~500 sessions **cannot forget** — yet it wanders right through them. And loss falls monotonically while
  the argmax flips → the net converges, the *decision* doesn't.

**Conclusion:** at γ=0 the wandering is **not a limit cycle** (max-bias needs the bootstrap; forgetting
ruled out) — it's **static argmax instability on a near-flat (thin-edge) Q-surface.** (Not dead globally —
max-bias could bite at γ>0 — but not the γ=0 story.)

---

## Test 9 — Four-axis performance evals (the deliverable) [CENTRAL]

The honest question: judged on *performance* (not the curve), is the learned bettor any good? Trained
agents vs discrete-Kelly + Flat on full sessions (growth-rate ± CI · ruin · drawdown · bankroll), each in
the regime it was **trained on** (in-distribution = fair; the other column = OOD cross-check, discounted).

| agent (in-dist) | curve | growth/hand ×1e-4 — agent / kelly / flat | ruin % | verdict |
|---|---|---|---|---|
| growth `bigbuf` (1000 sess, γ0) | over-bet | −0.557 / −0.035 / −0.127 | 0.5 | **worse than Flat** (over-bets) |
| ruin (1500 sess, γ0.95, **double-DQN**) | flat-4 | −6.39 [−7.2,−5.6] / −0.34 / −0.40 | **21** | **worse than Flat; ruins 21%** |
| converged (b4096, 5000 sess, γ0) | flat-1 | −0.233 / −0.035 / −0.127 | 0.3 | **≈ Flat** (marginally worse) |

- **Undertrained → over-bets → loses to Flat *and* ruins** (the expensive error; B2c's "over-betting
  ruins"). The ruin-trained agent, *even with γ>0 + double-DQN*, learned **flat-4** — no restraint, **21%
  ruin in-distribution**, CIs cleanly separated from Flat.
- **Converged (5000 sess) → flat-1 ≈ the Flat baseline** (growth −0.233 vs Flat −0.127; ruin ~0) — it
  *becomes* Flat, marginally worse (mild residual over-betting at off-probe bankrolls), and never beats it.

**Headline result (CI-backed): RL does NOT rediscover Kelly here.** It either wanders (undertrained → over-
bets → loses + ruins) or converges to **flat-minimum** (well-trained → becomes Flat) — never the count→size
ramp. Analytic Kelly varies with count and beats Flat; end-to-end RL converges *to* Flat. For a signal this
thin, the 20M-hand edge measurement is fundamentally more sample-efficient than value learning. (Lands on
B2c's "the skill is restraint" — the agent *can't* learn it.)

---

## Test 10 — γ>0 (ruin regime): double-DQN, scale, and the Huber-delta mechanism [REVISES Tests 4, 9]

Moved to the **ruin regime (200u, γ=0.95)** — where γ>0 is justified and the stabilizers (double-DQN,
Polyak) are *live*, not γ=0 no-ops. Batch 512, n=1000–2000. **This arc revises the "scale is cosmetic"
and "RL just converges to flat" conclusions.**

**(a) double-DQN unlocks count-dependence at γ>0.** A/B, scale1:
- double **OFF**: rigidly flat (`5 5 5 5 5 5 5`).
- double **ON**: the count *separates* — transient *correct* ramps appear (`1 1 1 1 8 8 8`), then wander.
The max-bias correction lets the count signal express (exactly the case double-DQN is for).

**(b) Scale is NOT cosmetic — it's the bigger lever, via the Huber loss `delta=1.0`.** We use
`F.smooth_l1_loss` (= Huber, delta=1). Reward magnitude vs delta sets the loss *regime*:
- **scale1**: TD errors ≪ delta → **quadratic** (MSE-like) → noisy per-hand *outliers* dominate the
  gradient → signal swamped → **flat**.
- **scale50/100**: TD errors ≳ delta → **linear/clipped** (Huber-robust) → outliers capped → the
  *consistent* count signal survives → the ramp *direction* forms (even with double **OFF**).
So **scale slides errors across the Huber delta**; that reshape is non-uniform → Adam can't normalize it
(Adam's ε is ruled out: ~5e-7, negligible at our gradients ~0.02). `reward_scale` and `delta` are two
knobs for the same thing (×100 scale ≈ ÷100 delta). **This is why Test 4's batch-128 scale comparison was
contaminated** — batch128 pinned everything flat, hiding scale's real effect (Arda's contamination point).

**(c) The coverage diagnosis — what's learnable vs not.** Consistent across all γ>0 runs:
- **Low/neutral counts (common, well-sampled) → lock to ~1 (correct, stable).** ✓ learnable.
- **High counts (rare) → never stabilize (wander 1↔8).** ✗ coverage-limited — the betting analog of
  Problem-A's rare-cell wall. lr-decay / batch / scale can't fix it (under-sampling, not optimization).
So the **achievable target = consistent minimum-betting around break-even** (the common part); the
high-count ramp *magnitude* is coverage-bound. (Flat-1, the γ=0 result, is actually the *coverage-robust*
policy — bet minimum where you can't estimate the rare states.)

**(d) Tuning toward the achievable (neutral) target:**
- **scale50 > scale100** for neutral cleanliness (delta ≈ noise floor → less leakage/saturation; loss
  *drops* to ~0.3 as well-sampled states resolve below delta, vs scale100 pinned ~1.0 in the linear region).
- **double-DQN *hurts* neutral consistency** (adds noise) — it only helps the coverage-limited high end we
  can't stabilize anyway, so for the neutral target prefer **OFF**.
- **Lower *constant* lr** (1e-4): smaller Adam steps → less argmax-jitter → steadier neutral lock; better
  than lr-*decay* (which freezes a random snapshot) — confirmed in the sweep below.

**Results (γ>0 ruin sweep, single-seed; "neutral" = the `[−4..+2]` region, want ~`1`; final = one
checkpoint, so for *wandering* runs it's partly luck — read the neutral column):**

| run | dbl | scale | batch | lr | n | final (−4…+8) | neutral | read |
|---|---|---|---|---|---|---|---|---|
| `dbloff` | off | 1 | 512 | 1e-3 | 1k | `5 5 5 5 5 5 5` | `5 5 5 5` | rigid flat (scale1) |
| `dblon` | on | 1 | 512 | 1e-3 | 1k | `5 5 5 2 2 2 2` | `5 5 5 2` | inverted |
| `dbloff_s100` | off | 100 | 512 | 1e-3 | 1k | `8 1 1 5 1 5 5` | `8 1 1 5` | wander snapshot |
| `dblon_s100` | on | 100 | 512 | 1e-3 | 1k | `1 1 1 1 1 5 6` | `1 1 1 1` | ramp-ish |
| `dbloff_s50` | off | 50 | 512 | 1e-3 | 1k | `1 1 1 1 3 1 1` | `1 1 1 1` | clean neutral |
| `dblon_s50` | on | 50 | 512 | 1e-3 | 1k | `2 2 2 2 2 2 2` | `2 2 2 2` | flat-2 |
| `dblon_decay` | on | 1 | 512 | dec | 1k | `1 1 1 1 1 1 1` | `1 1 1 1` | flat-1 (froze) |
| `dblon_s100_decay` | on | 100 | 512 | dec | 1k | `1 1 8 1 1 1 4` | `1 1 8 1` | neutral blip |
| `dbloff_s50_lr4` | off | 50 | 512 | 1e-4 | 2k | `3 3 1 1 1 1 3` | `3 3 1 1` | mostly clean |
| `dblon_s50_lr4` | on | 50 | 512 | 1e-4 | 2k | `1 1 2 1 1 3 3` | `1 1 2 1` | clean-ish |
| **`dbloff_s50_b256_lr4`** | off | 50 | 256 | 1e-4 | 2k | `1 1 1 1 1 1 1` | `1 1 1 1` | **cleanest** ✓ |
| `dblon_s50_b256_lr4` | on | 50 | 256 | 1e-4 | 2k | `4 3 3 1 6 6 6` | `4 3 3 1` | neutral blippy (dbl) |
| `*_lr5` (×4) | – | 50 | 256/512 | 1e-5 | 2k | `1 1 1 1 1 1 1` | `1 1 1 1` | flat-1 (under-converged) |
| `ruin_train` | on | 1 | 2048 | 1e-3 | 1.5k | `4 4 4 4 4 4 4` | `4 4 4 4` | flat-4 (scale1) |
| **Kelly** | | | | | | `1 1 1 2 5 8 8` | `1 1 1 2` | target |

**Conclusions:** (1) **scale1 → no structure** (the scale gate; even batch2048 `ruin_train` is flat-4);
(2) **scale50/100 → count structure forms but *wanders*** (finals vary); (3) **lr1e-4 = sweet spot** (clean
neutral + surviving high-end attempts), **lr1e-5 → under-converged flat-1**; (4) **double OFF cleaner** than
ON for the neutral target; (5) **batch256 ≈ batch512** for the neutral (cheaper). **Best cell for "consistent
around 0": `scale50 + double-OFF + lr1e-4` (batch256).** High-count *magnitude* stays coverage-bound (#10).

---

## Test 11 — Harmonic lr CONFIRMS §33 (orbit collapse) + "RL VISITS Kelly but won't hold it"

**2×2×2** (double{on,off} × lr-harmonic{on,off} × eps-decay{on,off}; ruin/γ0.95/scale50/batch512/lr1e-3/n2000).
"neutral" = `[−4..+2]` held across the last 4 ckpts.

| run | dbl | harm | eps | final (−4…+8) | neutral | loss | read |
|---|---|---|---|---|---|---|---|
| `dblon_base` | on | – | – | `2 2 2 2 2 2 2` | 1/1/1/**2** | 0.27 | **orbits** flat-1↔2 |
| `dblon_harm` | on | ✓ | – | `1 1 1 1 3 1 3` | **3**/3/1/1 | 0.17 | harmonic, double blips |
| `dblon_eps` | on | – | ✓ | `1 1 1 1 1 1 1` | 1/**2**/1/1 | 0.12 | orbits, clean final |
| **`dblon_harm_eps`** | on | ✓ | ✓ | `1 1 1 1 1 1 1` | **1/1/1/1** ✓ | **0.086** | solid neutral |
| `dbloff_base` | off | – | – | `1 1 1 1 1 1 1` | **2**/1/1/1 | 0.25 | orbits, settles |
| **`dbloff_harm`** | off | ✓ | – | `1 1 1 1 2 2 2` | **1/1/1/1** ✓ | 0.17 | solid + faint ramp |
| `dbloff_eps` | off | – | ✓ | `2 2 2 2 2 2 2` | 1/1/1/**2** | 0.13 | orbits (flat-2 end) |
| **`dbloff_harm_eps`** | off | ✓ | ✓ | `1 1 1 1 2 2 2` | **1/1/1/1** ✓ | **0.093** | solid + faint ramp |

**Harmonic confirms the Robbins–Monro prediction (§33):** every rock-solid-neutral cell is **harmonic**;
constant-lr cells still flip flat-1↔flat-2 at the end (orbit never closes). Harmonic also **slashes the
loss** (~0.09 vs ~0.27 — lr collapse tightens the fit). **ε-decay alone doesn't kill the orbit**
(`dbloff_eps` flips to flat-2 at the end) — it cuts exploration noise; harmonic *collapses*, ε-decay
*cleans* (complementary). double-OFF+harmonic shows a **faint high-end ramp** (`+4/6/8`→2–3).

**THE DISCOVERY (Arda, watching live): the orbit PASSES THROUGH near-Kelly ramps mid-run.** Scan of ALL
checkpoints vs Kelly `1 1 1 2 5 8 8` (L1):
- `dblon_base` **sess 700** = `1 1 1 2 5 6 8` (**dist 2 — essentially Kelly**); sess 400 = `2 1 1 1 5 8 8`.
- `ruin_dblon_s100_decay` hit ramp-shape **10/18** ckpts; `ruin_dblon_s100` **9/18**.
- (reproducibility: `dblon_base` ≡ `ruin_dblon_s50` for sess≤1000 — same seed.)

So **"RL converges to Flat" was partly an artifact of reading the FINAL checkpoint** — the trajectory
*visits* Kelly. (We only saved final weights → the sess-700 net is gone; curve logged, not the weights.)

### Open hypotheses to investigate (H-series) — the "why visits but won't hold" mechanism

**H1 — The Kelly ramp is NOT a loss-minimum; flat is the attractor; the ramp is a noise-driven transient.**
At the rare high counts, sparse data can't confidently support a high-count premium → loss-optimal value
there is ~flat → nothing holds the orbit at the ramp. *Evidence in hand:* **harmonic (noise-averaging)
collapses to flat, not the ramp** → the ramps are exactly the part that averages away.

**H2 — But the ramp signal is REAL (above chance), just weaker than the flat attractor.** A random argmax
curve over 8 levels almost never comes out monotone-rising, yet scale100+double ramps 9–10/18 → a genuine
count gradient pulls toward the ramp. Two competing forces: weak "high counts +EV → ramp" gradient vs
strong "sparse data → flat" attractor + noise. scale100/double *strengthen* the signal but don't make it win.

**H3 (discriminating — DIAGNOSTIC, not selection) — is the ramp a genuinely-better-but-unrewarded policy,
or a noise excursion?** Test: reproduce + capture the sess-700 ramp (deterministic, seed 0), four-axis eval.
- *higher growth + tolerable ruin* → the ramp IS better, but its advantage sits **below the per-hand noise
  floor it's estimated from** → TD can't separate ramp from flat → **"the signal distinguishing Kelly from
  flat is smaller than the noise."** (the coverage/objective wall, sharpened)
- *worse ruin / no better growth* → flat is correctly optimal; the "ramps" are partly a 1-D-slice artifact.

**Investigation plan:** (1) add checkpoint-saving (negligible — ~40KB/ckpt, regenerable via seed);
(2) reproduce + capture sess-700 `dblon_s50`; (3) four-axis eval the captured ramp (H3); (4) optional:
residence-time / checkpoint-autocorrelation to quantify the ramp-attractor strength (H2). **NOTE: the 3
evals running now are on FINAL (flat-ish) policies — they answer the double-OFF over-betting Q, NOT the
ramp (not in the final weights).** *Selection caveat (if a captured ramp ever becomes the deliverable, not
just a diagnostic): pick the checkpoint by a held-out val metric, NEVER by closeness-to-Kelly, and report
best-vs-final — else it's selection bias.*

---

## Test 12 — Longer sweep: more data doesn't break the flat attractor

2×2×2 (double{on,off} × regime{growth γ0, ruin γ0.95} × batch{512@5k, 2048@2k}) + rampiest (scale100),
base scale50 + harmonic + ε-decay, `--checkpoints`. **More data (2.5×) → flat-1 attractor; the ramp does
NOT stabilize.** Reading the *tails* (not just the final): **lows LOCK** (`1 1 1 1`), the **high end
perpetually WANDERS** through near-Kelly ramps and flat — e.g. `dbloff_r_b2048` @1500 = `1 1 1 1 6 8 8`
(dist-2 from Kelly), then back to flat by 2000. Harmonic shrinks the orbit but the *tied* high-count argmax
still flips even at low lr. So "more data → flat-1" was a **final-checkpoint artifact** — the ramp forms
*repeatedly*, never holds (the H-series, confirmed at 5000 sessions).
⚠ **Design slip:** double-DQN is a **no-op at γ=0**, so the growth double on/off cells came out bit-identical
(2 wasted runs — the warning was already in Test 8, and I repeated it).

## Test 13 — H3 (are the ramps real?) + multi-seed hardening [CENTRAL, deliverable]

**H3 — the visited ramps are NOT better policies.** Best-checkpoint four-axis evals (each run's
closest-to-Kelly checkpoint, incl. the dist-2 `1 1 1 1 6 8 8`) are *dominated by flat*: worse growth, FAR
higher drawdown. Overnight **multi-seed** hardening (native regime cell, mean ± std across seeds):

| config [phase] | n | growth ×1e4 | ruin % | dd % |
|---|---|---|---|---|
| ruin γ0.95 dbl-OFF [final] | 6 | −0.48 ± 0.04 | 0.03 | 1.39 ± 0.44 |
| ruin γ0.95 dbl-ON [final] | 6 | −0.52 ± 0.11 | 0.07 | **2.60 ± 3.05** |
| ruin γ0.9 dbl-OFF [final] | 3 | −0.79 ± 0.56 | 2.47 | 9.83 ± 14.3 |
| ruin γ0.99 dbl-OFF [final] | 3 | −0.52 ± 0.06 | 0.00 | 1.30 ± 0.44 |
| growth γ0 [final] | 6 | −0.19 ± 0.08 | 0.00 | 0.18 ± 0.29 |
| ruin γ0.95 dbl-OFF [best-ckpt] | 8 | −1.08 ± 0.88 | 2.09 | **13.9 ± 12.7** |
| ruin γ0.95 dbl-ON [best-ckpt] | 9 | −1.48 ± 1.21 | 4.94 | **18.3 ± 12.2** |
| *ruin flat / kelly* | | −0.42 / **−0.31** | 0 | 0.55 / 2.25 |
| *growth flat / kelly* | | −0.14 / **+0.04** | 0 | 0 |

**Hardened findings:**
1. **RL ≈ flat — tight across seeds** (finals just below flat, never Kelly; low std). The thesis holds *with
   error bars*.
2. **Best ramps robustly FAR worse than flat** (dd 14–18% vs 0.55%) — H3 decisive: the ramps are
   **over-betting artifacts** (Kelly-shaped in the 1-D probe, wealth-scaling in the full policy), NOT a
   better policy the objective failed to reward.
3. **⚠ REVISES Test 10/11 "double-ON is safe" — that was single-seed luck.** Multi-seed double-ON [final]
   dd = 2.60 ± **3.05**% (high variance, *some seeds over-bet*); **double-OFF is the consistent one**
   (1.39 ± 0.44%). Judge double by the eval across seeds, not one run's curve.
4. **γ characterized (the open item):** γ0.95/0.99 stable + safe (dd ~1.3%, low variance); **γ0.9 too low**
   — riskier, high-variance over-betting (dd 9.8 ± 14%). γ0.95 was a sound default; don't go below it.
5. **Structured Kelly is the only policy beating flat** (ruin −0.31, growth +0.04). The learned bettor
   never gets there.

*Eval infra (this session):* baseline **caching** (Kelly/Flat once) + `--regime` + `run_dir@session`
checkpoint loading + **structured result-saving** (`runs/<id>/eval_*.json`, provenance) — `eval_bet_agent.py`
/ `eval_best_checkpoints.py`. Fast eval = n2000 + cache ≈ 1 min/agent. Aggregation:
`scratchpad/aggregate_seeds.py`.

## Test 14 — Encoding ablation: wealth or thin edge? FALSIFIES the wealth hypothesis [CENTRAL]

The embedding (Test-11-era) suggested RL keyed on **wealth, not edge** (H-representational). Tested it
directly with a configurable bet encoder (`--bankroll-feature`, mirroring dqn's `encoding`/`with_splits`):
**raw** (bankroll/scale — current, D14), **logratio** (`log(W/ref)` — Kelly's natural scale), **none**
(bankroll **DROPPED** → count+depth only). Growth γ0, matched seeds 0–5, else identical to Test 13.

**Result — encoding-invariant, all ≈ flat (growth cell, mean ± std, ×1e4):**

| encoding | n | growth | dd % |
|---|---|---|---|
| raw | 6 | −0.19 ± 0.08 | 0.18 |
| logratio | 6 | −0.28 ± 0.16 | 0.13 |
| **none** (drop bankroll) | 6 | **−0.19 ± 0.04** | 0.37 |
| *flat / kelly* | | −0.14 / **+0.04** | 0 |

**`none` == `raw` exactly** — dropping wealth entirely changes nothing → **the wall is FUNDAMENTAL (thin
edge / coverage), NOT representational.** Native bet-vs-count across arms: no arm gates consistently (mostly
flat / weak; seed-5 gates in all three; `none`/`logratio` add *spurious negative-count* over-betting).

**Two overclaims corrected here (the honesty story):** (a) the embedding's "wealth organization" was an
**OOD-probe artifact** — it swept bankrolls 50–600u while the agent lived at 400u; at **native** bankroll
the agents are mostly **flat**, not wealth-scaling; (b) "learned half of Kelly" overstated — native is
mostly flat + occasional *coarse* gating (seed-dependent). **The arc:** RL≈flat → hypothesise wealth
(embedding) → **build the ablation, falsify it** → the thin-edge wall. Encoder seam:
`bet_agent.bankroll_feature` (`raw|logratio|none`) + `bet_feature_dim`.

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

1. **Code is sound** (oracle PASS) — the flatline is not a bug; with a *denoised* signal the ramp is
   clean **and stable**.
2. **γ governs stability, not the ramp** — γ=1 diverges (telescoping variance); γ<1 stable but still flat.
   Use γ=0 for growth (Kelly is a myopic optimum); intermediate γ reserved for the ruin regime.
3. **Batch (gradient-SNR) breaks the flatline, but doesn't reach Kelly.** ≤1024 stays flat; ≥2048 makes a
   ramp *appear* — but it **wanders**, and with enough data **converges to flat-minimum** (Test 7), never
   the count ramp.
4. **At γ=0: RL converges to flat-minimum, not Kelly (Test 9, CI-backed).** Undertrained agents over-bet
   → lose to Flat *and* ruin; well-trained agents → flat-1 ≈ Flat. **But this is the γ=0 story — at γ>0
   with proper reward scale, count-dependence *does* form (Test 10, #8–10).** So the headline is regime-
   dependent, not "RL can't ever learn it."
5. **Why — the thin-edge / SNR wall.** At γ=0, betting >1 only pays at the *rare* high counts and is −EV at
   the common low counts, so the loss-minimising policy is "bet the minimum." The count→size signal (~1e-4)
   is too thin and high counts too rare for value learning to resolve — the agent re-estimates the same edge
   curve that took **20M hands** to measure, through a noisier channel (betting-side analog of Problem-A's
   rare-cell coverage). The wandering is **static argmax instability on a near-flat surface** — *not* a
   limit cycle (Test 8 ruled out forgetting + showed double/Polyak are γ=0 no-ops).
6. **Meta-narrative (the report headline):** structured/analytic Kelly is **decisively more
   sample-efficient** than end-to-end RL here — RL converges *to* Flat; Kelly *beats* Flat. Lands on B2c's
   "the skill is restraint" (the agent can't learn it). Mirrors the Problem-A "less-structured vs
   principled" arc — with a *negative, principled* answer this time.
7. **Speed:** ~2.5–2.7 sess/s (γ=0, batch128); big batch / double-DQN / γ>0 are slower; threads don't help.
8. **[REVISES #5] Scale is NOT cosmetic — it's the Huber-delta regime (Test 10).** We use Huber loss
   (`smooth_l1`, delta=1); reward scale slides TD errors across delta. Too small (scale1) → quadratic →
   per-hand *outliers* swamp the signal → flat; large enough (scale50/100) → clipped/robust → the consistent
   signal survives → the count separates. Adam's ε ruled out (negligible at our gradients). The earlier
   "cosmetic" read was a **batch-128 contamination** artifact. Equivalent knob: lower the Huber **delta**.
9. **At γ>0, count-dependence FORMS (double-DQN and/or scale unlock it) — but doesn't fully stabilise.**
   The ramp *direction* appears (lows lock to ~1, highs elevate); the high-count *magnitude* still wanders.
10. **The wall is high-count COVERAGE; the achievable target is neutral consistency.** Low/neutral counts
    (common) lock reliably; high counts (rare) can't stabilise (under-sampling — the betting analog of
    Problem-A's rare cells). Honest deliverable = "reliable minimum-betting around break-even + a
    coverage-bound high end"; the fix for the high end is **oversampling / prioritised replay**, not scale
    /lr/batch. **scale50 + double-OFF + lower constant lr** is the current best cell for the neutral target.
11. **Harmonic lr collapses the orbit (§33, Robbins–Monro); loss = settledness, not correctness (Test 11).**
    Constant lr *orbits* the optimum forever (the wandering); harmonic `1/t` *collapses* it — lows lock
    rock-solid, but the high end still flips (coverage: the argmax is tied, so even tiny late-lr flips it).
12. **CONFIRMED across seeds — RL ≈ flat, and the visited ramps are FAR worse than flat (Test 13, H3).**
    The near-Kelly ramps the orbit visits are **over-betting artifacts** (dd 14–18% vs flat 0.55%, worse
    growth) — *not* a better policy the objective fails to reward. H3 answered: noise excursions. Only
    structured Kelly beats flat. (Mechanism: **wealth-scaling without edge-gating** — the net keyed on
    bankroll, not count; the PCA/t-SNE embedding shows clusters split by bankroll — REPORT_NOTES_B.)
13. **[REVISES the single-seed double reads] Multi-seed hardening (Test 13).** "double-ON is safe" was
    seed-luck — across seeds double-ON [final] dd = 2.60 ± 3.05% (variable, some over-bet) vs double-OFF
    1.39 ± 0.44% (consistent). γ characterized: γ0.95/0.99 stable+safe, γ0.9 too low. **The whole thesis
    now holds with CIs.**

**Status:** the headline is **CI-backed AND multi-seed hardened** (Test 13) — RL ≈ flat, the visited ramps
are far worse than flat, structured Kelly is the only winner; one single-seed claim (double-ON safety)
honestly revised; ruin-γ characterized. **The core investigation is complete.** **Next:** the *representational*
lever — the wealth-vs-edge diagnosis (embedding) → the `W_current/W_start` encoding experiment (a **design**
call, not a rescale — see REPORT_NOTES_B "wealth-scaling without edge-gating"), then report framing.

---

## Open items & decisions

- **⚠ WARNING (2026-06-30) — stabilisation levers are γ-DEPENDENT; double-DQN & Polyak are NO-OPS at
  γ=0.** They only touch the *bootstrap* term (`γ·max Q(s')`), which γ=0 zeroes — so a γ=0 double-DQN or
  Polyak/`target_tau` run is a **silent re-run of the hard-sync baseline** (caught via *bit-identical*
  output across `b2048_dbl` / `b2048_polyak`, killed unfinished). At γ=0 the only stability levers are
  **buffer size** (forgetting) and **lr / lr-decay**; double-DQN & Polyak only bite at **γ>0** (e.g. the
  ruin regime). Don't test them against a γ=0 run.
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
