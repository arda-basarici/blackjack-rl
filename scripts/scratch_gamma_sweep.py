"""Gamma-sweep diagnostic (B2d-3 scratch): does shortening the horizon recover the Kelly ramp?

Locked-baseline OFAT (CONCEPTS §32): growth_config with identical net / replay / epsilon / lr across
runs; the ONLY variable is the TD discount ``gamma`` in {0.0, 0.9, 0.99, 1.0}.

Hypothesis under test (telescoping variance): with gamma=1 the per-hand return telescopes to
log(W_final / W_t), so ~999 future hands' noise swamps the one-hand bet signal (SNR collapse). Kelly is
the *myopic* per-hand log-optimum, so a short horizon (gamma -> 0) should resolve the count->size ramp
in the growth regime (where hard ruin is dormant). Falsifier: if gamma=0 ALSO fails to learn the ramp,
the telescoping diagnosis is wrong -> pivot to a deeper audit (encoder / reward reconstruction / signal).

THROWAWAY diagnostic, NOT a committed deliverable. One gamma per invocation (launch 4 in parallel);
streams tagged checkpoint lines to logs/live.log and writes the final bet-vs-count curve to a JSON in
the scratch results dir for the aggregator.

    .venv\\Scripts\\python.exe scripts/scratch_gamma_sweep.py <n_sessions> <gamma>
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from blackjack_rl.core.paths import LOGS_DIR
from blackjack_rl.session.bet_agent import greedy_bet_curve
from blackjack_rl.session.env import growth_config
from blackjack_rl.session.references import load_edge_reference
from blackjack_rl.session.train import BetTrainConfig, train_bet

PROBE_COUNTS: tuple[int, ...] = (-4, -2, 0, 2, 4, 6, 8)  # matches train._PROBE_COUNTS
SEED = 0
EPSILON = 0.1
RESULTS_DIR = Path(__file__).resolve().parent.parent / "logs" / "gamma_sweep"


def _log(line: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main() -> None:
    n_sessions = int(sys.argv[1])
    gamma = float(sys.argv[2])
    tag = f"[g={gamma:g}]"

    cfg = BetTrainConfig(
        session=growth_config(), n_sessions=n_sessions, gamma=gamma, epsilon=EPSILON, seed=SEED,
    )
    env_decks = 6.0  # growth_config sim is 6 decks; probe at mid-shoe
    probe_bankroll = cfg.session.starting_bankroll
    probe_depth = env_decks / 2.0

    _log(f"{tag} START gamma={gamma:g} n_sessions={n_sessions} seed={SEED} eps={EPSILON}")

    def on_checkpoint(info: dict) -> None:
        curve = info["bet_by_count"]
        ramp = " ".join(f"{c:+d}:{curve.get(c, curve.get(str(c))):g}" for c in PROBE_COUNTS)
        _log(
            f"{tag} {info['session']}/{n_sessions} loss={info['recent_loss']} "
            f"eps={info['epsilon']} | bet[{ramp}]"
        )

    start = datetime.now()
    agent = train_bet(cfg, progress_every=max(1, n_sessions // 8), on_checkpoint=on_checkpoint)
    elapsed = (datetime.now() - start).total_seconds()

    curve = greedy_bet_curve(agent, PROBE_COUNTS, bankroll=probe_bankroll, decks_remaining=probe_depth)
    kelly = load_edge_reference().kelly_curve

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"gamma_{gamma:g}.json"
    out.write_text(json.dumps({
        "gamma": gamma, "n_sessions": n_sessions, "seed": SEED, "epsilon": EPSILON,
        "elapsed_s": round(elapsed, 1),
        "curve": {str(c): curve[c] for c in PROBE_COUNTS},
    }, indent=2))

    ramp = " ".join(f"{c:+d}:{curve[c]:g}" for c in PROBE_COUNTS)
    _log(f"{tag} DONE ({elapsed:.0f}s) final bet[{ramp}]  -> {out.name}")
    # kelly reference echoed once for context (same for all gammas)
    kref = " ".join(f"{c:+d}:{kelly.get(c, 0.0):.3f}" for c in PROBE_COUNTS)
    _log(f"{tag} kelly f*(TC) [{kref}]")


if __name__ == "__main__":
    main()
