"""Train the learned bet model (``BetAgent``) and persist it — the eval-ready B2d-3 runner.

A thin CLI around the committed ``train_bet`` (γ=0 by default — the growth-regime Kelly optimum; Kelly
is the myopic per-hand log-optimum). Trains on log-growth with fixed ``BasicStrategy`` play, streams the
bet-vs-count probe to ``logs/live.log``, and persists the trained agent via ``save_bet_run``
(``record.json`` + ``model.pt``) so every downstream wiring (four-axis eval, figures, B3) reuses it
without retraining.

Recipe (from B2d-3 diagnostics): **γ=0 + large batch (≥~2048)** breaks the count-independent flatline.
Set ``--batch`` to the value the characterization runs settle on. Background it (streams to live.log):

    .venv\\Scripts\\python.exe scripts/train_bet_agent.py --regime growth --sessions 5000 --batch 4096
"""
from __future__ import annotations

import argparse
from datetime import datetime

import torch

from blackjack_rl.core.paths import LOGS_DIR, RUNS_DIR
from blackjack_rl.session.bet_agent import greedy_bet_curve
from blackjack_rl.session.env import growth_config, ruin_config
from blackjack_rl.session.persistence import save_bet_run
from blackjack_rl.session.train import BetTrainConfig, train_bet

PROBE_COUNTS: tuple[int, ...] = (-4, -2, 0, 2, 4, 6, 8)
REGIMES = {"growth": growth_config, "ruin": ruin_config}


def _log(line: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    msg = f"[{datetime.now():%H:%M:%S}] {line}"
    print(msg, flush=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _ramp(curve: dict) -> str:
    return " ".join(f"{c:+d}:{curve.get(c, curve.get(str(c), 0)):g}" for c in PROBE_COUNTS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + persist a learned BetAgent (B2d-3).")
    ap.add_argument("--regime", choices=tuple(REGIMES), default="growth")
    ap.add_argument("--sessions", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--double", action="store_true")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--buffer", type=int, default=50_000)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay", action="store_true")  # linear lr -> 1e-5 over the run (else constant)
    ap.add_argument("--lr-harmonic", action="store_true")  # harmonic (1/t) lr decay -> lr_end (Robbins–Monro)
    ap.add_argument("--eps-decay", action="store_true")  # linear ε eps-start -> 0 over the run
    ap.add_argument("--eps-start", type=float, default=0.5)  # start ε for the decay (explore early)
    ap.add_argument("--checkpoints", action="store_true")  # persist q-net weights at every probe checkpoint
    args = ap.parse_args()

    config = BetTrainConfig(
        session=REGIMES[args.regime](),
        n_sessions=args.sessions,
        batch_size=args.batch,
        seed=args.seed,
        gamma=args.gamma,
        double_dqn=args.double,
        target_tau=args.tau,
        buffer_capacity=args.buffer,
        reward_scale=args.scale,
        lr=args.lr,
        lr_schedule=("harmonic" if args.lr_harmonic else "linear" if args.lr_decay else "constant"),
        epsilon_schedule="linear" if args.eps_decay else "constant",
        epsilon_start=args.eps_start,  # ε-decay goes eps_start -> 0 (explore early, exploit late)
    )
    tag = f"{args.regime}/b{args.batch}/s{args.seed}"
    _log(
        f"train_bet_agent START {tag}  sessions={args.sessions} gamma={config.gamma} "
        f"batch={config.batch_size} double={config.double_dqn} tau={config.target_tau} buffer={config.buffer_capacity}"
    )

    learning_curve: list[dict] = []
    checkpoints: list[tuple[int, dict]] = []  # (session, cpu state_dict) — the training trajectory

    def on_checkpoint(info: dict) -> None:
        _log(f"[{tag}] {info['session']} loss={info['recent_loss']} | bet[{_ramp(info['bet_by_count'])}]")
        learning_curve.append(info)

    def on_snapshot(session: int, state_dict: dict) -> None:
        checkpoints.append((session, state_dict))

    start = datetime.now()
    agent = train_bet(
        config, progress_every=max(1, args.sessions // 20), on_checkpoint=on_checkpoint,
        on_snapshot=on_snapshot if args.checkpoints else None,
    )
    elapsed = (datetime.now() - start).total_seconds()

    final_curve = greedy_bet_curve(
        agent, PROBE_COUNTS, bankroll=config.session.starting_bankroll, decks_remaining=agent.num_decks / 2.0
    )
    metrics = {
        "elapsed_s": round(elapsed, 1),
        "final_curve": {str(c): final_curve[c] for c in PROBE_COUNTS},
        "learning_curve": learning_curve,
        "checkpoint_sessions": [s for s, _ in checkpoints],
    }
    run_id = f"{start:%Y%m%d-%H%M%S}_bet-agent_{args.regime}_b{args.batch}_{args.sessions}sess"
    run_dir = save_bet_run(RUNS_DIR, agent, config, metrics, run_id=run_id)
    if checkpoints:  # the trajectory weights, beside the final model.pt (load via load_bet_checkpoint)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        for session, state_dict in checkpoints:
            torch.save(state_dict, ckpt_dir / f"ckpt_{session:05d}.pt")
        _log(f"  saved {len(checkpoints)} trajectory checkpoints -> {ckpt_dir}")
    _log(f"train_bet_agent DONE ({elapsed:.0f}s) final bet[{_ramp(final_curve)}]  saved -> {run_dir}")


if __name__ == "__main__":
    main()
