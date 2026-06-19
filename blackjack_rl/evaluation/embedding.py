"""End-of-run representation analysis: reload a trained DQN and read its learned features.

Persisted runs save the network weights (``model.pt``) beside the JSON record, so a trained agent
can be reconstructed without retraining. ``cell_embeddings`` reads the penultimate-layer activation
for every canonical decision cell — the learned representation the Q head sees — paired with per-cell
metadata (chosen action, basic-strategy action, diff category, decision margin) for coloring a
PCA / t-SNE scatter.

The point (CONCEPTS §18, §26): the input space is already 2-3 interpretable dimensions (the
basic-strategy heatmap plots it exactly), so this is *not* about re-deriving that. It visualizes the
*learned feature* geometry — where the shared-weight representation separates the actions and, more
tellingly, where the genuine errors live (the confusable region = the function-approximation floor
made visible). Prefer PCA (deterministic, robust at this small N=240); use t-SNE only as a secondary
nonlinear cross-check with the perplexity reported.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from blackjack_rl.agents.dqn import DQNAgent, encode_features
from blackjack_rl.evaluation.network_diff import _state_for, diff_network, enumerate_cells


def load_agent(run_dir: Path | str) -> DQNAgent:
    """Reconstruct a trained ``DQNAgent`` from a saved run directory (``record.json`` + ``model.pt``).

    Architecture is read from the record's config; weights are loaded onto CPU; epsilon is 0 (eval
    mode). A same-config rerun would reproduce the agent, but this reloads it directly.
    """
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    cfg = record["config"]
    agent = DQNAgent(
        epsilon=0.0,
        with_splits=bool(cfg.get("with_splits", False)),
        hidden=tuple(cfg.get("hidden", (64, 64))),
        encoding=cfg.get("encoding", "scalar"),
    )
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
    agent.q_net.load_state_dict(state_dict)
    agent.q_net.eval()
    return agent


@dataclass(frozen=True)
class CellEmbeddings:
    """Aligned learned representation + metadata; index ``i`` is one decision cell.

    ``embeddings[i]`` is the penultimate-layer activation vector for cell ``i``; ``cells[i]`` holds
    its coordinates and labels (chosen action, basic-strategy action, diff category, decision margin).
    """

    embeddings: list[list[float]]
    cells: list[dict]


def cell_embeddings(agent: DQNAgent, ev_tol: float = 0.02) -> CellEmbeddings:
    """Penultimate-layer activation for every canonical cell, paired with coloring metadata.

    Joins the learned representation with the policy diff vs basic strategy so a PCA / t-SNE scatter
    can be colored by chosen ``action``, by diff ``category`` (agree / near_equal_ev /
    genuine_disagreement), or by ``q_margin`` (Q(best) - Q(2nd) over legal actions — the net's own
    confidence). Cells are the 240 canonical (player_value, is_soft, dealer_upcard) no-split cells.
    """
    cells = enumerate_cells()
    states = [_state_for(v, s, u) for (v, s, u) in cells]
    feats = torch.tensor(
        [encode_features(st, agent.with_splits, agent.encoding) for st in states],
        dtype=torch.float32,
    )
    with torch.no_grad():
        emb = agent.q_net.features(feats)  # [n_cells, hidden_last]

    report = diff_network(agent, ev_tol=ev_tol)
    by_key = {(c.player_value, c.is_soft, c.dealer_upcard): c for c in report.cells}

    meta: list[dict] = []
    for (v, s, u), st in zip(cells, states):
        q = agent.q_values(st)
        legal_idx = [agent.actions.index(a) for a in st.legal_actions() if a in agent.actions]
        vals = sorted((float(q[i]) for i in legal_idx), reverse=True)
        margin = vals[0] - vals[1] if len(vals) > 1 else float("nan")
        d = by_key.get((v, s, u))
        meta.append(
            {
                "player_value": v,
                "is_soft": s,
                "dealer_upcard": u,
                "action": d.agent_action if d else agent.greedy_action(st),
                "basic_action": d.basic_action if d else None,
                "category": d.category if d else None,
                "q_margin": round(margin, 4),
            }
        )
    return CellEmbeddings(embeddings=emb.tolist(), cells=meta)
