"""Persist training runs so variants are saved, comparable, and reproducible.

Each run writes one ``record.json`` — its config, metrics, git hash, and timestamp — into its
own folder under ``runs/``, and never overwrites a previous run. Comparing two variants is
then reading two records, not scrolling terminal output. Mirrors the pathfinding-ml pattern
(D8).

This module is the generic I/O: it saves any JSON-able record. Assembling the record for a
specific run (config + metrics + Q-table + per-state visit counts, with state-key flattening)
lives with the trainer that produces those artifacts (Stage 2).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def git_hash() -> str:
    """Short current commit hash, or 'unknown' if git is missing / this isn't a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _unique_dir(runs_root: Path, run_id: str) -> Path:
    """A path under runs_root that does not yet exist (append -1, -2, ... on clash), so a
    save can never overwrite an earlier run."""
    candidate = runs_root / run_id
    suffix = 1
    while candidate.exists():
        candidate = runs_root / f"{run_id}-{suffix}"
        suffix += 1
    return candidate


def save_run(
    runs_root: Path | str, record: dict[str, Any], run_id: str | None = None
) -> Path:
    """Write ``record`` to a fresh run directory and return its path. Never overwrites.

    Provenance (run_id, timestamp, git_hash) is stamped in automatically, so a run can't be
    saved without it. If ``run_id`` is omitted it defaults to
    ``<timestamp>_seed<seed>_<hash>``, reading the seed from record["config"]["seed"].
    """
    now = datetime.now()
    ghash = git_hash()
    if run_id is None:
        seed = record.get("config", {}).get("seed", "NA")
        run_id = f"{now.strftime('%Y%m%d-%H%M%S')}_seed{seed}_{ghash}"

    run_dir = _unique_dir(Path(runs_root), run_id)
    run_dir.mkdir(parents=True)
    full = {
        "run_id": run_dir.name,
        "timestamp": now.isoformat(timespec="seconds"),
        "git_hash": ghash,
        **record,
    }
    (run_dir / "record.json").write_text(json.dumps(full, indent=2, default=str))
    return run_dir
