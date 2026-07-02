"""Execute notebook(s) top-to-bottom in a fresh kernel and write outputs back inline — the canonical way to
refresh an analysis chapter after editing it (figures + tables embedded, so the committed ``.ipynb`` reads
self-contained). Runs each notebook with CWD = repo root so ``sys.path.insert(0, '.')`` and ``runs/`` resolve,
and pins the kernel to this venv's ``python3`` spec (the deps live here).

    .venv\\Scripts\\python.exe scripts/refresh_notebook.py analysis/session/chapters/ch4_wealth_hypothesis.ipynb

Pass several paths to refresh them in sequence. Prefer this over editing a notebook's cells by index
(no stable cell ids → positional edits scramble); edit cell *sources* by id in a script, then refresh here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in scripts/)
KERNEL_TIMEOUT_S = 900


def refresh(nb_path: Path) -> None:
    """Run one notebook in a fresh venv kernel (cwd = repo root) and write its outputs back in place."""
    nb = nbformat.read(str(nb_path), as_version=4)
    NotebookClient(
        nb, timeout=KERNEL_TIMEOUT_S, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(nb, str(nb_path))
    rel = nb_path.relative_to(ROOT) if nb_path.is_relative_to(ROOT) else nb_path
    print(f"refreshed {rel} | {len(nb.cells)} cells executed", flush=True)


def main() -> None:
    paths = [Path(a).resolve() for a in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: refresh_notebook.py <notebook.ipynb> [more.ipynb ...]")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"notebook(s) not found: {', '.join(str(p) for p in missing)}")
    for p in paths:
        refresh(p)


if __name__ == "__main__":
    main()
