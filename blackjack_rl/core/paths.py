"""Project-root-anchored artifact paths — the single source of truth for where runs and logs live,
so the location can't drift between modules (e.g. the two experiment.py entry points). Anchored to
this file's location, so it is independent of the current working directory."""
from pathlib import Path

# blackjack_rl/core/paths.py -> parents: core -> blackjack_rl -> project root (blackjack-rl/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = PROJECT_ROOT / "logs"
