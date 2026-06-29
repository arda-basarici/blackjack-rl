"""Project-root-anchored artifact paths — the single source of truth for where runs and logs live,
so the location can't drift between modules (e.g. the two experiment.py entry points). Anchored to
this file's location, so it is independent of the current working directory."""
from pathlib import Path

# blackjack_rl/core/paths.py -> parents: core -> blackjack_rl -> project root (blackjack-rl/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Committed reference data shipped *inside* the package (frozen, versioned), as opposed to the
# git-ignored, regenerable artifacts under runs/. The edge-by-count reference curve lives here so the
# Problem-B baseline (KellyBet) and the signature figure read one canonical source (B2c, DESIGN D17).
SESSION_DATA_DIR = PROJECT_ROOT / "blackjack_rl" / "session" / "data"
EDGE_REFERENCE_PATH = SESSION_DATA_DIR / "edge_by_count_reference.json"
