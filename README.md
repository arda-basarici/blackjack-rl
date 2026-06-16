# Blackjack RL

An RL agent learns to play blackjack from win/loss rewards alone, then we check what it
learned against a strategy we have already proven optimal (the Phase 2 basic-strategy table).
The value is in the **audit**, not the agent: where does it rediscover the optimum, and where
doesn't it — and is a disagreement a *failure to learn* or genuinely *nothing to learn*?

- **Design & decisions:** [DESIGN.md](DESIGN.md) (the what and why, D1–D10)
- **Implementation notes:** [ARCHITECTURE.md](ARCHITECTURE.md) (the how, grown during the build)

> This README is written at the end of the project. Stub for now.

## Setup

This project uses its own virtual environment and depends on the Phase 2 engine via an
editable install (see DESIGN.md D10).

```powershell
# from this folder: phase3-deep-learning/blackjack-rl
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ..\..\phase2-data\blackjack-sim   # the Phase 2 engine + BasicStrategy
pip install -e ".[dev]"                           # this project + dev tools
```

### Dependencies

This project declares its dependencies in `pyproject.toml` (it's an installable package),
so there is no `requirements.txt`. Earlier phases used `requirements.txt` because they were
scripts/notebooks rather than packages; from Phase 3 on the projects are packaged, and
`pyproject.toml` is the single source of truth for dependencies.
