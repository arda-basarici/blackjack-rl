"""Auto-generate the API reference (HTML) for blackjack_rl with pdoc — never hand-written.

pdoc imports the package and renders docstrings + signatures, so the output doubles as a
docstring/type-hint checklist: gaps in the rendered pages = members still missing documentation.
Regenerate after changing public APIs. (pdoc imports every module; analysis_loader's import-time
ROOT walk falls back to the project root when runs/ is absent, so this is safe from a clean tree.)

Run from the repo root:   python scripts/make_docs.py
Needs the docs extra:     pip install -e ".[docs]"
Output (git-ignored):     docs/api/index.html
"""
import subprocess
import sys
from pathlib import Path

OUT = Path("docs/api")

print(f"generating API reference for blackjack_rl -> {OUT} ...")
subprocess.run(
    [sys.executable, "-m", "pdoc", "blackjack_rl", "-o", str(OUT)],
    check=True,
)
print(f"done — open {OUT / 'index.html'}")
