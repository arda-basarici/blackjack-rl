"""Export the (already-run) chapter notebooks to HTML and concatenate into one report_full.html.

Run `python scripts/run_notebooks.py` FIRST so the notebooks carry fresh outputs; this step does not execute
them, it just exports the saved outputs — fast and deterministic.

Run from the repo root:   python scripts/make_report.py
"""
import subprocess
import sys

from blackjack_rl.core.util import concat_notebook_html

CHAPTERS = [
    "ch1_result", "ch2_diagnosis", "ch3_cornering", "ch4_representation",
    "ch5_honesty", "ch6_complete_game", "ch7_synthesis",
]

for f in CHAPTERS:
    print(f"exporting {f} ...")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
         "--output-dir", "analysis/html", f"analysis/dqn/chapters/{f}.ipynb"],
        check=True,
    )

out = concat_notebook_html([f"analysis/html/{f}.html" for f in CHAPTERS], "analysis/html/report_full.html")
print(f"\nwrote {out}")
