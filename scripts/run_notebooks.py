"""Restart-kernel + run-all-cells + save — for every chapter notebook, from the terminal.

Equivalent to opening each notebook, "Restart Kernel and Run All", and saving: it executes the
notebook on a fresh kernel and writes the outputs back INTO the .ipynb (in place). Run this whenever
the data or code changed (e.g. after the reeval fold-in), then export with make_report.py.

Run from the repo root:   python scripts/run_notebooks.py
Needs the normal env (torch + the Phase-2 simulator on the path, jupyter/nbconvert).
"""
import subprocess
import sys

CHAPTERS = [
    "ch1_result", "ch2_diagnosis", "ch3_cornering", "ch4_representation",
    "ch5_honesty", "ch6_complete_game", "ch7_synthesis",
]

for f in CHAPTERS:
    print(f"restart + run-all + save: {f} ...")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=900", f"analysis/dqn/chapters/{f}.ipynb"],
        check=True,
    )
print("\ndone — all notebooks executed on a fresh kernel and saved with outputs")
