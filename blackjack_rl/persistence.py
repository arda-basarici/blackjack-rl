"""Run persistence — save config + seed + git hash + metrics + per-state visit counts.

Never overwrites a previous run (the pathfinding pattern). Visit counts are first-class, not
optional: distinguishing "failed to learn" (under-visited cell) from "nothing to learn"
(well-visited, near-equal-EV cell) depends on them. See DESIGN.md D8.
"""
