"""Tabular policy — a Q-table over the small discrete state, exposed as a Strategy.

The Problem A state is a tiny tuple, so a lookup table is exact and fully inspectable; a
network represents nothing here. This is the right-tool choice and the honesty core. See
DESIGN.md D1, D2.
"""
