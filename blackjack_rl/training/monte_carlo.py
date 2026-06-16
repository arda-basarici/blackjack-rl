"""Online Monte Carlo control — the credit-assignment core (Stage 2/3).

Generate complete episodes by playing hands with an exploring policy, then update the Q-table
from the returns. Monte Carlo updates after whole episodes, which fits the engine's atomic
play_hand directly. See DESIGN.md Stage 2, and the open MC-variant question in section 10.
"""
