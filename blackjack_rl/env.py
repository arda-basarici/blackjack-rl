"""Episode-capture wrapper around the Phase 2 engine.

The engine's HandSimulator.play_hand() is atomic: it plays a whole hand, calling
strategy.decide() at each point, and returns a HandResult carrying the trajectory and the
payout. So instead of a control-flipping reset()/step() env, we wrap an agent as a Strategy,
let it record the GameStates it is handed, run play_hand, and read the trajectory + reward
back. No engine changes. See DESIGN.md D7.
"""
