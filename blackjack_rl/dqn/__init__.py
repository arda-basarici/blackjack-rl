"""Step 2 — the deep Q-network agent: a small MLP approximating the action-value
function over the same state contract as the tabular table, with experience replay,
target networks, schedules, and the network-diff / embedding tooling that audits the
learned policy against the exact table. The deliberate 'cost of generalization' study."""
