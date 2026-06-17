"""Small shared utilities."""
from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Human-readable duration: '8.4s', '2m 05s', '1h 03m 07s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"
