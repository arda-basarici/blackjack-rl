"""Small shared utilities."""
from __future__ import annotations

import re
from pathlib import Path


def concat_notebook_html(html_paths, out_path, page_break: bool = True) -> str:
    """Stitch several exported-notebook HTML files into one HTML, back to back.

    Each file is first exported with ``jupyter nbconvert --to html <nb>.ipynb``.
    This keeps the *first* file's ``<head>`` (so nbconvert's CSS applies once) and
    concatenates every file's ``<body>`` in the given order. ``page_break`` inserts a
    print page-break between notebooks, which makes "Print to PDF" paginate cleanly.

    Args:
        html_paths: HTML files to join, in report order.
        out_path:   where to write the combined HTML.
        page_break: insert a page-break between notebooks (default True).

    Returns:
        ``out_path`` (as a string).
    """
    head, bodies = "", []
    for p in html_paths:
        html = Path(p).read_text(encoding="utf-8")
        if not head:
            m = re.search(r"<head\b[^>]*>(.*?)</head>", html, re.S | re.I)
            head = m.group(1) if m else ""
        m = re.search(r"<body\b[^>]*>(.*?)</body>", html, re.S | re.I)
        bodies.append(m.group(1) if m else html)
    sep = '\n<div style="page-break-before:always"></div>\n' if page_break else "\n"
    doc = ("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
           + head + "\n</head>\n<body>\n" + sep.join(bodies) + "\n</body>\n</html>\n")
    Path(out_path).write_text(doc, encoding="utf-8")
    return str(out_path)


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
