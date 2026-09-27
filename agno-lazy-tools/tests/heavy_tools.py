"""Stands in for a tool module that is expensive to import (big SDKs, clients...).

The lazy-import test registers `generate_report` by import path and checks that this
module is only imported once the tool is loaded.
"""


def generate_report(quarter: str) -> str:
    """Generate the quarterly sales report.

    Args:
        quarter: Quarter to report on, e.g. "2026-Q3".
    """
    return f"Report for {quarter}: revenue up 12%"
