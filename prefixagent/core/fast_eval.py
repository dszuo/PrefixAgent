"""Fast arrival-aware exact evaluator for prefix graphs.

Every non-input node (m, l) takes as upper parent the node (m, k) with
the SMALLEST k > l present in row m (the closest-upper-parent rule) and
as lower parent (k-1, l); arrival = max(parent arrivals) + 1.

dict + bisect DP in span-ascending order: both parents of a node span
strictly fewer bits than the node, so span-ascending is a valid
topological order.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable


def node_arrivals(
    nodeset: Iterable[tuple[int, int]], N: int, arrival: list[int]
) -> dict[tuple[int, int], int]:
    """Return arrival time for every computable node under the frozen rule."""
    if len(arrival) != N:
        raise ValueError(f"arrival length {len(arrival)} != N={N}")
    nodes = set(nodeset)
    rows: dict[int, list[int]] = {}
    for m, l in nodes:
        rows.setdefault(m, []).append(l)
    for lsbs in rows.values():
        lsbs.sort()

    A: dict[tuple[int, int], int] = {(i, i): arrival[i] for i in range(N)}
    for m, l in sorted(
        ((m, l) for m, l in nodes if m != l), key=lambda t: t[0] - t[1]
    ):
        row = rows[m]
        idx = bisect_right(row, l)  # smallest present k > l
        if idx >= len(row):
            continue  # no upper parent — the node is not computable; skipped
        k = row[idx]
        up, lp = (m, k), (k - 1, l)
        if up in A and lp in A:
            A[(m, l)] = max(A[up], A[lp]) + 1
    return A
