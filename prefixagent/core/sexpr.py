"""Timing-annotated S-expression rendering of a backbone.

Byte-for-byte equivalent to the original Backbone-pointer renderer (verified
by differential test over random backbones and arrival profiles): the tree is
traversed through the implied closest-upper-parent rule, and arrivals come
from the same frozen node_arrivals evaluation the game scores with.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from prefixagent.core import fast_eval as FE

Node = Tuple[int, int]


def backbone_to_sexpression(
    nodelist: Iterable[Node],
    N: int,
    arrival: list,
    indent: int = 4,
) -> str:
    """Render the backbone rooted at (N-1, 0) with per-node arrivals."""
    bset = frozenset(tuple(n) for n in nodelist)
    nodes = bset | {(i, i) for i in range(N)}
    A = FE.node_arrivals(nodes, N, list(arrival))

    def parents(m: int, l: int):
        best = None
        for (nm, nl) in nodes:
            if nm == m and nl > l and (best is None or nl < best):
                best = nl
        if best is None:
            return None, None
        return (m, best), (best - 1, l)

    def build(node: Node, depth: int = 0) -> str:
        m, l = node
        arr = A.get((m, l), 0)
        base = " " * (depth * indent)
        child = " " * ((depth + 1) * indent)
        if m == l:
            return f"input i{m} [arrival={arr}]"
        up, lp = parents(m, l)
        if up is None or lp is None or up not in nodes or lp not in nodes:
            return f"({m},{l}) [arrival={arr}] = incomplete"
        return (
            f"({m},{l}) [arrival={arr}] =\n"
            f"{base}group(\n"
            f"{child}{build(up, depth + 1)},\n"
            f"{child}{build(lp, depth + 1)})"
        )

    root = (N - 1, 0)
    if root not in nodes:
        return f"({N-1},0) missing: not a complete backbone"
    return build(root)
