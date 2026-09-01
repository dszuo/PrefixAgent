"""Graph operations: pure functions over frozenset nodesets.

Semantics (single source of truth, do not reinterpret): a prefix graph
over width N is a SET of coordinate nodes (m, l), 0 <= l <= m < N.  For a
non-input (m, l) the implied upper parent is (m, k) where k is the
SMALLEST lsb > l present in row m; the implied lower parent is (k-1, l).
Parents derive from the set; there are no stored pointers, so an edit
re-snaps exactly the nodes whose closest present upper neighbour changed.

This module carries the derived-parent helpers, the legality gate
(`legal`), the level_opt rewrite (`apply_graph_opt`), and fanout
counting.  Scoring is (L, S); fanout is informational.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable

from prefixagent.core.fast_eval import node_arrivals

Node = tuple[int, int]
NodeSet = frozenset[Node]


# ---------------------------------------------------------------------------
# derived-structure helpers
# ---------------------------------------------------------------------------


def _rows(nodeset: Iterable[Node]) -> dict[int, list[int]]:
    """Map msb -> sorted list of lsbs present in that row."""
    rows: dict[int, list[int]] = {}
    for m, l in nodeset:
        rows.setdefault(m, []).append(l)
    for lsbs in rows.values():
        lsbs.sort()
    return rows


def _implied_up_lsb(rows: dict[int, list[int]], m: int, l: int) -> int | None:
    """Smallest lsb > l present in row m (closest-upper-parent rule)."""
    row = rows.get(m)
    if not row:
        return None
    idx = bisect_right(row, l)
    if idx >= len(row):
        return None
    return row[idx]


def implied_parents(
    nodeset: Iterable[Node], node: Node
) -> tuple[Node | None, Node | None]:
    """Return (upper_parent, lower_parent) coords implied by the set.

    upper is None when row m has no lsb > l (or node is an input).
    lower is the COORDINATE (k-1, l) whenever upper exists — whether that
    coordinate is present in the set is the caller's concern.
    """
    m, l = node
    if m == l:
        return None, None
    k = _implied_up_lsb(_rows(nodeset), m, l)
    if k is None:
        return None, None
    return (m, k), (k - 1, l)


# ---------------------------------------------------------------------------
# legality gate
# ---------------------------------------------------------------------------


def legal(nodeset: Iterable[Node], N: int) -> bool:
    """True iff every output (i, 0) is PRESENT in the set and COMPUTABLE.

    Computable = reachable by the closest-upper-parent arrival recursion
    (fast_eval.node_arrivals). This is the gate that catches the hazard:
    node_arrivals silently skips nodes that are not computable, so an
    illegal nodeset can still yield arrivals that look flatteringly
    shallow. Check legal() before trusting any (L, S) read off them.

    Note: outputs-computable transitively forces every input (i, i) to be
    present in the set (each row's parent chain terminates at its input), so
    no separate input-presence check is needed for i >= 1; (0, 0) is covered
    by its own output-presence check.
    """
    nodes = frozenset(nodeset)
    A = node_arrivals(nodes, N, [0] * N)
    return all((i, 0) in nodes and (i, 0) in A for i in range(N))


# ---------------------------------------------------------------------------
# the level_opt rewrite (split a node's lower parent)
# ---------------------------------------------------------------------------


def apply_graph_opt(nodeset: Iterable[Node], N: int, p: Node) -> NodeSet | None:
    """The level_opt rewrite: split p's lower parent.

    Adds (p.msb, j), where j is the implied upper lsb of p's LOWER parent,
    so p re-snaps onto the pair ((m, j), (j-1, l)) — its lower side one
    level shallower, its upper side one step longer. Returns the new
    nodeset, or None where the rewrite is undefined at p.

    Legality is preserved by construction: row m contains no lsbs strictly
    inside (l, k), so the only node the insertion re-snaps is p itself,
    and p's new lower parent (j-1, l) is one of the preconditions below.
    """
    nodes = frozenset(nodeset)
    m, l = p
    if p not in nodes:
        return None  # nothing to rewrite
    if m == l:
        return None  # an input has no lower parent to split
    rows = _rows(nodes)
    k = _implied_up_lsb(rows, m, l)
    if k is None:
        return None  # p has no implied upper parent, so no lower parent
    ntf = (k - 1, l)
    if ntf not in nodes:
        return None  # p's lower parent is absent
    if k - 1 == l:
        return None  # p's lower parent is an input: nothing to split
    j = _implied_up_lsb(rows, k - 1, l)
    if j is None:
        return None  # the lower parent has no upper parent of its own
    if (j - 1, l) not in nodes:
        return None  # after the insert, p's lower parent would be (j-1, l);
        # requiring its presence is what keeps the rewrite legality-safe
    s = (m, j)
    # --- defensive; both unreachable given the guards above ----------------
    if s[0] < s[1]:
        return None  # j <= k-1 < k <= m always
    if s in nodes:
        return None  # l < j < k and k is the SMALLEST row-m lsb > l
    return nodes | {s}


# ---------------------------------------------------------------------------
# fanout (informational — scoring is (L, S) only)
# ---------------------------------------------------------------------------


def compute_fanout(nodeset: Iterable[Node], N: int) -> dict[Node, int]:
    """In-graph consumer counts under the implied closest-upper-parent rule.

    Every non-input node with an implied upper parent contributes one
    fanout to that parent, and one to its lower parent ONLY if that
    coordinate is present. Every node in the set appears as a key (0 if
    unconsumed).
    """
    nodes = frozenset(nodeset)
    rows = _rows(nodes)
    fan: dict[Node, int] = {node: 0 for node in nodes}
    for m, l in nodes:
        if m == l:
            continue
        k = _implied_up_lsb(rows, m, l)
        if k is None:
            continue  # broken node consumes nothing
        fan[(m, k)] += 1
        lp = (k - 1, l)
        if lp in fan:
            fan[lp] += 1
    return fan


def complete(nodeset: Iterable[Node], N: int) -> NodeSet:
    """Explicit completion of a backbone into a full adder: add every input
    (i, i) and every output (i, 0).  For a valid backbone no intermediate
    node is ever needed: each output (i, 0) takes upper parent (i, k) with
    k the smallest row-i lsb > 0 present (the input (i, i) at worst) and
    lower parent (k-1, 0), another output.  Differential-tested equal to the
    original Backbone.complete_adder."""
    nodes = frozenset(nodeset)
    return frozenset(nodes | {(i, i) for i in range(N)}
                     | {(i, 0) for i in range(N)})
