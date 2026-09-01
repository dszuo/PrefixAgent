"""Backbone (Phase 1) operations on plain node sets.

A backbone for an N-bit adder is the binary tree that computes the carry of
the most significant bit: N-1 non-input span nodes over the N input leaves,
rooted at (N-1, 0).  Nodes are (msb, lsb) tuples; input nodes (i, i) are
implicit and never stored.

`regroup` is the single Phase-1 rewrite: for an adjacent pair b, c with
b.msb == c.lsb - 1, both of whose rows' output nodes (b.msb, 0), (c.msb, 0)
take b resp. c as their closest upper parent, it removes (b.msb, 0) and adds
(c.msb, b.lsb).  This is the associativity rewrite of the prefix operator.

Behaviourally equivalent to the original PrefixAgent Backbone.regroup /
is_valid, verified by exhaustive differential test over every backbone
(all Catalan(N-1) span trees) for N = 4..8 and every unordered node pair.
"""
from __future__ import annotations

from itertools import combinations
from typing import FrozenSet, Iterable, Optional, Tuple

Node = Tuple[int, int]
NodeSet = FrozenSet[Node]


def _with_inputs(backbone: Iterable[Node], N: int) -> set:
    return set(backbone) | {(i, i) for i in range(N)}


def _upper_parent(nodes: set, m: int, l: int) -> Optional[Node]:
    """Closest upper parent of (m, l): (m, k) with the smallest k > l."""
    best = None
    for (nm, nl) in nodes:
        if nm == m and nl > l and (best is None or nl < best):
            best = nl
    return (m, best) if best is not None else None


def backbone_valid(backbone: Iterable[Node], N: int) -> bool:
    """Port of Backbone.is_valid: MSB output present, exactly N-1 non-input
    nodes, and every non-input node's implied parents exist in the set."""
    bset = frozenset(backbone)
    if (N - 1, 0) not in bset:
        return False
    if len(bset) != N - 1:
        return False
    nodes = _with_inputs(bset, N)
    for (m, l) in bset:
        up = _upper_parent(nodes, m, l)
        if up is None:
            return False
        lo = (up[1] - 1, l)
        if lo not in nodes:
            return False
    # Implied parents strictly decrease (msb - lsb), so levels always
    # resolve; the reference implementation's "level is None" branch is
    # unreachable here.
    return True


def regroup(backbone: Iterable[Node], N: int,
            n1: Node, n2: Node) -> Optional[NodeSet]:
    """Apply one regroup; None if illegal.  Mirrors Backbone.regroup exactly:

    1. both nodes exist (inputs count as nodes),
    2. orientation: b.msb == c.lsb - 1 (tried both ways),
    3. output nodes (b.msb, 0) and (c.msb, 0) exist,
    4. those outputs take b resp. c as their closest upper parent
       ("trivial fanout" -- note an output node can never be its own
       fanout, so b with lsb == 0 always fails this check),
    5. the new node (c.msb, b.lsb) does not already exist,
    6. the rewritten set passes backbone_valid, else the move is refused.
    """
    bset = frozenset(backbone)
    nodes = _with_inputs(bset, N)
    n1, n2 = tuple(n1), tuple(n2)
    if n1 not in nodes or n2 not in nodes:
        return None
    if n1[0] == n2[1] - 1:
        b, c = n1, n2
    elif n2[0] == n1[1] - 1:
        b, c = n2, n1
    else:
        return None
    bm, bl = b
    cm, cl = c
    if (bm, 0) not in nodes or (cm, 0) not in nodes:
        return None
    if b == (bm, 0) or _upper_parent(nodes, bm, 0) != b:
        return None
    if c == (cm, 0) or _upper_parent(nodes, cm, 0) != c:
        return None
    new, old = (cm, bl), (bm, 0)
    if new in nodes:
        return None
    out = frozenset((bset - {old}) | {new})
    if not backbone_valid(out, N):
        return None
    return out


def enumerate_regroups(backbone: Iterable[Node], N: int) -> list:
    """All legal regroup pairs, deterministically sorted.

    Mirrors enumerate_regroup_actions: unordered pairs over the full node
    set (inputs included), each reported as (min, max)."""
    bset = frozenset(backbone)
    keys = sorted(_with_inputs(bset, N))
    legal = []
    for a, b in combinations(keys, 2):
        if regroup(bset, N, a, b) is not None:
            legal.append((min(a, b), max(a, b)))
    legal.sort()
    return legal


def serial_backbone(N: int) -> NodeSet:
    """The all-serial start: (i, 0) for i = 1..N-1.  The only start from
    which every backbone is reachable by regroups."""
    return frozenset((i, 0) for i in range(1, N))


def all_backbones(N: int):
    """Yield every valid backbone (all Catalan(N-1) span trees) as a
    frozenset of non-input nodes.  For tests and exhaustive analyses."""
    def spans(l: int, m: int):
        if l == m:
            yield frozenset()
            return
        for k in range(l + 1, m + 1):
            for hi in spans(k, m):
                for lo in spans(l, k - 1):
                    yield hi | lo | {(m, l)}
    yield from spans(0, N - 1)
