"""The pushed per-node representation block.

The paper's full Enhanced Prefix Representation, rebuilt as a PURE function
over (nodeset, arrivals) so that C2 -- Phase 1 looking at the adder its
backbone WOULD complete to -- is the same code path as Phase 2 looking at
the graph it is standing in.  The block never evaluates anything itself:
the caller passes the arrivals dict, pays for it once on the harness
ledger, and everything rendered that turn reads the same physics.

Arrival-AWARE by design: the paper's own to_epr() printed structural depth,
but this game is scored on arrival levels, so the soul is ported, not the
bug.
"""
from __future__ import annotations

from prefixagent.core.graph_ops import compute_fanout, implied_parents

Node = tuple[int, int]

__all__ = ["render_block", "render_epr"]


def _assert_not_degenerate(vals) -> None:
    """An annotated block whose annotations are all equal is not annotated
    (an earlier harness shipped [arrival=0.0000] on every node for a whole
    experiment line).  Cheap to check, expensive to miss."""
    vals = list(vals)
    if len(vals) > 2 and len(set(vals)) == 1:
        raise AssertionError(
            f"representation rendered {len(vals)} identical annotations "
            f"(all {vals[0]!r}) -- degenerate view, refusing to ship it")


def _consumers(ns, N):
    """{node: (trivial_fanouts, non_trivial_fanouts)} by the paper's split:
    trivial = consumer shares the producer's msb (same row continuing)."""
    out = {n: ([], []) for n in ns}
    for n in sorted(ns):
        up, lo = implied_parents(ns, n)
        for p in (up, lo):
            if p is None or p not in out:
                continue
            (out[p][0] if p[0] == n[0] else out[p][1]).append(n)
    return out


def render_epr(ns, N: int, A: dict, L_max: int) -> str:
    ns = frozenset(ns)
    fo = compute_fanout(ns, N)
    cons = _consumers(ns, N)
    outs = [A[(i, 0)] for i in range(N) if (i, 0) in A]
    L = max(outs) if outs else None
    lines = ["[EPR]",
             f"bitwidth: {N}",
             f"non-input nodes: {sum(1 for m, l in ns if m != l)}",
             f"max arrival level: {L} (L_max {L_max})",
             f"max fanout: {max(fo.values()) if fo else 0}",
             "",
             "nodes: (msb,lsb) arrival, up, lp, tf, ntf, fanout"]
    ann = []
    for n in sorted(ns):
        m, l = n
        up, lo = implied_parents(ns, n)
        tf, ntf = cons[n]
        ann.append(A.get(n, "?"))
        lines.append(
            f"({m},{l}) a={A.get(n, '?')}"
            f" up={f'({up[0]},{up[1]})' if up else '-'}"
            f" lp={f'({lo[0]},{lo[1]})' if lo and lo in ns else '-'}"
            f" tf={'[' + ','.join(f'({a},{b})' for a, b in tf) + ']'}"
            f" ntf={'[' + ','.join(f'({a},{b})' for a, b in ntf) + ']'}"
            f" fo={fo.get(n, 0)}")
    _assert_not_degenerate(ann)
    return "\n".join(lines)


def render_block(kind: str, ns, N: int, A: dict, L_max: int) -> str:
    """Dispatch on the p2_view value.  Unknown kinds raise -- a view
    setting that silently renders nothing is the fake-axis failure again."""
    if kind == "none":
        return ""
    if kind == "epr":
        return render_epr(ns, N, A, L_max)
    raise ValueError(f"unknown representation block {kind!r}")
