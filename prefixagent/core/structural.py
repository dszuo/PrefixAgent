"""Structural primitives behind the two-phase tools, on plain node sets.

Same four-function API (and the same return shapes) as the original
copy-on-write wrappers around the research-code Backbone / PrefixGraph
classes; the implementations underneath are the clean node-set ports in
backbone_ops / graph_ops, each verified behaviourally equivalent to the
originals by exhaustive differential test.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from prefixagent.core import graph_ops
from prefixagent.core.backbone_ops import (backbone_valid, enumerate_regroups,
                                           regroup)

Node = Tuple[int, int]


def enumerate_regroup_actions(noninput_nodelist, n_bits: int) -> list:
    """All legal regroup actions, deterministically sorted by (node1, node2)."""
    return [{"action_type": "regroup", "node1": a, "node2": b}
            for a, b in enumerate_regroups(noninput_nodelist, n_bits)]


def apply_regroup(noninput_nodelist, n_bits: int, action: dict) -> dict:
    prev = frozenset(tuple(t) for t in noninput_nodelist)
    out = regroup(prev, n_bits, action["node1"], action["node2"])
    if out is None:
        return {"legal": False, "next_noninput_nodelist": None, "info": {}}
    next_nl = sorted(out)
    added = sorted(out - prev)
    removed = sorted(prev - out)
    return {
        "legal": True,
        "next_noninput_nodelist": next_nl,
        "info": {
            "is_valid": backbone_valid(out, n_bits),
            "message": "",
            "added_node": added[0] if len(added) == 1 else added,
            "removed_node": removed[0] if len(removed) == 1 else removed,
        },
    }


def enumerate_level_opt_actions(full_nodelist, n_bits: int) -> list:
    """All output nodes (i, 0) on which graph_opt applies, sorted."""
    ns = frozenset(tuple(t) for t in full_nodelist)
    legal = []
    for i in range(1, n_bits):
        if graph_ops.apply_graph_opt(ns, n_bits, (i, 0)) is not None:
            legal.append({"action_type": "level_opt", "target_node": (i, 0)})
    return legal


def apply_level_opt(full_nodelist, n_bits: int, action: dict) -> dict:
    ns = frozenset(tuple(t) for t in full_nodelist)
    out = graph_ops.apply_graph_opt(ns, n_bits, tuple(action["target_node"]))
    if out is None:
        return {"legal": False, "next_full_nodelist": None, "info": {}}
    return {
        "legal": True,
        "next_full_nodelist": sorted(out),
        "info": {"is_valid": graph_ops.legal(out, n_bits)},
    }
