"""Independent verification of a recorded episode.

Two checks that do not share code with the number under test:
  1. REPLAY the episode's recorded tool calls through a fresh environment
     at the EXACT recorded budgets (never padded: budget-sensitive tools
     branch on what is left, so a padded replay searches further than the
     real run did and manufactures false mismatches), asserting that every
     turn's (S, L, viol_count, viol_sum) reproduces the recorded value.
  2. Re-derive parents, levels and feasibility from the final node set with
     an evaluator written here from the definition -- not the engine's own
     evaluation path, which is the code that produced the claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prefixagent.runtime.game import game_config, game_config_hash
from prefixagent.runtime.episode import make_two_phase_episode
from prefixagent.runtime.tools import dispatch, game_tools

PKG = Path(__file__).resolve().parent


def independent_eval(nodes, N: int, arrival, L_max: int) -> dict:
    """Levels from the definition: upper parent = smallest present k > lsb
    with the same msb; lower parent = (k-1, lsb); input (i, i) starts at
    arrival[i]."""
    ns = set(map(tuple, nodes))
    for i in range(N):
        if (i, i) not in ns:
            return {"legal": False, "why": f"input ({i},{i}) missing"}
    lvl = {(i, i): arrival[i] for i in range(N)}
    todo = [n for n in ns if n[0] != n[1]]
    guard = 0
    while todo:
        guard += 1
        if guard > len(ns) + 5:
            return {"legal": False, "why": f"unresolvable nodes {sorted(todo)}"}
        progress = False
        for n in list(todo):
            m, l = n
            up = None
            for k in range(l + 1, m + 1):
                if (m, k) in ns:
                    up = (m, k)
                    break
            if up is None:
                return {"legal": False, "why": f"{n} has no upper parent"}
            lo = (up[1] - 1, l)
            if lo not in ns:
                return {"legal": False, "why": f"{n} lower parent {lo} absent"}
            if up in lvl and lo in lvl:
                lvl[n] = max(lvl[up], lvl[lo]) + 1
                todo.remove(n)
                progress = True
        if not progress:
            return {"legal": False, "why": f"cycle among {sorted(todo)}"}
    missing = [(i, 0) for i in range(N) if (i, 0) not in ns and i > 0]
    if missing:
        return {"legal": False, "why": f"outputs missing {missing}"}
    olv = {i: lvl[(i, 0)] if i > 0 else lvl[(0, 0)] for i in range(N)}
    viol = {i: v for i, v in olv.items() if v > L_max}
    return {"legal": True, "S": sum(1 for m, l in ns if m != l),
            "L": max(olv.values()), "feasible": not viol, "violators": viol}


def load_cell(cid: str, suite: str) -> dict:
    cells = {c["id"]: c for c in
             json.loads(Path(suite).read_text())["instances"]}
    cell = dict(cells[cid])
    assert "arrival" in cell and len(cell["arrival"]) == cell["N"]
    return cell


def verify_record(r: dict, suite: str) -> bool:
    cell = load_cell(r["cell"], suite)
    rec_hash = r.get("game_config_hash")
    if rec_hash and rec_hash != game_config_hash():
        # a replay through a different build is not the provenance contract:
        # say so rather than silently replaying anyway
        print(f"\n=== {r['cell']} / seed {r.get('seed')}  "
              f"game_config_hash mismatch: record {rec_hash}, "
              f"this build {game_config_hash()} -- the game or its "
              f"player-facing text changed since this episode ran")
    axes = game_config()
    env = {k: v for k, v in axes.items() if not k.startswith("context_")}
    b = r["budgets"]
    ep = make_two_phase_episode(
        cell, tools=game_tools(), seed=r["seed"],
        max_tool_calls=b["tool_calls"], max_evals=b["evals"],
        max_mutations=b["mutations"], **env)
    mism = []
    for t in r["turns"]:
        if not t.get("tool"):
            continue
        out = dispatch(ep, t["tool"], t.get("args") or {})
        for fld, key in (("S", "S"), ("L", "L"),
                         ("viol_count", "violations"),
                         ("viol_sum", "viol_sum")):
            if t.get(fld) is not None and out.get(key) != t[fld]:
                mism.append((t["n"], t["tool"], fld, t[fld], out.get(key)))
        if t.get("S") is not None and ep.S_now != t["S"]:
            mism.append((t["n"], t["tool"], "S_now", t["S"], ep.S_now))
    ns = ep.completed_nodeset()
    ind = independent_eval(ns, cell["N"], cell["arrival"], cell["L_max"])
    print(f"\n=== {r['cell']} / seed {r['seed']}  ({r.get('model')})")
    print(f"  recorded:    submitted_S={r.get('submitted_S')} "
          f"feasible={r.get('submitted_feasible')} final_L={r.get('final_L')}")
    print(f"  replay:      {len(r['turns'])} turns, "
          f"(S,L,viol_count,viol_sum,S_now) mismatches = {len(mism)}"
          + (f"  {mism[:3]}" if mism else ""))
    print(f"  independent: legal={ind.get('legal')} S={ind.get('S')} "
          f"L={ind.get('L')} feasible={ind.get('feasible')} "
          f"L_max={cell['L_max']}"
          + (f"  why={ind.get('why')}" if not ind.get("legal") else ""))
    if r.get("submitted_S") is not None:
        ok = (bool(ind.get("legal")) and bool(ind.get("feasible"))
              and ind.get("S") == r["submitted_S"]
              and ind.get("L") == r.get("final_L") and not mism)
    else:
        ok = not mism
    print(f"  ==> {'PASS' if ok else 'FAIL'}")
    return ok


def verify(path: str, suite: str) -> tuple[int, int]:
    """Verify every episode in one .jsonl file (the driver appends one JSON
    record per line; --out accumulates across runs)."""
    ok = n = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            n += 1
            ok += verify_record(json.loads(line), suite)
    return ok, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", help="episode .jsonl file(s)")
    ap.add_argument("--suite",
                    default=str(PKG / "suites/default_suite.json"),
                    help="the suite the episodes were run on (custom-suite "
                         "records need their own file here)")
    a = ap.parse_args()
    ok = n = 0
    for p in a.episodes:
        o, k = verify(p, a.suite)
        ok += o; n += k
    print(f"\n{ok}/{n} verified")
    return 0 if ok == n and n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
