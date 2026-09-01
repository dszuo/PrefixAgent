"""Tool surface for the two-phase environment.

The authoritative tool list is ALL_TOOLS below, derived from the phase
tuples -- this docstring deliberately does not repeat it by hand (a
hand-typed copy once silently dropped two Phase 2 tools).

Everything structural delegates to prefixagent.core.structural, whose
node-set ports of the paper's regroup and graph_opt were verified
behaviourally equivalent to the reference implementation by exhaustive
differential test.  Nothing here re-implements an operator.

Three properties the surface has to hold, each because of a specific way this
project has been burned:

  * A tool not advertised is REFUSED, never routed to a fallback.
    Letting guessed names execute collapses tool-set ablations onto each
    other.

  * Illegality returns an explicit error dict.  The reference regroup
    signalled failure by printing and returning False, and a swallowed
    rejection reads exactly like a move that legitimately did nothing.

  * Every rejection still costs a tool call.  A free retry loop turns a
    budget into a suggestion.

`preview` is the affordance the ridge theory earns: it reports the exact price
of the current backbone -- S_completed = 2(N-1) - ridge in closed form, and
L_completed for one evaluation -- so the player can see what a bridge would
cost before paying for it.
"""
from __future__ import annotations

from typing import Any, Callable

from prefixagent.core import graph_ops
from prefixagent.core import structural as ST
from prefixagent.runtime.episode import BudgetExceeded, TwoPhaseEpisode

Node = tuple[int, int]

PHASE1_TOOLS = ("list_regroups", "preview_regroups", "regroup",
                "regroup_wave", "preview", "bridge")
PHASE2_TOOLS = ("list_level_opts", "preview_level_opts",
                "level_opt", "level_opt_run", "level_opt_wave", "submit")
ANY_PHASE_TOOLS = ("get_state",)
ARCHIVE_TOOLS = ("archive_put", "archive_list", "fork")

ALL_TOOLS = PHASE1_TOOLS + PHASE2_TOOLS + ANY_PHASE_TOOLS + ARCHIVE_TOOLS



def game_tools() -> tuple[str, ...]:
    """The advertised tool set.

    Lives here rather than in the driver because it IS part of the
    environment: a tool list that only exists in a script is one the test
    suite cannot reach.  Such a list once was in a script, written by hand,
    and predated preview_level_opts and level_opt_run -- so an episode
    silently lost both Phase 2 affordances.

    Derived from the phase tuples, so anything added to PHASE2_TOOLS is
    advertised automatically.
    """
    return ALL_TOOLS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _node(arg: Any, name: str) -> Node:
    if isinstance(arg, (list, tuple)) and len(arg) == 2:
        return (int(arg[0]), int(arg[1]))
    raise ValueError(f"{name} must be a [msb, lsb] pair, got {arg!r}")


def _ok(ep: TwoPhaseEpisode, **extra) -> dict:
    out = {"ok": True, **ep.state()}
    out.update(extra)
    return out


def _err(ep: TwoPhaseEpisode, msg: str, **extra) -> dict:
    out = {"ok": False, "error": msg, **ep.state()}
    out.update(extra)
    return out


def _wave_conflicts(pairs: list[tuple[Node, Node]]) -> list[tuple[int, int]]:
    """Indices of regroup pairs that are NOT independent.

    The wave-exchange criterion: two regroups commute when the four nodes they
    name are pairwise distinct.  Sharing any node means one rewrites what the
    other reads, and the pair has to be sequenced rather than batched.
    """
    bad = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if set(pairs[i]) & set(pairs[j]):
                bad.append((i, j))
    return bad


def _pi_map(ns, N: int) -> dict[int, int]:
    """pi(i) = min{k-1 : (i,k) in G, k>0} -- the output-predecessor tree.

    Theorem (externally reviewed, verified 47/47 on the real operator):
    level_opt(i) is legal iff pi(i) > 0, and its whole effect is the
    grandparent shortcut pi(i) <- pi(pi(i)), inserting exactly (i, pi(pi(i))+1).
    This map therefore prices legality checks and independence checks at
    O(nodes), no evaluator pass and no PrefixGraph construction.
    """
    pi: dict[int, int] = {}
    for i in range(1, N):
        ls = [l for (m, l) in ns if m == i and l > 0]
        pi[i] = (min(ls) - 1) if ls else 0
    return pi


def _mis_wave(targets: list[int], pi: dict[int, int]) -> list[int]:
    """A maximum independent wave: MIS on the forest the pi tree induces on
    `targets`, by the linear tree DP (verified against
    brute force on 521 states, weighted included).  Returns the chosen
    target set, not just its size, because the menu's job is to hand the
    player a batch it can paste into level_opt_wave.
    """
    tset = set(targets)
    children: dict[int, list[int]] = {t: [] for t in targets}
    roots = []
    for t in targets:
        if pi[t] in tset:
            children[pi[t]].append(t)
        else:
            roots.append(t)

    def dp(v: int) -> tuple[tuple[int, list[int]], tuple[int, list[int]]]:
        take_w, take_set = 1, [v]
        skip_w, skip_set = 0, []
        for c in children[v]:
            (c1w, c1s), (c0w, c0s) = dp(c)
            take_w += c0w
            take_set = take_set + c0s
            if c1w >= c0w:
                skip_w += c1w
                skip_set = skip_set + c1s
            else:
                skip_w += c0w
                skip_set = skip_set + c0s
        return (take_w, take_set), (skip_w, skip_set)

    out: list[int] = []
    for r in roots:
        (w1, s1), (w0, s0) = dp(r)
        out.extend(s1 if w1 >= w0 else s0)
    return sorted(out)


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def _list_regroups(ep: TwoPhaseEpisode) -> dict:
    if ep.phase != 1:
        return _err(ep, "list_regroups is a Phase 1 tool; this episode has bridged")
    acts = ST.enumerate_regroup_actions(list(ep.backbone), ep.inst.N)
    return _ok(ep, regroups=[[list(a["node1"]), list(a["node2"])] for a in acts],
               n=len(acts))


def _apply_one_regroup(bb: tuple[Node, ...], N: int, a: Node, b: Node):
    """Returns (new_backbone, None) or (None, error_message)."""
    res = ST.apply_regroup(list(bb), N,
                           {"action_type": "regroup", "node1": a, "node2": b})
    if not res.get("legal"):
        # the structural layer signals refusal in-band;
        # surface it rather than letting it read as a no-op
        return None, (res.get("reason")
                      or f"regroup({list(a)}, {list(b)}) is not legal here")
    nl = res.get("next_noninput_nodelist")
    if not nl:
        return None, "regroup returned an empty node list"
    return tuple(sorted(tuple(t) for t in nl)), None


def _regroup(ep: TwoPhaseEpisode, a, b) -> dict:
    if ep.phase != 1:
        return _err(ep, "regroup is a Phase 1 tool; this episode has bridged")
    na, nb = _node(a, "a"), _node(b, "b")
    ep.charge_mutations(1)
    new, err = _apply_one_regroup(ep.backbone, ep.inst.N, na, nb)
    if err:
        return _err(ep, err)
    before = ep.ridge_len()
    ep.backbone = new
    ep.g += 1
    ep._check_identity()
    return _ok(ep, applied=[list(na), list(nb)],
               ridge_delta=ep.ridge_len() - before)


def _regroup_wave(ep: TwoPhaseEpisode, pairs) -> dict:
    """A set of pairwise-independent regroups, applied atomically.

    One tool call, len(pairs) mutations.  The point is the transcription tax:
    Phase I work here is dominated by the cost of naming moves, not by the
    moves, and g reaches 8-11 on the cells where Phase 1 matters at all.

    Atomic on purpose -- a partially applied wave would leave the player with
    a state they did not ask for and cannot name.
    """
    if ep.phase != 1:
        return _err(ep, "regroup_wave is a Phase 1 tool; this episode has bridged")
    if not isinstance(pairs, (list, tuple)) or not pairs:
        return _err(ep, "pairs must be a non-empty list of [[a_msb,a_lsb],[b_msb,b_lsb]]")
    try:
        want = [(_node(p[0], "a"), _node(p[1], "b")) for p in pairs]
    except (ValueError, IndexError, TypeError) as e:
        return _err(ep, f"malformed pairs: {e}")
    bad = _wave_conflicts(want)
    if bad:
        return _err(ep, f"wave is not independent: regroups {bad} share a node; "
                        "batched regroups must name four pairwise-distinct nodes",
                    conflicts=bad)
    ep.charge_mutations(len(want))
    bb, before = ep.backbone, ep.ridge_len()
    for i, (a, b) in enumerate(want):
        bb, err = _apply_one_regroup(bb, ep.inst.N, a, b)
        if err:
            return _err(ep, f"wave rejected at entry {i}: {err}; "
                            "no part of it was applied", failed_at=i)
    ep.backbone = bb
    ep.g += len(want)
    ep._check_identity()
    return _ok(ep, applied=len(want), ridge_delta=ep.ridge_len() - before)


#: rows a quote may return.  The driver caps a serialised TOOL_RESULT, and a
#: quote of the whole frontier at N=32 is several times that cap -- so without
#: a cap HERE the tail was cut mid-JSON and the player received a broken
#: fragment with no sign anything was missing.  preview_regroups serialises to
#: ~6200 chars (25 of 30 rows gone), and preview_level_opts had been losing
#: half its rows this way in live probes, in a heavily-called tool.
#: Capping in the tool makes the cut structural, keeps the best rows (they are
#: sorted), and REPORTS the omission.
QUOTE_ROW_CAP = 16


def _cap_rows(rows: list) -> tuple[list, int]:
    return rows[:QUOTE_ROW_CAP], max(0, len(rows) - QUOTE_ROW_CAP)


def _preview_regroups(ep: TwoPhaseEpisode, pairs=None) -> dict:
    """What every legal regroup would do to the COMPLETED adder, doing none.

    Phase 2 has had preview_level_opts since the pilot asked for it; Phase 1
    had nothing equivalent.  `preview` reports the completed L of the backbone
    the player is standing on, not what any individual regroup would change,
    so choosing a regroup meant reconstructing the backbone from the legal-move
    list and simulating level propagation in the model's head.  That is where
    one probe episode burned three consecutive turns at max_tokens.

    The paper's agent handed the model this outright -- a helper returned
    ((b), (c), level_reduction) triples -- and a later harness gave at
    least the count.  This environment at first gave neither
    while giving Phase 2 the per-candidate version, so the asymmetry was
    internal to one environment.

    Same two-currency rule as the Phase 2 quote: one evaluator pass per
    candidate, ZERO mutations, and no ratchet, or the tool would auto-play the
    game.  `pairs` narrows the probe so the eval cost is the player's to
    choose rather than fixed at the size of the frontier.
    """
    if ep.phase != 1:
        return _err(ep, "preview_regroups is a Phase 1 tool; this episode has bridged")
    want = None
    if pairs is not None:
        try:
            want = {(_node(q[0], "a"), _node(q[1], "b")) for q in pairs}
        except (ValueError, IndexError, TypeError) as e:
            return _err(ep, f"malformed pairs: {e}")
    N = ep.inst.N
    acts = ST.enumerate_regroup_actions(list(ep.backbone), N)
    rows, spent = [], 0
    for a in acts:
        n1, n2 = tuple(a["node1"]), tuple(a["node2"])
        if want is not None and (n1, n2) not in want and (n2, n1) not in want:
            continue
        if ep.evals_left <= 0:
            return _ok(ep, previews=rows, probed=spent, truncated=True,
                       note="evaluator budget ran out mid-preview")
        nb, err = _apply_one_regroup(ep.backbone, N, n1, n2)
        if err:
            continue
        ev = ep.probe(graph_ops.complete(list(nb), N))
        spent += 1
        rows.append({"pair": [list(n1), list(n2)],
                     "L_completed_after": ev.L,
                     "level_sum_after": ev.level_sum,
                     "violations_after": ev.viol_count,
                     "viol_sum_after": ev.viol_sum,
                     "feasible_after": ev.feasible,
                     "spine_after": sum(1 for (m, l) in nb if l == 0),
                     "S_completed_after": (N - 1) + ep.g + 1})
    # gap order: total excess over L_max first, because that is
    # the actual distance to feasibility; L alone is a small integer that
    # mostly ties.  level_sum stays as the deterministic tail tie-break.
    rows.sort(key=lambda d: (d["viol_sum_after"], d["violations_after"],
                             d["L_completed_after"] if d["L_completed_after"]
                             is not None else 10 ** 9, d["level_sum_after"]))
    shown, omitted = _cap_rows(rows)
    return _ok(ep, previews=shown, probed=spent, truncated=False,
               rows_omitted=omitted, rows_total=len(rows),
               note="nothing was applied; a preview never scores. Each of "
                    "these costs one node if you take it. They are ranked "
                    "against the CURRENT backbone -- taking any one of them "
                    "changes what the others would do, so this ranking is "
                    "only valid for the next single move."
                    + (f" Showing the best {len(shown)} of {len(rows)}; "
                       f"{omitted} more were probed and are not listed -- "
                       f"narrow with `pairs` to see specific ones."
                       if omitted else ""))


def _preview(ep: TwoPhaseEpisode) -> dict:
    """Exact price of bridging from here, without bridging.

    S_completed is closed form (ridge theorem 3: 2(N-1) - ridge, verified on
    429 and 4,862 backbones), so only L_completed costs an evaluation.
    """
    if ep.phase != 1:
        return _err(ep, "preview is a Phase 1 tool; this episode has bridged")
    try:
        ep.check_evals()
    except BudgetExceeded as e:
        return _err(ep, str(e))
    ev = ep.evaluate()
    N = ep.inst.N
    return _ok(ep, S_completed=2 * (N - 1) - ep.ridge_len(),
               L_completed=ev.L, feasible=ev.feasible,
               violations=ev.viol_count, level_sum=ev.level_sum)


def _bridge(ep: TwoPhaseEpisode) -> dict:
    """Complete the backbone and enter Phase 2.  Irreversible in this branch.

    A fork back to an archived Phase 1 snapshot is the only way back, which is
    what makes the archive load-bearing rather than a convenience.
    """
    if ep.phase != 1:
        return _err(ep, "already in Phase 2")
    ep.commit_backbone()
    try:
        ev = ep.evaluate()
    except BudgetExceeded as e:
        return _err(ep, str(e))
    return _ok(ep, L=ev.L, feasible=ev.feasible, violations=ev.viol_count,
               level_sum=ev.level_sum,
               note="Phase 2: only level_opt from here; each one adds one node")


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def _list_level_opts(ep: TwoPhaseEpisode) -> dict:
    if ep.phase != 2:
        return _err(ep, "list_level_opts is a Phase 2 tool; bridge first")
    ns = ep.completed_nodeset()
    acts = ST.enumerate_level_opt_actions(sorted(ns), ep.inst.N)
    targets = [a["target_node"][0] for a in acts]
    wave = _mis_wave(targets, _pi_map(ns, ep.inst.N))
    return _ok(ep, targets=[list(a["target_node"]) for a in acts], n=len(acts),
               max_wave=[[i, 0] for i in wave],
               max_wave_note="a LARGEST batch that level_opt_wave will accept "
                             "as one atomic move (pairwise independent in the "
                             "predecessor tree). Whether it is a GOOD batch "
                             "is a different question: single-move previews "
                             "do NOT add up (each move re-prices the others), "
                             "so a wave's only honest price is the state "
                             "after taking it -- archive_put first if unsure.")


def _preview_level_opts(ep: TwoPhaseEpisode, targets=None) -> dict:
    """What every legal level-opt would do, without doing any of them.

    Charges one evaluator pass per candidate and ZERO mutations.  The rule:
    the quote's evaluator work is charged honestly, but the states it visits
    neither ratchet nor count as mutations, or the tool would auto-play the
    game.

    This environment needs that rule badly.  S = (N-1) + g + k
    makes k the mutation count, so before this tool existed, looking at a move
    and making it cost exactly the same thing -- there was no way to think
    ahead that was cheaper than guessing, and the archive's fork/put machinery
    could not pay for itself.  An early probe episode played one straight
    greedy line and then said it wanted to go back and try a different
    backbone; it had no affordance that would have let it compare before
    committing.

    `targets` narrows the probe when only part of the frontier is interesting,
    so the eval cost is the player's to choose rather than fixed at N-1.

    The returned rows are ranked against the state the player is standing in
    RIGHT NOW, and applying any one of them re-ranks the others.  The note
    says so, because the obvious composition -- quote, then hand the top
    twenty-five straight to level_opt_run -- is a trap that this environment
    watched a model walk into: k went 0 -> 25 -> 47 -> 76 while the graded
    score never moved off the value it had after the third move.
    """
    if ep.phase != 2:
        return _err(ep, "preview_level_opts is a Phase 2 tool; bridge first")
    ns = ep.completed_nodeset()
    full = sorted(ns)
    acts = ST.enumerate_level_opt_actions(full, ep.inst.N)
    want = None
    if targets is not None:
        try:
            want = {_node(t, "target") for t in targets}
        except (ValueError, TypeError) as e:
            return _err(ep, f"malformed targets: {e}")
    rows, spent = [], 0
    for a in acts:
        t = tuple(a["target_node"])
        if want is not None and t not in want:
            continue
        if ep.evals_left <= 0:
            return _ok(ep, previews=rows, probed=spent, truncated=True,
                       note="evaluator budget ran out mid-preview")
        r = ST.apply_level_opt(full, ep.inst.N, a)
        if not r.get("legal") or not r.get("next_full_nodelist"):
            continue
        cand = frozenset(tuple(x) for x in r["next_full_nodelist"])
        ev = ep.probe(cand)
        spent += 1
        rows.append({"target": list(t), "L_after": ev.L,
                     "level_sum_after": ev.level_sum,
                     "violations_after": ev.viol_count,
                     "viol_sum_after": ev.viol_sum,
                     "feasible_after": ev.feasible,
                     # the realised count, which is exactly what the
                     # player minimises against
                     "S_after": ep.S_now + 1})
    # gap order, same key as the Phase 1 quote and the pushed menu
    rows.sort(key=lambda d: (d["viol_sum_after"], d["violations_after"],
                             d["L_after"] if d["L_after"] is not None
                             else 10 ** 9, d["level_sum_after"]))
    shown, omitted = _cap_rows(rows)
    return _ok(ep, previews=shown, probed=spent, truncated=False,
               rows_omitted=omitted, rows_total=len(rows),
               note="nothing was applied; a preview never scores. Each of "
                    "these costs one move if you take it. They are ranked "
                    "against the CURRENT graph -- taking any one of them "
                    "changes what the others would do, so this ranking is "
                    "only valid for the next single move."
                    + (f" Showing the best {len(shown)} of {len(rows)}; "
                       f"{omitted} more were probed and are not listed -- "
                       f"narrow with `targets` to see specific ones."
                       if omitted else ""))


def _level_opt(ep: TwoPhaseEpisode, target) -> dict:
    if ep.phase != 2:
        return _err(ep, "level_opt is a Phase 2 tool; bridge first")
    assert ep.nodeset is not None
    t = _node(target, "target")
    if t[1] != 0:
        return _err(ep, f"target {list(t)} is not an output; level_opt acts on (i, 0)")
    ep.charge_mutations(1)
    res = ST.apply_level_opt(sorted(ep.nodeset), ep.inst.N,
                             {"action_type": "level_opt", "target_node": t})
    if not res.get("legal") or not res.get("next_full_nodelist"):
        return _err(ep, f"level_opt on {list(t)} is not legal here")
    new = frozenset(tuple(x) for x in res["next_full_nodelist"])
    # Set-based certificate, not a length delta: a tampered apply that removes
    # one node and adds two has a net of +1 and would sail through a len()
    # check while silently rewriting the graph (external-review finding).
    added = new - ep.nodeset
    removed = ep.nodeset - new
    if len(added) != 1 or removed:
        return _err(ep, f"level_opt must add exactly one node and remove none, "
                        f"got added={sorted(added)} removed={sorted(removed)}; "
                        "refusing to score an operator that broke its own price")
    ep.nodeset = new
    ep.k += 1
    ep._check_identity()
    try:
        ev = ep.evaluate()
    except BudgetExceeded as e:
        return _err(ep, str(e))
    return _ok(ep, target=list(t), L=ev.L, feasible=ev.feasible,
               violations=ev.viol_count, level_sum=ev.level_sum)


def _level_opt_run(ep: TwoPhaseEpisode, targets) -> dict:
    """Several level-opts in one turn, applied IN ORDER, stopping at the first
    refusal.

    This exists to remove a confound, not to add power.  `regroup_wave` lets
    Phase 1 land ten moves in a single turn, and until now Phase 2 had no
    equivalent, so a turn of regrouping was worth roughly ten turns of
    level-opt.  An early probe episode leaned almost entirely on regroups; on
    instances where a level-opt-only line already does well, that mix
    is at least as easily explained by turn economics as by anything the model
    believed about the structure.  An arm comparison run on top of that
    asymmetry would measure the menu.

    Deliberately NOT a wave, and named so.  `level_opt_wave` is the atomic
    form, and its admission rule is a theorem (a target set that is
    an independent set in the current predecessor tree commutes, any order,
    same final graph) -- but that theorem is exactly why the wave REFUSES
    dependent targets.  This tool is the other half: a dependent chain
    (parent after child, or the same target twice) is sometimes what the
    player wants, and it genuinely is sequential -- the second target's
    meaning depends on the first having run.  Sequential, non-atomic, and it
    reports exactly how far it got.
    """
    if ep.phase != 2:
        return _err(ep, "level_opt_run is a Phase 2 tool; bridge first")
    if not isinstance(targets, (list, tuple)) or not targets:
        return _err(ep, "targets must be a non-empty list of [i, 0] outputs")
    applied: list[list[int]] = []
    steps: list[dict] = []
    for i, raw in enumerate(targets):
        r = _level_opt(ep, raw)
        if not r.get("ok"):
            return _err(ep, f"run stopped at entry {i}: {r.get('error')}; "
                            f"the {len(applied)} move(s) before it stand",
                        applied=applied, steps=steps, stopped_at=i)
        applied.append(r["target"])
        # per-entry, because the final state alone hides the shape of the run.
        # A quote's ranking is only valid for one move, so a long run usually
        # stops helping partway and then actively hurts; returning just the
        # end state makes that invisible and the player keeps batching.
        steps.append({"i": i, "target": r["target"], "L": r["L"],
                      "violations": r["violations"],
                      "level_sum": r["level_sum"]})
    ev = ep.evaluate()
    best = min(range(len(steps)), key=lambda j: (steps[j]["violations"],
                                                 steps[j]["L"], j)) \
        if steps else None
    return _ok(ep, applied=applied, n=len(applied), steps=steps, L=ev.L,
               feasible=ev.feasible, violations=ev.viol_count,
               level_sum=ev.level_sum, best_entry=best,
               note="applied in order; each one cost a node, same as one at a "
                    "time. `steps` is what each move actually did -- a quote "
                    "only ranks the NEXT move, so entries after the first are "
                    "acting on a ranking that is already stale. `best_entry` "
                    "is where in this run the graph was closest to feasible; "
                    "if it is not the last entry, the tail of the run cost "
                    "nodes and bought nothing.")


def _level_opt_wave(ep: TwoPhaseEpisode, targets) -> dict:
    """A set of pairwise-independent level-opts, applied atomically.

    Admission is the wave theorem (verified on the real operator:
    219 independent sets, every linearization, one final graph each; 97
    non-independent sets, every one diverged): the batch commutes if and only
    if no target is another target's predecessor in the CURRENT pi tree.
    Dependent batches are refused whole -- for those, level_opt_run exists
    and says it is sequential.

    Pricing (W3, counterexample on file): per-target one-step quotes do NOT
    add -- applying half a wave can pass through infeasible while the whole
    wave is feasible, and vice versa.  So this tool prices nothing per entry
    and re-evaluates the post-wave graph once; that evaluation is the only
    price statement it makes.

    One tool call, len(targets) mutations, one evaluator pass.
    """
    if ep.phase != 2:
        return _err(ep, "level_opt_wave is a Phase 2 tool; bridge first")
    assert ep.nodeset is not None
    if not isinstance(targets, (list, tuple)) or not targets:
        return _err(ep, "targets must be a non-empty list of [i, 0] outputs")
    try:
        want = [_node(t, "target") for t in targets]
    except (ValueError, TypeError) as e:
        return _err(ep, f"malformed targets: {e}")
    outs = []
    for t in want:
        if t[1] != 0:
            return _err(ep, f"target {list(t)} is not an output; "
                            "level_opt acts on (i, 0)")
        outs.append(t[0])
    if len(set(outs)) != len(outs):
        return _err(ep, "a wave names each target at most once; to hit the "
                        "same output twice use level_opt_run (sequential)")
    pi = _pi_map(ep.nodeset, ep.inst.N)
    illegal = [i for i in outs if pi.get(i, 0) <= 0]
    if illegal:
        return _err(ep, f"level_opt is not legal on {[[i, 0] for i in illegal]}"
                        " right now (no predecessor to shortcut)")
    dep = [(u, v) for u in outs for v in outs
           if u != v and pi[u] == v]
    if dep:
        return _err(ep, "wave is not independent: "
                        + ", ".join(f"({u},0) depends on ({v},0)"
                                    for u, v in dep)
                        + "; a wave must be an independent set in the "
                          "predecessor tree -- sequence those with "
                          "level_opt_run instead, or drop one side",
                    conflicts=[[u, v] for u, v in dep])
    ep.charge_mutations(len(outs))
    ns = ep.nodeset
    for i in outs:
        res = ST.apply_level_opt(sorted(ns), ep.inst.N,
                                 {"action_type": "level_opt",
                                  "target_node": (i, 0)})
        if not res.get("legal") or not res.get("next_full_nodelist"):
            return _err(ep, f"wave rejected: level_opt on [{i}, 0] failed "
                            "inside an admitted batch; no part was applied")
        new = frozenset(tuple(x) for x in res["next_full_nodelist"])
        added, removed = new - ns, ns - new
        if len(added) != 1 or removed:
            return _err(ep, f"level_opt on [{i}, 0] must add exactly one node "
                            f"and remove none, got added={sorted(added)} "
                            f"removed={sorted(removed)}; wave discarded")
        ns = new
    ep.nodeset = ns
    ep.k += len(outs)
    ep._check_identity()
    try:
        ev = ep.evaluate()
    except BudgetExceeded as e:
        return _err(ep, str(e))
    return _ok(ep, applied=[[i, 0] for i in sorted(outs)], n=len(outs),
               L=ev.L, feasible=ev.feasible, violations=ev.viol_count,
               viol_sum=ev.viol_sum, level_sum=ev.level_sum,
               note="atomic: all applied as one move batch, then the whole "
                    "graph was re-evaluated once. Per-target quotes from "
                    "before the wave no longer describe anything.")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def _submit(ep: TwoPhaseEpisode) -> dict:
    """Hand in the state you are standing on.  One-shot, terminal.

    Total on purpose: submitting is the one move that must never bounce.
    Grading re-uses the cached evaluation when the player is standing on an
    evaluated state (the common case -- so submit is free); an unevaluated
    state is graded with a player-budget pass if one remains, and on the
    harness ledger if not, because the game ending must not depend on having
    saved a pass to be scored with.  A feasible hand-in still ratchets in
    that last case -- it is a state the episode stood in, so all three
    ratchet gates hold and the best_S <= submitted_S invariant stays a fact
    rather than a hope.

    No guard on infeasible submits, by design: the briefing says what a
    non-feasible hand-in scores, and stopping the player from doing it would
    be the harness playing the game.
    """
    ns = ep.completed_nodeset()
    m = ep.meter
    if m._cache_key == ns and m._cache_val is not None:
        ev = m._cache_val
    else:
        try:
            ep.check_evals()
            ev = ep.evaluate()
        except BudgetExceeded:
            from prefixagent.runtime.episode import evaluate_nodeset
            ep.harness_evals += 1
            ev = evaluate_nodeset(ep.inst, ns)
            if ev.feasible and (m.best_S is None or ev.S < m.best_S):
                m.best_S, m.best_at_eval = ev.S, m.evals
                m.best_state = ns
                # keep the auto-archive promise on this path too: a new best is
                # auto-archived wherever it is set (external audit)
                snap = ep.snapshot(f"auto: best feasible S={ev.S}")
                ep.best_snapshot_sid = snap.sid
    ep.submitted_eval = ev
    ep.finish("submitted")
    return _ok(ep, submitted_S=(ev.S if ev.feasible else None),
               submitted_feasible=ev.feasible, submitted_L=ev.L,
               note="submitted: the episode is over. A feasible hand-in "
                    "scores its S; an infeasible one scores nothing.")


# ---------------------------------------------------------------------------
# any phase
# ---------------------------------------------------------------------------


def _get_state(ep: TwoPhaseEpisode) -> dict:
    return _ok(ep)


def _archive_put(ep: TwoPhaseEpisode, note: str = "") -> dict:
    s = ep.snapshot(str(note)[:120])
    return _ok(ep, snapshot=s.sid, phase=s.phase, g=s.g, k=s.k, S=s.S)


def _archive_list(ep: TwoPhaseEpisode) -> dict:
    return _ok(ep, snapshots=[
        {"id": s.sid, "phase": s.phase, "g": s.g, "k": s.k, "S": s.S,
         "L": s.L, "note": s.note}
        for s in ep.archive
    ])


def _fork(ep: TwoPhaseEpisode, snapshot: int) -> dict:
    """Rewind to an archived state -- including from Phase 2 back to Phase 1.

    Moves already spent are NOT refunded: g and k reset to the snapshot's
    values but the mutation budget does not, so exploring costs what it cost.
    """
    try:
        s = ep.restore(int(snapshot))
    except ValueError as e:
        return _err(ep, str(e))
    return _ok(ep, forked_to=s.sid, phase=s.phase, g=s.g, k=s.k,
               note="moves already spent are not refunded")


# ---------------------------------------------------------------------------
# registry + dispatch
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable[..., dict]] = {
    "get_state": _get_state,
    "list_regroups": _list_regroups,
    "regroup": _regroup,
    "regroup_wave": _regroup_wave,
    "preview_regroups": _preview_regroups,
    "preview": _preview,
    "bridge": _bridge,
    "list_level_opts": _list_level_opts,
    "preview_level_opts": _preview_level_opts,
    "level_opt": _level_opt,
    "level_opt_run": _level_opt_run,
    "level_opt_wave": _level_opt_wave,
    "submit": _submit,
    "archive_put": _archive_put,
    "archive_list": _archive_list,
    "fork": _fork,
}


def dispatch(ep: TwoPhaseEpisode, name: str, args: dict | None = None) -> dict:
    """Charge a turn, run the tool, record it.  Never raises on a bad move."""
    args = dict(args or {})
    if ep.done:
        return {"ok": False, "error": "EPISODE_DONE", **ep.state()}
    if ep.config.tools and name not in ep.config.tools:
        # hard wall: an un-advertised name must not execute, or tool-set
        # ablations quietly collapse onto each other
        return {"ok": False,
                "error": f"UNKNOWN_TOOL '{name}'; available: "
                         f"{', '.join(ep.config.tools)}",
                **ep.state()}
    fn = REGISTRY.get(name)
    if fn is None:
        return {"ok": False, "error": f"UNKNOWN_TOOL '{name}'", **ep.state()}
    try:
        ep.charge_tool_call()
    except BudgetExceeded as e:
        ep.finish("tool_calls_exhausted")
        return {"ok": False, "error": str(e), **ep.state()}

    try:
        out = fn(ep, **args)
    except BudgetExceeded as e:
        out = _err(ep, str(e))
    except TypeError as e:
        out = _err(ep, f"bad arguments for {name}: {e}")
    except (ValueError, KeyError, AssertionError) as e:
        out = _err(ep, f"{type(e).__name__}: {e}")

    ev = ep.current_eval()
    ep.records.append({
        "turn": ep.tool_calls, "tool": name, "args": args,
        "ok": bool(out.get("ok")), "error": out.get("error"),
        "phase": ep.phase, "g": ep.g, "k": ep.k,
        "S": ep.S_now,
        "L": ev.L if ev is not None else None,
        "viol_count": ev.viol_count if ev is not None else None,
        "viol_sum": ev.viol_sum if ev is not None else None,
        "best_S": ep.meter.best_S, "evals": ep.meter.evals,
        "mutations": ep.meter.mutations,
        "harness_evals": ep.harness_evals,
    })
    if not ep.done:
        stop = ep._budget_stop()
        if stop:
            ep.finish(stop)
            out["done"], out["stop_reason"] = True, stop
    return out
