"""What the player sees.

The two-phase game: Phase 1 and Phase 2 are named, the bridge is an
explicit move, and the view says which phase it is in.  The briefing is
generated from the environment's own facts -- objective, arrival profile,
operator semantics, price list -- so what the player is told and what the
engine enforces cannot drift apart.

The price list is stated up front, because it is not a hint -- it is the
rule of the game, and a player who does not know that a regroup costs a
node is not playing the game we are scoring.
"""
from __future__ import annotations

from prefixagent.runtime.episode import TwoPhaseEpisode

# The constraint has to read as a GATE, not as a second thing to be good at.
# The earlier wording ("...at most L_max, using as few nodes as possible.
# Score = the final node count S") let a lower S look like a better answer
# whether or not the graph was feasible, and an early research episode spent its last
# turn forking to a cheaper infeasible state.  It happened to be that
# episode's best state anyway, so nothing was lost -- but nothing in the text
# ruled the move out either.
_OBJECTIVE = (
    "Goal: build a legal prefix adder for {N}-bit inputs whose every output "
    "(i,0) has arrival level at most L_max = {L_max}, using as FEW non-input "
    "nodes as possible.\n"
    "  L <= {L_max} is a HARD GATE, not a preference. A graph with any output "
    "over {L_max} scores NOTHING, however few nodes it has -- an infeasible "
    "S=40 loses to a feasible S=80.\n"
    "  What you HAND IN is what is scored: `submit` ends the game and hands "
    "in the state you are standing on, and only a submitted, feasible graph "
    "scores at all -- its S, lower better. If the game ends any other way "
    "(a budget runs out), you score NOTHING; unspent budget is simply "
    "discarded, never rewarded.\n"
    "  Avoid over-designing the adder: once you hold a feasible graph, every "
    "further node you add is pure loss unless it buys a smaller feasible "
    "graph you actually hand in."
)

_ARRIVAL = (
    "Input arrival levels (index 0 = LSB): {arrival}\n"
    "A node's level is max(level of its two parents) + 1; an input (i,i) "
    "starts at its arrival level."
)

_PRICE = (
    "Price list (exact, not an estimate):\n"
    "  - You start from the serial backbone, which completes to S = {S0} nodes.\n"
    "  - Each `regroup` shortens the backbone's spine by exactly 1, which "
    "makes the completed adder exactly 1 node LARGER.\n"
    "  - Each `level_opt` adds exactly 1 node.\n"
    "  So S = {S0} + (regroups) + (level-opts).  Every move costs exactly one "
    "node, whichever kind it is; the only question is which move buys more "
    "depth for that node."
)

# --- the RULES, not strategy -----------------------------------------------
#
# These blocks state what the operators do and when they are refused.  They
# carry no advice about which move to pick -- that line is deliberate: a
# player who does not know that `regroup` removes (b.msb, 0) is not playing
# the game we are scoring, which is the same argument that already puts the
# price list up front.
#
# They live in the STATIC briefing because the system prompt is a cached
# prefix: its cost is paid once and amortised over the whole episode, while
# the per-turn view is re-rendered and never cached.  The measured cache hit
# rate on an early wave was 19%, so the harness was economising on the one
# channel where economising buys nothing, and making the model re-derive
# these rules every turn instead -- one episode burned three consecutive
# turns at max_tokens doing exactly that kind of re-derivation.

_PARENT_RULE = (
    "How parents are determined (this is the graph's definition, and it is "
    "what makes adding a node able to HURT):\n"
    "  A node (m, x) takes as its upper parent the node (m, k) with the "
    "SMALLEST present k > x, and as its lower parent (k-1, x).\n"
    "  So inserting a node at (m, k') with x < k' < k re-points (m, x) onto "
    "the new, nearer parent. A move that lowers one output's level can raise "
    "another's."
)

_OP_REGROUP = (
    "`regroup(a, b)` -- exact effect. The two named nodes are assigned to the "
    "roles b and c so that b.msb = c.lsb - 1 (give them in either order). It "
    "then REMOVES node (b.msb, 0) and ADDS node (c.msb, b.lsb).\n"
    "  Refused unless all of:\n"
    "    1. both named nodes exist in the backbone\n"
    "    2. b.msb == c.lsb - 1 holds for one of the two assignments\n"
    "    3. both (b.msb, 0) and (c.msb, 0) exist\n"
    "    4. each of b and c is the upper parent of its own output node --\n"
    "       (b.msb, 0) is a fanout of b, and (c.msb, 0) is a fanout of c\n"
    "    5. (c.msb, b.lsb) does not already exist\n"
    "  The spine is the set of nodes with lsb = 0. Condition 4 forces "
    "b.lsb >= 1, so the node removed is on the spine and the node added is "
    "not: every legal regroup shortens the spine by exactly 1. The spine "
    "cannot fall below 1 (the root ({rootmsb}, 0) is always present), so at "
    "most {gmax} regroups exist in total and S can never exceed {smax}."
)

_OP_LEVEL = (
    "`level_opt(target)` -- exact effect. Let p be the target, u its upper "
    "parent and l its lower parent. It creates s = (p.msb, k) where k is the "
    "lsb of l's upper parent, gives s the parents u and l's upper parent, and "
    "re-points p onto s and l's lower parent. That is the associativity "
    "rewrite: p's cone is re-bracketed so that one level comes out of it, at "
    "the cost of the one new node.\n"
    "  Refused unless p has both an upper and a lower parent, l itself has "
    "both parents, and s does not already exist."
)

_PHASE1 = (
    "PHASE 1 of 2 - backbone.  You are editing the backbone: the {n1} nodes "
    "that compute the top carry.  `regroup` is the only structural move here. "
    "When you are done, `bridge` completes the backbone into a full adder and "
    "moves you to Phase 2.  The bridge is one-way within a branch."
)

_PHASE2 = (
    "PHASE 2 of 2 - full adder.  The backbone is committed.  `level_opt` is "
    "the only structural move left; it targets an output (i,0) and restructures "
    "its cone, adding one node."
)


# --- the structural view ---------------------------------------------------
#
# Two things this renderer must not do, both learned from the harnesses that
# came before.
#
# It must not silently degrade.  An earlier harness called
# to_s_expression() without arrival_times, whose documented behaviour is "all
# arrival times will be set to 0.0", and shipped a view where every node read
# [arrival=0.0000].  It LOOKED like annotated structure and was bare
# structure, while the score -- computed from the real non-uniform
# profile -- saw everything the view hid.  So the renderer
# asserts its own output is non-degenerate.
#
# It must not hide its cost.  Structure is the expensive channel: it is
# re-rendered every turn and never cached, and the N=32 S-expression is about
# 2,100 tokens against a 60-token aggregate view.  A loss under a structure
# view has two live explanations -- structure does not help, context growth
# hurts -- and they cannot be separated after the fact, so every render
# reports its own size and the driver records it.

def _assert_annotations_vary(text: str) -> None:
    """An annotated view whose annotations are all equal is not annotated.

    An earlier harness rendered every node as [arrival=0.0000] for a whole
    line of experiments, because its renderer defaulted every arrival to
    0.0 when none is passed and nobody passed any.  It looked annotated
    and was bare, while the score -- computed from the real
    non-uniform profile -- saw everything the view had flattened.  A
    degenerate annotation is worse than none: it costs tokens and teaches the
    model that the profile is uniform.
    """
    import re
    vals = re.findall(r"(?:arrival=|@)(-?\d+)", text)
    if len(vals) > 2 and len(set(vals)) == 1:
        raise AssertionError(
            f"annotate='level' produced {len(vals)} identical annotations "
            f"(all {vals[0]}) -- the view is degenerate, not annotated")


def structure_block(ep) -> str:
    """Phase 1: the annotated S-expression of the backbone.

    Phase 2 renders nothing here: the full adder is a DAG, which has no
    S-expression, and the EPR block already carries the Phase 2 structure --
    rendering both would double-bill the same information.
    """
    if ep.phase == 2:
        return ""
    from prefixagent.core.sexpr import backbone_to_sexpression
    # indent=0, not the library default of 4.  Indentation is for human
    # eyes and this reader has none: the nesting is already given twice
    # over, once by the parentheses and again by the coordinates
    # themselves -- (4,0) says it spans bits 0..4, so depth is derivable
    # without any layout at all.  indent=0 keeps every newline (still one
    # node per line); at N=32 it is 7466 chars -> 1638, 2133 tokens -> 468.
    body = backbone_to_sexpression(sorted(ep.backbone), ep.inst.N,
                                   list(ep.inst.arrival), indent=0)
    _assert_annotations_vary(body)
    return "Backbone structure (S-expression, annotated with level):\n" + body


def repr_view(ep: TwoPhaseEpisode) -> str:
    """The p2_view representation block.

    Phase 2: the block describes the graph the player is standing in.
    Phase 1 with C2 on: the SAME block, rendered on the adder the current
    backbone WOULD complete to, labelled as such -- cross-phase visibility
    is the same code path, so the two views cannot drift apart.  All arrival
    computation is one harness-ledger pass per turn.
    """
    kind = getattr(ep.config, "p2_view", "none")
    if kind == "none":
        return ""
    from prefixagent.runtime.repr_blocks import render_block

    inst = ep.inst
    if ep.phase == 2:
        ns = ep.completed_nodeset()
        A = ep.harness_arrivals(ns)
        return render_block(kind, ns, inst.N, A, inst.L_max)
    if not getattr(ep.config, "c2", False):
        return ""
    ns = ep.completed_nodeset()
    A = ep.harness_arrivals(ns)
    return ("The full adder your current backbone would complete to, right "
            "now (nothing is committed by looking):\n"
            + render_block(kind, ns, inst.N, A, inst.L_max))


def briefing(ep: TwoPhaseEpisode) -> str:
    """The static part of the prompt: rules, objective, price list."""
    inst, cfg = ep.inst, ep.config
    parts = [_OBJECTIVE.format(N=inst.N, L_max=inst.L_max),
             _ARRIVAL.format(arrival=list(inst.arrival)),
             _PARENT_RULE]
    parts.append(_PRICE.format(S0=inst.N - 1))
    parts.append(_OP_REGROUP.format(rootmsb=inst.N - 1, gmax=inst.N - 2,
                                    smax=2 * (inst.N - 1) - 1))
    parts.append(_OP_LEVEL)
    parts.append(_PHASE1.format(n1=inst.N - 1))
    moves = ("unlimited moves (each is counted, none is refused)"
             if cfg.max_mutations >= 10 ** 8 else f"{cfg.max_mutations} moves")
    parts.append(
        f"Budget: {cfg.max_tool_calls} tool calls, {cfg.max_evals} evaluator "
        f"passes, {moves}.  One tool call per turn."
    )
    return "\n\n".join(parts)


def state_block(ep: TwoPhaseEpisode) -> str:
    """The per-turn view.  Conclusions with numbers, never a raw node dump."""
    st = ep.state()
    lines = []
    lines.append(_PHASE1.format(n1=ep.inst.N - 1) if ep.phase == 1
                 else _PHASE2)
    lines.append("")
    lines.append(f"moves so far: {ep.g} regroup, {ep.k} level-opt")
    lines.append(f"current S: {st['S']}")
    if st["L"] is not None:
        margin = ep.inst.L_max - st["L"]
        lines.append(
            f"current L: {st['L']} (L_max {ep.inst.L_max}, "
            + ("margin " + str(margin) if margin >= 0 else
               f"OVER by {-margin}") + ")"
        )
        # WHICH outputs are over, not just how many.  The count told the
        # player that 25 of 32 outputs miss the gate and nothing about where
        # to aim; StateEval carries the identities, but for a while they
        # only reached the tool results, not the view the player reads every
        # turn.  Listed as (output, its level) for the worst offenders.
        vio = st.get("violating_outputs") or []
        if vio:
            worst = sorted(vio, key=lambda v: (-v[2], v[0]))[:12]
            more = len(vio) - len(worst)
            # V1/V5 attribution (zero-eval, pure rendering): each violating
            # output is tagged by WHO put it there -- [bb] it is a backbone
            # node (on the spine the player shaped), [cmpl] it was added by
            # completion.  Attribution only, never a verdict on the backbone:
            # minimising the backbone's own L is a proven-invalid objective,
            # so the line must not read as "backbone passed/failed".
            spine = {m for (m, l) in ep.backbone if l == 0}

            def _tag(v):
                return f"({v[0]},0)@{v[1]}[{'bb' if v[0] in spine else 'cmpl'}]"

            lines.append(
                f"  over the gate [{len(vio)}], total excess "
                f"{st.get('viol_sum')}: "
                + " ".join(_tag(v) for v in worst)
                + (f"  +{more} more" if more else "")
            )
            nb = sum(1 for v in vio if v[0] in spine)
            lines.append(
                f"  attribution: {nb} violating output(s) are backbone "
                f"spine nodes, {len(vio) - nb} were added by completion")
    else:
        lines.append("current L: not evaluated yet")
    if ep.phase == 1:
        lines.append(
            f"spine length: {ep.ridge_len()}  ->  completing now would give "
            f"S = {2 * (ep.inst.N - 1) - ep.ridge_len()}"
        )
    # the gauge: best-so-far with a pointer at its auto-archived
    # snapshot, so "go back to the best thing I ever held" is one fork away.
    # An instrument, not a score -- the briefing already said only what is
    # handed in counts.
    if st["best_S"] is None:
        lines.append("best feasible so far: none yet -- no state has met L_max")
    else:
        ptr = (f" (snapshot {st['best_snapshot']})"
               if st.get("best_snapshot") is not None else "")
        lines.append(f"best feasible so far: S={st['best_S']}{ptr}")
    if ep.archive:
        lines.append(f"archive: {len(ep.archive)} snapshot(s)")
    b = st["budget"]
    moves = ("unlimited (counted)" if ep.config.max_mutations >= 10 ** 8
             else f"{b['mutations_left']}")
    lines.append(f"budget left: {b['tool_calls_left']} tool calls, "
                 f"{b['evals_left']} evaluator passes, {moves} moves")
    return "\n".join(lines)


# --- the pushed menus ------------------------------------------------------
#
# The paper's agent pushed its two tables (legal regroups with level
# reductions, legal level-opts) at the model every turn; a later harness made
# the model ask and pay.  This one puts the tables back: full enumerator, per-action
# 1-ply effect, ranked by the gap order (viol_sum, viol_count, L) -- distance
# to feasibility first -- capped at 16 rows with the omission REPORTED.
# Every evaluator pass here lands on the harness_evals side ledger, never the
# player's budget: the view must cost every arm the same, which is nothing.

MENU_ROW_CAP = 16


def _menu_rows_phase1(ep) -> list[dict]:
    from prefixagent.core import graph_ops
    from prefixagent.core import structural as ST
    from prefixagent.runtime.tools import _apply_one_regroup

    N = ep.inst.N
    rows = []
    for a in ST.enumerate_regroup_actions(list(ep.backbone), N):
        n1, n2 = tuple(a["node1"]), tuple(a["node2"])
        nb, err = _apply_one_regroup(ep.backbone, N, n1, n2)
        if err:
            continue
        ev = ep.harness_state_eval(graph_ops.complete(list(nb), N))
        rows.append({"call": f"regroup({list(n1)}, {list(n2)})",
                     "S": ev.S, "L": ev.L, "vc": ev.viol_count,
                     "vs": ev.viol_sum, "ls": ev.level_sum,
                     "feas": ev.feasible})
    return rows


def _menu_rows_phase2(ep) -> list[dict]:
    from prefixagent.core import structural as ST

    N = ep.inst.N
    ns = ep.completed_nodeset()
    full = sorted(ns)
    rows = []
    for a in ST.enumerate_level_opt_actions(full, N):
        t = tuple(a["target_node"])
        r = ST.apply_level_opt(full, N, a)
        if not r.get("legal") or not r.get("next_full_nodelist"):
            continue
        ev = ep.harness_state_eval(frozenset(tuple(x)
                                             for x in r["next_full_nodelist"]))
        rows.append({"call": f"level_opt({list(t)})",
                     "S": ev.S, "L": ev.L, "vc": ev.viol_count,
                     "vs": ev.viol_sum, "ls": ev.level_sum,
                     "feas": ev.feasible})
    return rows


def _render_menu(title: str, rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda d: (d["vs"], d["vc"],
                                       d["L"] if d["L"] is not None else 10 ** 9,
                                       d["ls"], d["call"]))
    shown, omitted = rows[:MENU_ROW_CAP], max(0, len(rows) - MENU_ROW_CAP)
    out = [title]
    for d in shown:
        out.append(f"  {d['call']} -> S={d['S']} L={d['L']} "
                   f"viol={d['vc']}(excess {d['vs']})"
                   + ("  FEASIBLE" if d["feas"] else ""))
    if omitted:
        out.append(f"  ... {omitted} more legal moves not shown (ranked "
                   "worse); narrow with the preview tools to see specific ones")
    if not rows:
        out.append("  (none)")
    return out


def menu_block(ep: TwoPhaseEpisode) -> str:
    """The per-turn pushed menu.  Rankings are 1-ply truths about the NEXT
    single move only -- taking any move re-prices all the others -- and the
    header says so once per turn."""
    if not getattr(ep.config, "menus", False):
        return ""
    parts: list[str] = []
    head = ("Legal moves this turn, each with its exact 1-ply effect "
            "(harness-computed, costs you nothing; ranked by distance to "
            "feasibility = (total excess, violating outputs, L). Valid for "
            "the next single move only -- any move re-prices the rest):")
    body: list[str] = []
    if ep.phase == 1:
        body += _render_menu("regroups (effect on the COMPLETED adder):",
                             _menu_rows_phase1(ep))
    else:
        body += _render_menu("level_opts:", _menu_rows_phase2(ep))
        from prefixagent.runtime.tools import _mis_wave, _pi_map
        pi = _pi_map(ep.completed_nodeset(), ep.inst.N)
        legal = sorted(i for i, p in pi.items() if p > 0)
        if legal:
            wave = _mis_wave(legal, pi)
            body.append(f"  largest independent batch for level_opt_wave: "
                        f"{[[i, 0] for i in wave]}")
    parts.append(head)
    parts += body
    return "\n".join(parts)


def tool_lines(ep: TwoPhaseEpisode) -> list[str]:
    """One line per advertised tool, phase-filtered so the menu never offers a
    move that would be refused."""
    from prefixagent.runtime import tools as T

    desc = {
        "get_state": "get_state() - re-read this view",
        "list_regroups": "list_regroups() - every legal regroup right now",
        "regroup": "regroup(a=[msb,lsb], b=[msb,lsb]) - one regroup; +1 node",
        "regroup_wave": ("regroup_wave(pairs=[[a,b],...]) - several regroups in "
                         "ONE turn; they must name four pairwise-distinct nodes"),
        "preview": "preview() - L of the completed adder, without committing",
        "preview_regroups": ("preview_regroups(pairs=None) - what EACH regroup "
                             "would do to the completed adder's L, without "
                             "doing any; costs one evaluator pass per "
                             "candidate and NO moves. Ranked against the "
                             "CURRENT backbone, so the ranking is only valid "
                             "for the next single move"),
        "bridge": "bridge() - complete the backbone and enter Phase 2 (one-way)",
        "list_level_opts": "list_level_opts() - every legal level-opt target",
        "preview_level_opts": ("preview_level_opts(targets=None) - what EACH "
                               "level-opt would do to L, without doing any; "
                               "costs one evaluator pass per candidate and NO moves"),
        "level_opt": "level_opt(target=[i,0]) - restructure that output; +1 node",
        "level_opt_run": ("level_opt_run(targets=[[i,0],...]) - several of them "
                          "in ONE turn, applied in order, stopping at the "
                          "first refusal; same price per move as one at a "
                          "time. Each move changes what the later ones do, so "
                          "a long list acts on a ranking that is already "
                          "stale; it returns what each move actually did"),
        "level_opt_wave": ("level_opt_wave(targets=[[i,0],...]) - a SET of "
                           "level_opts applied atomically in ONE turn; "
                           "accepted only if no target chains directly from "
                           "another (independent in the predecessor tree -- "
                           "the menu's 'largest independent batch' line "
                           "qualifies). Same price per move; the result is "
                           "re-evaluated once, after the whole wave"),
        "submit": ("submit() - stop and hand in the current graph. THIS is "
                   "what gets scored; one-shot, ends the game"),
        "archive_put": "archive_put(note='') - snapshot this state",
        "archive_list": "archive_list() - list snapshots",
        "fork": "fork(snapshot=id) - rewind to a snapshot (moves are not refunded)",
    }
    out = []
    for name in (ep.config.tools or T.ALL_TOOLS):
        d = desc.get(name)
        if d is None:
            continue
        # The menu must never offer a move that would be refused: in Phase 1
        # the whole point is that Phase 2 does not exist yet, and listing
        # level_opt there both wastes turns on guaranteed refusals and blurs
        # the phase structure the game is built on.
        if ep.phase == 2 and name in T.PHASE1_TOOLS:
            continue
        if ep.phase == 1 and name != "submit" and name in T.PHASE2_TOOLS:
            continue
        out.append("  " + d)
    return out


def render_turn(ep: TwoPhaseEpisode) -> str:
    parts = [state_block(ep)]
    sb = structure_block(ep)
    if sb:
        parts.append(sb)
    rv = repr_view(ep)
    if rv:
        parts.append(rv)
    mb = menu_block(ep)
    if mb:
        parts.append(mb)
    parts.append("Available moves:\n" + "\n".join(tool_lines(ep)))
    return "\n\n".join(parts)
