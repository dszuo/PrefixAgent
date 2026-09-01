"""Metered two-phase episode: Phase 1 regroup, bridge, Phase 2 level-opt.

State machine
-------------
Phase 1 holds a BACKBONE (the N-1 non-input nodes of a binary tree over N
leaves).  `regroup` rewrites it; the ridge shortens by exactly one each time.
`bridge` completes the backbone into a full adder and moves to Phase 2, which
holds a full NODESET and only grows it, one node per level-opt.

Scoring
-------
S = (N-1) + g + k, asserted against the realised node count on every
transition rather than trusted: every move is purely additive, so a bug in
either operator surfaces as an assertion rather than as a plausible score.
The ratchet keeps the minimum feasible S over every state evaluated, in
every branch.

Budgets
-------
Three (the CLI's defaults are the shipped game spec: 200 turns,
200,000 evaluator passes, unlimited-but-counted moves):
  max_tool_calls   turns
  max_evals        node_arrivals passes
  max_mutations    g + k, the moves themselves -- which is also the
                   amount by which S can rise

Archive
-------
`archive_put` / `fork` are how branching, backtracking and evolution happen:
a snapshot records the whole phase state, so a fork can rewind Phase 2 back
into Phase 1 and take a different backbone.  The interface between phases
is an archive, not a single structure, and a reference search measured why
it has to be: ranking backbones by L_completed and keeping the top-1 lost
cells that keeping the top-16 solved.

Failure surface
---------------
The reference `Backbone.regroup` reported illegality by printing and
returning False.  Every wrapper here converts refusal into an explicit error
dict, because a swallowed rejection is indistinguishable from a move that did
nothing -- the failure signature this project has been bitten by repeatedly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from prefixagent.core import graph_ops
from prefixagent.core import fast_eval as FE

Node = tuple[int, int]


# ---------------------------------------------------------------------------
# instance + config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoPhaseInstance:
    """One cell: width, arrival profile (index 0 = LSB), and the level cap."""

    cell_id: str
    N: int
    arrival: tuple[int, ...]
    L_max: int

    def __post_init__(self) -> None:
        if len(self.arrival) != self.N:
            raise ValueError(
                f"arrival has {len(self.arrival)} entries for N={self.N}"
            )


@dataclass
class TwoPhaseConfig:
    seed: int = 0
    max_tool_calls: int = 512
    max_evals: int = 200_000
    max_mutations: int = 4_096
    #: the advertised tools; a tool outside the set is refused, never run
    tools: tuple[str, ...] = ()
    # NOTE: hint/memory fields once declared here were removed: nothing
    # read them, and a field that looks configurable and does nothing is a
    # loaded gun -- set it, see no effect, draw a wrong conclusion.  If they
    # come back they come back wired to a renderer, not as decoration.
    # --- representation ----------------------------------------------------
    #
    #: the per-node representation Phase 2 pushes every turn: "epr" (the
    #: paper's full block, arrival-aware) | "none".
    p2_view: str = "epr"
    #: C2 cross-phase visibility: Phase 1 also sees the p2_view block of the
    #: adder its backbone WOULD complete to.
    c2: bool = True
    #: push the legal-move menus (full enumerator + 1-ply effects) into
    #: the view every turn.  Menu evals go to the harness_evals side ledger,
    #: never the player's budget.
    menus: bool = True


    def __post_init__(self) -> None:
        if self.p2_view not in ("epr", "none"):
            raise ValueError(f"p2_view must be epr|none, got {self.p2_view!r}")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateEval:
    """One node_arrivals pass over a full nodeset.

    `level_sum` is the sum over all outputs, not just the violating ones, and
    it is reported because L alone cannot order states.  L is a small integer,
    so a frontier is mostly ties on it and ranking by L alone amounts to
    picking arbitrarily among equals: a wide beam ordered on L reached 4
    of 16 cells here, while a reference search that reaches 11 orders on
    (L, level_sum).  Whether the tie-break alone closes that gap is measured,
    not assumed -- but withholding the signal the reference uses would leave a
    player unable to distinguish states the reference can distinguish, which
    makes the comparison meaningless rather than merely harder.
    """

    S: int
    legal: bool
    L: int | None
    viol_count: int
    viol_sum: int
    level_sum: int
    #: (output index, its level, how far over L_max) for every violating
    #: output.  A count says 25 of 32 are over the gate and nothing about
    #: where to aim; Phase 2's quote already reports per-candidate numbers,
    #: so aggregate-only was an asymmetry inside a single environment.
    violators: tuple[tuple[int, int, int], ...] = ()

    @property
    def feasible(self) -> bool:
        return self.legal and self.L is not None and self.viol_count == 0

    @property
    def gap(self) -> tuple[int, int, int, int] | None:
        """(viol_sum, viol_count, L, S) -- the graded distance to feasibility."""
        if not self.legal or self.L is None:
            return None
        return (self.viol_sum, self.viol_count, self.L, self.S)


def evaluate_nodeset(inst: TwoPhaseInstance, ns: frozenset) -> StateEval:
    """The one evaluation function: arrival-aware levels over a full nodeset.

    Pure -- no counters, no ratchet, no cache.  The Meter charges and
    ratchets around it for player-facing evaluation; the harness-side menu
    path calls it through TwoPhaseEpisode.harness_state_eval, which
    counts into the harness_evals side ledger instead of the player budget.
    One implementation, two ledgers, so the two paths cannot drift.
    """
    A = FE.node_arrivals(ns, inst.N, list(inst.arrival))
    legal = True
    L: int | None = None
    viol_count = viol_sum = level_sum = 0
    for i in range(inst.N):
        out = (i, 0)
        if out not in ns or out not in A:
            legal = False
            continue
        a = A[out]
        L = a if L is None else max(L, a)
        level_sum += a
        if a > inst.L_max:
            viol_count += 1
            viol_sum += a - inst.L_max
    S = sum(1 for m, l in ns if m != l)
    return StateEval(S, legal, L, viol_count, viol_sum, level_sum,
                     tuple((i, A[(i, 0)], A[(i, 0)] - inst.L_max)
                           for i in range(inst.N)
                           if (i, 0) in A and A[(i, 0)] > inst.L_max)
                     if legal else ())


class Meter:
    """Counts evaluator passes and mutations, and ratchets the best feasible S.

    The ratchet is gated on `feasible`, never on "we got a number back": an
    infeasible waypoint must not be recordable as a score.

    `best_gap` is a SEPARATE, graded record of how close an infeasible episode
    got, kept because on hard N=32 instances every blind policy measured so far
    ends infeasible: beam width 200 over 250k states, uniform-random restarts
    0/60, greedy-with-random-tiebreak 0/60.  If episodes also end infeasible,
    best_S alone cannot rank them and a whole run reads as a tie.  Key is
    (viol_sum, viol_count, L, S) -- total excess over L_max first, since that
    is the actual distance to feasibility.

    It is deliberately NOT a second scoring channel: it records only states the
    episode STOOD in, never probes, so preview_level_opts still cannot buy
    credit by enumeration, and it never touches best_S.
    """

    __slots__ = ("inst", "evals", "mutations", "best_S", "best_at_eval",
                 "best_state", "best_gap", "_cache_key", "_cache_val")

    def __init__(self, inst: TwoPhaseInstance) -> None:
        self.inst = inst
        self.evals = 0
        self.mutations = 0
        self.best_S: int | None = None
        self.best_at_eval: int | None = None
        self.best_state: Any = None
        self.best_gap: tuple[int, int, int, int] | None = None
        self._cache_key: frozenset[Node] | None = None
        self._cache_val: StateEval | None = None

    def evaluate(self, nodeset: Iterable[Node], *, record=None) -> StateEval:
        ns = frozenset(nodeset)
        if ns == self._cache_key and self._cache_val is not None:
            return self._cache_val  # free re-read of the current state
        self.evals += 1
        ev = evaluate_nodeset(self.inst, ns)
        S = ev.S
        if ev.feasible and (self.best_S is None or S < self.best_S):
            self.best_S = S
            self.best_at_eval = self.evals
            self.best_state = record if record is not None else ns
        gap = ev.gap
        if gap is not None and (self.best_gap is None or gap < self.best_gap):
            self.best_gap = gap
        self._cache_key, self._cache_val = ns, ev
        return ev

    def evaluate_no_ratchet(self, nodeset: Iterable[Node]) -> StateEval:
        """A metered look at a state the episode is NOT in.

        Charges one evaluator pass and nothing else: no ratchet, no mutation,
        and the current-state cache is left alone so a probe cannot make the
        next real read free (or, worse, return the probe's answer for the
        state the player is actually standing in).
        """
        ns = frozenset(nodeset)
        saved = (self._cache_key, self._cache_val)
        best = (self.best_S, self.best_at_eval, self.best_state)
        gap = self.best_gap
        try:
            ev = self.evaluate(ns)
        finally:
            self._cache_key, self._cache_val = saved
            self.best_S, self.best_at_eval, self.best_state = best
            self.best_gap = gap
        return ev

    def note_mutations(self, n: int) -> None:
        self.mutations += n


# ---------------------------------------------------------------------------
# episode
# ---------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Snapshot:
    sid: int
    phase: int
    backbone: tuple[Node, ...] | None
    nodeset: frozenset[Node] | None
    g: int
    k: int
    L: int | None
    S: int
    note: str


class TwoPhaseEpisode:
    """Phase 1 (regroup) -> bridge -> Phase 2 (level-opt), metered.

    `g` and `k` are the two move counters; the identity S = (N-1) + g + k is
    checked against the realised node count at every transition, so a bug in
    either operator surfaces as an assertion rather than as a plausible score.
    """

    def __init__(self, inst: TwoPhaseInstance, config: TwoPhaseConfig) -> None:
        self.inst = inst
        self.config = config
        self.meter = Meter(inst)
        self.tool_calls = 0
        self.phase = 1
        self.g = 0
        self.k = 0
        self.backbone: tuple[Node, ...] = tuple(
            sorted((i, 0) for i in range(1, inst.N))
        )
        self.nodeset: frozenset[Node] | None = None
        self.archive: list[Snapshot] = []
        self._next_sid = 0
        self.done = False
        self.stop_reason: str | None = None
        self.records: list[dict] = []
        #: side ledger: evaluator passes the HARNESS spends rendering the
        #: pushed menus and representation blocks.  Counted, reported, never
        #: charged to the player -- the player's budget prices only what the
        #: player asks for.
        self.harness_evals = 0
        #: when the best-feasible record improves, the state is
        #: auto-archived and the gauge points here.
        self.best_snapshot_sid: int | None = None
        #: set by submit: the grading eval of the handed-in state.
        self.submitted_eval: StateEval | None = None
        self._check_identity()

    def commit_backbone(self) -> None:
        """Complete the backbone and enter Phase 2 -- the `bridge` move.

        Snapshots the Phase-1 state on the way through, because the bridge is
        the one door in this environment that cannot be walked back and `fork`
        is advertised as available.  Without this, going back requires having
        called `archive_put` BEFORE the bridge -- a bookkeeping obligation the
        player has no way to anticipate, and an early probe episode ran
        straight into it: at turn 16, having reached a feasible S, it wrote
        "let me check whether an earlier snapshot exists to try a better-
        balanced backbone", found none, and submitted.  It wanted to branch,
        for the right reason, at the right moment, and was stopped by
        bookkeeping rather than by budget or by the game.

        The snapshot is not a decision made on the player's behalf -- whether
        to go back is still entirely theirs.
        """
        assert self.phase == 1, "already committed"
        self.snapshot("auto: backbone as it stood at the bridge")
        self.nodeset = frozenset(graph_ops.complete(list(self.backbone),
                                                    self.inst.N))
        self.phase = 2
        self._check_identity()

    # -- invariants ------------------------------------------------------

    def ridge_len(self) -> int:
        return sum(1 for (m, l) in self.backbone if l == 0)

    def _check_identity(self) -> None:
        """S = (N-1) + g + k, verified against the realised node count.

        In Phase 2 the realised count is the nodeset itself, so the check is
        against reality.  In Phase 1 it is the CLOSED FORM 2(N-1) - ridge,
        which means this assertion verifies the bookkeeping -- that g
        regroups shortened the spine by exactly g -- and NOT the closed form
        itself.  Saying otherwise (an earlier version of this docstring did)
        claims a check that is not performed.

        The closed form is verified in two other places instead: exhaustively
        over all Catalan(N-1) backbones for N = 4..9 with an actual
        completion counted each time, and again at every bridge, where
        commit_backbone completes for real and this assertion then runs in
        Phase 2 against the true node count.
        """
        N = self.inst.N
        if self.phase == 1:
            assert self.k == 0, "level-opts before the bridge"
            got = 2 * (N - 1) - self.ridge_len()
            assert len(self.backbone) == N - 1, (
                f"backbone has {len(self.backbone)} non-input nodes, want {N-1}"
            )
        else:
            assert self.nodeset is not None
            got = sum(1 for (m, l) in self.nodeset if m != l)
        # purely additive surface: every move costs exactly one node and
        # the identity is a theorem
        want = (N - 1) + self.g + self.k
        assert got == want, (
            f"S identity broken: realised {got}, (N-1)+g+k = {want} "
            f"(phase {self.phase}, g={self.g}, k={self.k})"
        )

    # -- budgets ---------------------------------------------------------

    @property
    def evals_left(self) -> int:
        return max(0, self.config.max_evals - self.meter.evals)

    @property
    def mutations_left(self) -> int:
        return max(0, self.config.max_mutations - self.meter.mutations)

    @property
    def tool_calls_left(self) -> int:
        return max(0, self.config.max_tool_calls - self.tool_calls)

    def charge_tool_call(self) -> None:
        if self.tool_calls >= self.config.max_tool_calls:
            raise BudgetExceeded(
                f"tool-call budget exhausted ({self.config.max_tool_calls})"
            )
        self.tool_calls += 1

    def charge_mutations(self, n: int) -> None:
        if n > self.mutations_left:
            raise BudgetExceeded(
                f"move costs {n} but only {self.mutations_left} remain "
                f"(budget {self.config.max_mutations})"
            )
        self.meter.note_mutations(n)

    def check_evals(self, n: int = 1) -> None:
        if self.evals_left < n:
            raise BudgetExceeded(
                f"evaluator budget exhausted ({self.config.max_evals})"
            )

    def _budget_stop(self) -> str | None:
        if self.tool_calls >= self.config.max_tool_calls:
            return "tool_calls_exhausted"
        if self.meter.evals >= self.config.max_evals:
            return "evals_exhausted"
        if self.mutations_left <= 0:
            return "mutations_exhausted"
        return None

    # -- evaluation ------------------------------------------------------

    @property
    def S_now(self) -> int:
        """Realised S of the state the player stands in: the completed-adder
        node count in Phase 1 (closed form), the actual count in Phase 2.
        Equal to (N-1)+g+k (theorem, asserted on every transition)."""
        if self.phase == 1:
            return 2 * (self.inst.N - 1) - self.ridge_len()
        assert self.nodeset is not None
        return sum(1 for (m, l) in self.nodeset if m != l)

    def completed_nodeset(self) -> frozenset[Node]:
        """The full adder implied by the current state, in either phase."""
        if self.phase == 2:
            assert self.nodeset is not None
            return self.nodeset
        return frozenset(graph_ops.complete(list(self.backbone), self.inst.N))

    def evaluate(self) -> StateEval:
        """One metered pass over the implied full adder.

        Auto-archive rides here: when this pass improves the best-feasible record,
        the state is snapshotted automatically and the gauge line points at
        the snapshot.  The three ratchet gates (feasible, strictly better,
        a state the episode STOOD in) are untouched -- this only archives
        what the ratchet already accepted.
        """
        self.check_evals()
        prev_best = self.meter.best_S
        ev = self.meter.evaluate(
            self.completed_nodeset(),
            record={"phase": self.phase, "g": self.g, "k": self.k,
                    "backbone": self.backbone if self.phase == 1 else None},
        )
        if self.meter.best_S is not None and self.meter.best_S != prev_best:
            snap = self.snapshot(f"auto: best feasible S={self.meter.best_S}")
            self.best_snapshot_sid = snap.sid
        return ev

    def harness_state_eval(self, nodeset) -> StateEval:
        """One evaluation on the HARNESS ledger.

        Used by the pushed menus and representation blocks: full price paid
        into harness_evals, zero into the player's budget, zero ratchet, and
        the meter's current-state cache is never touched.  The player's
        budget prices the player's own probes; the harness pricing its own
        rendering against the player would make the view itself a cost.
        """
        self.harness_evals += 1
        return evaluate_nodeset(self.inst, frozenset(nodeset))

    def harness_arrivals(self, nodeset) -> dict:
        """Per-node arrival levels on the harness ledger, for the pushed
        representation blocks (they annotate every node, which StateEval's
        aggregates cannot serve)."""
        self.harness_evals += 1
        ns = frozenset(nodeset)
        return FE.node_arrivals(ns, self.inst.N, list(self.inst.arrival))

    def probe(self, nodeset) -> StateEval:
        """Evaluate a hypothetical node set: costs an evaluator pass, costs no
        mutations, and does NOT ratchet.

        The rule: a probe charges its evaluator work but explicitly does
        not ratchet and does not count as mutations, or the tool would
        auto-play the game.  Without it the two currencies collapse into one:
        S = (N-1) + g + k means k IS the mutation count, so trying a move and
        undoing it still spends the score's own currency, and looking ahead
        was strictly more expensive than guessing.  Measured consequence: a
        wide beam goes bankrupt within a few plies under a small mutation budget
        while needing depth 9-16.

        Not ratcheting is the load-bearing half.  A probe that could record a
        score would let a player reach the optimum by enumeration without ever
        committing to it, which is the game playing itself.
        """
        self.check_evals()
        return self.meter.evaluate_no_ratchet(nodeset)

    # -- archive ---------------------------------------------------------

    def snapshot(self, note: str = "") -> Snapshot:
        ev = self.current_eval()
        snap = Snapshot(
            sid=self._next_sid,
            phase=self.phase,
            backbone=self.backbone if self.phase == 1 else None,
            nodeset=self.nodeset if self.phase == 2 else None,
            g=self.g,
            k=self.k,
            L=(ev.L if ev is not None else None),
            S=self.S_now,
            note=note,
        )
        self._next_sid += 1
        self.archive.append(snap)
        return snap

    def restore(self, sid: int) -> Snapshot:
        for s in self.archive:
            if s.sid == sid:
                break
        else:
            raise ValueError(f"no snapshot {sid}; have {[s.sid for s in self.archive]}")
        self.phase = s.phase
        self.g, self.k = s.g, s.k
        if s.phase == 1:
            assert s.backbone is not None
            self.backbone, self.nodeset = s.backbone, None
        else:
            assert s.nodeset is not None
            self.nodeset = s.nodeset
        self.meter._cache_key = self.meter._cache_val = None
        self._check_identity()
        return s

    # -- state view ------------------------------------------------------

    def fingerprint(self) -> str:
        """A short identity for the current structure.

        Without this the view says what a state SCORES but not which state it
        is, so no player -- scripted or otherwise -- can tell whether two
        different move sequences arrived at the same graph.  A beam built on
        this environment without it silently degenerates: its slots fill with
        re-derivations of the same few structures and its effective width
        collapses to about one, which is exactly how a wide beam came to
        score identically to width-1 greedy (3/16 both).

        Dedup is not an optimisation here, it is a precondition for search, so
        the identity is part of the state rather than something a player has
        to reconstruct.  Cheap: it hashes the node set, no evaluation.
        """
        import hashlib

        ns = self.nodeset if self.phase == 2 else self.backbone
        key = ",".join(f"{m}.{l}" for (m, l) in sorted(ns))
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    def current_eval(self) -> StateEval | None:
        """The cached evaluation ONLY if it describes the state being stood on.

        A Phase-1 regroup mutates without evaluating, so the meter's cache can
        lag the standing state; quoting it unguarded put the PREVIOUS state's
        L / violation numbers on 54-80%% of Phase-1 views in the research
        waves this package was extracted from.  Every reader goes through
        this: an absent number is honest, a stale one is not.
        """
        if (self.meter._cache_val is not None
                and self.meter._cache_key == self.completed_nodeset()):
            return self.meter._cache_val
        return None

    def state(self) -> dict:
        ev = self.current_eval()
        return {
            "fingerprint": self.fingerprint(),
            "cell": self.inst.cell_id,
            "N": self.inst.N,
            "L_max": self.inst.L_max,
            "phase": self.phase,
            "g": self.g,
            "k": self.k,
            "S": self.S_now,
            "ridge": self.ridge_len() if self.phase == 1 else None,
            "L": ev.L if ev is not None else None,
            "level_sum": ev.level_sum if ev is not None else None,
            "violations": ev.viol_count if ev is not None else None,
            "viol_sum": ev.viol_sum if ev is not None else None,
            # [output i, its level, how far over L_max]
            "violating_outputs": ([list(v) for v in ev.violators]
                                  if ev is not None else None),
            "best_S": self.meter.best_S,
            "best_gap": self.meter.best_gap,
            "best_snapshot": self.best_snapshot_sid,
            "harness_evals": self.harness_evals,
            "archive": len(self.archive),
            "budget": {
                "tool_calls_left": self.tool_calls_left,
                "evals_left": self.evals_left,
                "mutations_left": self.mutations_left,
            },
            "done": self.done,
            "stop_reason": self.stop_reason,
        }

    #: the five stop labels.  Everything the driver can die of
    #: -- transport, gateway declines, truncation runs -- is protocol_abort;
    #: the detail string lives in the run record, the label stays closed.
    STOP_REASONS = ("submitted", "tool_calls_exhausted", "evals_exhausted",
                    "mutations_exhausted", "protocol_abort")

    def finish(self, reason: str) -> None:
        assert reason in self.STOP_REASONS, f"unknown stop label {reason!r}"
        self.done = True
        self.stop_reason = reason

    def final_record(self) -> dict:
        """The dual-track scoring record, derived once at episode end.

        Delivery track: `submitted_S` exists ONLY when the episode ended by
        submit AND the handed-in state is feasible; every other ending is a
        non-submission (null = loss).  Best-so-far track: the ratchet's
        record, kept silently, never disclosed to the player as a score.
        `final_*` is the state the episode happened to be standing in when it
        stopped -- recorded for analysis, never scored: final_S does not pass
        the feasibility gate, and subtracting best_S from it would read the
        safety net as negative value.
        """
        fin = evaluate_nodeset(self.inst, self.completed_nodeset())
        sub = self.submitted_eval if self.stop_reason == "submitted" else None
        submitted_feasible = bool(sub.feasible) if sub is not None else None
        submitted_S = sub.S if (sub is not None and sub.feasible) else None
        submitted_gap = list(sub.gap) if (sub is not None and
                                          sub.gap is not None) else None
        if submitted_S is not None and self.meter.best_S is not None:
            assert self.meter.best_S <= submitted_S, (
                f"invariant broken: best_S={self.meter.best_S} > "
                f"submitted_S={submitted_S}")
        return {
            "stop_reason": self.stop_reason,
            "submitted_S": submitted_S,
            "submitted_feasible": submitted_feasible,
            "submitted_gap": submitted_gap,
            "best_S": self.meter.best_S,
            "best_at_eval": self.meter.best_at_eval,
            "best_gap": (list(self.meter.best_gap)
                         if self.meter.best_gap is not None else None),
            "best_snapshot": self.best_snapshot_sid,
            "final_S": fin.S,
            "final_L": fin.L,
            "final_feasible": fin.feasible,
            "final_gap": list(fin.gap) if fin.gap is not None else None,
            "final_g": self.g,
            "final_k": self.k,
            "final_phase": self.phase,
            "tool_calls": self.tool_calls,
            "evals": self.meter.evals,
            "mutations": self.meter.mutations,
            "harness_evals": self.harness_evals,
        }


def make_two_phase_episode(cell: dict, **kw) -> TwoPhaseEpisode:
    """Build an episode from a suite cell dict (id/N/arrival/L_max)."""
    inst = TwoPhaseInstance(
        cell_id=cell.get("id") or cell["cell_id"],
        N=int(cell["N"]),
        arrival=tuple(int(x) for x in cell["arrival"]),
        L_max=int(cell["L_max"]),
    )
    cfg = TwoPhaseConfig(**kw)
    return TwoPhaseEpisode(inst, cfg)
