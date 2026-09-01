"""The two-operator two-phase environment: regroup, then level-opt.

This is the paper's own decomposition, rebuilt on three layers:

  operators   prefixagent.core -- clean node-set ports of the paper's
              Backbone.regroup and PrefixGraph.graph_opt, each verified
              behaviourally equivalent to the reference implementation by
              exhaustive differential test before extraction.

  evaluation  prefixagent.core.fast_eval.node_arrivals -- arrival-aware
              integer levels.  Plain logic depth cannot express a per-bit
              arrival profile, and every L_max in the shipped suite is
              arrival-aware, so this layer is a capability requirement,
              not a preference.

  orchestration  this package -- the paper's phase split and tool surface,
              rebuilt on a metered episode with a ratchet, an archive and
              a per-turn record.

The objective is exact and asserted rather than assumed (checked
exhaustively over thousands of backbones in the test suite):

    S_final = (N-1) + g + k

g regroups and k level-opts.  The serial backbone has ridge N-1, hence
completes to S = N-1; every regroup shortens the ridge by exactly 1 and so
costs exactly one completion node; every level-opt adds exactly one node.
Both operators spend the same currency at the same rate, so the whole game
is `minimise g + k` subject to L <= L_max, and the score is literally the
move count.
"""

from prefixagent.runtime.episode import (
    TwoPhaseEpisode,
    TwoPhaseConfig,
    make_two_phase_episode,
)

__all__ = ["TwoPhaseEpisode", "TwoPhaseConfig", "make_two_phase_episode"]
