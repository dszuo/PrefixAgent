"""The game configuration, derived -- never hand-written.

Two incidents bought this module: a hand-typed tool list in a script
silently dropped two Phase 2 affordances, and a text tweak to the briefing
went unversioned, so two "identical" runs were not.  Hence the two rules
here:

  * the configuration exists in exactly one place (GAME) and reaches the
    episode only through `game_config()`;
  * `game_config_hash` covers both the configuration AND the rendered-text
    source (render.py, repr_blocks.py, tools.py, driver.py), so any change
    to what a player is shown -- briefing copy, menu wording, quote text --
    changes the hash and stale cross-run comparisons become visible
    instead of silent.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

#: the two-phase game: Phase 1 annotated S-expression, Phase 2 full
#: arrival-aware EPR, C2 cross-phase visibility, pushed menus with exact
#: 1-ply quotes, journal context with a 4-exchange verbatim window.
GAME = {
    "p2_view": "epr",
    "c2": True,
    "menus": True,
    "context_k": 4,
}


def game_config() -> dict:
    return dict(GAME)


@lru_cache(maxsize=1)
def _render_text_digest() -> str:
    """Digest of everything that can put words in front of the player.

    driver.py is included because SYSTEM and JOURNAL_HEAD live there --
    the first words of every prompt and the every-turn journal header.
    tools.py belongs here too: it writes player-facing text (the quotes'
    S_after, every refusal message).

    Cached for the life of the process: the hash must describe the code
    the process actually runs (the imported bytes), not whatever is on
    disk at call time.
    """
    here = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for fpath in (here / "render.py", here / "repr_blocks.py",
                  here / "tools.py", here.parent / "driver.py"):
        h.update(fpath.read_bytes())
    return h.hexdigest()


def game_config_hash() -> str:
    blob = json.dumps(GAME, sort_keys=True) + _render_text_digest()
    return hashlib.sha1(blob.encode()).hexdigest()[:12]
