"""prefixagent CLI.

  prefixagent run --model <m> --server <url> \
                  --api-key-env MY_API_KEY --out runs/x.jsonl
                                        LLM episodes (OpenAI-compatible API)
  prefixagent verify <episode.jsonl>    replay a recorded episode and
                                        re-derive its score from scratch
  prefixagent game                      the derived game configuration + hash
"""
from __future__ import annotations

import sys


def main(argv: list | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip()); return 0
    if argv[0] in ("--version", "-V"):
        from prefixagent import __version__
        print(__version__); return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "game":
        from prefixagent.runtime.game import GAME, game_config_hash
        print("game config:", ", ".join(f"{k}={v}" for k, v in GAME.items()))
        print("game_config_hash:", game_config_hash())
        return 0
    if cmd == "run":
        from prefixagent import driver
        sys.argv = ["prefixagent-run", *rest]
        return driver.main()
    if cmd == "verify":
        from prefixagent import verify
        sys.argv = ["prefixagent-verify", *rest]
        return verify.main()
    print(f"unknown command {cmd!r}; try --help"); return 2


if __name__ == "__main__":
    raise SystemExit(main())
