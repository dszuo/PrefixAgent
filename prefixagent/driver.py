#!/usr/bin/env python3
"""Run an LLM through the two-phase environment.

The game configuration is derived in prefixagent.runtime.game; the axes are
never set by hand here, so the driver cannot invent a variant by typo.

Context: three segments -- the system prompt (protocol + briefing, a
cached prefix paid once), an append-only one-line-per-turn journal at the
head of the first user message, and the last k exchanges verbatim, ending
with the current view.

Protocol: one `TOOL_CALL: {"name":..., "arguments": {...}}` per turn, first
parseable marker wins; extra markers are counted (multi_call_count) and the
feedback says only the first ran.  A reply we cannot parse is
fed back as a TOOL_RESULT, never silently retried; driver-side deaths
(transport, gateway declines, truncation runs) finish the EPISODE with the
fifth stop label, protocol_abort, with the detail in the run record --
"channel died" and "model stopped" must never pool.

The key is read from the environment and never printed, logged or passed in
argv; assert_no_key_in_argv enforces the argv half.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent

from prefixagent.net import (
    assert_env_var_name, assert_no_key_in_argv, chat, parse_tool_call,
    _cache_tokens, PARSE_ERROR_FEEDBACK, TransportError,
)
from prefixagent.runtime.game import game_config, game_config_hash
from prefixagent.runtime.episode import make_two_phase_episode
from prefixagent.runtime.render import briefing, render_turn
from prefixagent.runtime.tools import dispatch, game_tools

SYSTEM = (
    "You are optimising a parallel prefix adder.  Each turn, reply with "
    "exactly ONE line of the form\n"
    '  TOOL_CALL: {"name": "<tool>", "arguments": {...}}\n'
    "You may reason before it, but the line must be last and must be valid "
    "JSON on one line.  You will receive a TOOL_RESULT and the new state.\n\n"
)

#: backstop on a serialised TOOL_RESULT.  Generous on purpose: the quotes are
#: the reason the tools exist, and clipping them is how the harness lost most
#: of a frontier without saying so.
TOOL_RESULT_CAP = 8000

JOURNAL_HEAD = (
    "GAME JOURNAL -- one line per turn so far, oldest first. Views are not "
    "repeated here; the current view is at the end of this message thread. "
    "Lines marked *best* improved your best feasible state.\n")


def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    s = json.dumps(args, separators=(",", ":"))
    return s if len(s) <= 90 else s[:87] + "..."


def journal_line(turn_n: int, name: str | None, args: dict, out: dict,
                 ep, best_before) -> str:
    """One journal line: tool + args + ok/refusal code + g,k,S,L,viol + phase events +
    *best* marks.  Menus and views never enter the journal."""
    if name is None:
        return f"t{turn_n} PARSE_ERROR (reply had no usable TOOL_CALL)"
    bits = [f"t{turn_n}", f"P{out.get('phase', ep.phase)}",
            f"{name}({_fmt_args(args)})"]
    if out.get("ok"):
        bits.append("ok")
    else:
        err = str(out.get("error") or "refused")
        bits.append("REFUSED:" + err[:60].replace("\n", " "))
    bits.append(f"g={ep.g} k={ep.k} S={out.get('S')}")
    if out.get("L") is not None:
        bits.append(f"L={out.get('L')} viol={out.get('violations')}"
                    f"/{out.get('viol_sum')}")
    if name == "bridge" and out.get("ok"):
        bits.append("BRIDGED->P2")
    if name == "fork" and out.get("ok"):
        bits.append(f"forked->snapshot{out.get('forked_to')}")
    if name == "archive_put" and out.get("ok"):
        bits.append(f"snapshot{out.get('snapshot')}")
    if ep.meter.best_S is not None and ep.meter.best_S != best_before:
        bits.append(f"*best S={ep.meter.best_S}*")
    return " ".join(bits)


def build_messages(k: int, system: str, journal: list[str],
                   exchanges: list[dict], view: str) -> list[dict]:
    """Assemble the turn's message list: system prompt, the append-only
    journal at the head of the first user message (so the provider-side
    prefix cache keeps paying out through it), the last k exchanges
    verbatim, then the current view.  Strict user/assistant alternation
    throughout (Anthropic-fronting gateways require it).
    """
    msgs = [{"role": "system", "content": system}]
    head = JOURNAL_HEAD + "\n".join(journal) + "\n\n" if journal else ""
    win = exchanges[-k:] if k > 0 else []
    if not win:
        msgs.append({"role": "user", "content": head + view})
        return msgs
    msgs.append({"role": "user", "content":
                 head + "Recent turns, verbatim (older ones are in the "
                        "journal above):"})
    for i, e in enumerate(win):
        msgs.append({"role": "assistant", "content": e["a"]})
        u = e["u"]
        if i == len(win) - 1:
            u = u + "\n\n" + view
        msgs.append({"role": "user", "content": u})
    return msgs


def run_episode(cell, cfg, seed=0):
    axes = game_config()
    env_kw = {kk: v for kk, v in axes.items() if not kk.startswith("context_")}
    ctx_k = axes.get("context_k", 4)
    ep = make_two_phase_episode(
        cell, tools=game_tools(), seed=seed,
        max_tool_calls=cfg.max_tool_calls, max_evals=cfg.max_evals,
        max_mutations=cfg.max_mutations, **env_kw)
    system = SYSTEM + briefing(ep)
    journal: list[str] = []
    exchanges: list[dict] = []   # {"a": model reply, "u": tool feedback}
    rec = {"cell": cell.get("id") or cell.get("cell_id"),
           "seed": seed, "model": cfg.model,
           "axes": axes, "game_config_hash": game_config_hash(),
           "context": {"k": ctx_k},
           "budgets": {"tool_calls": cfg.max_tool_calls,
                       "evals": cfg.max_evals,
                       "mutations": cfg.max_mutations},
           "turns": [], "driver": {
               "http_errors": 0, "parse_errors": 0, "empty_replies": 0,
               "prompt_tokens": 0, "completion_tokens": 0,
               "cache_hit": 0, "cache_miss": 0, "multi_call_turns": 0}}
    t0 = time.time()
    consec_trunc = 0
    # A model that never produces a parseable call never spends a tool call,
    # so the episode budget alone cannot terminate it.  The driver guard is
    # generous -- twice the budget plus slack -- and is a protocol_abort,
    # charged to the channel, never to the game.
    llm_calls = 0
    llm_call_guard = 2 * cfg.max_tool_calls + 20
    abort_detail = None
    wall_cap = getattr(cfg, "wall_cap", 21600)
    while not ep.done:
        llm_calls += 1
        if llm_calls > llm_call_guard:
            abort_detail = (f"DRIVER: {llm_calls - 1} LLM calls against a "
                            f"{cfg.max_tool_calls}-turn budget -- the reply "
                            "stream is not converting into tool calls")
            break
        if wall_cap and time.time() - t0 > wall_cap:
            abort_detail = (f"DRIVER: episode exceeded the {wall_cap}s wall "
                            "cap -- a channel-layer lane protection, not a "
                            "game budget")
            break
        view = render_turn(ep)
        msgs = build_messages(ctx_k, system, journal, exchanges, view)
        payload = {"model": cfg.model, "messages": msgs,
                   "temperature": cfg.temperature,
                   "max_tokens": cfg.max_tokens}
        if cfg.effort:
            # some vendors reject `reasoning_effort` outright;
            # output_config.effort is the spelling that works there.
            payload["output_config"] = {"effort": cfg.effort}
        if getattr(cfg, "disable_thinking", False):
            # Local vLLM Qwen honours this cleanly; without it a thinking
            # model can burn the whole completion cap before its first
            # tool call.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = chat(cfg.server, payload, timeout=cfg.timeout,
                        api_key=os.environ[cfg.api_key_env],
                        connect_timeout=cfg.connect_timeout)
        except (TransportError, Exception) as e:      # noqa: BLE001
            rec["driver"]["http_errors"] += 1
            abort_detail = f"transport: {type(e).__name__}: {e}"[:1200]
            break
        ch = (resp.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        # Some vendors return the chain of thought in reasoning_content.
        # Captured, never fed back.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        finish_reason = ch.get("finish_reason")
        u = resp.get("usage") or {}
        hit, miss = _cache_tokens(u)
        rec["driver"]["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        rec["driver"]["completion_tokens"] += int(u.get("completion_tokens") or 0)
        rec["driver"]["cache_hit"] += hit
        rec["driver"]["cache_miss"] += miss
        if not content.strip():
            # Two very different causes, never pooled: completion_tokens == 0
            # is the gateway declining the request shape (our bug, stop);
            # > 0 is the model generating nothing usable.
            rec["driver"]["empty_replies"] += 1
            if int(u.get("completion_tokens") or 0) == 0:
                rec["driver"]["zero_generation"] = \
                    rec["driver"].get("zero_generation", 0) + 1
                if rec["driver"]["zero_generation"] >= 2:
                    abort_detail = (
                        "PAYLOAD: two replies with completion_tokens=0 -- "
                        "the gateway is declining this request shape, not "
                        "the model refusing")
                    break

        parse_src = content
        if getattr(cfg, "strip_think", False) and "</think>" in content:
            # Qwen-family serving writes the chain of thought INSIDE content,
            # closed by </think>.  A TOOL_CALL literal inside the thinking is
            # a move the model CONSIDERED, not the one it chose -- parsing
            # must start after the last close tag or the harness executes
            # deliberation.  Channel adaptation, uniform within a wave.
            parse_src = content.rsplit("</think>", 1)[1]
        name, args, err, n_markers = parse_tool_call(parse_src)
        if (err or not name) and (ch.get("message") or {}).get("tool_calls"):
            # The gateway sometimes answers in NATIVE function-call form even
            # though this driver never sends a `tools` array.  Reading only
            # the text marker would score a well-formed call as a protocol
            # failure (3 of 11 "parse errors" in one early research run).
            try:
                tc = (ch["message"]["tool_calls"] or [])[0]
                fn = tc.get("function") or {}
                cand = fn.get("name")
                raw = fn.get("arguments")
                cargs = json.loads(raw) if isinstance(raw, str) else (raw or {})
                if cand:
                    name, args, err = cand, cargs, None
                    rec["driver"]["native_tool_calls"] = \
                        rec["driver"].get("native_tool_calls", 0) + 1
            except (KeyError, IndexError, TypeError, ValueError,
                    json.JSONDecodeError):
                pass
        best_before = ep.meter.best_S
        if err or not name:
            # Parse failures have two causes that must not be pooled: a
            # malformed reply, and a reply CUT at the token cap (finish_
            # reason=length) -- the latter is our config, not the model.
            rec["driver"]["parse_errors"] += 1
            if finish_reason == "length":
                rec["driver"]["truncated_replies"] = \
                    rec["driver"].get("truncated_replies", 0) + 1
                consec_trunc += 1
                if consec_trunc >= 3:
                    abort_detail = (
                        f"CONFIG: {consec_trunc} consecutive replies hit "
                        f"max_tokens={cfg.max_tokens} (finish_reason=length) "
                        "-- the thinking budget is crowding out the answer; "
                        "raise --max-tokens rather than blaming the model")
                    break
            else:
                consec_trunc = 0
            out = {"ok": False, "error": "PARSE_ERROR"}
            feedback = PARSE_ERROR_FEEDBACK
        else:
            consec_trunc = 0
            out = dispatch(ep, name, args)
            blob = json.dumps({kk: v for kk, v in out.items()
                               if kk != "archive"})
            if len(blob) > TOOL_RESULT_CAP:
                blob = (blob[:TOOL_RESULT_CAP]
                        + f' ...[TRUNCATED: {len(blob) - TOOL_RESULT_CAP}'
                          f' more chars; this JSON is cut off]')
            feedback = "TOOL_RESULT: " + blob
            if n_markers > 1:
                rec["driver"]["multi_call_turns"] += 1
                feedback = (f"NOTE: your reply contained {n_markers} "
                            "TOOL_CALL lines; only the FIRST was executed. "
                            "One call per turn.\n" + feedback)
        journal.append(journal_line(len(rec["turns"]) + 1, name, args, out,
                                    ep, best_before))
        rec["turns"].append({
            "n": len(rec["turns"]) + 1, "ts": round(time.time(), 3),
            "content": content[:4000],
            "tool": name, "args": args, "ok": bool(out.get("ok")),
            "error": out.get("error"), "phase": ep.phase,
            "g": ep.g, "k": ep.k,
            "S": out.get("S"), "L": out.get("L"),
            "viol_count": out.get("violations"),
            "viol_sum": out.get("viol_sum"),
            "best_S": ep.meter.best_S,
            "best_bite": ep.meter.best_S != best_before,
            "best_gap": ep.meter.best_gap,
            "multi_call_count": n_markers,
            "finish_reason": finish_reason,
            "completion_tokens": int(u.get("completion_tokens") or 0),
            "prompt_tokens": int(u.get("prompt_tokens") or 0),
            "cache_hit": hit, "cache_miss": miss,
            "view_chars": len(view),
            "harness_evals": ep.harness_evals,
            "reasoning": reasoning[:8000],
        })
        print(f"    [{rec['cell']}/s{seed}] t{len(rec['turns']):>3} "
              f"{(name or 'PARSE_ERROR'):18s} "
              f"{'ok' if out.get('ok') else 'REFUSED':7s} "
              f"phase={ep.phase} g={ep.g} k={ep.k} "
              f"best_S={ep.meter.best_S} gap={ep.meter.best_gap} "
              f"fin={finish_reason} ctok={int(u.get('completion_tokens') or 0)} "
              f"{round(time.time() - t0)}s", flush=True)
        exchanges.append({"a": content, "u": feedback})

    if not ep.done:
        # driver-side death: the fifth stop label.  The game did not end;
        # the channel did.  Detail goes to the record, the label stays closed.
        ep.finish("protocol_abort")
    rec["driver_stop_detail"] = abort_detail
    rec.update(ep.final_record())
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def load_cells(args) -> list[dict]:
    """Load instances from one suite file; every cell carries its arrival
    profile inline, and a cell without one is an error, never a silent
    skip."""
    man = json.loads(Path(args.suite).read_text())["instances"]
    want = set(args.cells.split(",")) if args.cells else None
    out = []
    for c in man:
        if want and c["id"] not in want:
            continue
        assert "arrival" in c, f"{c['id']}: suite cell has no arrival profile"
        out.append({"id": c["id"], "N": int(c["N"]), "L_max": int(c["L_max"]),
                    "arrival": list(c["arrival"])})
    if want and len(out) != len(want):
        got = {c["id"] for c in out}
        sys.exit(f"asked for {len(want)} cells, loaded {len(out)}; missing "
                 f"{sorted(want - got)}")
    return out


def main() -> int:
    assert_no_key_in_argv(sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(PKG / "suites/default_suite.json"),
                    help="instance file (see suites/default_suite.json for "
                         "the four-field format; bring your own)")
    ap.add_argument("--cells", default="", help="comma-separated ids "
                    "(default: every cell in the suite)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="episodes per instance; the environment is "
                         "deterministic, so this labels repeats (variation "
                         "comes from the model's sampling)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--server", default="https://api.anthropic.com",
                    help="any OpenAI-compatible endpoint base URL (with or "
                         "without a trailing /v1 -- it is normalised); the "
                         "default is Anthropic's OpenAI-compatible endpoint")
    ap.add_argument("--api-key-env", default="ANTHROPIC_API_KEY",
                    help="NAME of the env var holding the key (never the key)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="completion cap per call; raise it for long-thinking "
                         "models, lower it if your endpoint's context is small")
    ap.add_argument("--effort", default="",
                    help="output_config.effort passthrough, for vendors "
                         "that support it (empty = omit)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--wall-cap", type=int, default=21600,
                    help="channel-layer per-episode wall clock cap (s); "
                         "breach = protocol_abort, never a game label")
    ap.add_argument("--strip-think", action="store_true",
                    help="parse TOOL_CALL only after the last </think> "
                         "(models that think inside content)")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="send chat_template_kwargs enable_thinking=false "
                         "(local vLLM Qwen)")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--max-tool-calls", type=int, default=200)
    ap.add_argument("--max-evals", type=int, default=200_000)
    ap.add_argument("--max-mutations", type=int, default=10 ** 9,
                    help="counted, not gated; the default IS the game "
                         "spec, not a placeholder")
    ap.add_argument("--out", default="")
    cfg = ap.parse_args()
    cfg.api_key_env = assert_env_var_name(cfg.api_key_env)
    if not os.environ.get(cfg.api_key_env):
        raise SystemExit(f"FATAL: env {cfg.api_key_env} is not set")

    if not cfg.out:
        raise SystemExit("FATAL: --out is required (one jsonl per job)")

    cells = load_cells(cfg)
    print(f"model={cfg.model} cells={len(cells)} "
          f"seeds={cfg.seeds} budget={cfg.max_tool_calls} turns", flush=True)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for c in cells:
            for s in range(cfg.seed_base, cfg.seed_base + cfg.seeds):
                r = run_episode(c, cfg, seed=s)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                d = r["driver"]
                print(f"  {c['id']:22s} s{s} -> "
                      f"submitted_S={r['submitted_S']} "
                      f"best_S={r['best_S']} (g={r['final_g']}, "
                      f"k={r['final_k']}) "
                      f"turns={r['tool_calls']} stop={r['stop_reason']} "
                      f"tok={d['prompt_tokens']}/{d['completion_tokens']} "
                      f"hit={d['cache_hit']} http={d['http_errors']} "
                      f"parse={d['parse_errors']} {r['wall_s']}s",
                      flush=True)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
