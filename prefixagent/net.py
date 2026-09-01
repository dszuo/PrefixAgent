"""Transport and protocol utilities: one retry loop, key hygiene, the
TOOL_CALL text protocol, and cache-token accounting across vendor spellings.
"""
from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

KEYLIKE_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")


def assert_no_key_in_argv(argv: list[str]) -> None:
    """Refuse to start if any argv token looks like an API key.

    Keys enter this process through an environment variable only (which one
    is selectable with --api-key-env, whose VALUE is a variable NAME, never
    a secret); there is no --api-key flag, and a key-looking token anywhere
    in argv aborts."""
    for tok in argv:
        if KEYLIKE_RE.search(tok):
            raise SystemExit(
                "FATAL: argv contains a key-looking token (sk-...). API keys "
                "must NEVER be command-line arguments (visible in `ps`/shell "
                "history). export MY_API_KEY=... and retry.")
        if tok.split("=", 1)[0] in ("--api-key", "--key", "--token"):
            raise SystemExit(
                "FATAL: this driver takes no key flag by design. "
                "export MY_API_KEY=... instead.")


def assert_env_var_name(name: str) -> str:
    """--api-key-env must name a variable, not carry one."""
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", name):
        raise SystemExit(
            f"FATAL: --api-key-env takes an environment VARIABLE NAME "
            f"(e.g. MY_API_KEY), got {name!r}.")
    return name



CALL_START_RE = re.compile(r"TOOL_CALL:\s*")

PARSE_ERROR_FEEDBACK = (
    'TOOL_RESULT: {"ok": false, "error": "PARSE_ERROR: your TOOL_CALL was '
    'not valid JSON. Emit exactly ONE line: TOOL_CALL: {\\"name\\": '
    '\\"<tool>\\", \\"arguments\\": {...}} using double quotes, with '
    'nothing after the closing brace."}')


def parse_tool_call(content: str) -> tuple[str | None, dict, str | None, int]:
    """First parseable TOOL_CALL -> (name, args, err, n_markers).

    raw_decode tolerates trailing prose after the JSON object (the failure
    mode that poisons naive parsers: a greedy {.*} regex spanning
    to the last brace of the message).  err = "unparseable" only when
    markers exist but none yields valid JSON; no marker at all is not an
    error (the model may be done talking)."""
    dec = json.JSONDecoder()
    starts = [m.end() for m in CALL_START_RE.finditer(content)]
    for pos in starts:
        try:
            obj, _end = dec.raw_decode(content, pos)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and obj.get("name"):
            return str(obj["name"]), obj.get("arguments") or {}, None, len(starts)
    if starts:
        return None, {}, "unparseable", len(starts)
    return None, {}, None, 0



CHAT_ATTEMPTS = 3
CHAT_RETRY_EXC = (OSError, urllib.error.URLError, TimeoutError,
                  json.JSONDecodeError, http.client.HTTPException)
# http.client.HTTPException covers IncompleteRead: the vendor closing a
# socket mid-body is a transport fault like any other, and NOT retrying
# it converts long healthy episodes into protocol aborts.


class TransportError(RuntimeError):
    pass


def _chat_via_http_client(server: str, body: str, headers: dict,
                          timeout: int, connect_timeout: float) -> dict:
    # Direct connection, env proxies ignored: a short connect timeout
    # defeats blackholed-IP routes some networks serve for API hosts.
    u = urllib.parse.urlsplit(server)
    conn_cls = (http.client.HTTPSConnection if u.scheme == "https"
                else http.client.HTTPConnection)
    path = (u.path.rstrip("/") if u.path else "") + "/v1/chat/completions"
    conn = conn_cls(u.netloc, timeout=connect_timeout)
    try:
        conn.connect()
        conn.sock.settimeout(timeout)
        conn.request("POST", path, body=body, headers=headers)
        r = conn.getresponse()
        data = r.read()
        if r.status != 200:
            raise urllib.error.URLError(f"HTTP {r.status}: {data[:200]!r}")
        return json.loads(data)
    finally:
        conn.close()   # finally, or concurrency leaks a socket per retry


def _chat_via_urllib(server: str, body: str, headers: dict,
                     timeout: int) -> dict:
    req = urllib.request.Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=body.encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _base(server: str) -> str:
    """Accept the base URL with or without a trailing /v1: the path below
    always appends /v1/chat/completions, and the OpenAI-SDK muscle memory
    of pasting ...:8000/v1 otherwise yields /v1/v1/... and a bare 404."""
    s = server.rstrip("/")
    return s[:-3] if s.endswith("/v1") else s


def chat(server: str, payload: dict, timeout: int = 300,
         api_key: str | None = None,
         connect_timeout: float | None = None) -> dict:
    """One chat completion; both transports share ONE retry loop --
    transports must differ only in proxy handling, never in attempt
    count or retryable-exception set."""
    server = _base(server)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload)
    for attempt in range(CHAT_ATTEMPTS):
        try:
            if connect_timeout:
                return _chat_via_http_client(server, body, headers, timeout,
                                             connect_timeout)
            return _chat_via_urllib(server, body, headers, timeout)
        except CHAT_RETRY_EXC as e:
            if attempt == CHAT_ATTEMPTS - 1:
                raise TransportError(f"{type(e).__name__}: {e}") from e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _cache_tokens(u: dict) -> tuple[int, int]:
    """(cache_hit, cache_miss) input tokens, across three vendor spellings.

    Reading only one vendor's spelling makes every other vendor's run
    report 0/0 and read as "caching is off" -- and some gateways inject
    cache_control automatically, so the field a vendor never sends is
    exactly the one that matters.

      * DeepSeek      prompt_cache_hit_tokens / prompt_cache_miss_tokens
      * OpenAI-compat prompt_tokens_details.cached_tokens (+ some gateways'
                      cached_creation_tokens for the write leg)
      * Anthropic     cache_read_input_tokens / cache_creation_input_tokens

    Miss is derived as prompt_tokens - hit when a vendor reports only the hit
    leg, so hit+miss always reconciles with prompt_tokens.
    """
    hit = int(u.get("prompt_cache_hit_tokens") or 0)
    miss = int(u.get("prompt_cache_miss_tokens") or 0)
    reported = ("prompt_cache_hit_tokens" in u
                or "prompt_cache_miss_tokens" in u)
    if not (hit or miss):
        det = u.get("prompt_tokens_details") or {}
        reported = reported or ("cached_tokens" in det
                                or "cached_creation_tokens" in det)
        hit = int(det.get("cached_tokens") or 0)
        # a cache WRITE is not a hit: it is billed at or above the cold rate,
        # so counting it as cached would flatter the hit rate exactly when
        # the prefix is churning and nothing is being reused.
        miss = int(det.get("cached_creation_tokens") or 0)
    if not (hit or miss):
        reported = reported or ("cache_read_input_tokens" in u
                                or "cache_creation_input_tokens" in u)
        hit = int(u.get("cache_read_input_tokens") or 0)
        miss = int(u.get("cache_creation_input_tokens") or 0)
    # Reconcile against prompt_tokens ALWAYS, not only when the miss leg is
    # absent.  A gateway may report a hit leg and a cache-WRITE leg while
    # leaving ordinary uncached tokens in neither bucket: measured on a real
    # probe, hit 103,163 + miss 230,726 = 333,889 against prompt_tokens
    # 400,409: hit/(hit+miss) reads 30.9% while the true cached share is
    # 25.8%.  A hit RATE computed off a short denominator flatters itself by
    # exactly the unaccounted tokens.  A vendor that reports NOTHING is
    # unknown, not 0% cached, so silence stays (0, 0) and analysis must
    # treat it as unmeasured.
    if not reported:
        return 0, 0
    total = int(u.get("prompt_tokens") or 0)
    if total and hit + miss != total:
        miss = max(0, total - hit)
    return hit, miss


