#!/usr/bin/env python3
"""Host-side key proxy for Claude, OpenAI and Gemini. Stdlib only; run on the HOST, not in the container.

The container sends requests to http://127.0.0.1:18765 (network_mode: host) with a dummy key.
The proxy swaps in the real key, forwards, logs every request to a ledger, and answers
HTTP 402 once the spend cap or the request cap is reached.

  route              provider   upstream                                              key in --env
  /v1/...            Claude     https://api.anthropic.com/v1/...                      CLAUDE_API_KEY or ANTHROPIC_API_KEY
  /openai/v1/...     OpenAI     https://api.openai.com/v1/...                         OPENAI_API_KEY
  /gemini/v1beta/... Gemini     https://generativelanguage.googleapis.com/v1beta/...  GEMINI_API_KEY or GOOGLE_API_KEY

  python3 keyproxy.py [--env ~/work/bench3/.env] [--port 18765] [--max-usd 10] [--max-requests 0]
                      [--ledger ~/.local/share/bench3/ledger.jsonl]
  curl -s 127.0.0.1:18765/_status                       # spend and requests per provider
  curl -s 127.0.0.1:18765/gemini/v1beta/models          # list models (free, not logged)
  curl -s 127.0.0.1:18765/openai/v1/models

The env file must be chmod 600. A missing key is fine: that route answers 503.
Cost is an ESTIMATE from the price table below (longest prefix match, $/MTok); the provider's
billing page is authoritative. Unknown models are charged at their provider's conservative fallback.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# provider: (path prefix to strip, upstream, auth header, header value format, env names)
ROUTES = {
    "openai": ("/openai", "https://api.openai.com", "authorization", "Bearer {}", ("OPENAI_API_KEY",)),
    "gemini": ("/gemini", "https://generativelanguage.googleapis.com", "x-goog-api-key", "{}", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "claude": ("", "https://api.anthropic.com", "x-api-key", "{}", ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")),
}
# input, output, cache read, cache write ($/MTok). Estimates; check the provider's price page.
PRICES = {
    "claude-opus-5": (4.0, 20.0, 0.20, 5.0),
    "gpt-5-mini": (0.25, 2.0, 0.025, 0.0),
    "gpt-5-nano": (0.05, 0.4, 0.005, 0.0),
    "gpt-5": (1.25, 10.0, 0.125, 0.0),
    "gpt-5.": (5.0, 40.0, 0.5, 0.0),  # gpt-5.x: price not in this table, conservative
    "gemini-2.5-flash": (0.30, 2.50, 0.03, 0.0),
    "gemini-2.5-pro": (1.25, 10.0, 0.125, 0.0),
    "gemini-3": (2.0, 12.0, 0.20, 0.0),
}
FALLBACK = {"claude": (4.0, 20.0, 0.20, 5.0), "openai": (5.0, 40.0, 0.5, 0.0), "gemini": (4.0, 18.0, 0.4, 0.0)}
DROP = {"host", "x-api-key", "authorization", "x-goog-api-key", "content-length", "accept-encoding", "connection"}


def load_keys(path: pathlib.Path) -> dict[str, str]:
    path = path.expanduser()
    if path.stat().st_mode & 0o077:
        raise SystemExit(f"{path} must be chmod 600")
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    keys = {}
    for prov, route_ in ROUTES.items():
        v = next((env[n] for n in route_[4] if env.get(n)), "")
        if v:
            keys[prov] = v
    if not keys:
        raise SystemExit(f"no API key in {path} (expected one of {[n for r in ROUTES.values() for n in r[4]]})")
    return keys


def route(path: str) -> str:
    for prov, route_ in ROUTES.items():
        prefix = route_[0]
        if prefix and (path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?")):
            return prov
    return "claude"


def usage_of(prov: str, j: dict) -> tuple[str, dict] | None:
    """(model, {input, output, cache_read, cache_write}) from a response body, or None."""
    if prov == "claude" and isinstance(j.get("usage"), dict):
        u = j["usage"]
        return j.get("model", ""), {"input": u.get("input_tokens", 0) or 0, "output": u.get("output_tokens", 0) or 0, "cache_read": u.get("cache_read_input_tokens") or 0, "cache_write": u.get("cache_creation_input_tokens") or 0}
    if prov == "openai" and isinstance(j.get("usage"), dict):
        u = j["usage"]
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        inp = u.get("prompt_tokens", u.get("input_tokens", 0)) or 0
        return j.get("model", ""), {"input": inp - cached, "output": u.get("completion_tokens", u.get("output_tokens", 0)) or 0, "cache_read": cached, "cache_write": 0}
    if prov == "gemini" and isinstance(j.get("usageMetadata"), dict):
        u = j["usageMetadata"]
        cached = u.get("cachedContentTokenCount", 0) or 0
        return j.get("modelVersion", ""), {"input": (u.get("promptTokenCount", 0) or 0) - cached, "output": (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0), "cache_read": cached, "cache_write": 0}
    return None


def cost(prov: str, model: str, u: dict) -> float:
    best = max((k for k in PRICES if model.startswith(k)), key=len, default=None)
    p = PRICES[best] if best else FALLBACK[prov]
    return (u["input"] * p[0] + u["output"] * p[1] + u["cache_read"] * p[2] + u["cache_write"] * p[3]) / 1e6


class State:
    def __init__(self, keys: dict[str, str], max_usd: float, max_requests: int, ledger: pathlib.Path):
        self.keys, self.max_usd, self.max_requests, self.ledger = keys, max_usd, max_requests, ledger
        self.lock = threading.Lock()
        self.spent = 0.0
        self.by = {p: {"requests": 0, "usd": 0.0} for p in ROUTES}
        if ledger.is_file():
            for line in ledger.read_text().splitlines():
                try:
                    self._add(json.loads(line))
                except (json.JSONDecodeError, AttributeError, ValueError):
                    pass

    def _add(self, r: dict) -> None:
        usd = float(r.get("usd", 0.0) or 0.0)
        self.spent += usd
        b = self.by.setdefault(r.get("provider", "claude"), {"requests": 0, "usd": 0.0})
        b["usd"] += usd
        if r.get("billable"):
            b["requests"] += 1

    @property
    def requests(self) -> int:
        return sum(b["requests"] for b in self.by.values())

    def record(self, rec: dict) -> None:
        with self.lock:
            self._add(rec)
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


def make_handler(st: State):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet; the ledger is the log
            pass

        def _send(self, code: int, body: bytes, headers: dict | None = None):
            self.send_response(code)
            for k, v in (headers or {"content-type": "application/json"}).items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code: int, kind: str, msg: str):
            return self._send(code, json.dumps({"type": "error", "error": {"type": kind, "message": msg, "code": code}}).encode())

        def do_GET(self):
            if self.path == "/_status":
                by = {p: {"requests": b["requests"], "usd": round(b["usd"], 4), "key": p in st.keys} for p, b in st.by.items()}
                return self._send(200, json.dumps({"spent_usd": round(st.spent, 4), "max_usd": st.max_usd, "requests": st.requests, "max_requests": st.max_requests, "by_provider": by}).encode())
            return self._forward(b"")

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            return self._forward(self.rfile.read(n))

        def _forward(self, body: bytes):
            prov = route(self.path)
            prefix, upstream, hname, hfmt, names = ROUTES[prov]
            billable = self.command == "POST"
            if billable and st.spent >= st.max_usd:
                st.record({"t": time.time(), "provider": prov, "path": self.path, "status": 402, "usd": 0.0})
                return self._err(402, "budget", f"bench3 key proxy cap reached: ${st.spent:.2f} >= ${st.max_usd:.2f}")
            if billable and st.max_requests and st.requests >= st.max_requests:
                st.record({"t": time.time(), "provider": prov, "path": self.path, "status": 402, "usd": 0.0})
                return self._err(402, "budget", f"bench3 key proxy request cap reached: {st.requests} >= {st.max_requests}")
            key = st.keys.get(prov)
            if not key:
                return self._err(503, "proxy", f"no {prov} key in the proxy env file ({' or '.join(names)})")
            # drop any key the client put in the query string; the proxy sends the real one as a header
            parts = urllib.parse.urlsplit(self.path[len(prefix):] or "/")
            query = urllib.parse.urlencode([(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True) if k != "key"])
            url = upstream + (parts.path or "/") + (f"?{query}" if query else "")
            hdrs = {k: v for k, v in self.headers.items() if k.lower() not in DROP}
            hdrs[hname] = hfmt.format(key)
            req = urllib.request.Request(url, data=body or None, headers=hdrs, method=self.command)
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    code, data, rh = r.status, r.read(), dict(r.headers)
            except urllib.error.HTTPError as e:
                code, data, rh = e.code, e.read(), dict(e.headers)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                st.record({"t": t0, "provider": prov, "path": parts.path, "status": 502, "error": str(e), "usd": 0.0})
                return self._err(502, "proxy", str(e))
            rec = {"t": t0, "dt": round(time.time() - t0, 3), "provider": prov, "path": parts.path, "status": code, "billable": billable, "usd": 0.0}
            try:
                j = json.loads(data)
                mu = usage_of(prov, j) if isinstance(j, dict) else None
                if mu:
                    rec["model"], rec["usage"] = mu
                    rec["usd"] = cost(prov, mu[0], mu[1])
                if code >= 400 and isinstance(j, dict):
                    rec["error"] = str(j.get("error", ""))[:300]
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            if billable or code >= 400:  # successful GETs (model lists) are free and not logged
                st.record(rec)
            keep = {k: v for k, v in rh.items() if k.lower() in ("content-type", "request-id", "x-request-id", "anthropic-ratelimit-requests-remaining", "retry-after")}
            return self._send(code, data, keep)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--max-usd", type=float, default=float(os.environ.get("FVLM_MAX_USD", 10)))
    ap.add_argument("--max-requests", type=int, default=0, help="stop after this many POSTs in the ledger (0 = no cap)")
    ap.add_argument("--env", "--keys", dest="env", default=str(pathlib.Path.home() / "work/bench3/.env"))
    ap.add_argument("--ledger", default=str(pathlib.Path.home() / ".local/share/bench3/ledger.jsonl"))
    a = ap.parse_args()
    keys = load_keys(pathlib.Path(a.env))
    st = State(keys, a.max_usd, a.max_requests, pathlib.Path(a.ledger).expanduser())
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st))
    cap = f", {st.requests} of {a.max_requests} requests" if a.max_requests else ""
    print(f"keyproxy on 127.0.0.1:{a.port}  keys: {', '.join(sorted(keys))}  spent ${st.spent:.2f} of ${a.max_usd:.2f}{cap}  ledger {a.ledger}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
