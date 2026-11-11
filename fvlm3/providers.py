"""Model providers. Each takes one request and returns a parsed reply dict.

claude : Anthropic Messages API (SDK), structured output (JSON schema), images as base64 JPEG.
         Credentials: ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL.
openai : OpenAI Chat Completions over plain HTTP (stdlib), response_format json_schema (strict).
         Credentials: OPENAI_API_KEY / OPENAI_BASE_URL (default https://api.openai.com/v1).
gemini : Gemini generateContent over plain HTTP (stdlib), responseJsonSchema.
         Credentials: GEMINI_API_KEY / GEMINI_BASE_URL (default https://generativelanguage.googleapis.com).
mock   : free, deterministic, simulated latency: drives toward the farthest open floor.

In the container all three point at the host key proxy (tools/keyproxy.py), which holds the real
keys; the container only gets dummy keys. The prompt, images and schema are the same for every
provider; only the wire format differs.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import os
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

_log = logging.getLogger("fvlm3.provider")


@dataclasses.dataclass
class Request:
    system: str
    text: str
    images: list[np.ndarray]  # BGR uint8
    schema: dict
    # for the mock only
    depth: np.ndarray | None = None
    cam: object | None = None


@dataclasses.dataclass
class Reply:
    ok: bool
    data: dict
    latency_s: float
    usage: dict = dataclasses.field(default_factory=dict)
    model: str = ""
    error: str = ""
    stop_reason: str = ""


def _b64(img: np.ndarray, quality: int = 88) -> tuple[str, str]:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return "image/jpeg", base64.standard_b64encode(buf.tobytes()).decode()


class ClaudeProvider:
    def __init__(self, model: str, effort: str = "low", max_tokens: int = 8000, timeout_s: float = 30.0, fallbacks: str = "default"):
        import anthropic  # noqa: PLC0415

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=1)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks if fallbacks and fallbacks != "off" else ""

    def __call__(self, req: Request) -> Reply:
        content: list[dict] = []
        for img in req.images:
            mt, data = _b64(img)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        content.append({"type": "text", "text": req.text})
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": req.schema}},
        )
        t0 = time.monotonic()
        try:
            if self.fallbacks:
                try:
                    resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks=self.fallbacks, **kwargs)
                except TypeError:  # SDK release without the typed `fallbacks` argument
                    resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": self.fallbacks}, **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as e:
            return Reply(False, {}, time.monotonic() - t0, error=f"rate_limit: {e}")
        except self._anthropic.APIStatusError as e:
            return Reply(False, {}, time.monotonic() - t0, error=f"http_{e.status_code}: {str(e)[:300]}")
        except self._anthropic.APIConnectionError as e:
            return Reply(False, {}, time.monotonic() - t0, error=f"connection: {e}")
        dt = time.monotonic() - t0
        u = resp.usage
        usage = {
            "input": getattr(u, "input_tokens", 0) or 0,
            "output": getattr(u, "output_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        if resp.stop_reason == "refusal":
            return Reply(False, {}, dt, usage, resp.model, error="refusal", stop_reason="refusal")
        if resp.stop_reason == "max_tokens":
            return Reply(False, {}, dt, usage, resp.model, error="max_tokens", stop_reason="max_tokens")
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return Reply(False, {}, dt, usage, resp.model, error=f"bad_json: {e}", stop_reason=resp.stop_reason or "")
        return Reply(True, data, dt, usage, resp.model, stop_reason=resp.stop_reason or "")


TRANSIENT = {0, 500, 502, 503, 504}


def _http_json(url: str, headers: dict, body: dict, timeout_s: float, retries: int = 1) -> tuple[int, dict, str]:
    """POST JSON; returns (status, parsed body or {}, error text). status 0 = no connection.
    Retries once on a connection error / 5xx (DNS hiccups, overloaded upstream)."""
    for k in range(retries + 1):
        code, j, err = _http_json_once(url, headers, body, timeout_s)
        if code not in TRANSIENT or k == retries:
            return code, j, err
        _log.warning("transient error %s (%s); retrying once", code, err[:100])
        time.sleep(0.5)
    return code, j, err


def _http_json_once(url: str, headers: dict, body: dict, timeout_s: float) -> tuple[int, dict, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw, code = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {}, f"connection: {e}"
    try:
        j = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return code, {}, raw[:300].decode(errors="replace")
    err = ""
    if code >= 400:
        e = j.get("error", j) if isinstance(j, dict) else j
        err = (e.get("message") if isinstance(e, dict) else None) or json.dumps(e)[:300]
    return code, j if isinstance(j, dict) else {}, err


def _http_error(code: int, msg: str) -> str:
    if code == 0:
        return msg
    if code == 429:
        return f"rate_limit: {msg[:300]}"
    return f"http_{code}: {msg[:300]}"


def _parse_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):  # tolerate a fenced answer
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    return json.loads(t)


def _strip_keys(schema, drop=("additionalProperties",)):
    if isinstance(schema, dict):
        return {k: _strip_keys(v, drop) for k, v in schema.items() if k not in drop}
    if isinstance(schema, list):
        return [_strip_keys(v, drop) for v in schema]
    return schema


class OpenAIProvider:
    """Chat Completions; same system prompt, images and JSON schema as Claude."""

    def __init__(self, model: str, effort: str = "low", max_tokens: int = 8000, timeout_s: float = 30.0):
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.model, self.effort, self.max_tokens, self.timeout_s = model, effort, max_tokens, timeout_s
        self.use_effort = bool(effort) and effort != "off"
        self.strict = True

    def _body(self, req: Request) -> dict:
        content: list[dict] = []
        for img in req.images:
            mt, data = _b64(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}})
        content.append({"type": "text", "text": req.text})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": content}],
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "reply", "schema": req.schema, "strict": self.strict}},
        }
        if self.use_effort:
            body["reasoning_effort"] = self.effort
        return body

    def __call__(self, req: Request) -> Reply:
        t0 = time.monotonic()
        hdr = {"authorization": f"Bearer {self.key}"}
        for _ in range(3):
            code, j, err = _http_json(f"{self.base}/chat/completions", hdr, self._body(req), self.timeout_s)
            if code == 400 and self.use_effort and "reasoning_effort" in err:
                _log.warning("openai: %s rejects reasoning_effort=%s; retrying without", self.model, self.effort)
                self.use_effort = False
                continue
            if code == 400 and self.strict and ("strict" in err or "json_schema" in err):
                _log.warning("openai: strict schema rejected (%s); retrying non-strict", err[:120])
                self.strict = False
                continue
            break
        dt = time.monotonic() - t0
        if code != 200:
            return Reply(False, {}, dt, error=_http_error(code, err))
        u = j.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        usage = {"input": (u.get("prompt_tokens") or 0) - cached, "output": u.get("completion_tokens") or 0, "cache_read": cached, "cache_write": 0}
        model = j.get("model", self.model)
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        fin = ch.get("finish_reason") or ""
        if msg.get("refusal"):
            return Reply(False, {}, dt, usage, model, error=f"refusal: {str(msg['refusal'])[:200]}", stop_reason="refusal")
        if fin == "length":
            return Reply(False, {}, dt, usage, model, error="max_tokens", stop_reason="max_tokens")
        if fin == "content_filter":
            return Reply(False, {}, dt, usage, model, error="refusal: content_filter", stop_reason="refusal")
        try:
            data = _parse_json(msg.get("content") or "")
        except json.JSONDecodeError as e:
            return Reply(False, {}, dt, usage, model, error=f"bad_json: {e}", stop_reason=fin)
        return Reply(True, data, dt, usage, model, stop_reason=fin)


class GeminiProvider:
    """generateContent; same system prompt, images and JSON schema as Claude."""

    REFUSE = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}

    def __init__(self, model: str, effort: str = "low", max_tokens: int = 8000, timeout_s: float = 30.0):
        self.base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        self.model, self.effort, self.max_tokens, self.timeout_s = model, effort, max_tokens, timeout_s
        self.use_thinking = bool(effort) and effort != "off"
        self.json_schema = True  # responseJsonSchema; False = legacy responseSchema (OpenAPI subset)

    def _thinking(self) -> dict:
        if self.model.startswith("gemini-2"):  # 2.x: token budget; -1 = dynamic
            return {"thinkingBudget": {"minimal": 128, "low": 1024, "medium": 4096}.get(self.effort, -1)}
        return {"thinkingLevel": self.effort}

    def _body(self, req: Request) -> dict:
        parts: list[dict] = []
        for img in req.images:
            mt, data = _b64(img)
            parts.append({"inline_data": {"mime_type": mt, "data": data}})
        parts.append({"text": req.text})
        gen: dict = {"maxOutputTokens": self.max_tokens, "responseMimeType": "application/json"}
        if self.json_schema:
            gen["responseJsonSchema"] = req.schema
        else:
            gen["responseSchema"] = _strip_keys(req.schema)
        if self.use_thinking:
            gen["thinkingConfig"] = self._thinking()
        return {"systemInstruction": {"parts": [{"text": req.system}]}, "contents": [{"role": "user", "parts": parts}], "generationConfig": gen}

    def __call__(self, req: Request) -> Reply:
        t0 = time.monotonic()
        url = f"{self.base}/v1beta/models/{self.model}:generateContent"
        hdr = {"x-goog-api-key": self.key}
        for _ in range(3):
            code, j, err = _http_json(url, hdr, self._body(req), self.timeout_s)
            low = err.lower()
            if code == 400 and self.use_thinking and "thinking" in low:
                _log.warning("gemini: %s rejects thinking config (%s); retrying without", self.model, err[:120])
                self.use_thinking = False
                continue
            if code == 400 and self.json_schema and ("responsejsonschema" in low or "response_json_schema" in low):
                _log.warning("gemini: responseJsonSchema rejected (%s); retrying with responseSchema", err[:120])
                self.json_schema = False
                continue
            break
        dt = time.monotonic() - t0
        if code != 200:
            return Reply(False, {}, dt, error=_http_error(code, err))
        u = j.get("usageMetadata") or {}
        cached = u.get("cachedContentTokenCount") or 0
        usage = {"input": (u.get("promptTokenCount") or 0) - cached, "output": (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0), "cache_read": cached, "cache_write": 0}
        model = j.get("modelVersion") or self.model
        block = (j.get("promptFeedback") or {}).get("blockReason")
        if block:
            return Reply(False, {}, dt, usage, model, error=f"refusal: prompt blocked ({block})", stop_reason="refusal")
        cand = (j.get("candidates") or [{}])[0]
        fin = cand.get("finishReason") or ""
        if fin in self.REFUSE:
            return Reply(False, {}, dt, usage, model, error=f"refusal: {fin}", stop_reason="refusal")
        if fin == "MAX_TOKENS":
            return Reply(False, {}, dt, usage, model, error="max_tokens", stop_reason="max_tokens")
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
        try:
            data = _parse_json(text)
        except json.JSONDecodeError as e:
            return Reply(False, {}, dt, usage, model, error=f"bad_json: {e}", stop_reason=fin)
        return Reply(True, data, dt, usage, model, stop_reason=fin)


class MockProvider:
    """Heuristic stand-in: aims at the deepest floor column; turns when blocked."""

    def __init__(self, latency_s: float = 2.5, seed: int = 0):
        self.latency_s = latency_s
        self.rng = np.random.default_rng(seed)
        self.model = "mock"

    def __call__(self, req: Request) -> Reply:
        t0 = time.monotonic()
        time.sleep(max(0.0, self.latency_s * float(self.rng.uniform(0.7, 1.3))))
        cam = req.cam
        depth = req.depth
        props = req.schema.get("properties", {})
        if "target" in props and "route" in props:  # vlnav MISSION: the target room named in the task, route = its door
            zones = props["target"]["enum"]
            names = props["route"]["items"]["enum"]
            low = req.text.splitlines()[0].lower().replace(" ", "_")  # the task line only
            target = next((z for z in sorted(zones, key=len, reverse=True) if z in low or z.replace("_", "") in low.replace("_", "")), zones[0])
            door = next((n for n in names if n.startswith(target) and n.endswith("door")), None) or next((n for n in names if target.split("_")[0] in n and "door" in n), None)
            route = [n for n in ("central_hallway", door, target) if n]
            return Reply(True, {"interpretation": f"mock: go to {target}", "target": target, "route": route, "subtasks": ["mock"], "why": "mock"}, time.monotonic() - t0, model="mock")
        if "choice" in props:  # vlnav WAYPOINT: follow the route hint
            import re  # noqa: PLC0415
            m = re.search(r"route follower would pick (\d+)", req.text)
            inside = "(the target room)" in req.text
            data = {"scene": "mock", "people": [], "action": "arrived" if inside else ("goto" if m else "turn_left"), "choice": int(m.group(1)) if (m and not inside) else 0, "expects": "mock", "why": "mock: route hint"}
            if "rules_applied" in props:
                data["rules_applied"] = []
            return Reply(True, data, time.monotonic() - t0, model="mock")
        data = {"scene": "mock", "progress": "searching", "action": "goto", "subgoals": [], "path_reasoning": "mock: farthest open floor"}
        if "people" in props:
            data["people"] = []
        if "heading_to" in props:
            data["heading_to"] = "unknown"
        if "rules_applied" in props:
            data["rules_applied"] = []
        best = None
        if cam is not None and depth is not None and depth.size:
            # rows below the horizon; for each column the floor distance at a fixed row
            for un in range(150, 851, 50):
                for vn in (620, 700, 800):
                    u, v = cam.norm_to_px(un, vn)
                    d = cam.sample_depth(depth, u, v)
                    p = cam.pixel_to_robot(u, v, depth)
                    if p.source in ("depth", "floor", "depth_clamped") and d is not None:
                        score = min(p.range_m, 5.0) - 0.004 * abs(un - 500)
                        if best is None or score > best[0]:
                            best = (score, un, vn, p.range_m)
        if best is None or best[3] < 0.8:
            data["action"] = "turn_left"
        else:
            _, un, vn, _ = best
            data["subgoals"] = [{"u": int(un), "v": int(vn), "why": "open floor"}]
        return Reply(True, data, time.monotonic() - t0, model="mock")


DEFAULT_MODEL = {"claude": "claude-opus-5-5", "openai": "gpt-5.5", "gemini": "gemini-3.1-pro-preview"}


def make_provider(cfg: dict):
    kind = str(cfg.get("provider", "claude"))
    if kind == "mock":
        return MockProvider(float(cfg.get("mock_latency_s", 2.5)))
    if kind not in DEFAULT_MODEL:
        raise ValueError(f"unknown provider {kind!r} (claude | openai | gemini | mock)")
    model = str(cfg.get("model") or "") or DEFAULT_MODEL[kind]
    cfg["model"] = model
    common = dict(model=model, effort=str(cfg.get("effort", "low")), max_tokens=int(cfg.get("max_tokens", 8000)), timeout_s=float(cfg.get("timeout_s", 30)))
    if kind == "claude":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_BASE_URL")):
            _log.warning("no ANTHROPIC_* credentials in the environment; calls will fail (start tools/keyproxy.py, see README)")
        return ClaudeProvider(fallbacks=str(cfg.get("fallbacks", "default")), **common)
    env = {"openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"), "gemini": ("GEMINI_API_KEY", "GEMINI_BASE_URL")}[kind]
    if not any(os.environ.get(v) for v in env):
        _log.warning("no %s / %s in the environment; calls will fail (start tools/keyproxy.py, see README)", *env)
    return (OpenAIProvider if kind == "openai" else GeminiProvider)(**common)


# USD per million tokens (input, output, cache read, cache write), for the per-episode ESTIMATE in
# the logs; longest prefix wins. The proxy ledger and the provider billing pages are authoritative.
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
FALLBACK_PRICES = {"claude": (4.0, 20.0, 0.20, 5.0), "gpt": (5.0, 40.0, 0.5, 0.0), "o": (5.0, 40.0, 0.5, 0.0), "gemini": (4.0, 18.0, 0.4, 0.0)}


def cost_usd(model: str, usage: dict) -> float:
    if model == "mock":
        return 0.0
    best = max((k for k in PRICES if model.startswith(k)), key=len, default=None)
    if best:
        p = PRICES[best]
    else:
        p = next((v for k, v in FALLBACK_PRICES.items() if model.startswith(k)), FALLBACK_PRICES["claude"])
    pin, pout, pread, pwrite = p
    return (usage.get("input", 0) * pin + usage.get("output", 0) * pout + usage.get("cache_read", 0) * pread + usage.get("cache_write", 0) * pwrite) / 1e6


__all__ = ["ClaudeProvider", "GeminiProvider", "MockProvider", "OpenAIProvider", "Reply", "Request", "cost_usd", "make_provider"]
