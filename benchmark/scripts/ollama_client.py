"""
ollama_client.py — Thin wrapper around the Ollama HTTP API (`/api/chat`) that:

* Issues a single non-streaming `chat` request so we get a complete response
  containing the official timing fields (`total_duration`, `load_duration`,
  `prompt_eval_duration`, `eval_duration`, `eval_count`) per md §7.
* Supports the **top-level `think: false` toggle** required by Ollama 0.23+
  for thinking-capable models (qwen3.5, qwen3-vl, qwen3-thinking,
  deepseek-r1, lfm2.5-thinking, etc.). The `ollama-python` SDK does NOT expose
  this field — it must be set at the JSON top level of the request body.
* Returns the **raw** response object so callers can persist every field
  verbatim. Per md §17 we must keep `response_text` and every performance
  field, not just the derived metrics.
* Never touches wall-clock time on our side for the throughput denominator —
  the denominator is always `eval_duration` returned by Ollama (md §7).

Per-call audit fields added for the v2 orchestrated benchmark:
  * `client_request_sent_unix`     wall-clock just before urlopen
  * `client_response_received_unix` wall-clock just after urlopen returns
  * `client_total_wall_ms`         response - request
  * `client_queue_wait_ms`         first byte delta - request_sent
  * `client_response_headers`      dict of HTTP response headers

Lifecycle helpers (used by the orchestrator between models):
  * `list_loaded_models()`         GET /api/ps — which models are loaded now
  * `stop_model(model)`            unload a model by sending a keep_alive=0
                                   request, equivalent to `ollama stop`
  * `wait_for_model_unloaded(...)` poll /api/ps until the model is gone
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import ollama  # kept only for is_model_available() — uses `ollama list`


_THINKING_CAP_CACHE: dict[str, bool] = {}


def _supports_thinking(model: str) -> bool:
    """Return True iff Ollama reports this model has 'thinking' capability.

    Cached per-process. `think=true` for a non-thinking model → HTTP 400;
    `think=false` for a thinking model → silent CoT disable. The safe
    default is to omit `think` entirely when the model doesn't advertise it.
    """
    if model in _THINKING_CAP_CACHE:
        return _THINKING_CAP_CACHE[model]
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=json.dumps({"name": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        caps = body.get("capabilities", []) or []
        is_thinking = "thinking" in caps
    except Exception:
        is_thinking = False
    _THINKING_CAP_CACHE[model] = is_thinking
    return is_thinking


@dataclass
class OllamaCallResult:
    """Wrap an Ollama `chat` response with the fields we need."""

    model: str
    response_text: str
    raw_response: dict
    thinking_text: str
    # the four durations in nanoseconds
    total_duration_ns: Optional[int]
    load_duration_ns: Optional[int]
    prompt_eval_duration_ns: Optional[int]
    eval_duration_ns: Optional[int]
    eval_count: Optional[int]
    prompt_eval_count: Optional[int]
    done_reason: Optional[str]
    # client-side audit fields (new in v2)
    client_request_sent_unix: float = 0.0
    client_response_received_unix: float = 0.0
    client_total_wall_ms: float = 0.0
    client_first_byte_unix: float = 0.0
    client_queue_wait_ms: float = 0.0
    client_response_headers: dict = field(default_factory=dict)
    client_http_status: int = 0


def _http_post(
    url: str, payload: dict, timeout_s: int
) -> tuple[bytes, float, float, float, dict, int]:
    """POST JSON, return (body, request_sent_unix, first_byte_unix,
    response_received_unix, headers_dict, http_status).

    `first_byte_unix` is captured by reading the response object lazily;
    urllib delivers headers/body in one shot so we approximate by stamping
    `first_byte_unix` right after urlopen returns.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    request_sent = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        first_byte = time.time()
        body = resp.read()
        response_received = time.time()
        headers = {k: v for k, v in resp.headers.items()}
        status = resp.status
    return body, request_sent, first_byte, response_received, headers, status


def chat(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    seed: int = 42,
    num_ctx: int = 4096,
    num_predict: int = 128,
    keep_alive: str = "5m",
    timeout_s: int = 600,
    think: Optional[bool] = None,
    ollama_host: str = "http://localhost:11434",
) -> OllamaCallResult:
    """Single Ollama chat call. Returns the raw response as a dict.

    `think` defaults to False for known thinking models (so `qwen3.5`,
    `deepseek-r1`, `lfm2.5-thinking`, etc. produce content instead of
    spending the entire `num_predict` budget on a hidden chain of thought).
    Pass `think=True` explicitly if you want the chain of thought back.
    """
    is_thinking = _supports_thinking(model)
    if think is None:
        think = False if is_thinking else True

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "keep_alive": keep_alive,
    }
    if is_thinking:
        payload["think"] = think

    try:
        body, sent_unix, first_unix, recv_unix, headers, status = _http_post(
            f"{ollama_host.rstrip('/')}/api/chat", payload, timeout_s
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama HTTP call failed for {model!r}: {exc}") from exc

    raw = json.loads(body)
    msg = raw.get("message", {}) or {}
    text = msg.get("content", "") or ""
    thinking_text = msg.get("thinking", "") or ""

    if thinking_text and thinking_text in text:
        text = text.replace(thinking_text, "").strip()
    else:
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"</?think>", "", text).strip()

    return OllamaCallResult(
        model=raw.get("model", model),
        response_text=text,
        raw_response=raw,
        thinking_text=thinking_text,
        total_duration_ns=raw.get("total_duration"),
        load_duration_ns=raw.get("load_duration"),
        prompt_eval_duration_ns=raw.get("prompt_eval_duration"),
        eval_duration_ns=raw.get("eval_duration"),
        eval_count=raw.get("eval_count"),
        prompt_eval_count=raw.get("prompt_eval_count"),
        done_reason=raw.get("done_reason"),
        client_request_sent_unix=sent_unix,
        client_response_received_unix=recv_unix,
        client_total_wall_ms=(recv_unix - sent_unix) * 1000.0,
        client_first_byte_unix=first_unix,
        client_queue_wait_ms=(first_unix - sent_unix) * 1000.0,
        client_response_headers=headers,
        client_http_status=status,
    )


def is_model_available(model: str) -> bool:
    try:
        listed = ollama.list()
        models = listed.get("models", []) if isinstance(listed, dict) else listed.models
    except Exception:
        return False
    for m in models:
        name = m.get("name") if isinstance(m, dict) else getattr(m, "model", None) or getattr(m, "name", None)
        if name == model or (name and name.split(":")[0] == model.split(":")[0]):
            return True
    return False


def list_loaded_models(
    ollama_host: str = "http://localhost:11434",
    timeout_s: int = 5,
) -> list[dict]:
    """Return currently-loaded models from `/api/ps`.

    Each entry: {"name": ..., "size": ..., "size_vram": ..., "expires_at": ...}.
    Returns [] on transport error so the caller can decide whether to retry.
    """
    try:
        req = urllib.request.Request(
            f"{ollama_host.rstrip('/')}/api/ps",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read())
        return list(body.get("models", []) or [])
    except Exception:
        return []


def stop_model(
    model: str,
    *,
    ollama_host: str = "http://localhost:11434",
    timeout_s: int = 10,
) -> bool:
    """Unload a model by issuing a `generate` request with `keep_alive=0`.

    Equivalent to `ollama stop <model>` on the CLI. Returns True if Ollama
    acknowledged, False on transport error. Idempotent — calling it on an
    already-unloaded model is a no-op that still returns True.
    """
    payload = {
        "model": model,
        "prompt": "",
        "keep_alive": 0,
        "stream": False,
    }
    try:
        body, _, _, _, _, _ = _http_post(
            f"{ollama_host.rstrip('/')}/api/generate", payload, timeout_s
        )
        return True
    except Exception:
        return False


def wait_for_model_unloaded(
    model: str,
    *,
    poll_interval_s: float = 1.0,
    timeout_s: float = 60.0,
    ollama_host: str = "http://localhost:11434",
) -> tuple[bool, float]:
    """Poll `/api/ps` until `model` is gone. Returns (success, elapsed_seconds).

    `success=True` if model unloaded within timeout, else False. The caller
    is expected to log a warning but not abort on timeout — the next model
    load will overwrite VRAM anyway.
    """
    t0 = time.time()
    while True:
        loaded = list_loaded_models(ollama_host)
        names = {m.get("name") for m in loaded}
        if model not in names and not any(
            model.split(":")[0] == n.split(":")[0] for n in names if n
        ):
            return True, time.time() - t0
        if time.time() - t0 > timeout_s:
            return False, time.time() - t0
        time.sleep(poll_interval_s)


if __name__ == "__main__":
    print("loaded:", list_loaded_models())
    for m in ["qwen3:0.6b", "llama3.2:1b"]:
        ok = stop_model(m)
        print(f"stop {m}: {ok}")
    print("loaded after stop:", list_loaded_models())