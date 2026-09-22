"""
model_orchestrator.py — Lifecycle manager for one model run.

A ModelRun wraps everything the benchmark needs to drive one model through
all of its languages and samples in a controlled, repeatable way:

  * `preflight()`      assert Ollama is reachable, no other model is loaded,
                       and the requested model is in the local model store.
  * `ensure_loaded()`  trigger a tiny `keep_alive="5m"` warm-up call so
                       Ollama actually brings the model into VRAM. Times the
                       load_duration.
  * `unload()`         ollama stop + poll /api/ps until the model is gone
                       (capped; logs a warning on timeout but does not raise).
  * `invoke(...)`      one chat call wrapped in client-side timing,
                       pre/post telemetry, retry-with-backoff. Returns the
                       raw JSON payload (including the `host` audit block).
  * `summary()`        per-model rollup (cells attempted, cells saved,
                       retries, peak VRAM, total wall time).

The orchestrator is intentionally *stateless* across models — its only
mutable state is `self.peak_vram_mib`, `self.cells_attempted`,
`self.cells_saved`, `self.cells_failed`, `self.retry_count`,
`self.load_duration_ns`, which get reset by `reset_metrics()` when a new
model takes over.

The point of putting all this in one place is that `run_benchmark.py` no
longer has to know about Ollama lifecycle, retries, or telemetry — it just
calls `orchestrator.invoke(...)` for each cell and `orchestrator.unload()`
between models.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from scripts.host_telemetry import host_snapshot
from scripts.ollama_client import (
    OllamaCallResult,
    chat,
    is_model_available,
    list_loaded_models,
    stop_model,
    wait_for_model_unloaded,
)


LogFn = Callable[[str], None]


@dataclass
class ModelRunSummary:
    model: str
    started_unix: float = 0.0
    ended_unix: float = 0.0
    cells_attempted: int = 0
    cells_saved: int = 0
    cells_failed: int = 0
    cells_skipped: int = 0
    retries: int = 0
    warmups_attempted: int = 0
    warmups_failed: int = 0
    peak_vram_mib: float = 0.0
    initial_load_duration_ns: Optional[int] = None
    unload_succeeded: bool = False
    unload_elapsed_s: float = 0.0
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


class ModelOrchestrator:
    """Drive one model through the full experiment matrix.

    Usage:
        orch = ModelOrchestrator(model="qwen3:0.6b", log=print)
        orch.preflight()
        orch.ensure_loaded()
        for lang in languages:
            for sample in samples:
                payload = orch.invoke(lang=lang, sample_id=..., repeat=r, ...)
                save_raw(...)
        orch.unload(timeout_s=60)
        save_model_summary(orch.summary())
    """

    def __init__(
        self,
        model: str,
        log: LogFn,
        *,
        retry_attempts: int = 3,
        retry_backoff_s: float = 2.0,
        unload_timeout_s: float = 60.0,
        unload_poll_interval_s: float = 1.0,
    ) -> None:
        self.model = model
        self.log = log
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_s = retry_backoff_s
        self.unload_timeout_s = unload_timeout_s
        self.unload_poll_interval_s = unload_poll_interval_s

        self.peak_vram_mib = 0.0
        self.cells_attempted = 0
        self.cells_saved = 0
        self.cells_failed = 0
        self.cells_skipped = 0
        self.retry_count = 0
        self.warmups_attempted = 0
        self.warmups_failed = 0
        self.initial_load_duration_ns: Optional[int] = None
        self.unload_succeeded = False
        self.unload_elapsed_s = 0.0
        self.errors: list[dict] = []
        self._started_unix = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle phases
    # ------------------------------------------------------------------ #
    def preflight(self) -> dict:
        """Sanity-check Ollama is up and `model` exists in its store.

        Returns a dict with `loaded_before` (what was loaded before we
        started) so the orchestrator can decide whether an explicit unload
        is required at the end.
        """
        loaded = list_loaded_models()
        loaded_names = [m.get("name") for m in loaded]
        available = is_model_available(self.model)
        snap = host_snapshot()
        vram = _peak_vram(snap)
        if vram > self.peak_vram_mib:
            self.peak_vram_mib = vram
        if not available:
            raise RuntimeError(
                f"preflight failed: {self.model!r} not in `ollama list`. "
                f"Pull it first or remove it from config.yaml."
            )
        self.log(
            f"  [preflight] {self.model!r} available={available} "
            f"VRAM={vram:.0f}MiB loaded_before={[n for n in loaded_names if n]}"
        )
        return {
            "loaded_before": [m for m in loaded if m.get("name")],
            "vram_mib": vram,
        }

    def ensure_loaded(self) -> Optional[int]:
        """Force Ollama to load the model by issuing a tiny chat call.

        Returns the `load_duration` (ns) Ollama spent loading. Records the
        model as loaded for the lifetime of this orchestrator instance.
        """
        self.warmups_attempted += 1
        snap_pre = host_snapshot()
        vram_pre = _peak_vram(snap_pre)
        if vram_pre > self.peak_vram_mib:
            self.peak_vram_mib = vram_pre
        try:
            result = chat(
                model=self.model,
                system="",
                user="ok",
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                seed=42,
                num_ctx=512,
                num_predict=1,
                keep_alive="5m",
                timeout_s=600,
                think=False,
            )
        except Exception as exc:
            self.warmups_failed += 1
            self.errors.append({
                "phase": "ensure_loaded",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_clock_unix": time.time(),
            })
            self.log(f"  [ensure_loaded] FAILED: {exc}")
            return None
        load_ns = result.load_duration_ns
        if load_ns is None or load_ns == 0:
            load_ns = result.total_duration_ns
        self.initial_load_duration_ns = load_ns
        snap_post = host_snapshot()
        vram_post = _peak_vram(snap_post)
        if vram_post > self.peak_vram_mib:
            self.peak_vram_mib = vram_post
        self.log(
            f"  [ensure_loaded] ok — load_duration={_ns_to_ms(load_ns):.0f}ms "
            f"VRAM: {vram_pre:.0f} -> {vram_post:.0f} MiB"
        )
        return load_ns

    def unload(self) -> bool:
        """Stop the model in Ollama, then poll until it's gone from /api/ps.

        Logs a warning on timeout but never raises — the next model load
        will overwrite VRAM anyway.
        """
        loaded_before = {m.get("name") for m in list_loaded_models()}
        if self.model not in loaded_before and not any(
            self.model.split(":")[0] == n.split(":")[0]
            for n in loaded_before if n
        ):
            self.unload_succeeded = True
            self.unload_elapsed_s = 0.0
            self.log(f"  [unload] {self.model!r} not loaded — skip")
            return True
        t0 = time.time()
        ok = stop_model(self.model)
        if not ok:
            self.log(f"  [unload] stop_model({self.model!r}) returned False")
        success, elapsed = wait_for_model_unloaded(
            self.model,
            poll_interval_s=self.unload_poll_interval_s,
            timeout_s=self.unload_timeout_s,
        )
        self.unload_succeeded = success
        self.unload_elapsed_s = elapsed
        snap_post = host_snapshot()
        vram_post = _peak_vram(snap_post)
        self.log(
            f"  [unload] {self.model!r} "
            f"success={success} elapsed={elapsed:.1f}s VRAM_after={vram_post:.0f}MiB"
        )
        if not success:
            self.errors.append({
                "phase": "unload",
                "error": f"model still loaded after {elapsed:.1f}s",
                "wall_clock_unix": time.time(),
            })
        return success

    # ------------------------------------------------------------------ #
    # Per-call invocation (with telemetry + retry)
    # ------------------------------------------------------------------ #
    def invoke(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        num_ctx: int,
        num_predict: int,
        keep_alive: str,
        timeout_s: int,
        think: Optional[bool],
        prompt_user_text: str,
        prompt_source_text: str,
        sample_id: str,
        repeat: int,
        language: str,
        experiment: str,
        extra_options: Optional[dict] = None,
    ) -> Optional[dict]:
        """One Ollama chat call, wrapped in telemetry + retry.

        Returns the raw payload (caller persists it) or None if every attempt
        failed. Increments `cells_attempted` / `cells_saved` / `cells_failed`
        / `retry_count`.
        """
        self.cells_attempted += 1
        snap_pre = host_snapshot()
        vram_pre = _peak_vram(snap_pre)
        if vram_pre > self.peak_vram_mib:
            self.peak_vram_mib = vram_pre

        last_err: Optional[Exception] = None
        for try_idx in range(self.retry_attempts):
            try:
                t_send = time.time()
                result: OllamaCallResult = chat(
                    model=self.model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    keep_alive=keep_alive,
                    timeout_s=timeout_s,
                    think=think,
                )
                snap_post = host_snapshot()
                vram_post = _peak_vram(snap_post)
                if vram_post > self.peak_vram_mib:
                    self.peak_vram_mib = vram_post
                t_recv = time.time()

                options_snapshot = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "seed": seed,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                    **(extra_options or {}),
                }
                payload = {
                    "experiment": experiment,
                    "model": self.model,
                    "language": language,
                    "sample_id": sample_id,
                    "repeat": repeat,
                    "prompt": {
                        "system": system,
                        "user": prompt_user_text,
                        "source_text": prompt_source_text,
                    },
                    "options": options_snapshot,
                    "response_text": result.response_text,
                    "ollama": {
                        "model": result.model,
                        "total_duration": result.total_duration_ns,
                        "load_duration": result.load_duration_ns,
                        "prompt_eval_duration": result.prompt_eval_duration_ns,
                        "eval_duration": result.eval_duration_ns,
                        "eval_count": result.eval_count,
                        "prompt_eval_count": result.prompt_eval_count,
                        "done_reason": result.done_reason,
                    },
                    "raw_response": result.raw_response,
                    "host": {
                        "pre_call": {
                            "timestamp_unix": snap_pre["timestamp_unix"],
                            "nvidia_smi": snap_pre["nvidia_smi"],
                            "psutil": snap_pre["psutil"],
                            "proc": snap_pre["proc"],
                        },
                        "post_call": {
                            "timestamp_unix": snap_post["timestamp_unix"],
                            "nvidia_smi": snap_post["nvidia_smi"],
                            "psutil": snap_post["psutil"],
                            "proc": snap_post["proc"],
                        },
                        "client": {
                            "request_sent_unix": result.client_request_sent_unix,
                            "response_received_unix": result.client_response_received_unix,
                            "first_byte_unix": result.client_first_byte_unix,
                            "total_wall_ms": result.client_total_wall_ms,
                            "queue_wait_ms": result.client_queue_wait_ms,
                            "http_status": result.client_http_status,
                            "response_headers": result.client_response_headers,
                            "payload_received_unix": t_recv,
                            "vram_delta_mib": vram_post - vram_pre,
                        },
                        "retry": {
                            "attempt": try_idx + 1,
                            "max_attempts": self.retry_attempts,
                            "wall_clock_request_unix": t_send,
                        },
                    },
                    "wall_clock_unix": t_recv,
                }
                self.cells_saved += 1
                return payload
            except Exception as exc:
                last_err = exc
                if try_idx < self.retry_attempts - 1:
                    self.retry_count += 1
                    backoff = self.retry_backoff_s * (2 ** try_idx)
                    self.log(
                        f"  retry {try_idx + 1}/{self.retry_attempts - 1} "
                        f"after {backoff:.1f}s: {exc}"
                    )
                    time.sleep(backoff)
                else:
                    break

        self.cells_failed += 1
        err_record = {
            "phase": "invoke",
            "sample_id": sample_id,
            "repeat": repeat,
            "language": language,
            "error": str(last_err) if last_err else "unknown",
            "traceback": traceback.format_exc(),
            "wall_clock_unix": time.time(),
        }
        self.errors.append(err_record)
        snap_post = host_snapshot()
        self.peak_vram_mib = max(
            self.peak_vram_mib, _peak_vram(snap_post)
        )
        self.log(
            f"  [invoke] FAILED sample={sample_id} r={repeat} "
            f"lang={language}: {last_err}"
        )
        return None

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def start_timer(self) -> None:
        self._started_unix = time.time()

    def summary(self) -> ModelRunSummary:
        return ModelRunSummary(
            model=self.model,
            started_unix=self._started_unix,
            ended_unix=time.time(),
            cells_attempted=self.cells_attempted,
            cells_saved=self.cells_saved,
            cells_failed=self.cells_failed,
            cells_skipped=self.cells_skipped,
            retries=self.retry_count,
            warmups_attempted=self.warmups_attempted,
            warmups_failed=self.warmups_failed,
            peak_vram_mib=self.peak_vram_mib,
            initial_load_duration_ns=self.initial_load_duration_ns,
            unload_succeeded=self.unload_succeeded,
            unload_elapsed_s=self.unload_elapsed_s,
            errors=list(self.errors),
        )

    def record_skipped(self, n: int = 1) -> None:
        self.cells_skipped += n


# --------------------------------------------------------------------------- #
def _peak_vram(snap: dict) -> float:
    gpus = (snap.get("nvidia_smi") or {}).get("gpus") or []
    if not gpus:
        return 0.0
    return float(gpus[0].get("vram_used_mib") or 0.0)


def _ns_to_ms(ns: Optional[int]) -> float:
    if ns is None:
        return 0.0
    return ns / 1e6


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "granite4:350m-h"
    orch = ModelOrchestrator(model=target, log=lambda s: print(s, flush=True))
    print(json.dumps(orch.preflight(), indent=2, ensure_ascii=False, default=str))
    orch.ensure_loaded()
    payload = orch.invoke(
        system="You are a helpful assistant.",
        user="Say hi in 5 words.",
        temperature=0.0, top_p=1.0, top_k=1, seed=42,
        num_ctx=512, num_predict=32, keep_alive="5m", timeout_s=60, think=False,
        prompt_user_text="Say hi in 5 words.",
        prompt_source_text="",
        sample_id="smoke", repeat=0, language="eng_Latn", experiment="smoke",
    )
    print("\n[summary]\n", json.dumps(orch.summary().to_dict(), indent=2, default=str))
    orch.unload()