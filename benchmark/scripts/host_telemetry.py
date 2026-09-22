"""
host_telemetry.py — Snapshots of host-side state around each Ollama call.

Three snapshot kinds, all returning plain dicts so they can be serialised
straight into a raw JSON's `host` field:

  * `nvidia_snapshot()`     pynvml + nvidia-smi fallback. Per-GPU: name,
                            vram_used_mib, vram_total_mib, util_pct,
                            power_w, temp_c, sm_clock_mhz, mem_clock_mhz.
  * `psutil_snapshot()`     Per-process CPU%, RSS, VMS, ctx_switches,
                            num_threads. Falls back gracefully when psutil
                            can't find the pid.
  * `proc_snapshot(pid)`    Read /proc/<pid>/{status,io,stat,cmdline} to get
                            state, voluntary/nonvoluntary ctx switches,
                            read_bytes, write_bytes, threads, fds.

All functions are tolerant: any failure returns `{"error": "..."}` rather
than raising, so a missing `nvidia-smi` or absent pid never aborts the run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


def _now_unix() -> float:
    return time.time()


def nvidia_snapshot() -> dict:
    """Return a snapshot of every visible NVIDIA GPU.

    Tries pynvml first (one NVML call per GPU, no subprocess); falls back to
    `nvidia-smi --query-gpu=... --format=csv` if NVML import failed.
    """
    snap: dict = {"timestamp_unix": _now_unix(), "gpus": [], "method": None}
    try:
        import pynvml  # type: ignore

        try:
            pynvml.nvmlInit()
        except Exception as exc:
            snap["error"] = f"nvmlInit failed: {exc}"
            return snap
        snap["method"] = "pynvml"
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    power_w = None
                try:
                    temp_c = pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp_c = None
                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(
                        h, pynvml.NVML_CLOCK_SM
                    )
                except Exception:
                    sm_clock = None
                try:
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(
                        h, pynvml.NVML_CLOCK_MEM
                    )
                except Exception:
                    mem_clock = None
                snap["gpus"].append({
                    "index": i,
                    "name": pynvml.nvmlDeviceGetName(h).decode()
                    if isinstance(pynvml.nvmlDeviceGetName(h), bytes)
                    else str(pynvml.nvmlDeviceGetName(h)),
                    "vram_used_mib": mem.used / (1024 * 1024),
                    "vram_total_mib": mem.total / (1024 * 1024),
                    "util_pct": util.gpu,
                    "mem_util_pct": util.memory,
                    "power_w": power_w,
                    "temp_c": temp_c,
                    "sm_clock_mhz": sm_clock,
                    "mem_clock_mhz": mem_clock,
                })
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return snap
    except Exception as exc:
        snap["nvml_error"] = str(exc)

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,"
                    "utilization.gpu,utilization.memory,power.draw,"
                    "temperature.gpu,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
                stderr=subprocess.STDOUT,
            ).decode()
            snap["method"] = "nvidia-smi"
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 10:
                    continue
                gpu: dict = {"index": int(parts[0]), "name": parts[1]}
                for key_name, idx, parser in [
                    ("vram_used_mib", 2, float),
                    ("vram_total_mib", 3, float),
                    ("util_pct", 4, lambda v: float(v) if v != "[Not Supported]" else None),
                    ("mem_util_pct", 5, lambda v: float(v) if v != "[Not Supported]" else None),
                    ("power_w", 6, lambda v: float(v) if v != "[Not Supported]" else None),
                    ("temp_c", 7, lambda v: float(v) if v != "[Not Supported]" else None),
                    ("sm_clock_mhz", 8, lambda v: float(v) if v != "[Not Supported]" else None),
                    ("mem_clock_mhz", 9, lambda v: float(v) if v != "[Not Supported]" else None),
                ]:
                    try:
                        gpu[key_name] = parser(parts[idx])
                    except Exception:
                        gpu[key_name] = None
                snap["gpus"].append(gpu)
            return snap
        except Exception as exc:
            snap["nvidia_smi_error"] = str(exc)
    else:
        snap["error"] = snap.get("error", "no nvidia-smi binary found")
    return snap


def _ollama_pid() -> Optional[int]:
    """Locate the ollama serve process via psutil, falling back to pgrep."""
    try:
        import psutil  # type: ignore

        for p in psutil.process_iter(attrs=["name", "cmdline"]):
            name = (p.info.get("name") or "").lower()
            cmdline = " ".join(p.info.get("cmdline") or []).lower()
            if "ollama" in name and "serve" in cmdline:
                return p.pid
            if name == "ollama":
                return p.pid
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "ollama"], timeout=3, stderr=subprocess.STDOUT
        ).decode()
        for line in out.strip().splitlines():
            try:
                return int(line.strip())
            except ValueError:
                continue
    except Exception:
        pass
    return None


def psutil_snapshot(pid: Optional[int] = None) -> dict:
    """Per-process CPU%, RSS, VMS, ctx_switches via psutil.

    Walks the whole process tree under the ollama serve process so that the
    `ollama runner` subprocess (which actually owns the loaded model) is
    captured too. Aggregates RSS / CPU across the tree.
    """
    snap: dict = {"timestamp_unix": _now_unix()}
    if pid is None:
        pid = _ollama_pid()
        snap["ollama_pid"] = pid
    if pid is None:
        snap["error"] = "ollama process not found"
        return snap
    try:
        import psutil  # type: ignore

        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
        agg_cpu = 0.0
        agg_rss = 0
        agg_threads = 0
        children: list[dict] = []
        for p in procs:
            try:
                with p.oneshot():
                    agg_cpu += p.cpu_percent(interval=None) or 0.0
                    agg_rss += p.memory_info().rss
                    agg_threads += p.num_threads() or 0
                    children.append({
                        "pid": p.pid,
                        "name": p.name(),
                        "cmdline": " ".join(p.cmdline() or []),
                        "rss_mib": p.memory_info().rss / (1024 * 1024),
                        "status": p.status(),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        with root.oneshot():
            snap["pid"] = pid
            snap["cpu_pct"] = root.cpu_percent(interval=None)
            mem = root.memory_info()
            snap["rss_mib"] = mem.rss / (1024 * 1024)
            snap["vms_mib"] = mem.vms / (1024 * 1024)
            try:
                snap["num_threads"] = root.num_threads()
            except Exception:
                snap["num_threads"] = None
            try:
                snap["status"] = root.status()
            except Exception:
                snap["status"] = None
            try:
                snap["nice"] = root.nice()
            except Exception:
                snap["nice"] = None
            try:
                snap["ctx_switches"] = sum(root.num_ctx_switches())
            except Exception:
                snap["ctx_switches"] = None
        snap["tree_cpu_pct"] = agg_cpu
        snap["tree_rss_mib"] = agg_rss / (1024 * 1024)
        snap["tree_threads"] = agg_threads
        snap["tree_process_count"] = len(procs)
        snap["children"] = children
    except Exception as exc:
        snap["error"] = f"psutil failed: {exc}"
    return snap


def proc_snapshot(pid: Optional[int] = None) -> dict:
    """Read /proc/<pid>/{status,io,stat,cmdline} for kernel-level details.

    Linux-only. Returns `{"error": ...}` on non-Linux or missing pid.
    """
    snap: dict = {"timestamp_unix": _now_unix()}
    if pid is None:
        pid = _ollama_pid()
        snap["ollama_pid"] = pid
    if pid is None:
        snap["error"] = "ollama process not found"
        return snap
    if not hasattr(os, "readlink"):
        snap["error"] = "/proc not available on this platform"
        return snap
    proc_root = Path(f"/proc/{pid}")
    if not proc_root.exists():
        snap["error"] = f"/proc/{pid} does not exist"
        return snap

    try:
        status = (proc_root / "status").read_text()
        for line in status.splitlines():
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "State":
                snap["proc_state"] = val.split()[0] if val else None
            elif key == "Threads":
                try:
                    snap["proc_threads"] = int(val)
                except ValueError:
                    pass
            elif key == "VmRSS":
                snap["proc_vm_rss_kib"] = _parse_kib(val)
            elif key == "VmSize":
                snap["proc_vm_size_kib"] = _parse_kib(val)
            elif key == "voluntary_ctxt_switches":
                snap["proc_vol_ctx"] = int(val)
            elif key == "nonvoluntary_ctxt_switches":
                snap["proc_invol_ctx"] = int(val)
    except Exception as exc:
        snap["status_error"] = str(exc)

    try:
        io = (proc_root / "io").read_text()
        for line in io.splitlines():
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key in ("read_bytes", "write_bytes", "rchar", "wchar"):
                try:
                    snap[f"proc_{key}"] = int(val)
                except ValueError:
                    pass
    except Exception as exc:
        snap["io_error"] = str(exc)

    try:
        fd_dir = proc_root / "fd"
        if fd_dir.is_dir():
            snap["proc_fd_count"] = sum(1 for _ in fd_dir.iterdir())
    except Exception:
        pass

    try:
        cmdline_raw = (proc_root / "cmdline").read_bytes()
        snap["proc_cmdline"] = cmdline_raw.replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
    except Exception:
        pass

    return snap


def host_snapshot(pid: Optional[int] = None) -> dict:
    """Convenience: all three snapshots in one dict, suitable for `host.pre_call`."""
    return {
        "timestamp_unix": _now_unix(),
        "nvidia_smi": nvidia_snapshot(),
        "psutil": psutil_snapshot(pid),
        "proc": proc_snapshot(pid),
    }


def _parse_kib(val: str) -> Optional[int]:
    val = val.strip()
    if not val:
        return None
    try:
        if val.endswith("kB"):
            return int(val[:-2].strip())
        return int(val)
    except ValueError:
        return None


if __name__ == "__main__":
    print(json.dumps(host_snapshot(), indent=2, ensure_ascii=False, default=str))