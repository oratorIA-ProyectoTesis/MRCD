"""System probes without psutil (not installed): total CPU load, peak working set, hardware, top CPU processes."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


class _FT(ctypes.Structure):
    _fields_ = [("lo", wt.DWORD), ("hi", wt.DWORD)]


def _ft(x):
    return (x.hi << 32) | x.lo


def _times():
    i, k, u = _FT(), _FT(), _FT()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
    return _ft(i), _ft(k), _ft(u)  # idle, kernel (includes idle), user


def system_cpu_load(seconds: float = 3.0) -> float:
    """Whole-machine CPU utilisation in percent over ``seconds`` (all processes)."""
    i0, k0, u0 = _times()
    time.sleep(seconds)
    i1, k1, u1 = _times()
    tot = (k1 - k0) + (u1 - u0)
    return round(100.0 * (tot - (i1 - i0)) / tot, 2) if tot else 0.0


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def mem_mb() -> dict:
    """Current and peak working set of THIS process (MB)."""
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(pmc)
    k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wt.DWORD]
    ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return {"rss_mb": round(pmc.WorkingSetSize / 1e6, 1), "peak_rss_mb": round(pmc.PeakWorkingSetSize / 1e6, 1)}


def process_cpu_s() -> float:
    return time.process_time()


def top_processes(n: int = 6) -> list[dict]:
    """Top CPU-time processes (cumulative), via PowerShell without profile; informational."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-Process | Sort-Object CPU -Descending | Select-Object -First {n} Name,CPU | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out)
        return data if isinstance(data, list) else [data]
    except Exception as e:  # noqa: BLE001
        return [{"error": repr(e)}]


def heavy_now(n: int = 5, seconds: float = 3.0) -> list[dict]:
    """Processes that used the most CPU time during ``seconds`` (delta of cumulative CPU)."""
    cmd = ("$a=Get-Process|Select Id,Name,CPU; Start-Sleep -Seconds %d; $b=Get-Process|Select Id,Name,CPU;"
           "$h=@{};$a|%%{$h[$_.Id]=$_.CPU};"
           "$b|%%{[pscustomobject]@{Name=$_.Name;Delta=[math]::Round(($_.CPU-$h[$_.Id]),2)}}|Sort Delta -Desc|Select -First %d|ConvertTo-Json -Compress") % (seconds, n)
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60).stdout
        d = json.loads(out)
        return d if isinstance(d, list) else [d]
    except Exception as e:  # noqa: BLE001
        return [{"error": repr(e)}]


def background_report(seconds: float = 3.0) -> dict:
    return {"system_cpu_pct": system_cpu_load(seconds), "heaviest_cpu_s_delta": heavy_now(5, 3)}


def hardware() -> dict:
    import os

    d = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
         "cpu_count_logical": os.cpu_count(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor|Select Name,NumberOfCores,NumberOfLogicalProcessors|ConvertTo-Json -Compress);"
                              "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        d["cpu"] = json.loads(out[0]); d["ram_gb"] = float(out[1])
    except Exception as e:  # noqa: BLE001
        d["cpu_error"] = repr(e)
    for m in ("torch", "numpy", "librosa", "faster_whisper", "ctranslate2", "scipy", "soundfile"):
        try:
            d[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            d[m] = None
    return d


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha(root: Path, patterns=("mrcd/**/*.py",)) -> str:
    h = hashlib.sha256()
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            h.update(str(p.relative_to(root)).encode()); h.update(sha256_file(p).encode())
    return h.hexdigest()
