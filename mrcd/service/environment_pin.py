"""Runtime check of the execution environment the service contract pins (TODO_PLAN action 3.4).

The pinned state (DESIGN_B_INCREMENTAL 1.A.1, measured): ``OMP_NUM_THREADS=4`` in the environment BEFORE numpy is
imported, ASR device ``cpu``, every profile preloaded and THEN a warm-up that imports Silero, which leaves
``torch.get_num_threads() == 1``; loading another engine afterwards puts the shared pool back to 4.

``evaluate`` is pure (inputs injected) so it is testable; ``environment_is_pinned`` is True only when every check
positively passes. A value that cannot be verified is reported as not pinned with the reason, never as pinned.
Nothing here reads or logs audio, text or credentials.
"""

from __future__ import annotations

EXPECTED = {
    "omp_num_threads": "4",
    "device": "cpu",
    "torch_threads_after_silero": 1,
    "load_order": ["preload", "warmup"],
}


def _check(expected, observed, ok: bool, reason: str | None = None) -> dict:
    return {
        "expected": expected,
        "observed": observed,
        "ok": ok,
        "reason": None if ok else reason,
    }


def evaluate(
    *,
    environ: dict,
    omp_status: dict,
    device: str | None,
    torch_threads: int | None,
    load_events: list[str],
    engines_now: list[str],
    engines_at_warmup: list[str] | None,
) -> dict:
    """Returns ``{"environment_is_pinned": bool, "checks": {...}, "warnings": [{"code", "message"}]}``."""
    checks: dict[str, dict] = {}
    omp = environ.get("OMP_NUM_THREADS")
    checks["omp_num_threads"] = _check(
        EXPECTED["omp_num_threads"],
        omp,
        omp == EXPECTED["omp_num_threads"],
        "OMP_NUM_THREADS is not 4",
    )
    eff = omp_status.get("effective")
    checks["omp_set_before_numpy"] = _check(
        True,
        omp_status.get("set_before_numpy"),
        eff is True,
        "not verifiable (numpy imported before mrcd)"
        if eff is None
        else "OMP_NUM_THREADS was not set before numpy was imported",
    )
    checks["device"] = _check(
        EXPECTED["device"],
        device,
        device == EXPECTED["device"],
        "ASR device is not pinned to cpu (use --device cpu)",
    )
    warmed = "warmup" in load_events
    if torch_threads is None or not warmed:
        t_ok, t_reason = (
            False,
            "torch threads not verified: warm-up (Silero import) has not run",
        )
    else:
        t_ok = torch_threads == EXPECTED["torch_threads_after_silero"]
        t_reason = "torch thread count differs after Silero"
    checks["torch_threads"] = _check(
        EXPECTED["torch_threads_after_silero"], torch_threads, t_ok, t_reason
    )
    order_ok = (
        load_events[:2] == EXPECTED["load_order"] and engines_at_warmup is not None
    )
    checks["load_order"] = _check(
        EXPECTED["load_order"],
        list(load_events),
        order_ok,
        "start-up order is not preload -> warmup",
    )
    late = engines_at_warmup is not None and sorted(engines_now) != sorted(
        engines_at_warmup
    )
    checks["no_engine_loaded_after_warmup"] = _check(
        True,
        not late,
        not late,
        "an engine was loaded after the warm-up (torch thread count can drift)",
    )
    pinned = all(c["ok"] for c in checks.values())
    warnings = [
        {"code": f"environment_not_pinned:{name}", "message": c["reason"]}
        for name, c in checks.items()
        if not c["ok"]
    ]
    return {"environment_is_pinned": pinned, "checks": checks, "warnings": warnings}
