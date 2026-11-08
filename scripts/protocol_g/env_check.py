"""Run-environment and weight checks of the protocol G harness (section 14): everything that must hold BEFORE the lock.

Nothing here downloads: snapshots are located with ``local_files_only``. Locators are injectable so tests use fakes.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from scripts.protocol_g import common as C

H = C.H


def import_probe(mods=C.REQUIRED_IMPORTS) -> dict[str, str | None]:
    """name -> None when the module really imports, else the error text (a missing package is not found by metadata alone
    when the install is broken, so the module is imported)."""
    out: dict[str, str | None] = {}
    for m in mods:
        try:
            importlib.import_module(m)
            out[m] = None
        except Exception as exc:  # noqa: BLE001 - any import failure is a missing requirement
            out[m] = f"{type(exc).__name__}: {exc}"
    return out


def sanitize_argv(argv: list[str], repo: Path) -> list[str]:
    """argv with no absolute path: a path inside the repo becomes repo-relative, any other absolute path is redacted."""
    out = []
    for a in argv:
        key, eq, val = a.partition("=")
        cand = val if eq and key.startswith("--") else a
        looks_abs = Path(cand).is_absolute() or (
            len(cand) > 2 and cand[1:3] in (":\\", ":/")
        )
        if looks_abs:
            try:
                new = Path(cand).resolve().relative_to(Path(repo).resolve()).as_posix()
            except ValueError:
                new = "<absolute-path-redacted>"
            out.append(f"{key}={new}" if eq and key.startswith("--") else new)
        else:
            out.append(a)
    return out


# ------------------------------------------------------------------------------------------------ weight locators
def snapshot_manifest(path: Path) -> dict:
    """sha256 of every file of a model snapshot directory (symlinks resolved) and one combined digest (the definition
    used by the protocol F cw2 annex)."""
    files = {}
    for p in sorted(Path(path).iterdir()):
        if p.is_file():
            files[p.name] = H.sha256_file(p.resolve())
    digest = H.sha256_bytes(
        "".join(f"{n}\t{h}\n" for n, h in sorted(files.items())).encode()
    )
    return {"files": files, "combined_sha256": digest}


def real_locator(ft_model_bin: Path | None = None):
    """-> callable giving the local weight locations (``None`` = not found locally). Never downloads."""

    def locate() -> dict:
        out: dict = {
            "whisper_model_bin": None,
            "cw2_snapshot": None,
            "ft_model_bin": ft_model_bin,
            "aligner_snapshot": None,
        }
        try:
            from mrcd.service.whisper_cache import cached_model_dir

            d = cached_model_dir("small")
            out["whisper_model_bin"] = (d / "model.bin") if d else None
        except Exception:  # noqa: BLE001
            pass
        try:
            from huggingface_hub import snapshot_download
            from mrcd import asr_profiles as AP

            out["cw2_snapshot"] = Path(
                snapshot_download(
                    AP.CW2_MODEL_ID,
                    revision=AP.CW2_MODEL_REVISION,
                    local_files_only=True,
                )
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            from scripts.protocol_g.ctc_align import locate_snapshot

            out["aligner_snapshot"] = locate_snapshot()
        except Exception:  # noqa: BLE001
            pass
        return out

    return locate


def check_aligner_snapshot(snapshot: Path) -> dict[str, str]:
    """Hash every pinned file of the aligner snapshot (weights, config, vocabulary, feature extractor) and, for a hub-cache
    layout, require the revision directory name to be the pinned revision. Refuses on any difference."""
    snapshot = Path(snapshot)
    if snapshot.parent.name == "snapshots" and snapshot.name != C.ALIGNER_REVISION:
        raise C.Refusal(
            "aligner_revision",
            f"aligner snapshot revision {snapshot.name} differs from the pinned {C.ALIGNER_REVISION}",
        )
    got = {}
    for name, want in sorted(C.aligner_expected_files().items()):
        f = snapshot / name
        if not f.is_file():
            raise C.Refusal(
                "aligner_weights", f"aligner file {name} missing in the snapshot"
            )
        got[name] = H.sha256_file(f)
        if got[name] != want:
            raise C.Refusal(
                "aligner_weights",
                f"aligner file {name} sha256 differs from the pinned value",
            )
    return got


def verify_weights(
    sf: dict,
    locate,
    *,
    include_ft: bool,
    need_aligner: bool = True,
    allow_cw2_unavailable: bool = False,
) -> dict:
    """Hash the weight files that will actually be loaded and compare them with the frozen record and the pinned values.

    Whisper small CT2 ``model.bin`` (p0 / p1; must be found), aligner files (must be found when ``need_aligner``), ft
    ``model.bin`` (must be given when ``include_ft``), cw2 snapshot (an absent snapshot REFUSES unless
    ``allow_cw2_unavailable`` is set explicitly, in which case the reason is recorded and the exploratory arm ends
    ``not_available``; a present one must match). Any mismatch refuses."""
    loc = locate()
    report: dict = {}
    wb = loc.get("whisper_model_bin")
    if not wb or not Path(wb).is_file():
        raise C.Refusal(
            "whisper_weights", "Whisper small CT2 model.bin not found locally"
        )
    got = H.sha256_file(Path(wb))
    want = sf["asr"]["whisper_small_ct2_int8_model_bin_sha256"]
    if got != want or got != C.WHISPER_SMALL_MODEL_BIN_SHA256:
        raise C.Refusal(
            "whisper_weights",
            "the loaded Whisper small CT2 model.bin differs from the frozen / pinned sha256",
        )
    report["whisper_small_model_bin"] = "match"
    snap = loc.get("aligner_snapshot")
    if need_aligner:
        if not snap:
            raise C.Refusal(
                "aligner_weights",
                "aligner snapshot not available locally (no download is attempted)",
            )
        check_aligner_snapshot(Path(snap))
        report["aligner_files"] = "match"
    if include_ft:
        fb = loc.get("ft_model_bin")
        if not fb or not Path(fb).is_file():
            raise C.Refusal(
                "ft_weights",
                "OD-4 includes the ft arm but its model.bin was not given / found",
            )
        if H.sha256_file(Path(fb)) != sf["od4"]["ft_model_bin_sha256"]:
            raise C.Refusal(
                "ft_weights", "the ft model.bin differs from the OD-4 frozen sha256"
            )
        report["ft_model_bin"] = "match"
    cw = loc.get("cw2_snapshot")
    if cw is None:
        if not allow_cw2_unavailable:
            raise C.Refusal(
                "cw2_weights",
                "the local cw2 snapshot is not available (no download is attempted); pass --allow-cw2-unavailable "
                "to run with the exploratory cw2 arm recorded as not_available",
            )
        report["cw2_snapshot"] = "not_available_locally"
        report["cw2_unavailable_allowed"] = True
        report["cw2_unavailable_reason"] = (
            "snapshot absent locally; --allow-cw2-unavailable given explicitly; exploratory cw2 arm not_available"
        )
    else:
        man = snapshot_manifest(Path(cw))
        if (
            man["combined_sha256"] != C.CW2_WEIGHTS_MANIFEST_SHA256
            or man["combined_sha256"] != sf["asr"]["cw2"]["weights_manifest_sha256"]
        ):
            raise C.Refusal(
                "cw2_weights",
                "the local cw2 snapshot differs from the frozen / pinned weights manifest",
            )
        report["cw2_snapshot"] = "match"
    return report
