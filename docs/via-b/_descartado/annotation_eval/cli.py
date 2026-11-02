"""Command line for the Via B toolkit: ``python -m mrcd.annotation_eval <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mrcd.annotation_eval import power
from mrcd.annotation_eval.schema import ALL_CLASSES, CORE_CLASSES


def _p(s: str) -> Path:
    return Path(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m mrcd.annotation_eval", description=__doc__
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(
        "make-eaf", help="create an empty EAF for one annotator and recording"
    )
    s.add_argument("--recording", required=True)
    s.add_argument("--annotator", required=True, help="A1, A2 (or ADJ)")
    s.add_argument("--wav", type=_p, required=True, help="16 kHz mono WAV used in ELAN")
    s.add_argument("--out", type=_p, required=True)
    s.add_argument("--guide-version", default="0.1-borrador")

    s = sub.add_parser(
        "declare-coverage",
        help="declare the classes fully reviewed (close the file in ELAN first)",
    )
    s.add_argument("eaf", type=_p)
    s.add_argument(
        "--complete", nargs="+", default=list(CORE_CLASSES), choices=ALL_CLASSES
    )
    s.add_argument("--not-annotated", nargs="*", default=[], choices=ALL_CLASSES)

    s = sub.add_parser("validate", help="validate EAF files")
    s.add_argument("eaf", type=_p, nargs="+")
    s.add_argument("--annotator")
    s.add_argument("--wav", type=_p, help="WAV whose duration is authoritative")
    s.add_argument("--require-complete", nargs="*", default=[])

    s = sub.add_parser(
        "agreement",
        help="inter-annotator agreement between two directories of EAF files",
    )
    s.add_argument("--a1", type=_p, required=True)
    s.add_argument("--a2", type=_p, required=True)
    s.add_argument("--out", type=_p, required=True)

    s = sub.add_parser(
        "adjudicate", help="build the adjudication EAF and the disagreement list"
    )
    s.add_argument("--a1", type=_p, required=True)
    s.add_argument("--a2", type=_p, required=True)
    s.add_argument("--wav-name", required=True)
    s.add_argument("--out", type=_p, required=True)

    s = sub.add_parser(
        "adjudication-log", help="change log of a finished adjudication EAF"
    )
    s.add_argument("adj", type=_p)
    s.add_argument("--out", type=_p, required=True)

    s = sub.add_parser(
        "import", help="import adjudicated EAF files into the corpora schema"
    )
    s.add_argument("--adj-dir", type=_p, required=True)
    s.add_argument("--media-dir", type=_p, required=True)
    s.add_argument("--out", type=_p, required=True)
    s.add_argument(
        "--root",
        type=_p,
        help="path root that must contain media and EAF (default: parent of adj-dir)",
    )
    s.add_argument("--agreement", type=_p, help="agreement.json of the TEST recordings")

    s = sub.add_parser("evaluate", help="pre-registered confirmatory evaluation")
    s.add_argument("--corpus", type=_p, required=True)
    s.add_argument("--pred", action="append", required=True, metavar="SYSTEM=FILE")
    s.add_argument("--freeze", type=_p)
    s.add_argument("--gate", type=_p, help="admitted segments of the frozen S6 gate")
    s.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="exploratory dry run; report is stamped NOT CONFIRMATORY",
    )
    s.add_argument("--out", type=_p, required=True)

    s = sub.add_parser(
        "freeze", help="freeze the pre-registration and the evaluation code"
    )
    s.add_argument("--out", type=_p, required=True)
    s.add_argument("--tag", default="preregistro")
    s = sub.add_parser("verify-freeze")
    s.add_argument("freeze", type=_p)

    s = sub.add_parser("power", help="expected positives for a rate and duration")
    s.add_argument(
        "--rate", type=float, required=True, help="events per minute of speech"
    )
    s.add_argument("--minutes", type=float, required=True)
    s.add_argument("--retention", type=float, default=1.0)

    s = sub.add_parser(
        "verify", help="technical verification of recordings (stores no content)"
    )
    s.add_argument("files", nargs="+")
    s.add_argument("--no-face", action="store_true")
    s.add_argument("--face-model")

    a = ap.parse_args(argv)

    if a.cmd == "make-eaf":
        from mrcd.annotation_eval.eaf import build_eaf, write_eaf
        from mrcd.corpora.base import wav_info

        dur = int(round(wav_info(a.wav, rate=16000, channels=1) * 1000))
        write_eaf(
            build_eaf(
                a.recording,
                [a.annotator],
                a.wav.name,
                dur,
                guide_version=a.guide_version,
            ),
            a.out,
        )
        print(f"{a.out} ({dur} ms)")
        return 0
    if a.cmd == "declare-coverage":
        from mrcd.annotation_eval.eaf import declare_coverage

        declare_coverage(a.eaf, a.complete, not_annotated=a.not_annotated)
        return 0
    if a.cmd == "validate":
        from mrcd.annotation_eval.eaf import validate_eaf
        from mrcd.corpora.base import wav_info

        dur = (
            int(round(wav_info(a.wav, rate=16000, channels=1) * 1000))
            if a.wav
            else None
        )
        bad = 0
        for f in a.eaf:
            rep = validate_eaf(
                f,
                expected_annotator=a.annotator,
                audio_duration_ms=dur,
                require_complete=a.require_complete,
            )
            print(
                f"{f}: {'OK' if rep.ok else 'ERRORES'} ({len(rep.errors)} errores, {len(rep.warnings)} avisos)"
            )
            for i in rep.issues:
                print("  ", i)
            bad += not rep.ok
        return 1 if bad else 0
    if a.cmd == "agreement":
        from mrcd.annotation_eval.agreement import agreement_from_dirs

        rep = agreement_from_dirs(a.a1, a.a2)
        a.out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
        for c, r in rep["classes"].items():
            g = r.get("gate", {})
            print(
                f"{c:18s} gate={r.get('passes_gate')} {g.get('failed', g.get('reason', ''))}"
            )
        return 0
    if a.cmd == "adjudicate":
        from mrcd.annotation_eval.adjudication import build_adjudication, write_csv

        dis = build_adjudication(a.a1, a.a2, a.out, a.wav_name)
        write_csv(dis, a.out.with_suffix(".desacuerdos.csv"))
        print(f"{a.out}: {len(dis)} desacuerdos")
        return 0
    if a.cmd == "adjudication-log":
        from mrcd.annotation_eval.adjudication import adjudication_log, write_csv

        write_csv(adjudication_log(a.adj), a.out)
        return 0
    if a.cmd == "import":
        from mrcd.annotation_eval.importer import import_corpus

        agr = (
            json.loads(a.agreement.read_text(encoding="utf-8")) if a.agreement else None
        )
        m = import_corpus(a.adj_dir, a.media_dir, a.out, root=a.root, agreement=agr)
        print(
            json.dumps(
                {k: m[k] for k in ("n_recordings", "n_speakers", "duration_s")},
                indent=2,
            )
        )
        for c, v in m["classes"].items():
            print(
                f"{c:18s} positives={v['positives']:4d} speakers={v['speakers_with_positives']:3d} gate={v['agreement_gate_passed']}"
            )
        return 0
    if a.cmd == "evaluate":
        from mrcd.annotation_eval.evaluate import evaluate, render_markdown

        preds = dict(x.split("=", 1) for x in a.pred)
        res = evaluate(
            a.corpus,
            {k: Path(v) for k, v in preds.items()},
            freeze_file=a.freeze,
            allow_unfrozen=a.allow_unfrozen,
            gate_file=a.gate,
        )
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "evaluation.json").write_text(
            json.dumps(res, indent=2, default=float), encoding="utf-8"
        )
        (a.out / "evaluation.md").write_text(render_markdown(res), encoding="utf-8")
        print(res["status"])
        return 0
    if a.cmd == "freeze":
        from mrcd.annotation_eval.freeze import freeze

        rec = freeze(a.out, tag=a.tag)
        print(
            json.dumps(
                {k: rec[k] for k in ("frozen_at_utc", "config_hash", "git_head")},
                indent=2,
            )
        )
        return 0
    if a.cmd == "verify-freeze":
        from mrcd.annotation_eval.freeze import verify_freeze

        bad = verify_freeze(a.freeze)
        print("INTACTO" if not bad else "\n".join(bad))
        return 1 if bad else 0
    if a.cmd == "power":
        e = power.expected_positives(a.rate, a.minutes, a.retention)
        print(
            f"expected positives: {e:.1f}; minutes for 40: {power.minutes_needed(a.rate, 40, a.retention)}"
        )
        return 0
    if a.cmd == "verify":
        from mrcd.annotation_eval.verify_recording import main as vmain

        return vmain(
            a.files
            + (["--no-face"] if a.no_face else [])
            + (["--face-model", a.face_model] if a.face_model else [])
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
