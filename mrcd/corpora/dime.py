"""DIME importer (20 dialogues as unlabeled material + utt1/utt74 verification samples).

What the audit established and this module enforces:

* TXT transcripts are cp1252, lower-cased, punctuation-stripped (only ``?``), untimed;
  they keep fillers, literal repeats and hyphen-joined fragments, but NOT lengthening
  or blocks, and their ``<sil>`` pauses carry no duration (and are incomplete).
* The XLS files are dialogue acts / obligations, never disfluency labels.
* The PHN samples (``MillisecondsPerFrame: 1.0``, times in ms) are the only timed tiers.
* utt1 / utt74 are segments of d19 / d12 and inherit that group.
* There are no speaker ids: ``speaker_id`` is always ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mrcd.corpora.base import (
    CLASSES,
    Annotation,
    CorpusFormatError,
    Example,
    InvalidBoundsError,
    SourceFile,
    ms_to_s,
    norm_label,
    relative_posix,
    sha256_file,
    wav_info,
)
from mrcd.corpora.groups import UnresolvedGroupError, group_for_name

CORPUS = "dime"
TXT_ENCODING = "cp1252"
PHN_ENCODING = "latin-1"
_DIALOG_ID = re.compile(r"d\d{2}")
_UTT_LINE = re.compile(r"^(utt\d+)\s*:\s*(?:([su]):)?\s*(.*?)\s*$")
_MARKER = re.compile(r"<[^<>\s]+>")
_HYPHEN_FRAGMENT = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")
_BRACKET_TOKEN = re.compile(r"\S*\[[^\]]*\]\S*")

NORMALISATION_NOTES = (
    "TXT is lower-case, punctuation-stripped except '?', untimed; lengthening and blocks "
    "are not transcribed; <sil> has no duration and is incomplete (d12 utt74 has a 267 ms "
    "silence the TXT omits); [..] reductions are written in canonical orthography. "
    "input_view removes <markers>, '+', s:/u: turn tags and bracket characters, and splits "
    "hyphen-joined fragments ('pue-puede' -> 'pue puede'); it never reconstructs anything."
)
_TXT_LABEL_DOC = (
    "Undocumented in the shipped material; kept only as an untimed text cue."
)


class XlsNotLabelsError(CorpusFormatError):
    """DIME XLS files are dialogue acts / charges, not disfluency labels."""


def refuse_xls_as_labels(path: str | Path) -> None:
    """Always raises: an XLS can never be turned into disfluency labels."""
    raise XlsNotLabelsError(
        f"{Path(path).name}: DIME XLS hold dialogue acts (DAMSL), obligations or modality; "
        "they contain no disfluency labels and must not be used as such"
    )


# --------------------------------------------------------------------------- TXT
@dataclass(frozen=True)
class DimeUtterance:
    utt_id: str
    turn_tag: str | None  # 's' / 'u' on turn-initial lines only; NOT a speaker identity
    raw_text: str


@dataclass(frozen=True)
class DimeDialog:
    header: dict[str, str]
    utterances: tuple[DimeUtterance, ...]


def parse_dime_txt(path: str | Path) -> DimeDialog:
    """Parse a DIME TXT (cp1252). Raises on any line that is neither header nor utterance."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode(TXT_ENCODING)
    except UnicodeDecodeError as e:
        raise CorpusFormatError(f"{path}: not valid {TXT_ENCODING}") from e
    header: dict[str, str] = {}
    utts: list[DimeUtterance] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = _UTT_LINE.match(line)
        if m:
            utts.append(DimeUtterance(m.group(1), m.group(2), m.group(3)))
        elif not utts and ":" in line:
            k, v = line.split(":", 1)
            header[k.strip()] = v.strip()
        else:
            raise CorpusFormatError(f"{path}:{n}: unrecognised line {line[:40]!r}")
    if not utts:
        raise CorpusFormatError(f"{path}: no utterance lines")
    return DimeDialog(header, tuple(utts))


def dime_input_view(utterance_text: str) -> str:
    """Spoken words of one utterance without DIME markers (see NORMALISATION_NOTES)."""
    t = _MARKER.sub(" ", utterance_text)
    t = t.replace("+", " ").replace("[", "").replace("]", "")
    t = _HYPHEN_FRAGMENT.sub(" ", t)
    return " ".join(t.split())


def _txt_cues(utts: tuple[DimeUtterance, ...]) -> tuple[Annotation, ...]:
    """Untimed text cues, kept apart from the input and never mapped to a class."""
    out: list[Annotation] = []
    for u in utts:
        loc = {"utterance": u.utt_id}
        for m in _MARKER.finditer(u.raw_text):
            out.append(
                Annotation(
                    "dime_txt_marker",
                    m.group(0),
                    "unknown",
                    text=m.group(0),
                    locator=loc,
                    justification=_TXT_LABEL_DOC,
                )
            )
        for tok in u.raw_text.split():
            if _HYPHEN_FRAGMENT.search(tok):
                out.append(
                    Annotation(
                        "dime_txt_hyphen_fragment",
                        "hyphen_fragment",
                        "unknown",
                        text=tok,
                        locator=loc,
                        justification="Candidate repeat/restart only after adjudication.",
                    )
                )
    return tuple(out)


# --------------------------------------------------------------------------- PHN
@dataclass(frozen=True)
class PhnInterval:
    start_s: float
    end_s: float
    label: str


def parse_phn(path_or_text: str | Path, *, dense: bool = False) -> list[PhnInterval]:
    """Parse a PHN tier; times are converted ms -> s using ``MillisecondsPerFrame``.

    ``dense=True`` additionally rejects overlaps (sparse tiers may have gaps anyway).
    """
    p = Path(str(path_or_text)) if "\n" not in str(path_or_text) else None
    text = (
        p.read_bytes().decode(PHN_ENCODING)
        if p is not None and p.exists()
        else str(path_or_text)
    )
    lines = text.splitlines()
    mpf: float | None = None
    i = 0
    for i, line in enumerate(lines):
        if line.strip() == "END OF HEADER":
            break
        if line.startswith("MillisecondsPerFrame:"):
            try:
                mpf = float(line.split(":", 1)[1])
            except ValueError as e:
                raise CorpusFormatError(f"Bad MillisecondsPerFrame: {line!r}") from e
    else:
        raise CorpusFormatError("PHN without END OF HEADER")
    if mpf is None:
        raise CorpusFormatError("PHN without MillisecondsPerFrame")
    out: list[PhnInterval] = []
    for n, line in enumerate(lines[i + 1 :], i + 2):
        if not line.strip():
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            raise CorpusFormatError(f"PHN line {n}: expected 'start end label'")
        try:
            s, e = ms_to_s(float(parts[0]), mpf), ms_to_s(float(parts[1]), mpf)
        except ValueError as ex:
            raise CorpusFormatError(f"PHN line {n}: non-numeric time") from ex
        if not (0 <= s < e):
            raise InvalidBoundsError(f"PHN line {n}: non-positive interval {s}-{e}")
        out.append(PhnInterval(s, e, parts[2].strip()))
    if dense:
        for a, b in zip(out, out[1:]):
            if b.start_s < a.end_s - 1e-6:
                raise InvalidBoundsError(f"Overlap in dense tier: {a} / {b}")
    return out


# ----------------------------------------------------------------- sample parents
def _group_for_sample(sample: str, sample_parents: dict[str, str]) -> str:
    parent = sample_parents.get(sample.lower())
    if parent is None:
        raise UnresolvedGroupError(
            f"{sample}: parent dialogue not established by the audit"
        )
    return parent


# --------------------------------------------------------------------- examples
def _all_unknown() -> dict[str, str]:
    return {c: "unknown" for c in CLASSES}


def import_dime_dialog(
    dialog_id: str, dime_dir: str | Path, root: str | Path
) -> Example:
    """One whole dialogue as UNLABELED material: every class is 'unknown'.

    Paths: ``<dime_dir>/transcripciones_y_actos/<id>.txt`` and
    ``<dime_dir>/audio_wav_16k_mono/<id>.wav``. XLS siblings are never read.
    """
    if not _DIALOG_ID.fullmatch(dialog_id):
        raise CorpusFormatError(f"Unsafe dialogue id {dialog_id!r}")
    dime_dir, root = Path(dime_dir), Path(root)
    txt = dime_dir / "transcripciones_y_actos" / f"{dialog_id}.txt"
    wav = dime_dir / "audio_wav_16k_mono" / f"{dialog_id}.wav"
    dialog = parse_dime_txt(txt)
    duration = wav_info(wav)
    lines = [dime_input_view(u.raw_text) for u in dialog.utterances]
    claimed = dialog.header.get("Number of utterances files")
    return Example(
        corpus=CORPUS,
        recording_id=dialog_id,
        speaker_id=None,
        group_id=group_for_name(wav.name),
        source_files=tuple(
            SourceFile(relative_posix(root, p), sha256_file(p)) for p in (txt, wav)
        ),
        original_text="\n".join(f"{u.utt_id}: {u.raw_text}" for u in dialog.utterances),
        original_annotations=_txt_cues(dialog.utterances),
        temporal_unit="seconds",
        start_s=0.0,
        end_s=duration,
        source_label=None,
        proposed_mrcd_mapping=None,
        mapping_justification="No disfluency labels exist for this dialogue (TXT cues are untimed).",
        coverage=_all_unknown(),
        negative_basis={},
        provenance="unknown",
        input_view="\n".join(x for x in lines if x),
        audio_duration_s=duration,
        extra={
            "normalisation": NORMALISATION_NOTES,
            "header_dialogue_name": dialog.header.get("Dialogue"),
            "header_utterances_declared": claimed,
            "utterances_parsed": len(dialog.utterances),
            "header_length_s": dialog.header.get("Length of dialogue"),
            "xls_not_used": True,
        },
    )


_REPAIR_MAP = {
    "modification repair": (
        "revision",
        "exact",
        "DIME 'modification repair' (non-empty reparandum, restart/replacement of the planned "
        "phrase) is the closest MRCD reading: revision. Repetition is only a secondary, "
        "non-adjacent reading (reparandum 'un' recurs after a 2-word insertion) and is not "
        "emitted as an event.",
    ),
}


def _words_inside(words: list[PhnInterval], lo: float, hi: float, tol: float = 0.0015):
    return [
        w.label
        for w in words
        if w.start_s >= lo - tol and w.end_s <= hi + tol and not w.label.startswith(".")
    ]


def import_dime_sample(
    sample: str,
    muestras_dir: str | Path,
    root: str | Path,
    sample_parents: dict[str, str],
    *,
    word_tier: str,
    repair_tiers: tuple[str, str] | None = None,
) -> Example:
    """Import a time-aligned DIME sample (verification only, n=1; never an evaluation).

    ``repair_tiers`` = (rep_tipo file, rep_est file). Without them nothing is labeled and
    all classes stay 'unknown' (utt1 carries no repair tiers).
    """
    if not re.fullmatch(r"utt\d+", sample):
        raise CorpusFormatError(f"Unsafe sample id {sample!r}")
    mdir, root = Path(muestras_dir), Path(root)
    wav = mdir / f"{sample}.wav"
    duration = wav_info(wav)
    group = _group_for_sample(sample, sample_parents)
    words = parse_phn(mdir / word_tier, dense=True)
    if words[-1].end_s > duration + 0.002:
        raise InvalidBoundsError(f"{sample}: word tier ends after the audio")
    files = [wav, mdir / word_tier]
    annotations: list[Annotation] = []
    label_text = None
    mapping_cls = None
    just = "No repair tiers shipped for this sample: nothing is labeled."
    # untimed context: the word tier itself is an alignment, not a disfluency label
    sil = [w for w in words if w.label == ".sil" and (w.end_s - w.start_s) > 0.0]
    if repair_tiers:
        tipo_f, est_f = repair_tiers
        files += [mdir / tipo_f, mdir / est_f]
        tipo = parse_phn(mdir / tipo_f)
        est = {norm_label(i.label): i for i in parse_phn(mdir / est_f)}
        if len(tipo) != 1 or {"reparandum", "alteracion"} - set(est):
            raise CorpusFormatError(
                f"{sample}: expected one repair type and reparandum+alteracion"
            )
        label_text = tipo[0].label.strip().strip('"').strip().lower()
        rep, alt = est["reparandum"], est["alteracion"]
        if not rep.end_s <= alt.start_s:
            raise InvalidBoundsError(
                f"{sample}: reparandum must end before the alteration starts"
            )
        if alt.end_s > duration + 0.002:
            raise InvalidBoundsError(f"{sample}: alteration exceeds the audio")
        cls, strength, just = _REPAIR_MAP.get(label_text, (None, "none", ""))
        if cls is None:
            just = f"Repair type {label_text!r} has not been audited; not mapped."
        mapping_cls = cls
        edit_silence = [
            w
            for w in sil
            if rep.end_s - 1e-3 <= w.start_s and w.end_s <= alt.start_s + 1e-3
        ]
        annotations.append(
            Annotation(
                "dime_phn_repair",
                label_text,
                "human",
                start_s=rep.start_s,
                end_s=alt.end_s,
                locator={
                    "reparandum": [rep.start_s, rep.end_s],
                    "alteration": [alt.start_s, alt.end_s],
                    "reparandum_words": _words_inside(words, rep.start_s, rep.end_s),
                    "alteration_words": _words_inside(words, alt.start_s, alt.end_s),
                    "editing_silence_s": [[w.start_s, w.end_s] for w in edit_silence],
                    "provenance_basis": "paper: repair tiers were hand-tagged; not stated per file",
                },
                proposed_class=cls,
                mapping_strength=strength if cls else "none",
                justification=just if cls else "",
            )
        )
    for w in sil:  # timed silences: real, but no functional class can be decided
        annotations.append(
            Annotation(
                "dime_phn_silence",
                "sil",
                "unknown",
                start_s=w.start_s,
                end_s=w.end_s,
                justification="",
                locator={
                    "note": "silence without rhetorical/neutral function; not mapped"
                },
            )
        )
    view = " ".join(w.label for w in words if not w.label.startswith("."))
    cov = _all_unknown()
    if mapping_cls:
        cov[mapping_cls] = "pos"
    return Example(
        corpus=CORPUS,
        recording_id=sample,
        speaker_id=None,
        group_id=group,
        source_files=tuple(
            SourceFile(relative_posix(root, p), sha256_file(p)) for p in files
        ),
        original_text=view,
        original_annotations=tuple(annotations),
        temporal_unit="seconds",
        start_s=0.0,
        end_s=duration,
        source_label=label_text,
        proposed_mrcd_mapping=mapping_cls,
        mapping_justification=just,
        coverage=cov,
        negative_basis={},
        provenance="human" if mapping_cls else "unknown",
        input_view=view,
        audio_duration_s=duration,
        extra={
            "verification_only": True,
            "parent_dialogue": group,
            "coverage_note": "the sample declares no negative coverage: other classes unknown",
        },
    )


def import_utt74(muestras_dir, root, sample_parents) -> Example:
    return import_dime_sample(
        "utt74",
        muestras_dir,
        root,
        sample_parents,
        word_tier="utt74_pal.phn",
        repair_tiers=("utt74_rep_tipo.phn", "utt74_rep_est.phn"),
    )


def import_utt1(muestras_dir, root, sample_parents) -> Example:
    return import_dime_sample(
        "utt1", muestras_dir, root, sample_parents, word_tier="palabrautt1.phn"
    )


def list_dialog_ids(dime_dir: str | Path) -> list[str]:
    """Dialogue ids that have BOTH a TXT and a WAV (anything else is ignored, not guessed)."""
    d = Path(dime_dir)
    txt = {p.stem for p in (d / "transcripciones_y_actos").glob("d??.txt")}
    wav = {p.stem for p in (d / "audio_wav_16k_mono").glob("d??.wav")}
    return sorted(i for i in txt & wav if _DIALOG_ID.fullmatch(i))
