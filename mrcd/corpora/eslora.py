"""ESLORA importer (public text only: no audio, no times, no verified negatives).

* ``alargamiento`` -> ``prolongation`` ONLY as a partial, text-level approximation
  (word-level, untimed, no duration).
* ``palabra_cortada`` is a word fragment: never mapped to revision or block.
* ``pausa`` / ``pausa_larga`` / ``silencio`` have no duration and no function: never
  pauses of any MRCD class. ``vacilacion`` is undefined in the shipped documentation.
* Speaker labels (hab1..) are unique per recording; ``speaker_id`` stays ``None``.
* License is ambiguous (LGPLLR vs CC BY-NC 4.0): the strict reading applies.
"""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from mrcd.corpora.base import (
    CLASSES,
    Annotation,
    CorpusFormatError,
    Example,
    SourceFile,
    UnsafeArchiveError,
    norm_label,
    sha256_bytes,
    sha256_file,
)

CORPUS = "eslora"
MAX_MEMBER_BYTES = 64 * 1024 * 1024
LICENSE_NOTE = "Conflicting statements (LGPLLR vs CC BY-NC 4.0): apply attribution + non-commercial."
NORMALISATION_NOTES = (
    "Closest to verbatim text with conventional orthography, capitals and punctuation added "
    "by transcribers; names replaced by <ficticio>. input_view drops every tag, keeps the "
    "words inside <alargamiento> and <palabra_cortada> (what was said), keeps literal "
    "'eeh'/'mm' (speech, not a delimiter), drops the habN speaker label. Lengthening cannot "
    "be seen in the remaining word. Tags replaced by a space; wrappers by nothing."
)

_TAG = re.compile(r"<(/?)([^\s<>/]+)([^<>]*?)(/?)>")
_PAIRED = {"alargamiento", "palabra_cortada", "ficticio"}
_ANNOTATED = {
    "alargamiento",
    "palabra_cortada",
    "pausa",
    "pausa_larga",
    "silencio",
    "vacilacion",
}
_TURN = re.compile(r"^(hab\d+)\t(.*)$")
_RECORDING = re.compile(r"SCOM_[A-Z0-9]+_\d+")


def safe_member_name(name: str) -> str:
    """Validate an archive member name (zip-slip / absolute / drive / backslash tricks)."""
    if not name or "\x00" in name or "\\" in name:
        raise UnsafeArchiveError(f"Unsafe member name {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or re.match(r"^[A-Za-z]:", name) or ".." in p.parts:
        raise UnsafeArchiveError(f"Unsafe member name {name!r}")
    return name


def read_zip_members(zip_path: str | Path) -> dict[str, bytes]:
    """Read every member in memory (nothing is extracted to disk), validating each name/size."""
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                safe_member_name(info.filename.rstrip("/") or "/")
                continue
            safe_member_name(info.filename)
            if info.file_size > MAX_MEMBER_BYTES:
                raise UnsafeArchiveError(
                    f"{info.filename}: {info.file_size} bytes exceeds cap"
                )
            with z.open(info) as f:
                data = f.read(MAX_MEMBER_BYTES + 1)
            if len(data) > MAX_MEMBER_BYTES:
                raise UnsafeArchiveError(f"{info.filename}: declared size understated")
            out[info.filename] = data
    return out


def _decode(data: bytes, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise CorpusFormatError(f"{name}: not valid UTF-8") from e


def parse_head(text: str) -> dict:
    """Recording metadata and the within-recording speaker labels (habN)."""
    meta: dict[str, str] = {}
    speakers: set[str] = set()
    for line in text.splitlines():
        if "\t" not in line:
            continue
        k, v = line.split("\t", 1)
        if re.fullmatch(r"hab\d+", k):
            speakers.add(k)
        else:
            meta[k.strip()] = v.strip()
    return {"meta": meta, "speaker_labels": sorted(speakers)}


def convert_turn(
    speaker_label: str, raw: str, turn_index: int
) -> tuple[str, list[Annotation]]:
    """(input_text, separate annotations) for one turn; raises on unknown/broken markup."""
    pieces: list[str] = []
    anns: list[Annotation] = []
    open_at: dict[str, tuple[int, int]] = {}  # tag -> (token index, piece index)
    pos = 0

    def n_tokens() -> int:
        return len("".join(pieces).split())

    def emit(tag, text, t0, t1):
        key = norm_label(tag)
        if key not in _ANNOTATED:
            return
        if key == "alargamiento":
            ann = Annotation(
                "eslora_tag",
                tag,
                "human",
                text=text,
                locator={
                    "turn": turn_index,
                    "speaker_label": speaker_label,
                    "input_tokens": [t0, t1],
                },
                proposed_class="prolongation",
                mapping_strength="approximate",
                justification="Word-level human tag of increased sound quantity; untimed, no "
                "duration or acoustic confirmation: partial text-level approximation only.",
            )
        else:
            ann = Annotation(
                "eslora_tag",
                tag,
                "human",
                text=text,
                locator={
                    "turn": turn_index,
                    "speaker_label": speaker_label,
                    "input_tokens": [t0, t1],
                },
            )
        anns.append(ann)

    for m in _TAG.finditer(raw):
        pieces.append(raw[pos : m.start()])
        pos = m.end()
        closing, name, _attrs, selfclose = (
            m.group(1) == "/",
            m.group(2),
            m.group(3),
            m.group(4) == "/",
        )
        if selfclose or (name.endswith(("_inicio", "_fin")) and not closing):
            emit(name, None, n_tokens(), n_tokens())
            pieces.append(" ")
        elif name in _PAIRED:
            if closing:
                if name not in open_at:
                    raise CorpusFormatError(f"Unmatched </{name}> in turn {turn_index}")
                t0, p0 = open_at.pop(name)
                emit(name, "".join(pieces[p0:]).strip() or None, t0, n_tokens())
            else:
                if name in open_at:
                    raise CorpusFormatError(f"Nested <{name}> in turn {turn_index}")
                open_at[name] = (n_tokens(), len(pieces))
        else:
            raise CorpusFormatError(
                f"Unknown markup <{closing and '/' or ''}{name}> in turn {turn_index}"
            )
    pieces.append(raw[pos:])
    if open_at:
        raise CorpusFormatError(f"Unclosed {sorted(open_at)} in turn {turn_index}")
    return " ".join("".join(pieces).split()), anns


def parse_content(text: str) -> list[tuple[str, str]]:
    turns = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = _TURN.match(line)
        if not m:
            raise CorpusFormatError(f"content line {n}: not 'habN<TAB>text'")
        turns.append((m.group(1), m.group(2)))
    if not turns:
        raise CorpusFormatError("empty content")
    return turns


def import_eslora_recording(
    rec_id: str, head: bytes, content: bytes, zip_rel: str, zip_sha: str
) -> Example:
    if not _RECORDING.fullmatch(rec_id):
        raise CorpusFormatError(f"Unsafe recording id {rec_id!r}")
    head_t, content_t = (
        _decode(head, f"{rec_id}_head"),
        _decode(content, f"{rec_id}_content"),
    )
    info = parse_head(head_t)
    audio_name = info["meta"].get("Nombre fichero de audio")
    if audio_name != rec_id:
        raise CorpusFormatError(f"{rec_id}: head names audio {audio_name!r}")
    views, anns = [], []
    other: Counter = Counter()
    for i, (spk, raw) in enumerate(parse_content(content_t)):
        view, a = convert_turn(spk, raw, i)
        anns += a
        for m in _TAG.finditer(raw):
            nm = norm_label(m.group(2))
            if m.group(1) != "/" and nm not in _ANNOTATED:
                other[nm] += 1
        if view:
            views.append(view)
    has_prol = any(a.proposed_class == "prolongation" for a in anns)
    cov = {c: "unknown" for c in CLASSES}
    if has_prol:
        cov["prolongation"] = "pos"
    n_words = sum(len(v.split()) for v in views)
    return Example(
        corpus=CORPUS,
        recording_id=rec_id,
        speaker_id=None,
        group_id=rec_id,
        source_files=(
            SourceFile(f"{zip_rel}", zip_sha, member=f"{rec_id}_head.txt"),
            SourceFile(
                f"{zip_rel}", sha256_bytes(content), member=f"{rec_id}_content.txt"
            ),
        ),
        original_text=content_t,
        original_annotations=tuple(anns),
        temporal_unit="untimed",
        start_s=None,
        end_s=None,
        source_label="alargamiento" if has_prol else None,
        proposed_mrcd_mapping="prolongation" if has_prol else None,
        mapping_justification="alargamiento -> prolongation as a partial text-level approximation."
        if has_prol
        else "",
        coverage=cov,
        negative_basis={},
        provenance="human" if has_prol else "unknown",
        input_view="\n".join(views),
        extra={
            "normalisation": NORMALISATION_NOTES,
            "license": LICENSE_NOTE,
            "n_words": n_words,
            "n_turns": len(views),
            "within_recording_speaker_labels": info["speaker_labels"],
            "other_tag_counts": dict(other),
            "head_meta": info["meta"],
        },
    )


def import_eslora_zip(zip_path: str | Path, root: str | Path) -> list[Example]:
    zip_path = Path(zip_path)
    from mrcd.corpora.base import relative_posix

    members = read_zip_members(zip_path)
    zsha, rel = sha256_file(zip_path), relative_posix(Path(root), zip_path)
    ids = sorted(
        {n[: -len("_content.txt")] for n in members if n.endswith("_content.txt")}
    )
    out = []
    for rid in ids:
        if f"{rid}_head.txt" not in members:
            raise CorpusFormatError(f"{rid}: content without head")
        out.append(
            import_eslora_recording(
                rid,
                members[f"{rid}_head.txt"],
                members[f"{rid}_content.txt"],
                rel,
                zsha,
            )
        )
    heads = {n[: -len("_head.txt")] for n in members if n.endswith("_head.txt")}
    if heads - set(ids):
        raise CorpusFormatError(f"head without content: {sorted(heads - set(ids))[:3]}")
    return out


def eslora_stats(examples: list[Example]) -> dict:
    """Text-only statistics (per 1000 words); not evaluation, not labels."""
    words = sum(e.extra["n_words"] for e in examples)
    tags: Counter = Counter()
    for e in examples:
        tags.update(norm_label(a.source_label) for a in e.original_annotations)
    return {
        "recordings": len(examples),
        "turns": sum(e.extra["n_turns"] for e in examples),
        "words": words,
        "tag_counts": dict(tags),
        "tags_per_1000_words": {k: round(1000 * v / words, 2) for k, v in tags.items()},
        "recordings_with_prolongation_tag": sum(
            e.coverage["prolongation"] == "pos" for e in examples
        ),
        "note": "text-level only; no audio, no times, no verified negatives",
    }
