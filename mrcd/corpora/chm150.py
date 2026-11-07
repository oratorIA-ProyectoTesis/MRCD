"""CHM150 importer (LDC2016S04, "Corpus Hecho en Mexico", Mexican Spanish, CC BY-SA 4.0).

External test corpus only: nothing here is used for fitting, selection or threshold tuning.

* Layout (as shipped): ``CHM150/data/transcriptions/CHM150.transcription`` (one line per utterance:
  ``<orthographic text> <utterance id>``), ``CHM150/docs/CHM150.paths`` (``/speech/<date>/<speaker>/<id>.flac``,
  relative to ``CHM150/data``) and 2 663 FLAC files (16 kHz, mono).
* Group = speaker folder (``F04MAY1239``: sex + recording day/month + session time). The README states 150
  speakers; the folder is the only speaker identifier shipped, so ``speaker_id = group_id = folder``.
* The transcription is orthographic. It writes filled pauses and some fragments as words, but it has NO
  disfluency annotation (no repetition, revision, prolongation or filler marks, no cut-word marks). Every class
  therefore stays ``unknown`` here; any evaluation reference is derived elsewhere
  (``mrcd.context_eval.transcript_reference``) and is declared as derived, never as human disfluency labels.
* Downloaded data is untrusted: paths are validated against a strict pattern and must resolve inside the
  corpus root; nothing inside the corpus is executed.
"""

from __future__ import annotations

import re
from pathlib import Path

from mrcd.corpora.base import (
    CLASSES,
    CorpusFormatError,
    Example,
    SourceFile,
    UnsafeArchiveError,
    relative_posix,
    sha256_file,
)

CORPUS = "chm150"
LICENSE_NOTE = (
    "CHM150 Corpus by Carlos Daniel Hernandez Mena (UNAM, 2012), LDC2016S04, licensed under CC BY-SA 4.0. "
    "Attribution required; derived datasets must keep the same license. Audio and transcriptions are not "
    "redistributed by this repository."
)
NO_ANNOTATION_NOTE = (
    "Orthographic transcription without disfluency annotation: no class has a human positive or a declared "
    "negative coverage."
)

TRANSCRIPTION = Path("data/transcriptions/CHM150.transcription")
PATHS = Path("docs/CHM150.paths")
_SPK = r"[FM]\d{2}[A-Z]{3}\d{4}"
UTT_RE = re.compile(rf"^({_SPK})_(\d{{4}})$")
_PATH_RE = re.compile(
    rf"^/speech/(\d{{2}}[A-Z]{{3}}\d{{4}})/({_SPK})/(({_SPK})_\d{{4}})\.flac$"
)


def speaker_of(utt_id: str) -> str:
    m = UTT_RE.match(utt_id)
    if not m:
        raise CorpusFormatError(f"Invalid CHM150 utterance id {utt_id!r}")
    return m.group(1)


def parse_transcription(text: str) -> list[tuple[str, str]]:
    """``[(utt_id, orthographic_text)]`` in file order. Fails loud on malformed lines or duplicate ids."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        uid, words = parts[-1], parts[:-1]
        if not UTT_RE.match(uid):
            raise CorpusFormatError(
                f"line {n}: last field {uid!r} is not an utterance id"
            )
        if not words:
            raise CorpusFormatError(f"line {n}: empty transcription for {uid}")
        if uid in seen:
            raise CorpusFormatError(f"line {n}: duplicate utterance id {uid}")
        seen.add(uid)
        out.append((uid, " ".join(words)))
    if not out:
        raise CorpusFormatError("empty transcription file")
    return out


def parse_paths(text: str) -> dict[str, tuple[str, str]]:
    """``{utt_id: (relative path under data/, session date)}``; the file stem must match its speaker folder."""
    out: dict[str, tuple[str, str]] = {}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        m = _PATH_RE.match(line)
        if not m:
            raise UnsafeArchiveError(f"paths line {n}: unexpected path {line!r}")
        date, folder, uid, spk = m.groups()
        if folder != spk:
            raise CorpusFormatError(
                f"paths line {n}: {uid} is filed under speaker folder {folder}"
            )
        if uid in out:
            raise CorpusFormatError(f"paths line {n}: duplicate path for {uid}")
        out[uid] = (line.lstrip("/"), date)
    return out


def _audio_info(path: Path) -> float:
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except RuntimeError as e:
        raise CorpusFormatError(f"Unreadable audio {path.name}: {e}") from e
    if info.samplerate != 16000 or info.channels != 1:
        raise CorpusFormatError(
            f"{path.name}: {info.samplerate} Hz / {info.channels} ch != 16000 Hz mono"
        )
    return round(info.frames / info.samplerate, 6)


def import_chm150(root: str | Path) -> list[Example]:
    """All CHM150 utterances. ``root`` is the extracted ``CHM150`` directory (the one holding ``data/`` and ``docs/``)."""
    root = Path(root)
    tr_path, paths_path = root / TRANSCRIPTION, root / PATHS
    for p in (tr_path, paths_path):
        if not p.is_file():
            raise CorpusFormatError(f"missing {p}")
    try:
        utts = parse_transcription(tr_path.read_text(encoding="utf-8"))
        paths = parse_paths(paths_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as e:
        raise CorpusFormatError(f"CHM150 text files must be UTF-8: {e}") from e
    ids = {u for u, _ in utts}
    if ids != set(paths):
        raise CorpusFormatError(
            f"transcription/paths mismatch: {len(ids - set(paths))} ids without audio, "
            f"{len(set(paths) - ids)} audio files without transcription"
        )
    tr_src = SourceFile(relative_posix(root, tr_path), sha256_file(tr_path))
    data = root / "data"
    out = []
    for uid, text in utts:
        rel, date = paths[uid]
        audio = data / rel
        # relative_posix refuses anything resolving outside the corpus root (symlink / traversal)
        a_rel = relative_posix(root, audio)
        if not audio.is_file():
            raise CorpusFormatError(f"missing audio for {uid}")
        spk = speaker_of(uid)
        a_src = SourceFile(a_rel, sha256_file(audio))
        out.append(
            Example(
                corpus=CORPUS,
                recording_id=uid,
                speaker_id=spk,
                group_id=spk,
                source_files=(a_src, tr_src),
                original_text=text,
                original_annotations=(),
                temporal_unit="untimed",
                start_s=None,
                end_s=None,
                source_label=None,
                proposed_mrcd_mapping=None,
                mapping_justification=NO_ANNOTATION_NOTE,
                coverage={c: "unknown" for c in CLASSES},
                negative_basis={},
                provenance="human",
                input_view=text,
                audio_duration_s=_audio_info(audio),
                extra={"sex": spk[0], "session_date": date, "audio_relpath": rel},
            )
        )
    return out
