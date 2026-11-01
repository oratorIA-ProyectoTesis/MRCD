"""Common example schema, validation and leakage guards for corpus importers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mrcd.constants import TAXONOMY

CLASSES: tuple[str, ...] = TAXONOMY
PROVENANCES = ("human", "automatic", "heuristic", "unknown")
COVERAGE_STATES = ("pos", "neg", "unknown")
MAPPING_STRENGTHS = ("exact", "approximate", "none")
TEMPORAL_UNITS = ("seconds", "untimed")
DURATION_TOL_S = 0.002  # PHN tiers end up to ~1 ms away from the WAV header duration


class CorpusFormatError(ValueError):
    """The source file does not have the documented structure."""


class UnsafeArchiveError(CorpusFormatError):
    """An archive member would escape the extraction root or is oversized."""


class InvalidBoundsError(ValueError):
    """Temporal bounds are inconsistent (order, audio duration, overlap)."""


class ForbiddenMappingError(ValueError):
    """A source label was mapped to an MRCD class that the audit forbids."""


class LeakError(ValueError):
    """A response token, identifier or target column reached model-visible input."""


def norm_label(s: str) -> str:
    return (
        unicodedata.normalize("NFD", s)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


# Basic confusables folded to their Latin look-alike BEFORE any comparison (Cyrillic/Greek letters that render
# like Latin ones, long s, dotless i). NFKC already folds full-width and compatibility forms.
_CONFUSABLES = str.maketrans(
    {
        "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0443": "y",
        "\u0445": "x", "\u0456": "i", "\u0458": "j", "\u0455": "s", "\u043d": "h", "\u043a": "k",
        "\u0442": "t", "\u043c": "m", "\u0432": "b",
        "\u03bf": "o", "\u03b1": "a", "\u03b5": "e", "\u03bd": "v", "\u03c1": "p", "\u03c4": "t",
        "\u03b9": "i", "\u03ba": "k", "\u03c5": "u", "\u03b2": "b",
        "\u017f": "s", "\u0131": "i", "\u0269": "i", "\u01c0": "l",
    }
)


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise ForbiddenMappingError(
            f"{what} must be a plain str, got {type(value).__name__}"
        )
    return value


def fold_label(s: str) -> str:
    """NFKC, basic confusables folded, case-folded, accents/combining marks and zero-width characters removed."""
    t = unicodedata.normalize("NFKC", _require_str(s, "label"))
    t = t.casefold().translate(_CONFUSABLES)
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) not in ("Mn", "Cf", "Cc"))


def canon_label(s: str) -> str:
    """Guard key: folded (see ``fold_label``) and ONLY letters/digits (spaces, hyphens, underscores, dots, ``<>`` and
    any other punctuation removed), so ``"Palabra-cortada"``, ``"palabra cortada"``, ``".sil"`` and ``"<sil>"``
    collapse. Non-str input is refused (no ``str()`` coercion of tuples/dicts)."""
    return re.sub(r"[\W_]+", "", fold_label(s))


_PAUSE_CLASSES = frozenset({"block", "rhetorical_pause", "neutral_pause"})
# Canonical source label (see canon_label) -> classes it must never be mapped to (audit decisions).
FORBIDDEN_MAPPINGS: dict[str, frozenset[str]] = {
    "palabracortada": frozenset({"revision", "block"}),
    "fragmento": frozenset({"revision", "block"}),
    "pausa": _PAUSE_CLASSES,
    "pausacorta": _PAUSE_CLASSES,
    "pausalarga": _PAUSE_CLASSES,
    "silencio": _PAUSE_CLASSES,
    "silence": _PAUSE_CLASSES,
    "sil": _PAUSE_CLASSES,  # DIME <sil>, .sil
    "sp": _PAUSE_CLASSES,  # HTK-style short pause
}
# Family stems (defence in depth behind the positive allow-list): a label that CONTAINS one of these, or that
# equals a short token, is in the family whatever its plural, numeric suffix or prefix ('pausas', 'pausa 2',
# 'silent pausa', 'fragmentos', 'pause').
_FAMILY_CONTAINS: dict[str, frozenset[str]] = {
    "palabracortada": FORBIDDEN_MAPPINGS["palabracortada"],
    "fragment": FORBIDDEN_MAPPINGS["fragmento"],
    "pausa": _PAUSE_CLASSES,
    "pause": _PAUSE_CLASSES,
    "silenc": _PAUSE_CLASSES,
    "silent": _PAUSE_CLASSES,
}
_FAMILY_EXACT: dict[str, frozenset[str]] = {"sil": _PAUSE_CLASSES, "sp": _PAUSE_CLASSES}

# POSITIVE list: the only (source_kind, source_label) pairs that may carry a class, and the classes they may carry.
# Behaviour (decided in revision 6, fail-loud): an Annotation that asks for a class for a pair NOT listed here
# RAISES ForbiddenMappingError; nothing downgrades it to "unmapped" automatically. An importer that meets a new
# legitimate label must either emit it with proposed_class=None itself or extend this table (an audit decision).
# Keys are canon_label forms. The table must not contain a forbidden family (checked below at import time).
ALLOWED_MAPPINGS: dict[str, dict[str, frozenset[str]]] = {
    "dihanatag": {
        "fa": frozenset({"filler_word"}),
        "fe": frozenset({"filler_word"}),
        "fm": frozenset({"filler_word"}),
        "fsinidentificar": frozenset({"filler_word"}),
        "a": frozenset({"prolongation"}),
        "rrepeticion": frozenset({"repetition"}),
        "rsustitucion": frozenset({"revision"}),
        "rinsercion": frozenset({"revision"}),
        "rborrado": frozenset({"revision"}),
    },
    "esloratag": {"alargamiento": frozenset({"prolongation"})},
    "dimephnrepair": {"modificationrepair": frozenset({"revision"})},
}
# Sources that are not disfluency labels at all: any mapping is refused.
FORBIDDEN_SOURCE_KINDS = frozenset(
    {"dime_xls_dialogue_act", "dime_xls_cargo_abono", "dime_xls_modalidad"}
)


def is_forbidden_source_kind(kind: str) -> bool:
    """Case-insensitive; every DIME XLS kind (and a bare ``xls``) is forbidden, whatever its suffix."""
    c = canon_label(kind)
    return c in {canon_label(k) for k in FORBIDDEN_SOURCE_KINDS} or "xls" in c


def forbidden_classes(label: str | None) -> frozenset[str]:
    """Classes a label's FAMILY must never map to (exact deny-list + plurals/suffixes/prefixes of the families)."""
    if label is None:
        return frozenset()
    c = canon_label(label)
    base = c.rstrip("0123456789")
    out: set[str] = set(FORBIDDEN_MAPPINGS.get(c, ())) | set(FORBIDDEN_MAPPINGS.get(base, ()))
    for key in {c, base}:
        out |= _FAMILY_EXACT.get(key, frozenset())
    for stem, classes in _FAMILY_CONTAINS.items():
        if stem in c:
            out |= classes
    return frozenset(out)


#: Categories never allowed in a stored label: any control/format/surrogate/private/unassigned (C*), any combining
#: mark left after NFKC (M*: U+034F, U+FE0F...), any separator other than the ASCII space (Z*: U+1680, U+2028/9...),
#: and modifier/other symbols (Sk/So: U+2800 braille blank). Letters, digits, ASCII punctuation, ``<>`` are fine.
_REJECTED_CATEGORIES = ("C", "M", "Zl", "Zp", "Zs", "Sk", "So")


def is_plain_label(s: str) -> bool:
    """True when ``s`` is stored exactly as written: no compatibility/full-width forms (NFKC changes it), no
    confusable look-alike letters, and no invisible/format characters (see ``_REJECTED_CATEGORIES``; the plain
    ASCII space is the only separator allowed). Precomposed accents are fine. This is hygiene; the guarantee
    that a stored mapped label is the canonical spelling is ``is_canonical_spelling``."""
    s = _require_str(s, "label")
    return (
        unicodedata.normalize("NFKC", s) == s
        and not any(ord(c) in _CONFUSABLES for c in s.casefold())
        and not any(
            c != " " and unicodedata.category(c).startswith(_REJECTED_CATEGORIES) for c in s
        )
        and s == s.strip()
    )


#: The ONLY raw spellings that may carry a class. Keys are canon_label forms (same as ALLOWED_MAPPINGS).
CANONICAL_SPELLINGS: dict[str, dict[str, str]] = {
    "dihanatag": {
        "fa": "f:a", "fe": "f:e", "fm": "f:m", "fsinidentificar": "f:sin_identificar", "a": "a",
        "rrepeticion": "r:repeticion", "rsustitucion": "r:sustitucion", "rinsercion": "r:insercion",
        "rborrado": "r:borrado",
    },
    "esloratag": {"alargamiento": "alargamiento"},
    "dimephnrepair": {"modificationrepair": "modification repair"},
}
CANONICAL_KINDS = {"dihanatag": "dihana_tag", "esloratag": "eslora_tag", "dimephnrepair": "dime_phn_repair"}


def is_canonical_spelling(source_kind: str, source_label: str) -> bool:
    """True only when the RAW (kind, label) strings equal the canonical spelling of an allow-listed pair
    (so ``FA``, ``f.a``, ``<f:a>``, ``f:a.`` or any invisible-character variant are refused)."""
    k = canon_label(source_kind)
    return (
        CANONICAL_KINDS.get(k) == source_kind
        and CANONICAL_SPELLINGS.get(k, {}).get(canon_label(source_label)) == source_label
    )


def allowed_classes(source_kind: str, source_label: str) -> frozenset[str]:
    """Classes the positive list allows for this exact (kind, label); empty when the pair is not listed."""
    return ALLOWED_MAPPINGS.get(canon_label(source_kind), {}).get(
        canon_label(source_label), frozenset()
    )


def label_allowed_for_class(source_label: str, cls: str) -> bool:
    """Example-level check: some listed source kind maps this label to ``cls``."""
    c = canon_label(source_label)
    return any(cls in labels.get(c, ()) for labels in ALLOWED_MAPPINGS.values())


def _assert_allow_list_is_clean() -> None:
    for kind, labels in ALLOWED_MAPPINGS.items():
        for lab, classes in labels.items():
            bad = classes & forbidden_classes(lab)
            if bad:
                raise AssertionError(f"ALLOWED_MAPPINGS[{kind!r}][{lab!r}] allows forbidden {sorted(bad)}")


_assert_allow_list_is_clean()


_TAG_RE = re.compile(r"<[^<>]*>")
# Response vocabulary that must never be visible to a model (per corpus, on top of any <tag>).
RESPONSE_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "common": (_TAG_RE, re.compile(r"[<>]")),
    "dime": (
        re.compile(r"\+"),
        re.compile(r"[\[\]]"),
        re.compile(r"(?<![A-Za-z0-9])[su]:"),
        re.compile(r"[^\W\d_]-[^\W\d_]"),
    ),
    "eslora": (re.compile(r"(?<![A-Za-z0-9])hab\d+(?![A-Za-z0-9])"),),
    "dihana": (),
}
FORBIDDEN_FEATURE_KEY_PARTS = (
    "label",
    "target",
    "annot",
    "coverage",
    "provenance",
    "file",
    "path",
    "group",
    "recording",
    "speaker",
    "class",
    "sha",
    "mapping",
    "fold",
    "split",
    "event",
)


def ms_to_s(value: float, ms_per_frame: float = 1.0) -> float:
    """Convert a PHN time value (frames of ``ms_per_frame`` ms) to seconds, 1 us precision."""
    if ms_per_frame <= 0:
        raise InvalidBoundsError(
            f"MillisecondsPerFrame must be positive, got {ms_per_frame}"
        )
    return round(float(value) * ms_per_frame / 1000.0, 6)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_info(
    path: str | Path, *, rate: int | None = 16000, channels: int | None = 1
) -> float:
    """Duration (s) from the WAV header; refuses a wrong rate/channel layout."""
    try:
        with wave.open(str(path), "rb") as w:
            r, c, n = w.getframerate(), w.getnchannels(), w.getnframes()
    except (wave.Error, EOFError) as e:
        raise CorpusFormatError(f"Unreadable WAV {path}: {e}") from e
    if rate is not None and r != rate:
        raise CorpusFormatError(f"{path}: sample rate {r} != expected {rate}")
    if channels is not None and c != channels:
        raise CorpusFormatError(f"{path}: {c} channels != expected {channels}")
    return round(n / r, 6)


def relative_posix(root: Path, p: Path) -> str:
    """Path of ``p`` relative to ``root``; refuses anything that resolves outside it."""
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as e:
        raise UnsafeArchiveError(f"{p} is outside the corpus root {root}") from e


@dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str
    member: str | None = None  # name inside a container (ZIP) when applicable

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise CorpusFormatError(f"Invalid sha256 for {self.path}")


@dataclass(frozen=True)
class Annotation:
    """One source annotation (kept apart from the model input) and its audited mapping."""

    source_kind: str  # e.g. "dime_txt_marker", "eslora_tag", "dime_phn_repair"
    source_label: str
    provenance: str
    start_s: float | None = None
    end_s: float | None = None
    text: str | None = None
    locator: dict[str, Any] = field(default_factory=dict)
    proposed_class: str | None = None
    mapping_strength: str = "none"
    justification: str = ""

    def __post_init__(self):
        _require_str(self.source_kind, "source_kind")
        _require_str(self.source_label, "source_label")
        _require_str(self.justification, "justification")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"Invalid provenance {self.provenance!r}")
        if self.mapping_strength not in MAPPING_STRENGTHS:
            raise ValueError(f"Invalid mapping strength {self.mapping_strength!r}")
        if (self.start_s is None) != (self.end_s is None):
            raise InvalidBoundsError("start_s and end_s must be both set or both None")
        if self.start_s is not None and not (0 <= self.start_s < self.end_s):
            raise InvalidBoundsError(
                f"Invalid interval [{self.start_s}, {self.end_s}] for {self.source_label}"
            )
        if self.proposed_class is None:
            if self.mapping_strength != "none":
                raise ValueError(
                    "mapping_strength must be 'none' without a proposed class"
                )
            return
        if is_forbidden_source_kind(self.source_kind):
            raise ForbiddenMappingError(
                f"{self.source_kind} is not a disfluency label source; cannot map to "
                f"{self.proposed_class}"
            )
        if self.proposed_class not in CLASSES:
            raise ForbiddenMappingError(f"Unknown MRCD class {self.proposed_class!r}")
        if self.proposed_class in forbidden_classes(self.source_label):
            raise ForbiddenMappingError(
                f"{self.source_label!r} must not map to {self.proposed_class!r}"
            )
        if self.proposed_class not in allowed_classes(self.source_kind, self.source_label):
            raise ForbiddenMappingError(
                f"({self.source_kind!r}, {self.source_label!r}) is not in the allow-list of mappings to "
                f"{self.proposed_class!r}; rejected (an unlisted pair is never downgraded to unmapped "
                f"automatically: emit proposed_class=None or extend ALLOWED_MAPPINGS)"
            )
        if not (
            is_plain_label(self.source_kind)
            and is_plain_label(self.source_label)
            and is_canonical_spelling(self.source_kind, self.source_label)
        ):
            raise ForbiddenMappingError(
                f"({self.source_kind!r}, {self.source_label!r}) matches an allow-listed pair only through "
                f"homoglyph, full-width, invisible characters or a non-canonical spelling; refusing to store it"
            )
        if self.mapping_strength == "none" or not self.justification.strip():
            raise ForbiddenMappingError(
                "A mapping needs a strength and a justification"
            )

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def find_response_tokens(text: str, corpus: str) -> list[str]:
    """Delimiters/tags/labels of ``corpus`` still present in model-visible text."""
    hits: list[str] = []
    for pat in RESPONSE_PATTERNS["common"] + RESPONSE_PATTERNS.get(corpus, ()):
        hits += [m.group(0) for m in pat.finditer(text)]
    return hits


@dataclass(frozen=True)
class Example:
    corpus: str
    recording_id: str
    speaker_id: str | None
    group_id: str
    source_files: tuple[SourceFile, ...]
    original_text: str
    original_annotations: tuple[Annotation, ...]
    temporal_unit: str
    start_s: float | None
    end_s: float | None
    source_label: str | None
    proposed_mrcd_mapping: str | None
    mapping_justification: str
    coverage: dict[str, str]
    negative_basis: dict[str, str]
    provenance: str
    input_view: str
    audio_duration_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (self.corpus and self.recording_id and self.group_id):
            raise CorpusFormatError("corpus, recording_id and group_id are required")
        if self.speaker_id is not None and not self.speaker_id.strip():
            raise ValueError(
                "speaker_id must be None or a real identifier, never invented"
            )
        if not self.source_files:
            raise CorpusFormatError("At least one hashed source file is required")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"Invalid provenance {self.provenance!r}")
        if self.temporal_unit not in TEMPORAL_UNITS:
            raise ValueError(f"Invalid temporal unit {self.temporal_unit!r}")
        self._check_time()
        self._check_coverage()
        self._check_example_mapping()
        leaks = find_response_tokens(self.input_view, self.corpus)
        if leaks:
            raise LeakError(
                f"{self.recording_id}: response tokens in input_view: {leaks[:5]}"
            )

    def _check_time(self):
        if self.temporal_unit == "untimed":
            if self.start_s is not None or self.end_s is not None:
                raise InvalidBoundsError("Untimed examples must not carry bounds")
            if any(a.start_s is not None for a in self.original_annotations):
                raise InvalidBoundsError(
                    "Untimed examples must not carry timed annotations"
                )
            return
        if (
            self.start_s is None
            or self.end_s is None
            or not (0 <= self.start_s < self.end_s)
        ):
            raise InvalidBoundsError(f"{self.recording_id}: invalid bounds")
        if self.audio_duration_s is None:
            raise InvalidBoundsError(
                "Timed examples need the audio duration to be verified"
            )
        if self.end_s > self.audio_duration_s + DURATION_TOL_S:
            raise InvalidBoundsError(
                f"{self.recording_id}: end {self.end_s} exceeds audio {self.audio_duration_s}"
            )
        by_class: dict[str, list[tuple[float, float]]] = {}
        for a in self.original_annotations:
            if a.start_s is None:
                continue
            if a.start_s < self.start_s - 1e-9 or a.end_s > self.end_s + DURATION_TOL_S:
                raise InvalidBoundsError(
                    f"Annotation {a.source_label} outside example bounds"
                )
            if a.proposed_class:
                by_class.setdefault(a.proposed_class, []).append((a.start_s, a.end_s))
        for cls, ivs in by_class.items():
            ivs.sort()
            for (s1, e1), (s2, e2) in zip(ivs, ivs[1:]):
                if s2 < e1:
                    raise InvalidBoundsError(
                        f"Overlapping {cls} events: {(s1, e1)} / {(s2, e2)}"
                    )

    def _check_example_mapping(self):
        """``source_label`` / ``proposed_mrcd_mapping`` of the Example obey the same guard as the annotations."""
        if self.source_label is not None:
            _require_str(self.source_label, "source_label")
        if self.proposed_mrcd_mapping is None:
            return
        if self.proposed_mrcd_mapping not in CLASSES:
            raise ForbiddenMappingError(
                f"Unknown MRCD class {self.proposed_mrcd_mapping!r}"
            )
        if self.proposed_mrcd_mapping in forbidden_classes(self.source_label):
            raise ForbiddenMappingError(
                f"{self.source_label!r} must not map to {self.proposed_mrcd_mapping!r}"
            )
        if self.source_label is None or not label_allowed_for_class(
            self.source_label, self.proposed_mrcd_mapping
        ):
            raise ForbiddenMappingError(
                f"Example-level label {self.source_label!r} is not in the allow-list for "
                f"{self.proposed_mrcd_mapping!r}"
            )
        if not (
            is_plain_label(self.source_label)
            and any(self.source_label in d.values() for d in CANONICAL_SPELLINGS.values())
        ):
            raise ForbiddenMappingError(
                f"Example-level label {self.source_label!r} is not a canonical spelling "
                f"(homoglyph, full-width, invisible characters or a variant)"
            )
        mapped = {a.proposed_class for a in self.original_annotations if a.proposed_class}
        if self.proposed_mrcd_mapping not in mapped:
            raise ForbiddenMappingError(
                f"Example-level mapping {self.proposed_mrcd_mapping!r} has no annotation mapped to it"
            )

    def _check_coverage(self):
        if set(self.coverage) != set(CLASSES):
            raise ValueError(f"coverage must have exactly the classes {CLASSES}")
        mapped = {
            a.proposed_class for a in self.original_annotations if a.proposed_class
        }
        for cls, state in self.coverage.items():
            if state not in COVERAGE_STATES:
                raise ValueError(f"Invalid coverage state {state!r} for {cls}")
            if (cls in mapped) != (state == "pos"):
                raise ValueError(
                    f"Coverage {state!r} for {cls} inconsistent with annotations"
                )
            if state == "neg" and not self.negative_basis.get(cls, "").strip():
                raise ValueError(
                    f"'neg' for {cls} needs a documented declaration of sufficient coverage"
                )

    # ------------------------------------------------------------------ views
    def model_features(self) -> dict[str, str]:
        """The only thing a model/baseline may see. Keys and text are leak-checked."""
        feats = {"input_text": self.input_view}
        assert_features_clean(feats, self)
        return feats

    def manifest_human_annotation(self) -> dict[str, str]:
        """Value for a manifest ``human_annotation`` entry, without relaxing the contract.

        ``coverage='complete'`` is returned only when every class is pos/neg and the
        provenance is human; otherwise 'partial', which ``build_recording_windows`` rejects.
        """
        complete = self.provenance == "human" and all(
            s in ("pos", "neg") for s in self.coverage.values()
        )
        return {
            "coverage": "complete" if complete else "partial",
            "provenance": f"{self.corpus}:{self.provenance}",
        }

    def events_as_dataset_dicts(self) -> list[dict]:
        """Timed, mapped annotations in the shape of ``read_human_events`` (ms)."""
        return [
            dict(
                category=a.proposed_class,
                start_ms=int(round(a.start_s * 1000)),
                end_ms=int(round(a.end_s * 1000)),
                confidence=1.0,
                source=a.provenance,
            )
            for a in self.original_annotations
            if a.proposed_class and a.start_s is not None
        ]

    def to_dict(self, *, include_annotations: bool = True) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["source_files"] = [vars(s) for s in self.source_files]
        d["original_annotations"] = (
            [a.to_dict() for a in self.original_annotations]
            if include_annotations
            else {"count": len(self.original_annotations)}
        )
        return d


def assert_features_clean(features: dict[str, Any], example: Example) -> None:
    """Raise LeakError if features expose a target column, file/group names or response tokens."""
    for key in features:
        k = str(key).lower()
        if any(part in k for part in FORBIDDEN_FEATURE_KEY_PARTS):
            raise LeakError(f"Feature key {key!r} looks like a target/metadata column")
    identifiers = {example.recording_id, example.group_id}
    for sf in example.source_files:
        base = (sf.member or sf.path).replace("\\", "/").rsplit("/", 1)[-1]
        identifiers.update({base, base.rsplit(".", 1)[0]})
    for key, val in features.items():
        if not isinstance(val, str):
            continue
        toks = find_response_tokens(val, example.corpus)
        if toks:
            raise LeakError(f"Response tokens in feature {key!r}: {toks[:5]}")
        for ident in identifiers:
            if len(ident) >= 3 and re.search(
                rf"(?<![A-Za-z0-9]){re.escape(ident)}(?![A-Za-z0-9])", val
            ):
                raise LeakError(f"Identifier {ident!r} leaked into feature {key!r}")
