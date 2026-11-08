"""Items, the human reference H and the transcript-derived reference R_T (protocol G sections 3, 4, 6).

Pure functions on already-loaded data: nothing here opens audio, a transcript or a model.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.annotation.markers import MARKER_CLASS, check_marker
from scripts.protocol_g import common as C
from scripts.protocol_g.ctc_align import AlignOutput

H = C.H


# ------------------------------------------------------------------------------------------------ items
@dataclass(frozen=True)
class Item:
    item_id: str
    recording: str  # file stem
    file: str
    sha256: str
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_manifest(path: Path) -> list[Item]:
    items = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            stem = Path(r["file"]).stem
            spk = (r.get("speaker_if_in_name") or "").strip() or H.nocando_speaker(stem)
            items.append(
                Item(
                    r["item_id"],
                    stem,
                    r["file"],
                    r["sha256"],
                    float(r["start_s"]),
                    float(r["end_s"]),
                    spk,
                )
            )
    return items


def recordings_of(items: list[Item]) -> dict[str, str]:
    """recording -> speaker."""
    return {i.recording: i.speaker for i in items}


# ------------------------------------------------------------------------------------------------ human reference
@dataclass
class HumanRef:
    events: dict[
        str, list[dict]
    ]  # item_id -> events (class, state, start, end) in RECORDING time
    markers: list[
        dict
    ]  # item_id, recording, speaker, token, function, start, end (recording time)
    inaudible_excluded: set[str]
    n_unknown_rows: int = 0
    n_items: int = 0


def _union_len(iv: list[tuple[float, float]]) -> float:
    iv = sorted(iv)
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def load_human(path: Path, items: list[Item]) -> HumanRef:
    by_id = {i.item_id: i for i in items}
    events: dict[str, list[dict]] = {i.item_id: [] for i in items}
    markers, inaud, unknown = [], {}, 0
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for n, r in enumerate(csv.DictReader(fh), start=2):
            it = by_id.get(r["item_id"])
            if it is None:
                unknown += 1
                continue
            cls = (r["class"] or "").strip().lower()
            if cls == "none":
                continue
            s, e = float(r["start_s"]), float(r["end_s"])
            if cls == "inaudible":
                inaud.setdefault(it.item_id, []).append(
                    (max(0.0, s), min(it.duration, e))
                )
                continue
            if cls == MARKER_CLASS:
                try:
                    tok, fn = check_marker(
                        r.get("marker_token"), r.get("marker_function")
                    )
                except ValueError as exc:
                    raise ValueError(f"{Path(path).name} line {n}: {exc}") from None
                markers.append(
                    {
                        "item_id": it.item_id,
                        "recording": it.recording,
                        "speaker": it.speaker,
                        "token": tok,
                        "function": fn,
                        "start": it.start + s,
                        "end": it.start + e,
                    }
                )
                continue
            events[it.item_id].append(
                {
                    "class": cls,
                    "state": (r["state"] or "").strip().lower(),
                    "start": it.start + s,
                    "end": it.start + e,
                    "item_id": it.item_id,
                    "recording": it.recording,
                    "speaker": it.speaker,
                }
            )
    excluded = {
        i
        for i, iv in inaud.items()
        if _union_len(iv) >= C.INAUDIBLE_SHARE * by_id[i].duration
    }
    return HumanRef(events, markers, excluded, unknown, len(items))


def human_labels(
    h: HumanRef, items: list[Item], cls: str, mode: str
) -> dict[str, int | None]:
    """Item-level label per item: 1 / 0 / None (not scorable).

    mode 'lit'  : >= 1 event of ``cls`` in ANY state (primary H_lit for repetition)
         'disf' : >= 1 event with state disfluent; items with no disfluent and >= 1 uncertain event of ``cls`` -> None
         'unc_pos' / 'unc_neg': as 'disf' but uncertain-only items counted 1 / 0 (S-UNC-POS / S-UNC-NEG)
    An item with inaudible >= 50 % of its duration is None in every mode."""
    out: dict[str, int | None] = {}
    for it in items:
        if it.item_id in h.inaudible_excluded:
            out[it.item_id] = None
            continue
        ev = [
            e
            for e in h.events.get(it.item_id, [])
            if e["class"] == cls
            and e["state"] in ("disfluent", "functional", "uncertain")
        ]
        if mode == "lit":
            out[it.item_id] = int(bool(ev))
        else:
            if any(e["state"] == "disfluent" for e in ev):
                out[it.item_id] = 1
            elif any(e["state"] == "uncertain" for e in ev):
                # protocol 4.1: no disfluent event and >= 1 uncertain event (alone OR together with functional ones) ->
                # the H_disf label is undetermined -> not scorable (S-UNC-POS / S-UNC-NEG score the same items 1 / 0)
                out[it.item_id] = {"disf": None, "unc_pos": 1, "unc_neg": 0}[mode]
            else:
                out[it.item_id] = 0
    return out


def human_event_list(h: HumanRef, cls: str, states: tuple[str, ...]) -> list[dict]:
    return [
        e
        for evs in h.events.values()
        for e in evs
        if e["class"] == cls
        and e["state"] in states
        and e["item_id"] not in h.inaudible_excluded
    ]


# ------------------------------------------------------------------------------------------------ R_T
@dataclass
class RTRec:
    available: bool
    reason: str | None = None
    gate: dict = field(default_factory=dict)
    rep_events: list[tuple[float, float]] = field(default_factory=list)
    fill_events: list[tuple[float, float]] = field(default_factory=list)
    n_tokens: int = 0
    n_lexical: int = 0
    dropped_chars: int = 0
    interpolated_tokens: int = 0
    gate_passed: bool = True

    def usable(self, gate_on: bool = True) -> bool:
        return self.available and (self.gate_passed or not gate_on)


class TranscriptFormatError(ValueError):
    """The ``<stem>_ts.txt`` file does not have the declared layout (protocol 4.3, parse rule): R_T for that recording is
    not_available (it counts towards the "more than 1 of 19" rule); nothing is guessed."""


# a self-closing ``<segment .../>`` is an EMPTY segment (one empty string); tried first so it never swallows successors
_SEGMENT = re.compile(
    r"<segment\b[^<>]*/>|<segment\b[^<>]*>(.*?)</segment\s*>", re.S | re.I
)
_TAG_ANY = re.compile(r"<[^<>]*>")


def read_transcript(raw: bytes) -> str:
    """Decode the transcript bytes exactly as protocol F decoded TextGrid bytes (BOM -> UTF-16; else strict UTF-8 with an
    optional UTF-8 BOM; else Latin-1). The layout is parsed by ``transcript_segments``."""
    return H.decode_textgrid(raw)


def transcript_segments(text: str) -> list[str]:
    """Parse rule of ``<stem>_ts.txt`` (protocol 4.3; derived from the file structure of a SPENT-cell recording, never
    from a file of this cell). The file is a markup document: a ``<recording ...>`` element with descriptive attributes
    contains a ``<transcript>`` element made of ``<segment id="...">TEXT</segment>`` elements, with no times.

    - The transcript text is the character content of every ``segment`` element, in document order; one string per
      segment (a self-closing ``<segment .../>`` is one empty segment). Tag names, attribute values (recording id, story, speaker name, dates, transcriber) and anything outside
      a segment element are never text. Character references / entities in the content are decoded (``html.unescape``).
    - A file with no segment element, with markup (``<`` or ``>``) left inside a segment, or with non-blank character data
      outside every element raises ``TranscriptFormatError``.
    """
    import html

    segs = _SEGMENT.findall(text)
    if not segs:
        raise TranscriptFormatError("no <segment> element")
    if any("<" in s or ">" in s for s in segs):
        raise TranscriptFormatError("markup inside a <segment> element")
    rest = _TAG_ANY.sub(" ", _SEGMENT.sub(" ", text))
    if rest.strip():
        raise TranscriptFormatError("character data outside every <segment> element")
    return [html.unescape(s) for s in segs]


def transcript_tokens(text: str) -> tuple[list[str], list[bool]]:
    """(tokens, is_filler flags) of ONE segment text with the vendored rule; the flag looks at the next token of the same
    segment, exactly as protocol F's ``split_fillers`` looked within one TextGrid interval."""
    toks = H.norm_tokens(text)
    return toks, [
        H.is_filler(t, toks[i + 1] if i + 1 < len(toks) else None)
        for i, t in enumerate(toks)
    ]


def segment_tokens(segments: list[str]) -> tuple[list[str], list[bool], list[int]]:
    """Tokens of all segments in order (this sequence is the aligner target), their filler flags (per segment) and the
    segment index of each token."""
    toks, fl, seg = [], [], []
    for k, s in enumerate(segments):
        t, f = transcript_tokens(s)
        toks += t
        fl += f
        seg += [k] * len(t)
    return toks, fl, seg


def rt_from_alignment(segments: list[str], out: AlignOutput) -> RTRec:
    """R_T events and the aligner-confidence gate from one alignment of ``segment_tokens(segments)``. Repetitions are
    detected within a segment only (protocol F detected them within one transcript unit)."""
    from scripts.protocol_g.ctc_align import confidence_gate

    toks, fl, seg = segment_tokens(segments)
    assert [t.tok for t in out.tokens] == toks
    keep_idx = [i for i, f in enumerate(fl) if not f]
    rep = []
    for k in sorted(set(seg)):
        kk = [i for i in keep_idx if seg[i] == k]
        for pos, n in H.repeat_positions([toks[i] for i in kk]):
            a, b = out.tokens[kk[pos]], out.tokens[kk[pos + n - 1]]
            rep.append((a.start, b.end))
    fill = [(out.tokens[i].start, out.tokens[i].end) for i, f in enumerate(fl) if f]
    gate = confidence_gate([out.tokens[i].conf for i in keep_idx])
    return RTRec(
        True,
        None,
        gate,
        rep,
        fill,
        n_tokens=len(toks),
        n_lexical=len(keep_idx),
        dropped_chars=out.dropped_chars,
        interpolated_tokens=out.interpolated_tokens,
        gate_passed=gate["passed"],
    )


def p0_anchor_gate(out: AlignOutput, segments: list[str]) -> dict:
    """S-P0ANCHOR own gate: share of lexical tokens exactly matched to a p0 word >= 70 %."""
    toks, fl, _seg = segment_tokens(segments)
    lex = [i for i, f in enumerate(fl) if not f]
    n_ok = sum(1 for i in lex if out.tokens[i].timed and out.tokens[i].conf == 1.0)
    share = n_ok / len(lex) if lex else None
    return {
        "passed": bool(lex and share >= C.P0ANCHOR_MATCH_SHARE),
        "share": share,
        "n_lexical": len(lex),
    }


def rt_labels(
    rt: dict[str, RTRec], items: list[Item], kind: str, gate_on: bool = True
) -> dict[str, int | None]:
    """Item label under R_T: 1 iff an event midpoint falls in the item; None when the recording is not usable."""
    out = {}
    for it in items:
        r = rt.get(it.recording)
        if r is None or not r.usable(gate_on):
            out[it.item_id] = None
            continue
        evs = r.rep_events if kind == "repetition" else r.fill_events
        out[it.item_id] = int(
            any(
                it.start <= (s + max(e, s + C.EVENT_MIN_WIDTH_S)) / 2 < it.end
                for s, e in evs
            )
        )
    return out


# ------------------------------------------------------------------------------------------------ systems -> items
def system_item_preds(
    events_by_rec: dict[str, dict[str, list]],
    items: list[Item],
    cls: str,
    failed: set[str],
    overlap: bool = False,
) -> tuple[dict[str, int | None], int]:
    """Event -> item by midpoint (events shorter than 20 ms widened); S-OVERLAP: any overlap >= 1 ms. A failed recording
    gives None (excluded), never all-zero. Returns (labels, events in no item)."""
    by_rec: dict[str, list[Item]] = {}
    for it in items:
        by_rec.setdefault(it.recording, []).append(it)
    out: dict[str, int | None] = {}
    dropped = 0
    for rec, its in by_rec.items():
        if rec in failed:
            for it in its:
                out[it.item_id] = None
            continue
        evs = (events_by_rec.get(rec) or {}).get(cls, [])
        hit = set()
        for s, e in evs:
            e = max(e, s + C.EVENT_MIN_WIDTH_S)
            mid = (s + e) / 2
            got = False
            for it in its:
                if (
                    (min(e, it.end) - max(s, it.start) >= 0.001)
                    if overlap
                    else (it.start <= mid < it.end)
                ):
                    hit.add(it.item_id)
                    got = True
            dropped += not got
        for it in its:
            out[it.item_id] = int(it.item_id in hit)
    return out, dropped


def always_present(items: list[Item]) -> dict[str, int]:
    return {i.item_id: 1 for i in items}


# ------------------------------------------------------------------------------------------------ rule on ASR words
def _ns(w):
    from types import SimpleNamespace

    return SimpleNamespace(
        text=w["text"] if "text" in w else w["norm"], start=w["start"], end=w["end"]
    )


_TAG = re.compile(r"^\[([^\[\]]*)\]$")
#: parity with ``mrcd.verbatim.FP_TAG_NORM`` is asserted in tests/test_protocol_g_*.py
FP_TAG_NORM = {
    "uh": "eh",
    "um": "em",
    "eh": "eh",
    "em": "em",
    "er": "eh",
    "ah": "ah",
    "ahh": "ah",
    "hm": "mm",
    "hmm": "mm",
    "mm": "mm",
    "mmm": "mm",
}


def asr_tokens(words: list[dict]) -> list[tuple[str, float, float]]:
    """Normalised tokens with word times; bracketed tags are mapped by FP_TAG_NORM and other tags dropped (cw2)."""
    out = []
    for w in words or []:
        raw = str(w.get("text", w.get("norm", ""))).strip()
        m = _TAG.match(raw)
        if m:
            mapped = FP_TAG_NORM.get(m.group(1).strip().lower())
            if mapped:
                out.append((mapped, float(w["start"]), float(w["end"])))
            continue
        for t in H.norm_tokens(raw):
            out.append((t, float(w["start"]), float(w["end"])))
    return out


def rule_events(words: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """Protocol F ``rule_events`` on the words (tags mapped first) -> classes repetition / filled_pause."""
    toks = asr_tokens(words)
    fillers, rest = [], []
    for i, (t, s, e) in enumerate(toks):
        nxt = toks[i + 1][0] if i + 1 < len(toks) else None
        (fillers if H.is_filler(t, nxt) else rest).append((t, s, e))
    rep = [
        (rest[p][1], rest[p + n - 1][2])
        for p, n in H.repeat_positions([t for t, _, _ in rest])
    ]
    return {"repetition": rep, "filled_pause": [(s, e) for _, s, e in fillers]}


def doc_events_g(doc: dict) -> dict[str, list[tuple[float, float]]]:
    """Analyzer document -> events by class; ``filler_word`` split by subclass (lexical_filler never pooled with filled pauses)."""
    out: dict[str, list[tuple[float, float]]] = {}
    for ev in doc.get("events") or []:
        if ev.get("in_scope", True) is False:
            continue
        cls = ev["class"]
        if cls == "filler_word":
            sub = (ev.get("context") or {}).get("subclass")
            cls = "lexical_filler" if sub == "lexical_filler" else "filled_pause"
        out.setdefault(cls, []).append((float(ev["start_s"]), float(ev["end_s"])))
    return out


def cell_arrays(items: list[Item]):
    return (
        np.array([i.recording for i in items]),
        np.array([i.speaker for i in items]),
    )
