"""Pre-etiquetado heurístico en cascada (propuestas para revisión humana).

IMPORTANTE (metodología): estas etiquetas son CANDIDATAS. Sirven para acelerar la
anotación humana en ELAN; no son ground truth. Entrenar y evaluar el modelo de
fusión sobre ellas es circular (el modelo reaprendería las reglas, que además
usan rasgos cinésicos, lo que favorece artificialmente al modo trimodal).

Cascada:
  1. Acústica (siempre): silencios VAD > 600 ms, F0 (pYIN), RMS.
  2. ASR verbatim: muletillas léxicas, repeticiones, revisiones, fronteras de cláusula.
  3. Cinésica (si hay video): tensión perioral, entrecejo y estabilidad cefálica
     para desambiguar pausa retórica / bloqueo / pausa neutra.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from core.constants import (BACKGROUND, CLASSIFICATION_OF, EDIT_TERMS, FUNCTION_WORDS, H,
                            KIDX, STRIDE_S, TAXONOMY, WINDOW_S)
from core.extractors.acoustic import AcousticTrack
from core.extractors.linguistic import Word, lexicon_for

CLAUSE_PUNCT = {".", "?", "!", "…", "...", ";", ":", ","}
STRONG_PUNCT = {".", "?", "!", "…", "...", ";", ":"}


@dataclass
class Candidate:
    category: str
    start: float
    end: float
    confidence: float
    evidence: dict = field(default_factory=dict)
    label: str = ""
    source: str = "auto"

    @property
    def classification(self) -> str:
        return CLASSIFICATION_OF.get(self.category, "DISFLUENCY")

    def to_dict(self):
        d = asdict(self)
        d["classification"] = self.classification
        d["start_ms"], d["end_ms"] = int(self.start * 1000), int(self.end * 1000)
        return d


# ------------------------------------------------------------ utilidades
def _kin_stats(kin_t, kin_v, kin_det, a, b):
    if kin_t is None:
        return None
    m = (kin_t >= a) & (kin_t <= b) & kin_det
    if m.sum() < 2:
        return None
    v = kin_v[m]
    mp = np.nanmean((v[:, KIDX["mouth_press_left"]] + v[:, KIDX["mouth_press_right"]]) / 2)
    return dict(
        mouth_press=float(mp),
        brow_down=float(np.nanmean(v[:, KIDX["brow_down_avg"]])),
        head_std=float(np.nanmax(np.nanstd(v[:, 0:3], 0))),
        gaze_away=float(np.nanmean(np.nanmax(v[:, 3:7], 1))),
        frames=int(m.sum()),
    )


def _f0_slope_st(ac: AcousticTrack, t_end: float, span: float = 0.30) -> float | None:
    """Pendiente de F0 (semitonos/s) en los últimos `span` s sonoros antes de t_end."""
    m = (ac.times >= t_end - span) & (ac.times < t_end) & ~np.isnan(ac.f0)
    if m.sum() < 5:
        return None
    t = ac.times[m]; st = 12 * np.log2(ac.f0[m] / 100.0)
    return float(np.polyfit(t, st, 1)[0])


def _longest_flat_run(ac: AcousticTrack, a: float, b: float) -> tuple[float, float, float]:
    """Mayor tramo sonoro con rango de F0 <= flat_f0_max_hz y RMS estable. -> (dur, t0, t1)."""
    m = np.where((ac.times >= a) & (ac.times <= b))[0]
    best = (0.0, a, a)
    i = 0
    f0, rms, tt = ac.f0, ac.rms, ac.times
    while i < len(m):
        j = i
        lo = hi = f0[m[i]]
        if np.isnan(lo):
            i += 1; continue
        while j + 1 < len(m) and not np.isnan(f0[m[j + 1]]):
            nlo, nhi = min(lo, f0[m[j + 1]]), max(hi, f0[m[j + 1]])
            if nhi - nlo > H["flat_f0_max_hz"]:
                break
            lo, hi, j = nlo, nhi, j + 1
        seg = rms[m[i]:m[j] + 1]
        stable = seg.mean() > 0 and seg.std() / (seg.mean() + 1e-9) < 0.35
        dur = tt[m[j]] - tt[m[i]] + 0.01
        if stable and dur > best[0]:
            best = (float(dur), float(tt[m[i]]), float(tt[m[j]] + 0.01))
        i = j + 1
    return best


# ------------------------------------------------------------- detectores
def detect_pauses(ac, words, kin):
    out = []
    if not words:
        return out
    first, last = words[0].start, words[-1].end
    for a, b in ac.silence_segments:
        dur = b - a
        if dur < H["silence_min_s"] or a < first or b > last:
            continue
        prev = max((w for w in words if w.end <= a + 0.15), key=lambda w: w.end, default=None)
        nxt = min((w for w in words if w.start >= b - 0.15), key=lambda w: w.start, default=None)
        if prev is None or nxt is None:
            continue
        ks = _kin_stats(*kin, a, b) if kin else None
        slope = _f0_slope_st(ac, a)
        boundary = prev.punct_after in CLAUSE_PUNCT
        strong_boundary = prev.punct_after in STRONG_PUNCT
        intra = (prev.norm in FUNCTION_WORDS) and not boundary
        tone_ok = slope is None or slope < -1.0 or abs(slope) <= 1.0     # L% descendente o H% suspendido
        tense = ks is not None and (ks["mouth_press"] >= H["mouth_press_tense"] or ks["brow_down"] >= H["brow_down_tense"])
        relaxed = ks is not None and ks["mouth_press"] < H["mouth_press_relaxed"] and ks["head_std"] <= H["head_stable_deg"]
        ev = dict(duration_s=round(dur, 3), prev_word=prev.text, next_word=nxt.text,
                  clause_boundary=boundary, intra_constituent=intra,
                  f0_slope_st_per_s=None if slope is None else round(slope, 2), kinesic=ks)
        if intra and tense:
            cat, conf = "block", 0.80
        elif intra and ks is None:
            cat, conf = "block", 0.50          # sin evidencia facial: candidato débil
        elif boundary and H["rhetorical_min_s"] <= dur <= H["rhetorical_max_s"] and tone_ok and (relaxed or ks is None):
            cat, conf = "rhetorical_pause", (0.80 if relaxed and strong_boundary else 0.65 if relaxed else 0.55)
        elif tense and not boundary:
            cat, conf = "block", 0.60
        else:
            cat, conf = "neutral_pause", 0.60
        out.append(Candidate(cat, a, b, conf, ev))
    return out


def detect_low_conf_pauses(ac, words, kin):
    """Silencios 0.35–0.6 s tras palabra funcional: posibles bloqueos breves (solo revisión)."""
    out = []
    for a, b in ac.silence_segments:
        if not (0.35 <= b - a < H["silence_min_s"]):
            continue
        prev = max((w for w in words if w.end <= a + 0.15), key=lambda w: w.end, default=None)
        if prev is None or prev.norm not in FUNCTION_WORDS or prev.punct_after in CLAUSE_PUNCT:
            continue
        ks = _kin_stats(*kin, a, b) if kin else None
        out.append(Candidate("block", a, b, 0.40, dict(duration_s=round(b - a, 3), prev_word=prev.text, kinesic=ks),
                             source="auto_low_conf"))
    return out


def detect_fillers(ac, words, variety):
    lex = lexicon_for(variety)
    ambiguous = {"este", "bueno", "pues", "tipo", "o sea", "ya", "a ver", "digamos", "pe"}
    out, skip = [], set()
    for i, w in enumerate(words):
        if i in skip:
            continue
        nxt = words[i + 1] if i + 1 < len(words) else None
        prev = words[i - 1] if i else None
        tok, end = w.norm, w.end
        if nxt and f"{w.norm} {nxt.norm}" in lex:
            tok, end = f"{w.norm} {nxt.norm}", nxt.end
            skip.add(i + 1)
        if tok not in lex:
            continue
        gap_b = (w.start - prev.end) if prev else 1.0
        gap_a = (nxt.start - end) if nxt and tok == w.norm else ((words[i + 2].start - end) if i + 2 < len(words) else 1.0)
        dur = end - w.start
        if tok in ambiguous:
            # Palabras con uso léxico pleno: exigir pausa adyacente, alargamiento o coma
            if not (gap_b >= 0.15 or gap_a >= 0.15 or dur >= 0.35 or w.punct_after in {",", "...", "…"}):
                continue
            conf = 0.60 + 0.15 * min(1.0, dur / 0.6)
        else:
            conf = 0.85
        out.append(Candidate("filler_word", w.start, end, round(min(conf, 0.95) * (0.7 + 0.3 * w.prob), 3),
                             dict(token=tok, duration_s=round(dur, 3), gap_before=round(gap_b, 3),
                                  gap_after=round(gap_a, 3)), label=w.text))
    # Vacilaciones vocálicas ("eeeh") que el ASR omitió: tramo sonoro plano sin palabra
    covered = np.zeros(len(ac.times), bool)
    for w in words:
        covered[(ac.times >= w.start - 0.05) & (ac.times <= w.end + 0.05)] = True
    voiced = (~np.isnan(ac.f0)) & (ac.vad == 1) & ~covered
    run_start = None
    for k, v in enumerate(np.append(voiced, False)):
        if v and run_start is None:
            run_start = k
        elif not v and run_start is not None:
            a, b = ac.times[run_start], ac.times[k - 1] + 0.01
            if b - a >= 0.25:
                dur, _, _ = _longest_flat_run(ac, a, b)
                if dur >= 0.20:
                    out.append(Candidate("filler_word", float(a), float(b), 0.60,
                                         dict(token="<vocalic>", duration_s=round(b - a, 3), asr_missed=True),
                                         label="<eh>"))
            run_start = None
    return out


def _voiced_fraction(ac: AcousticTrack, a: float, b: float) -> float:
    """Proporción del intervalo con habla activa y energía sonora o fricativa."""
    m = (ac.times >= a) & (ac.times < b)
    if not m.any():
        return 0.0
    active = (ac.vad[m] == 1) & (~np.isnan(ac.f0[m]) | ((ac.hf_ratio[m] >= 0.4) if ac.hf_ratio is not None else False))
    return float(active.mean())


def _longest_fricative_run(ac: AcousticTrack, a: float, b: float) -> tuple[float, float, float]:
    """Mayor tramo SORDO con energía de alta frecuencia estable (/s/, /f/, /x/) y RMS estable."""
    if ac.hf_ratio is None:
        return (0.0, a, a)
    m = np.where((ac.times >= a) & (ac.times <= b))[0]
    ok = np.isnan(ac.f0[m]) & (ac.hf_ratio[m] >= 0.5) & (ac.vad[m] == 1)
    best, i = (0.0, a, a), 0
    while i < len(m):
        if not ok[i]:
            i += 1; continue
        j = i
        while j + 1 < len(m) and ok[j + 1]:
            j += 1
        seg_r, seg_h = ac.rms[m[i]:m[j] + 1], ac.hf_ratio[m[i]:m[j] + 1]
        stable = seg_r.mean() > 0 and seg_r.std() / (seg_r.mean() + 1e-9) < 0.35 and seg_h.std() < 0.10
        dur = ac.times[m[j]] - ac.times[m[i]] + 0.01
        if stable and dur > best[0]:
            best = (float(dur), float(ac.times[m[i]]), float(ac.times[m[j]] + 0.01))
        i = j + 1
    return best


def detect_prolongations(ac, words, variety, fillers, recall_mode: bool = False):
    """Vocálicas (F0 plano) y fricativas (energía HF estable) >= 350 ms, fuera de cierre oracional."""
    lex = lexicon_for(variety)
    filler_spans = [(c.start, c.end) for c in fillers]
    out = []
    for w in words:
        if w.norm in lex or any(a <= w.start < b for a, b in filler_spans):
            continue
        if w.punct_after in STRONG_PUNCT:          # alargamiento final de frase = prosodia normal
            continue
        per_char = (w.end - w.start) / max(len(w.norm), 1)
        v_dur, v0, v1 = _longest_flat_run(ac, w.start, w.end)
        f_dur, f0_, f1_ = _longest_fricative_run(ac, w.start, w.end)
        kind, dur, t0, t1 = ("vocalic", v_dur, v0, v1) if v_dur >= f_dur else ("fricative", f_dur, f0_, f1_)
        if dur >= H["prolongation_min_s"] and per_char >= 0.12:
            out.append(Candidate("prolongation", t0, t1, 0.70,
                                 dict(word=w.text, kind=kind, sustained_s=round(dur, 3), sec_per_char=round(per_char, 3)),
                                 label=w.text))
        elif (w.end - w.start) >= 0.6 and per_char >= 0.18 and len(w.norm) >= 3 and _voiced_fraction(ac, w.start, w.end) >= 0.7:
            # el intervalo del ASR debe estar REALMENTE lleno de voz: un timestamp largo sobre una
            # palabra corta suele incluir el silencio posterior, no un alargamiento.
            out.append(Candidate("prolongation", w.start, w.end, 0.50,
                                 dict(word=w.text, kind="duration_only", sec_per_char=round(per_char, 3),
                                      voiced_fraction=round(_voiced_fraction(ac, w.start, w.end), 2)), label=w.text))
        elif recall_mode and dur >= 0.25 and per_char >= 0.10:
            out.append(Candidate("prolongation", t0, t1, 0.40, dict(word=w.text, kind=kind, sustained_s=round(dur, 3)),
                                 label=w.text, source="auto_low_conf"))
    return out


def detect_repetitions(words):
    out, i = [], 0
    while i < len(words) - 1:
        j = i
        partial = False
        while j + 1 < len(words) and words[j + 1].start - words[j].end <= H["repetition_max_gap_s"]:
            a, b = words[j].norm, words[j + 1].norm
            if a == b:
                j += 1
            elif a and b.startswith(a) and len(b) > len(a) and (
                    words[j].text.rstrip().endswith(("-", "…", "...")) or (len(a) <= 3 and a not in FUNCTION_WORDS)):
                partial = True; j += 1; break   # fragmento + palabra completa ("pe- pero")
            else:
                break
        if j > i:
            out.append(Candidate("repetition", words[i].start, words[j].end, 0.70 if partial else 0.80,
                                 dict(tokens=[w.text for w in words[i:j + 1]], repetition_count=j - i + 1,
                                      partial=partial), label=" ".join(w.text for w in words[i:j + 1])))
            i = j + 1
        else:
            i += 1
    return out


def detect_revisions(words):
    """Marcador de edición precedido de cláusula incompleta, o fragmento + reinicio."""
    out = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        big = f"{w.norm} {nxt.norm}" if nxt else ""
        term = big if big in EDIT_TERMS else w.norm if w.norm in EDIT_TERMS else None
        prev3 = words[max(0, i - 3):i]
        incomplete = bool(prev3) and not any(x.punct_after in STRONG_PUNCT for x in prev3)
        fragment = w.text.rstrip().endswith("-")
        if term and incomplete:
            k = i + (2 if term == big else 1)                      # primera palabra de la reparación
            if k >= len(words):
                continue
            e = words[min(len(words) - 1, k + 1)].end
            out.append(Candidate("revision", prev3[0].start, e, 0.55,
                                 dict(edit_term=term, reparandum=[x.text for x in prev3],
                                      repair=[x.text for x in words[k:k + 2]]), label=term))
        elif fragment and nxt and not nxt.norm.startswith(w.norm):
            out.append(Candidate("revision", w.start, nxt.end, 0.50, dict(fragment=w.text, restart=nxt.text), label=w.text))
    return out


# ---------------------------------------------------------------- orquestador
def label_recording(ac: AcousticTrack, words: list[Word], kin_track=None, variety: str = "es-PE",
                    recall_mode: bool = False) -> list[Candidate]:
    """recall_mode=True añade candidatos de confianza 0.35–0.5 (source='auto_low_conf') SOLO para
    revisión humana; build_windows los ignora como etiquetas."""
    kin = None
    if kin_track is not None and len(kin_track.times):
        kin = (kin_track.times, kin_track.vectors, kin_track.detected)
    fillers = detect_fillers(ac, words, variety)
    cands = (detect_pauses(ac, words, kin) + fillers + detect_prolongations(ac, words, variety, fillers, recall_mode)
             + detect_repetitions(words) + detect_revisions(words))
    if recall_mode:
        cands += detect_low_conf_pauses(ac, words, kin)
    # enriquecer con estadísticas cinésicas los eventos no-pausa
    for c in cands:
        if kin and "kinesic" not in c.evidence:
            c.evidence["kinesic"] = _kin_stats(*kin, c.start, c.end)
    return sorted(cands, key=lambda c: (c.start, c.end))


def window_labels(cands: list[Candidate], duration_s: float, window_s: float = WINDOW_S,
                  stride_s: float = STRIDE_S, min_conf: float = 0.5) -> list[dict]:
    """Etiqueta multi-hot + primaria por ventana (3 s / 500 ms)."""
    rows = []
    n = int(np.floor((duration_s - window_s) / stride_s)) + 1 if duration_s >= window_s else 0
    for k in range(n):
        a = round(k * stride_s, 6); b = a + window_s
        multi = {c: 0 for c in TAXONOMY}
        best, best_score = BACKGROUND, 0.0
        for c in cands:
            if c.confidence < min_conf or c.source != "auto":
                continue
            ov = min(b, c.end) - max(a, c.start)
            if ov <= 0 or ov < min(0.30, 0.5 * (c.end - c.start)):
                continue
            multi[c.category] = 1
            score = ov * c.confidence
            if score > best_score:
                best, best_score = c.category, score
        rows.append(dict(start_ms=int(a * 1000), end_ms=int(b * 1000), primary=best, multi=multi))
    return rows


# ------------------------------------------------------------------ export
def write_eaf(path: str | Path, cands: list[Candidate], words: list[Word], silences, media: dict[str, str],
              author: str = "MRCD auto_labeler") -> None:
    """Exporta a ELAN (EAF 3.0) con tiers de candidatos, evidencia, ASR y VAD + tier humano vacío."""
    from xml.sax.saxutils import escape

    slots: dict[int, str] = {}

    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        if ms not in slots:
            slots[ms] = f"ts{len(slots) + 1}"
        return slots[ms]

    aid = [0]

    def ann(a, b, val):
        aid[0] += 1
        return (f'<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a{aid[0]}" TIME_SLOT_REF1="{ts(a)}" '
                f'TIME_SLOT_REF2="{ts(max(b, a + 0.001))}"><ANNOTATION_VALUE>{escape(str(val))}</ANNOTATION_VALUE>'
                f'</ALIGNABLE_ANNOTATION></ANNOTATION>')

    hi = [c for c in cands if c.source == "auto"]
    lo = [c for c in cands if c.source != "auto"]
    tiers = {
        "auto_candidates": [ann(c.start, c.end, c.category) for c in hi],
        "auto_confidence": [ann(c.start, c.end, f"{c.confidence:.2f}") for c in hi],
        "auto_evidence": [ann(c.start, c.end, json.dumps(c.evidence, ensure_ascii=False, default=str)) for c in hi],
        "auto_low_conf": [ann(c.start, c.end, f"{c.category} ({c.confidence:.2f})") for c in lo],
        "asr_words": [ann(w.start, w.end, w.text) for w in words],
        "vad_silence": [ann(a, b, f"sil {b - a:.2f}s") for a, b in silences if b - a >= 0.2],
        "human_disfluency": [],
    }
    media_xml = "".join(
        f'<MEDIA_DESCRIPTOR MEDIA_URL="file:///{escape(Path(p).resolve().as_posix())}" MIME_TYPE="{mt}" '
        f'RELATIVE_MEDIA_URL="./{escape(Path(p).name)}"/>' for p, mt in media.items())
    order = "".join(f'<TIME_SLOT TIME_SLOT_ID="{sid}" TIME_VALUE="{ms}"/>' for ms, sid in sorted(slots.items()))
    cv_tiers = {"auto_candidates", "human_disfluency"}
    auto_attr = ' ANNOTATOR="auto"'
    tier_xml = "".join(
        f'<TIER TIER_ID="{name}" LINGUISTIC_TYPE_REF="{"taxonomy" if name in cv_tiers else "default-lt"}"'
        f'{auto_attr if name.startswith(("auto", "asr", "vad")) else ""}>{"".join(anns)}</TIER>'
        for name, anns in tiers.items())
    cv = "".join(f'<CV_ENTRY_ML CVE_ID="{c}"><CVE_VALUE LANG_REF="spa">{c}</CVE_VALUE></CV_ENTRY_ML>' for c in TAXONOMY)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<ANNOTATION_DOCUMENT AUTHOR="{escape(author)}" DATE="2026-01-01T00:00:00Z" FORMAT="3.0" VERSION="3.0" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">'
        f'<HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">{media_xml}</HEADER>'
        f'<TIME_ORDER>{order}</TIME_ORDER>{tier_xml}'
        '<LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default-lt" TIME_ALIGNABLE="true"/>'
        '<LINGUISTIC_TYPE CONTROLLED_VOCABULARY_REF="mrcd_taxonomy" GRAPHIC_REFERENCES="false" '
        'LINGUISTIC_TYPE_ID="taxonomy" TIME_ALIGNABLE="true"/>'
        '<LANGUAGE LANG_ID="spa" LANG_LABEL="Spanish (spa)"/>'
        f'<CONTROLLED_VOCABULARY CV_ID="mrcd_taxonomy"><DESCRIPTION LANG_REF="spa">Taxonomía MRCD (7 clases)</DESCRIPTION>{cv}</CONTROLLED_VOCABULARY>'
        '</ANNOTATION_DOCUMENT>'
    )
    Path(path).write_text(xml, encoding="utf-8")
