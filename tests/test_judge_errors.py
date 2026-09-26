"""Pruebas del manejo de fallos del juez.

El bug que originó este archivo: `call_openai` leía sólo `message.content`. Cuando
el modelo se niega, OpenAI deja ese campo en null y pone el motivo en
`message.refusal`. El resultado era una cadena vacía que reventaba río abajo como
«ValueError: substring not found», y el bucle descartaba el error sin registrar
nada. Los fallos eran invisibles y su causa, indeterminable a posteriori.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.llm_judge as J


# ---------------------------------------------------- lectura de la respuesta
class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_openai(monkeypatch, payload):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(payload))


def test_openai_refusal_is_surfaced_not_swallowed(monkeypatch):
    """El caso real que rompió la adjudicación."""
    _patch_openai(monkeypatch, {"choices": [{"finish_reason": "stop", "message": {
        "content": None, "refusal": "I'm sorry, I can't help with identifying people in images."}}]})
    with pytest.raises(J.JudgeRefusal) as ei:
        J.call_openai("gpt-4o", "p", [b"x"], "k")
    assert "identifying people" in ei.value.reason


def test_openai_content_filter_is_surfaced(monkeypatch):
    _patch_openai(monkeypatch, {"choices": [{"finish_reason": "content_filter",
                                             "message": {"content": None}}]})
    with pytest.raises(J.JudgeRefusal):
        J.call_openai("gpt-4o", "p", [], "k")


def test_openai_empty_content_is_surfaced(monkeypatch):
    _patch_openai(monkeypatch, {"choices": [{"finish_reason": "stop", "message": {"content": "   "}}]})
    with pytest.raises(J.JudgeRefusal):
        J.call_openai("gpt-4o", "p", [], "k")


def test_openai_normal_response_still_works(monkeypatch):
    _patch_openai(monkeypatch, {"choices": [{"finish_reason": "stop", "message": {
        "content": '{"classification":"block","confidence":0.8,"rationale":"tensión"}'}}]})
    assert "block" in J.call_openai("gpt-4o", "p", [], "k")


def test_anthropic_refusal_is_surfaced(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _Resp({"stop_reason": "refusal", "content": []}))
    with pytest.raises(J.JudgeRefusal):
        J.call_anthropic("claude-x", "p", [b"i"], "k")


# --------------------------------------------------------- parseo del veredicto
def test_parse_plain_json():
    v = J.parse_verdict('{"classification":"filler_word","confidence":0.9}', J.CLASSES)
    assert v["classification"] == "filler_word"


def test_parse_strips_code_fences():
    raw = '```json\n{"classification":"neutral_pause","confidence":0.5}\n```'
    assert J.parse_verdict(raw, J.CLASSES)["classification"] == "neutral_pause"


def test_parse_tolerates_surrounding_prose():
    raw = 'Claro, aquí tienes:\n{"classification":"block","confidence":0.7}\nEspero que sirva.'
    assert J.parse_verdict(raw, J.CLASSES)["classification"] == "block"


def test_parse_accepts_spanish_key():
    assert J.parse_verdict('{"clase":"revision","confidence":0.6}', J.CLASSES)["classification"] == "revision"


def test_parse_keeps_raw_text_when_there_is_no_json():
    """Lo que faltaba: poder saber DESPUÉS qué había devuelto el modelo."""
    raw = "I'm sorry, I can't assist with that request."
    with pytest.raises(J.VerdictError) as ei:
        J.parse_verdict(raw, J.CLASSES)
    assert ei.value.raw == raw, "el texto crudo debe viajar en la excepción"
    assert "no contiene JSON" in ei.value.reason


def test_parse_rejects_a_class_outside_the_taxonomy():
    with pytest.raises(J.VerdictError) as ei:
        J.parse_verdict('{"classification":"tartamudeo","confidence":0.9}', J.CLASSES)
    assert "tartamudeo" in ei.value.reason


def test_parse_reports_invalid_json_with_its_text():
    with pytest.raises(J.VerdictError) as ei:
        J.parse_verdict('{"classification": block,}', J.CLASSES)
    assert "JSON inválido" in ei.value.reason and ei.value.raw


# --------------------------------------------------------------- escalado
def test_escalation_drops_frames_after_two_refusals():
    """Si el modelo rechaza las imágenes, el tercer intento va sin ellas y se marca."""
    seen = []

    def caller(model, prompt, imgs, key):
        seen.append(len(imgs))
        if imgs:
            raise J.JudgeRefusal("no puedo analizar imágenes de personas", "openai")
        return '{"classification":"neutral_pause","confidence":0.4}'

    v, n_used, note = J.ask_judge(caller, "gpt-4o", "p", [b"a", b"b", b"c"], "k", J.CLASSES)
    assert v["classification"] == "neutral_pause"
    assert n_used == 0, "debe informar que no usó fotogramas"
    assert note == "sin_fotogramas"
    assert seen == [3, 3, 0], "dos intentos con imágenes, el tercero sin ellas"


def test_escalation_retries_format_before_dropping_frames():
    calls = []

    def caller(model, prompt, imgs, key):
        calls.append(prompt)
        if len(calls) == 1:
            return "Claro, con gusto te ayudo."          # prosa, sin JSON
        return '{"classification":"block","confidence":0.9}'

    v, n_used, note = J.ask_judge(caller, "m", "p", [b"a"], "k", J.CLASSES)
    assert v["classification"] == "block"
    assert n_used == 1, "el reintento de formato NO debe descartar los fotogramas"
    assert note == "reintento_formato"
    assert "SOLO el objeto JSON" in calls[1]


def test_escalation_succeeds_first_try_reports_ok():
    def caller(*a, **k):
        return '{"classification":"fluent","confidence":0.99}'

    v, n_used, note = J.ask_judge(caller, "m", "p", [b"a", b"b"], "k", J.CLASSES)
    assert note == "ok" and n_used == 2 and v["classification"] == "fluent"


def test_no_frames_first_try_reports_audio_text_only_fallback():
    def caller(*args):
        return '{"classification":"fluent","confidence":0.9}'

    verdict, n_used, note = J.ask_judge(caller, "m", "p", [], "k", J.CLASSES)
    assert verdict["classification"] == "fluent"
    assert n_used == 0 and note == "sin_fotogramas"


def test_escalation_reraises_when_everything_fails():
    def caller(*a, **k):
        raise J.JudgeRefusal("siempre no", "openai")

    with pytest.raises(J.JudgeRefusal):
        J.ask_judge(caller, "m", "p", [b"a"], "k", J.CLASSES)


# ------------------------------------------------------------- registro
def test_log_failure_writes_the_raw_response(tmp_path):
    p = tmp_path / "errors.jsonl"
    J.log_failure(p, rid="r1", kind="VerdictError", reason="sin JSON",
                  raw="I'm sorry, I can't help with that.")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["raw"].startswith("I'm sorry")
    assert rows[0]["rid"] == "r1" and "ts" in rows[0]


def test_log_failure_appends(tmp_path):
    p = tmp_path / "e.jsonl"
    J.log_failure(p, a=1); J.log_failure(p, a=2)
    assert len(p.read_text(encoding="utf-8").strip().splitlines()) == 2


# ----------------------------------- el informe separa lo que vio vídeo
def test_report_excludes_frameless_judgments_from_the_video_verdict(tmp_path, monkeypatch):
    """Arbitrar si el vídeo aporta con juicios que no vieron vídeo es inválido."""
    import scripts.qa_report as Q

    W = []
    for i in range(4):
        W.append({"timestamp_start": float(i), "timestamp_end": float(i) + 3.0,
                  "transcription": "x",
                  "features": {"f0_reset": 1.0, "max_delta_mouth_press": 0.2, "is_dangling": False},
                  "features_extra": {"sil_dur": 0.2},
                  "predictions": {"audio_only": {"class": "fluent", "prob": .5},
                                  "audio_text": {"class": "neutral_pause", "prob": .5},
                                  "trimodal": {"class": "block", "prob": .5}}})
    results = {"windows": W, "events": {m: [] for m in Q.MODES}, "labels": ["fluent", "block"],
               "url": None, "title": "t", "duration_s": 60, "window_s": 3.0, "stride_s": 0.5,
               "window_disagreement_rate": 1.0, "model_provenance": {}}
    rp = tmp_path / "results_X.json"
    rp.write_text(json.dumps(results), encoding="utf-8")

    case = tmp_path / "case"; case.mkdir()
    # 1 juicio con fotogramas (gana trimodal) y 3 sin ellos (ganan a audio_text)
    rows = [{"window_index": 0, "label": "block", "n_frames": 3}]
    rows += [{"window_index": i, "label": "neutral_pause", "n_frames": 0} for i in (1, 2, 3)]
    (case / "X.adjudication.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    monkeypatch.setattr(Q, "CASE", case)
    monkeypatch.setattr(sys, "argv", ["qa_report.py", "--vid", "X",
                                      "--results", str(rp), "--out", str(tmp_path / "r.md")])
    Q.main()
    md = (tmp_path / "r.md").read_text(encoding="utf-8")

    assert "**3** se resolvieron SIN fotogramas" in md
    assert "quedan **excluidas**" in md.lower() or "excluidas" in md
    # el recuento principal usa 1 sola ventana, no 4
    assert "Recuento sobre las **1**" in md
    assert "cabe dentro del ruido" in md, "debe advertir del tamaño de muestra"


# ------------------- registros sin clase utilizable (bug de clave equivocada)
def test_dedupe_drops_records_without_a_usable_class():
    """38 de 60 registros salieron con label=null porque una versión leía
    `clase`/`class` mientras el juez devuelve `classification`. Contarlos en el
    denominador rebajaba todos los porcentajes por un factor de casi tres."""
    from scripts.qa_report import dedupe_judgments

    rows = [{"window_index": i, "label": None, "confidence": 0.8} for i in range(38)]
    rows += [{"window_index": 100 + i, "label": "block"} for i in range(22)]
    usable, n_bad = dedupe_judgments(rows)
    assert len(usable) == 22, "sólo cuentan los que traen clase"
    assert n_bad == 38


def test_dedupe_prefers_the_rejudged_record_for_the_same_window():
    from scripts.qa_report import dedupe_judgments

    rows = [{"window_index": 7, "label": None},
            {"window_index": 7, "label": "filler_word", "n_frames": 3}]
    usable, n_bad = dedupe_judgments(rows)
    assert len(usable) == 1 and usable[0]["label"] == "filler_word"
    assert n_bad == 0, "la ventana quedó resuelta, no debe contarse como rota"


def test_dedupe_keeps_one_record_per_window():
    from scripts.qa_report import dedupe_judgments

    rows = [{"window_index": 3, "label": "block"}, {"window_index": 3, "label": "revision"}]
    usable, _ = dedupe_judgments(rows)
    assert len(usable) == 1, "una ventana, un veredicto"


def test_parse_verdict_accepts_the_key_the_rubric_asks_for():
    """La rúbrica pide `classification`; el lector debe leer esa clave."""
    import json as _json

    import scripts.llm_judge as _J
    assert '"classification"' in _J.RUBRIC
    v = _J.parse_verdict(_json.dumps({"classification": "block", "confidence": 0.9}), _J.CLASSES)
    assert v["classification"] == "block"


def test_adjudicate_treats_labelless_records_as_pending():
    """Rejuzgar exige que un registro sin clase NO cuente como hecho."""
    prev = [{"window_index": 1, "label": None}, {"window_index": 2, "label": "block"}]
    done = {r["window_index"] for r in prev if r.get("label")}
    broken = {r["window_index"] for r in prev if not r.get("label")} - done
    assert done == {2} and broken == {1}


# ------------------------------------------------- contraste pareado (McNemar)
def _W(preds):
    return [{"predictions": {m: {"class": c, "prob": 0.5} for m, c in p.items()}} for p in preds]


def test_mcnemar_counts_only_discordant_pairs():
    from scripts.qa_report import mcnemar

    # v0: trimodal acierta; v1: audio_text acierta; v2: los dos aciertan (no cuenta)
    W = _W([{"audio_only": "x", "audio_text": "x", "trimodal": "block"},
            {"audio_only": "x", "audio_text": "block", "trimodal": "x"},
            {"audio_only": "block", "audio_text": "block", "trimodal": "block"}])
    rows = [{"window_index": i, "label": "block"} for i in range(3)]
    ab, ba, p = mcnemar(rows, W, "trimodal", "audio_text")
    assert (ab, ba) == (1, 1), "el empate no debe entrar en el contraste"
    assert p == 1.0


def test_mcnemar_does_not_call_a_small_lead_significant():
    """El caso real: 21 contra 12 parece una victoria y no lo es."""
    from scripts.qa_report import mcnemar

    W, rows = [], []
    for i in range(21):
        W.append(_W([{"audio_only": "x", "audio_text": "x", "trimodal": "block"}])[0])
        rows.append({"window_index": i, "label": "block"})
    for i in range(21, 33):
        W.append(_W([{"audio_only": "x", "audio_text": "block", "trimodal": "x"}])[0])
        rows.append({"window_index": i, "label": "block"})
    ab, ba, p = mcnemar(rows, W, "trimodal", "audio_text")
    assert (ab, ba) == (21, 12)
    assert p > 0.05, "21-12 sobre 33 pares no alcanza significación"
    assert abs(p - 0.163) < 0.01


def test_mcnemar_detects_a_real_difference():
    from scripts.qa_report import mcnemar

    W, rows = [], []
    for i in range(18):
        W.append(_W([{"audio_only": "x", "audio_text": "x", "trimodal": "block"}])[0])
        rows.append({"window_index": i, "label": "block"})
    for i in range(18, 22):
        W.append(_W([{"audio_only": "x", "audio_text": "block", "trimodal": "x"}])[0])
        rows.append({"window_index": i, "label": "block"})
    _, _, p = mcnemar(rows, W, "trimodal", "audio_text")
    assert p < 0.05, "18-4 sí debe salir significativo"


def test_mcnemar_handles_no_discordant_pairs():
    from scripts.qa_report import mcnemar

    W = _W([{"audio_only": "a", "audio_text": "a", "trimodal": "a"}])
    ab, ba, p = mcnemar([{"window_index": 0, "label": "a"}], W, "trimodal", "audio_text")
    assert (ab, ba, p) == (0, 0, 1.0)


def test_report_warns_that_the_queue_is_biased_toward_trimodal(tmp_path, monkeypatch):
    """La cola se ordena por evidencia cinésica/prosódica y se trunca: favorece
    al trimodal. El informe debe decirlo para que nadie generalice un triunfo."""
    import scripts.qa_report as Q

    W = [{"timestamp_start": float(i), "timestamp_end": float(i) + 3.0, "transcription": "x",
          "features": {"f0_reset": 1.0, "max_delta_mouth_press": 0.2, "is_dangling": False},
          "features_extra": {"sil_dur": 0.2},
          "predictions": {"audio_only": {"class": "fluent", "prob": .5},
                          "audio_text": {"class": "neutral_pause", "prob": .5},
                          "trimodal": {"class": "block", "prob": .5}}} for i in range(6)]
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"windows": W, "events": {m: [] for m in Q.MODES},
                              "labels": ["fluent", "block"], "url": None, "title": "t",
                              "duration_s": 60, "window_s": 3.0, "stride_s": 0.5,
                              "window_disagreement_rate": 1.0, "model_provenance": {}}),
                  encoding="utf-8")
    case = tmp_path / "case"; case.mkdir()
    (case / "X.adjudication.jsonl").write_text("\n".join(
        json.dumps({"window_index": i, "label": "block", "n_frames": 3}) for i in range(6)),
        encoding="utf-8")
    monkeypatch.setattr(Q, "CASE", case)
    monkeypatch.setattr(sys, "argv", ["qa_report.py", "--vid", "X", "--results", str(rp),
                                      "--out", str(tmp_path / "o.md")])
    Q.main()
    md = (tmp_path / "o.md").read_text(encoding="utf-8")
    assert "McNemar" in md
    assert "no son una muestra aleatoria" in md
    assert "NO se generaliza" in md
