#!/usr/bin/env python3
"""Juez multimodal ciego (VLM-as-a-Judge) -> etiquetas GOLD independientes de las heurísticas.

Diseño (importante para la validez del benchmark):
  * El juez NO ve las etiquetas ni la evidencia heurística: recibe frames, transcripción de
    contexto y MEDICIONES objetivas (duración, silencio, F0, energía). Es una segunda opinión,
    no una repetición de las reglas.
  * Además de los candidatos, se le envían ventanas SIN candidato (muestreo negativo). Sin esto,
    el test heredaría la cobertura de las heurísticas y no podría medir lo que estas no ven.
  * Resultado: data/features/{id}.gold.json + data/judge/judgments.jsonl (caché reanudable).

Uso:
  set ANTHROPIC_API_KEY=...    (o OPENAI_API_KEY)
  python scripts/llm_judge.py --per-class 60 --negatives 120 --model claude-sonnet-4-5
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import ACOUSTIC_HOP_S, BACKGROUND, RANDOM_SEED, TAXONOMY, WINDOW_S  # noqa: E402

JUDGE_DIR = ROOT / "data/judge"
CACHE = JUDGE_DIR / "judgments.jsonl"
ERRORS = JUDGE_DIR / "errors.jsonl"
CLASSES = list(TAXONOMY) + [BACKGROUND]

SYSTEM = """Eres un lingüista y fonoaudiólogo experto en análisis del discurso oral en español.
Clasificas segmentos de presentaciones orales según una taxonomía cerrada de fluidez verbal.
Trabajas a ciegas: nadie te dice qué esperar. Si el segmento no contiene ningún fenómeno de la
taxonomía, respondes "fluent". Sé conservador: ante duda real, baja la confianza.

Sobre los fotogramas: la tarea es describir el MOVIMIENTO ARTICULATORIO visible (apertura
mandibular, tensión de labios, dirección de la mirada) durante el habla. No se te pide
identificar, nombrar, describir ni inferir nada sobre la persona que aparece, ni su
apariencia, edad, origen ni estado. Ignora todo eso. Si los fotogramas no te resultan
útiles, clasifica igualmente usando la transcripción y las mediciones acústicas."""

RUBRIC = """TAXONOMÍA (elige exactamente una):
1. "filler_word": muletilla léxica o vocálica sin función semántica ("este", "eh", "o sea", "mmm", "bueno").
2. "block": interrupción con tensión visible (labios apretados, ceño fruncido, esfuerzo), típicamente
   dentro de un constituyente y no en frontera de frase.
3. "prolongation": alargamiento anómalo de un fonema (vocal o consonante continua) que no corresponde
   al alargamiento natural del final de una frase.
4. "repetition": repetición involuntaria de un sonido, sílaba o palabra.
5. "revision": falso inicio o reformulación de una idea a medio terminar.
6. "rhetorical_pause": pausa deliberada en frontera sintáctica, rostro relajado, mirada al frente.
7. "neutral_pause": silencio fisiológico o de transición, sin tensión ni intención enfática.
8. "fluent": no hay ningún fenómeno de la lista en el intervalo evaluado.

Reglas de decisión:
- Distingue rhetorical_pause de block por la tensión facial y por la posición sintáctica.
- Una pausa de respiración corriente es neutral_pause, no rhetorical_pause.
- El alargamiento final de frase NO es prolongation.

FORMATO DE RESPUESTA (JSON estricto, sin texto alrededor):
{"classification": "<una de las 8>", "confidence": <0.0-1.0>, "rationale": "<máx. 25 palabras>"}"""


# ------------------------------------------------------------------ proveedores
class JudgeRefusal(Exception):
    """El modelo se negó a responder (típicamente por las imágenes de un rostro real)."""

    def __init__(self, reason: str, provider: str = ""):
        super().__init__(f"{provider} rechazó: {reason[:200]}")
        self.reason, self.provider = reason, provider


class VerdictError(Exception):
    """La respuesta llegó pero no es un veredicto utilizable. Conserva el texto crudo."""

    def __init__(self, reason: str, raw: str):
        super().__init__(f"{reason}: {raw[:200]!r}")
        self.reason, self.raw = reason, raw


def call_anthropic(model: str, prompt: str, images: list[bytes], api_key: str, timeout: int = 90) -> str:
    import urllib.request
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(im).decode()}} for im in images]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": model, "max_tokens": 300, "system": SYSTEM,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if data.get("stop_reason") == "refusal":
        raise JudgeRefusal("stop_reason=refusal", "anthropic")
    blocks = [b.get("text", "") for b in (data.get("content") or []) if b.get("type") == "text"]
    txt = "".join(blocks)
    if not txt.strip():
        raise JudgeRefusal(f"respuesta vacía (stop_reason={data.get('stop_reason')})", "anthropic")
    return txt


def call_openai(model: str, prompt: str, images: list[bytes], api_key: str, timeout: int = 90) -> str:
    """Devuelve el texto de la respuesta.

    Cuando el modelo se niega, OpenAI deja `content` en null y pone el motivo en
    `message.refusal`. Leer sólo `content` devolvía una cadena vacía y el error
    aparecía río abajo como «substring not found», escondiendo la causa real.
    """
    import urllib.request
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(im).decode()}}
               for im in images]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": model, "max_tokens": 300, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
        "content-type": "application/json", "authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    if msg.get("refusal"):
        raise JudgeRefusal(str(msg["refusal"]), "openai")
    if ch.get("finish_reason") == "content_filter":
        raise JudgeRefusal("finish_reason=content_filter", "openai")
    txt = msg.get("content")
    if not txt or not str(txt).strip():
        raise JudgeRefusal(f"respuesta vacía (finish_reason={ch.get('finish_reason')})", "openai")
    return str(txt)


def call_gemini(model: str, prompt: str, images: list[bytes], api_key: str, timeout: int = 90) -> str:
    """Call Gemini through its REST API and return the candidate text."""
    import urllib.parse
    import urllib.request

    parts = [{"inline_data": {"mime_type": "image/jpeg",
                              "data": base64.b64encode(im).decode()}} for im in images]
    parts.append({"text": prompt})
    body = json.dumps({"system_instruction": {"parts": [{"text": SYSTEM}]},
                       "contents": [{"role": "user", "parts": parts}],
                       "generationConfig": {"responseMimeType": "application/json"}}).encode()
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           + urllib.parse.quote(model, safe="") + ":generateContent?"
           + urllib.parse.urlencode({"key": api_key}))
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Gemini HTTP {exc.code} para {model}: {detail}") from None
    feedback = data.get("promptFeedback") or {}
    candidates = data.get("candidates") or []
    if not candidates:
        raise JudgeRefusal(str(feedback.get("blockReason") or "sin candidatos"), "gemini")
    candidate = candidates[0]
    reason = candidate.get("finishReason")
    if reason in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT"}:
        raise JudgeRefusal(str(reason), "gemini")
    text = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", []))
    if not text.strip():
        raise JudgeRefusal(str(reason or "respuesta vacía"), "gemini")
    return text


def call_groq(model: str, prompt: str, images: list[bytes], api_key: str, timeout: int = 90) -> str:
    """Call Groq's OpenAI-compatible chat endpoint."""
    import urllib.request

    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(im).decode()}}
               for im in images]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": model, "max_tokens": 300,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=body, headers={
        "content-type": "application/json", "authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Groq HTTP {exc.code}: {detail}") from None
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content")
    if not text or not str(text).strip():
        raise JudgeRefusal(str(message.get("refusal") or choice.get("finish_reason") or "respuesta vacía"), "groq")
    return str(text)


def call_ollama(model: str, prompt: str, images: list[bytes], api_key: str = "ollama",
                base_url: str = "http://localhost:11434/v1", timeout: int = 180) -> str:
    """Call Ollama's OpenAI-compatible local chat endpoint."""
    import urllib.request

    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(im).decode()}}
               for im in images]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": model, "max_tokens": 300,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers={
        "content-type": "application/json", "authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from None
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content")
    if not text or not str(text).strip():
        raise JudgeRefusal(str(choice.get("finish_reason") or "respuesta vacía"), "ollama")
    return str(text)


def parse_verdict(txt: str, classes: list[str]) -> dict:
    """Extrae el veredicto. Si falla, conserva el texto crudo en la excepción."""
    t = (txt or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    t = t.removeprefix("json").strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise VerdictError("la respuesta no contiene JSON", txt or "")
    try:
        out = json.loads(t[i:j + 1])
    except json.JSONDecodeError as exc:
        raise VerdictError(f"JSON inválido ({exc.msg})", txt or "") from None
    cls = out.get("classification") or out.get("clase") or out.get("class")
    if cls not in classes:
        raise VerdictError(f"clase fuera de la taxonomía: {cls!r}", txt or "")
    out["classification"] = cls
    return out


def ask_judge(caller, model: str, prompt: str, images: list[bytes], api_key: str,
              classes: list[str]) -> tuple[dict, int, str]:
    """Pide un veredicto con escalado. Devuelve (veredicto, nº de imágenes usadas, nota).

    Escalado, en orden:
      1. prompt + imágenes.
      2. prompt + recordatorio de formato + imágenes (por si devolvió prosa).
      3. prompt SIN imágenes.

    El tercer intento existe porque los modelos con visión rechazan con cierta
    frecuencia los fotogramas de una persona real identificable. Un juicio sin
    fotogramas sigue siendo utilizable —usa acústica y transcripción— pero NO es
    equivalente a uno con vídeo, así que se devuelve el número de imágenes
    realmente usadas para que el registro lo deje escrito y nadie mezcle ambos.
    """
    last: Exception | None = None
    for attempt in range(3):
        imgs = images if attempt < 2 else []
        p = prompt if attempt == 0 else prompt + "\n\nRecuerda: responde SOLO el objeto JSON."
        try:
            return parse_verdict(caller(model, p, imgs, api_key), classes), len(imgs), (
                "ok" if attempt == 0 else ("reintento_formato" if attempt == 1 else "sin_fotogramas"))
        except JudgeRefusal as exc:
            last = exc
            if attempt < 2:
                time.sleep(1)
                continue
        except VerdictError as exc:
            last = exc
            if attempt < 2:
                time.sleep(1)
                continue
        except Exception as exc:                     # red, límite de tasa, saldo
            last = exc
            if attempt < 2:
                time.sleep(2 + 3 * attempt)
                continue
    raise last if last else RuntimeError("fallo desconocido del juez")


def log_failure(path: Path, **fields) -> None:
    """Registra por qué falló un ítem. Sin esto los errores son invisibles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields},
                            ensure_ascii=False) + "\n")


def load_env_file() -> None:
    """Lee KEY=VALOR de un archivo .env local (nunca se versiona) y lo pasa al entorno."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def pick_provider(model: str) -> tuple[str, str]:
    load_env_file()
    if model.startswith("claude") and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ["ANTHROPIC_API_KEY"]
    if model.startswith("gemini") and os.environ.get("GEMINI_API_KEY"):
        return "gemini", os.environ["GEMINI_API_KEY"]
    if os.environ.get("OPENAI_API_KEY") and not model.startswith("claude"):
        return "openai", os.environ["OPENAI_API_KEY"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ["ANTHROPIC_API_KEY"]
    sys.exit(
        "Falta la API key. Dos formas (elige una):\n"
        "  A) Crea el archivo .env en esta carpeta con una línea:  OPENAI_API_KEY=sk-...\n"
        "  B) En la misma consola donde lanzas el script:  $env:OPENAI_API_KEY=\"sk-...\"\n"
        "Con OpenAI usa --model gpt-4o; con Gemini, GEMINI_API_KEY y --model gemini-2.0-flash; "
        "con Anthropic, ANTHROPIC_API_KEY y --model claude-sonnet-4-5.")


# ------------------------------------------------------------------ contexto
def frames_at(mp4: Path, times_s: list[float], max_side: int = 512) -> list[bytes]:
    import cv2
    out = []
    cap = cv2.VideoCapture(str(mp4))
    for t in times_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        sc = max_side / max(h, w)
        if sc < 1:
            frame = cv2.resize(frame, (int(w * sc), int(h * sc)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            out.append(buf.tobytes())
    cap.release()
    return out


def acoustic_summary(z, a: float, b: float) -> dict:
    t, f0, rms, vad = z["ac_times"], z["f0"], z["rms"], z["vad"]
    m = (t >= a) & (t < b)
    if not m.any():
        return {}
    f0m = f0[m][~np.isnan(f0[m])]
    return dict(duracion_s=round(b - a, 2),
                proporcion_silencio=round(float((vad[m] == 0).mean()), 2),
                f0_rango_hz=round(float(f0m.max() - f0m.min()), 1) if len(f0m) > 3 else None,
                f0_pendiente_hz_s=round(float(np.polyfit(t[m][~np.isnan(f0[m])], f0m, 1)[0]), 1) if len(f0m) > 5 else None,
                energia_variacion=round(float(rms[m].std() / (rms[m].mean() + 1e-9)), 2))


def context_text(words: list[dict], a: float, b: float, span: float = 4.0) -> tuple[str, str, str]:
    prev = " ".join(w["text"] for w in words if a - span <= w["end"] <= a + 0.05)
    seg = " ".join(w["text"] for w in words if w["end"] > a and w["start"] < b)
    nxt = " ".join(w["text"] for w in words if b - 0.05 <= w["start"] <= b + span)
    return prev[-300:], seg[:200] or "(sin palabras: silencio)", nxt[:300]


def build_prompt(prev: str, seg: str, nxt: str, meas: dict) -> str:
    return (f"{RUBRIC}\n\nSEGMENTO A EVALUAR\n"
            f"- Transcripción previa: \"{prev}\"\n- Intervalo evaluado: \"{seg}\"\n- Transcripción posterior: \"{nxt}\"\n"
            f"- Mediciones objetivas: {json.dumps(meas, ensure_ascii=False)}\n"
            f"- Imágenes: fotogramas de inicio, medio y final del intervalo (en ese orden).\n\n"
            f"Responde SOLO el JSON.")


# ------------------------------------------------------------------ muestreo
def sample_items(man: dict, per_class: int, low_conf: int, negatives: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list] = defaultdict(list)
    per_rec_events: dict[str, list] = {}
    for r in man["recordings"]:
        ev = json.loads((ROOT / r["events_json"]).read_text(encoding="utf-8"))
        per_rec_events[r["recording_id"]] = ev
        for i, e in enumerate(ev):
            key = e["category"] if e.get("source") == "auto" else "_low"
            by_class[key].append(dict(rid=r["recording_id"], idx=i, start=e["start_ms"] / 1000,
                                      end=e["end_ms"] / 1000, origin="candidate" if key != "_low" else "low_conf",
                                      heur=e["category"]))
    items = []
    for c in TAXONOMY:
        pool = by_class.get(c, [])
        rng.shuffle(pool)
        # reparto equitativo entre oradores
        pool.sort(key=lambda x: x["rid"])
        seen: dict[str, int] = defaultdict(int)
        picked = []
        for it in sorted(pool, key=lambda x: (seen[x["rid"]], rng.random())):
            if len(picked) >= per_class:
                break
            seen[it["rid"]] += 1
            picked.append(it)
        items += picked
    lc = by_class.get("_low", [])
    rng.shuffle(lc)
    items += lc[:low_conf]
    # negativos: ventanas de 3 s sin candidato
    negs = []
    for r in man["recordings"]:
        ev = per_rec_events[r["recording_id"]]
        spans = [(e["start_ms"] / 1000, e["end_ms"] / 1000) for e in ev]
        for _ in range(6):
            if r["duration_s"] < WINDOW_S + 1:
                break
            a = rng.uniform(0.5, max(0.6, r["duration_s"] - WINDOW_S - 0.5))
            b = a + WINDOW_S
            if any(min(b, s2) - max(a, s1) > 0 for s1, s2 in spans):
                continue
            negs.append(dict(rid=r["recording_id"], idx=-1, start=a, end=b, origin="negative", heur=BACKGROUND))
    rng.shuffle(negs)
    items += negs[:negatives]
    rng.shuffle(items)
    return items


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--low-conf", type=int, default=60)
    ap.add_argument("--negatives", type=int, default=120)
    ap.add_argument("--max-events", type=int, default=0, help="0 = sin tope")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true", help="solo muestra el muestreo y un prompt de ejemplo")
    a = ap.parse_args()

    man = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    recs = {r["recording_id"]: r for r in man["recordings"]}
    items = sample_items(man, a.per_class, a.low_conf, a.negatives, a.seed)
    if a.max_events:
        items = items[: a.max_events]
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line); done[(j["rid"], round(j["start"], 2), round(j["end"], 2))] = j
    print(f"[judge] {len(items)} ítems muestreados | ya juzgados: {len(done)} | modelo {a.model}")
    counts = defaultdict(int)
    for it in items:
        counts[it["origin"] + ":" + it["heur"]] += 1
    print("[judge] muestreo:", dict(sorted(counts.items())))
    if a.dry_run:
        it = items[0]; r = recs[it["rid"]]
        z = np.load(ROOT / r["features_npz"]); w = json.loads((ROOT / r["words_json"]).read_text(encoding="utf-8"))
        p, s, n = context_text(w, it["start"], it["end"])
        print("\n--- PROMPT DE EJEMPLO ---\n" + build_prompt(p, s, n, acoustic_summary(z, it["start"], it["end"])))
        return

    provider, key = pick_provider(a.model)
    caller = call_anthropic if provider == "anthropic" else call_openai
    npz_cache: dict[str, object] = {}
    words_cache: dict[str, list] = {}
    t0, n_new, errors = time.time(), 0, 0
    fail_kinds: Counter = Counter()
    notes: Counter = Counter()
    with CACHE.open("a", encoding="utf-8") as fh:
        for k, it in enumerate(items):
            key_t = (it["rid"], round(it["start"], 2), round(it["end"], 2))
            if key_t in done:
                continue
            r = recs[it["rid"]]
            if it["rid"] not in npz_cache:
                npz_cache[it["rid"]] = np.load(ROOT / r["features_npz"])
                words_cache[it["rid"]] = json.loads((ROOT / r["words_json"]).read_text(encoding="utf-8"))
            z, words = npz_cache[it["rid"]], words_cache[it["rid"]]
            prev, seg, nxt = context_text(words, it["start"], it["end"])
            meas = acoustic_summary(z, it["start"], it["end"])
            mid = (it["start"] + it["end"]) / 2
            imgs = frames_at(ROOT / r["mp4"], [it["start"], mid, max(it["end"] - 0.05, mid)]) if r.get("mp4") else []
            prompt = build_prompt(prev, seg, nxt, meas)
            try:
                out, n_used, note = ask_judge(caller, a.model, prompt, imgs, key, CLASSES)
            except Exception as exc:
                errors += 1
                fail_kinds[type(exc).__name__] += 1
                log_failure(ERRORS, rid=it["rid"], start=it["start"], end=it["end"],
                            kind=type(exc).__name__, reason=str(exc)[:300],
                            raw=getattr(exc, "raw", "")[:500], n_frames=len(imgs))
                continue
            if note != "ok":
                notes[note] += 1
            rec = dict(rid=it["rid"], speaker_id=r["speaker_id"], start=it["start"], end=it["end"],
                       origin=it["origin"], heuristic=it["heur"], judge=out["classification"],
                       confidence=float(out.get("confidence", 0.0)), rationale=out.get("rationale", "")[:200],
                       model=a.model, n_frames=n_used, note=note, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            done[key_t] = rec; n_new += 1
            if n_new % 20 == 0:
                print(f"   {n_new} nuevos | {len(done)}/{len(items)} | {(time.time()-t0)/max(n_new,1):.1f} s/ítem")
            time.sleep(a.sleep)
    print(f"[judge] listo: {n_new} nuevos, {errors} errores, {len(done)} en caché")
    if fail_kinds:
        print(f"[judge] errores por tipo: {dict(fail_kinds)}")
        print(f"[judge] detalle con la respuesta cruda en {ERRORS.relative_to(ROOT)}")
    if notes:
        print(f"[judge] juicios que necesitaron escalado: {dict(notes)}")
        if notes.get("sin_fotogramas"):
            print(f"[judge] AVISO: {notes['sin_fotogramas']} juicios se resolvieron SIN fotogramas "
                  "(el modelo rechazó las imágenes). Quedan marcados con n_frames=0 y note="
                  "'sin_fotogramas': no son equivalentes a un juicio multimodal.")
    write_gold(man, done)


def write_gold(man: dict, done: dict) -> None:
    """Escribe {id}.gold.json: SOLO los intervalos juzgados (el resto queda sin etiqueta)."""
    by_rec: dict[str, list] = defaultdict(list)
    for j in done.values():
        if j["judge"] == BACKGROUND:
            cat = BACKGROUND
        else:
            cat = j["judge"]
        by_rec[j["rid"]].append(dict(category=cat, start_ms=int(j["start"] * 1000), end_ms=int(j["end"] * 1000),
                                     confidence=j["confidence"], source="gold_llm", model=j["model"],
                                     heuristic=j["heuristic"], origin=j["origin"], rationale=j["rationale"]))
    n = 0
    for r in man["recordings"]:
        rid = r["recording_id"]
        p = ROOT / r["events_json"].replace(".events.json", ".gold.json")
        p.write_text(json.dumps(sorted(by_rec.get(rid, []), key=lambda e: e["start_ms"]), ensure_ascii=False),
                     encoding="utf-8")
        n += len(by_rec.get(rid, []))
    stats = defaultdict(int)
    for j in done.values():
        stats[j["judge"]] += 1
    print(f"[judge] {n} juicios escritos en data/features/*.gold.json | distribución: {dict(stats)}")


if __name__ == "__main__":
    main()
