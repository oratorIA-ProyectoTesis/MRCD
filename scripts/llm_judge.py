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
import hashlib
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
from core.contracts import validate_raw_features  # noqa: E402
from core.dataset import (JUDGE_INTERVAL_VERSION, gold_windows, judgment_window_index,
                          seconds_to_ms, window_starts)  # noqa: E402

JUDGE_DIR = ROOT / "data/judge"
CACHE = JUDGE_DIR / "judgments.jsonl"
ERRORS = JUDGE_DIR / "errors.jsonl"
CLASSES = list(TAXONOMY) + [BACKGROUND]
JUDGE_INPUT_VERSION = "2"

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
                "sin_fotogramas" if not imgs else "ok" if attempt == 0 else "reintento_formato")
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


PROVIDER_CALLERS = {"anthropic": call_anthropic, "openai": call_openai,
                    "gemini": call_gemini, "groq": call_groq, "ollama": call_ollama}
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
                 "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


def provider_for_model(model: str, provider: str = "auto") -> str:
    """Resolve transport without requiring credentials (also used by dry-run/cache)."""
    if provider == "auto":
        provider = ("anthropic" if model.startswith("claude") else
                    "gemini" if model.startswith("gemini") else
                    "groq" if model.startswith(("llama-", "mixtral-", "gemma-")) else
                    "openai")
    if provider not in PROVIDER_CALLERS:
        raise ValueError(f"proveedor no soportado: {provider}")
    return provider


def pick_provider(model: str, provider: str = "auto") -> tuple[str, str]:
    """Bind a model to exactly one provider; never use another provider's key."""
    load_env_file()
    provider = provider_for_model(model, provider)
    if provider == "ollama":
        return provider, "ollama"
    key_name = PROVIDER_KEYS[provider]
    key = os.environ.get(key_name)
    if not key:
        sys.exit(f"Falta {key_name} para el proveedor {provider}; configura la variable o .env local.")
    return provider, key


def caller_for(provider: str):
    try:
        return PROVIDER_CALLERS[provider]
    except KeyError:
        raise ValueError(f"proveedor no soportado: {provider}") from None


def judgment_input_fingerprint(prompt: str, images: list[bytes], mp4: Path | None) -> str:
    """Bind a verdict to the exact rubric, context, frame bytes, and media identity."""
    media = None
    if mp4 is not None:
        media = {"path": str(mp4.resolve()), "exists": mp4.exists()}
        if mp4.exists():
            stat = mp4.stat()
            media.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    payload = {"version": JUDGE_INPUT_VERSION, "system": SYSTEM, "prompt": prompt,
               "media": media, "frame_sha256": [hashlib.sha256(im).hexdigest() for im in images]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def judgment_cache_key(row: dict, provider: str, model: str) -> tuple | None:
    """Legacy rows without provider/fingerprint are intentionally not reusable."""
    if (row.get("provider") != provider or row.get("model") != model
            or not row.get("input_fingerprint")
            or row.get("interval_identity_version") != JUDGE_INTERVAL_VERSION):
        return None
    return (provider, model, row["rid"], seconds_to_ms(row["start"]),
            seconds_to_ms(row["end"]), row["input_fingerprint"])


def load_judgment_cache(path: Path, provider: str, model: str) -> dict:
    """Read only verdicts from the exact provider/model experiment."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = judgment_cache_key(row, provider, model)
                if key is not None:
                    done[key] = row
    return done


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
    """Sample by class and speaker without assigning two calls to one test window."""
    rng = random.Random(seed)
    by_class: dict[str, list] = defaultdict(list)
    per_rec_events: dict[str, list] = {}
    grids = {r["recording_id"]: window_starts(r["duration_s"]) for r in man["recordings"]}
    for r in man["recordings"]:
        ev = json.loads((ROOT / r["events_json"]).read_text(encoding="utf-8"))
        per_rec_events[r["recording_id"]] = ev
        for i, e in enumerate(ev):
            key = e["category"] if e.get("source") == "auto" else "_low"
            by_class[key].append(dict(rid=r["recording_id"], idx=i, start=e["start_ms"] / 1000,
                                      end=e["end_ms"] / 1000, origin="candidate" if key != "_low" else "low_conf",
                                      heur=e["category"]))

    occupied: set[tuple[str, int]] = set()

    def reserve(item: dict) -> bool:
        grid = grids[item["rid"]]
        if not len(grid):
            return False
        # Mirror write_gold's milliseconds conversion and gold_windows' nearest-centre rule.
        index = judgment_window_index(seconds_to_ms(item["start"]), seconds_to_ms(item["end"]), grid)
        key = (item["rid"], index)
        if key in occupied:
            return False
        occupied.add(key)
        return True

    def balanced_pick(pool: list[dict], limit: int) -> list[dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for item in pool:
            groups[item["rid"]].append(item)
        speakers = sorted(groups)
        rng.shuffle(speakers)
        for items in groups.values():
            rng.shuffle(items)
        selected = []
        while len(selected) < limit and speakers:
            remaining = []
            for rid in speakers:
                candidates = groups[rid]
                while candidates:
                    item = candidates.pop()
                    if reserve(item):
                        selected.append(item)
                        break
                if candidates:
                    remaining.append(rid)
                if len(selected) >= limit:
                    break
            speakers = remaining
        return selected

    items = []
    def capacity(category: str) -> int:
        return len({(item["rid"], judgment_window_index(seconds_to_ms(item["start"]),
                                                         seconds_to_ms(item["end"]), grids[item["rid"]]))
                    for item in by_class.get(category, []) if len(grids[item["rid"]])})

    # Fill scarce classes first, so a class with one possible window is not
    # displaced by another class that has a non-colliding alternative.
    for category in sorted(TAXONOMY, key=capacity):
        items += balanced_pick(by_class.get(category, []), per_class)
    items += balanced_pick(by_class.get("_low", []), low_conf)

    # Negative examples are actual 3 s dataset windows that overlap no heuristic
    # event; this avoids random intervals mapping to an already selected window.
    negative_pool = []
    for r in man["recordings"]:
        rid = r["recording_id"]
        spans = [(e["start_ms"] / 1000, e["end_ms"] / 1000) for e in per_rec_events[rid]]
        for start in grids[rid]:
            a, b = float(start), float(start + WINDOW_S)
            if any(min(b, s2) - max(a, s1) > 0 for s1, s2 in spans):
                continue
            negative_pool.append(dict(rid=rid, idx=-1, start=a, end=b,
                                      origin="negative", heur=BACKGROUND))
    items += balanced_pick(negative_pool, negatives)
    rng.shuffle(items)
    return items


def validate_sampled_windows(man: dict, items: list[dict]) -> None:
    """Fail before provider calls if a future sampler change reintroduces collisions."""
    by_recording: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_recording[item["rid"]].append(dict(category=BACKGROUND,
                                              start_ms=seconds_to_ms(item["start"]),
                                              end_ms=seconds_to_ms(item["end"])))
    for recording in man["recordings"]:
        rid = recording["recording_id"]
        try:
            gold_windows(by_recording[rid], window_starts(recording["duration_s"]))
        except ValueError as exc:
            raise ValueError(f"{rid}: sampled judge intervals are not exportable: {exc}") from exc


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--provider", choices=["auto", *PROVIDER_CALLERS], default="auto",
                    help="proveedor explícito; usa ollama para un modelo de visión local")
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
    validate_sampled_windows(man, items)
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    provider = provider_for_model(a.model, a.provider)
    done = load_judgment_cache(CACHE, provider, a.model)
    print(f"[judge] {len(items)} ítems muestreados | veredictos versionados en caché: {len(done)} | "
          f"proveedor {provider} | modelo {a.model}")
    counts = defaultdict(int)
    for it in items:
        counts[it["origin"] + ":" + it["heur"]] += 1
    print("[judge] muestreo:", dict(sorted(counts.items())))
    if a.dry_run:
        it = items[0]; r = recs[it["rid"]]
        z = np.load(ROOT / r["features_npz"])
        validate_raw_features(z, path=str(ROOT / r["features_npz"]))
        w = json.loads((ROOT / r["words_json"]).read_text(encoding="utf-8"))
        p, s, n = context_text(w, it["start"], it["end"])
        print("\n--- PROMPT DE EJEMPLO ---\n" + build_prompt(p, s, n, acoustic_summary(z, it["start"], it["end"])))
        return

    provider, key = pick_provider(a.model, provider)
    caller = caller_for(provider)
    npz_cache: dict[str, object] = {}
    words_cache: dict[str, list] = {}
    t0, n_new, errors = time.time(), 0, 0
    fail_kinds: Counter = Counter()
    notes: Counter = Counter()
    current_done: dict[tuple, dict] = {}
    with CACHE.open("a", encoding="utf-8") as fh:
        for k, it in enumerate(items):
            r = recs[it["rid"]]
            if it["rid"] not in npz_cache:
                npz_cache[it["rid"]] = np.load(ROOT / r["features_npz"])
                validate_raw_features(npz_cache[it["rid"]], path=str(ROOT / r["features_npz"]))
                words_cache[it["rid"]] = json.loads((ROOT / r["words_json"]).read_text(encoding="utf-8"))
            z, words = npz_cache[it["rid"]], words_cache[it["rid"]]
            prev, seg, nxt = context_text(words, it["start"], it["end"])
            meas = acoustic_summary(z, it["start"], it["end"])
            mid = (it["start"] + it["end"]) / 2
            mp4 = ROOT / r["mp4"] if r.get("mp4") else None
            imgs = frames_at(mp4, [it["start"], mid, max(it["end"] - 0.05, mid)]) if mp4 else []
            prompt = build_prompt(prev, seg, nxt, meas)
            fingerprint = judgment_input_fingerprint(prompt, imgs, mp4)
            key_t = (provider, a.model, it["rid"], seconds_to_ms(it["start"]),
                     seconds_to_ms(it["end"]), fingerprint)
            if key_t in done:
                current_done[key_t] = done[key_t]
                continue
            try:
                out, n_used, note = ask_judge(caller, a.model, prompt, imgs, key, CLASSES)
            except Exception as exc:
                errors += 1
                fail_kinds[type(exc).__name__] += 1
                log_failure(ERRORS, rid=it["rid"], start=it["start"], end=it["end"],
                            provider=provider, model=a.model,
                            kind=type(exc).__name__, reason=str(exc)[:300],
                            raw=getattr(exc, "raw", "")[:500], n_frames=len(imgs))
                continue
            if note != "ok":
                notes[note] += 1
            rec = dict(rid=it["rid"], speaker_id=r["speaker_id"], start=it["start"], end=it["end"],
                       origin=it["origin"], heuristic=it["heur"], judge=out["classification"],
                       confidence=float(out.get("confidence", 0.0)), rationale=out.get("rationale", "")[:200],
                       provider=provider, model=a.model, n_frames=n_used, note=note,
                       input_fingerprint=fingerprint, interval_identity_version=JUDGE_INTERVAL_VERSION,
                       ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            done[key_t] = rec; current_done[key_t] = rec; n_new += 1
            if n_new % 20 == 0:
                print(f"   {n_new} nuevos | {len(current_done)}/{len(items)} | {(time.time()-t0)/max(n_new,1):.1f} s/ítem")
            time.sleep(a.sleep)
    print(f"[judge] listo: {n_new} nuevos, {errors} errores, {len(current_done)} actuales")
    if fail_kinds:
        print(f"[judge] errores por tipo: {dict(fail_kinds)}")
        print(f"[judge] detalle con la respuesta cruda en {ERRORS.relative_to(ROOT)}")
    if notes:
        print(f"[judge] juicios que necesitaron escalado: {dict(notes)}")
        if notes.get("sin_fotogramas"):
            print(f"[judge] AVISO: {notes['sin_fotogramas']} juicios se resolvieron SIN fotogramas "
                  "(el modelo rechazó las imágenes). Quedan marcados con n_frames=0 y note="
                  "'sin_fotogramas': no son equivalentes a un juicio multimodal.")
    write_gold(man, current_done)


def write_gold(man: dict, done: dict) -> None:
    """Escribe {id}.gold.json: SOLO los intervalos juzgados (el resto queda sin etiqueta)."""
    by_rec: dict[str, list] = defaultdict(list)
    for j in done.values():
        if j.get("interval_identity_version") != JUDGE_INTERVAL_VERSION:
            raise ValueError("legacy judgment uses obsolete interval identity; rerun llm_judge.py")
        if j["judge"] == BACKGROUND:
            cat = BACKGROUND
        else:
            cat = j["judge"]
        n_frames = j.get("n_frames")
        by_rec[j["rid"]].append(dict(category=cat, start_ms=seconds_to_ms(j["start"]),
                                     end_ms=seconds_to_ms(j["end"]),
                                     confidence=j["confidence"], source="gold_llm", model=j["model"],
                                     provider=j.get("provider"), n_frames=n_frames,
                                     input_fingerprint=j.get("input_fingerprint"),
                                     interval_identity_version=j.get("interval_identity_version"),
                                     visual_evidence_used=(n_frames > 0 if n_frames is not None else None),
                                     fallback_note=j.get("note", "legacy_unknown"),
                                     heuristic=j["heuristic"], origin=j["origin"], rationale=j["rationale"]))
    # Validate the entire export before touching any current GOLD file. Two
    # judgments on one model window cannot both become independent test items.
    for r in man["recordings"]:
        rid = r["recording_id"]
        try:
            gold_windows(by_rec.get(rid, []), window_starts(r["duration_s"]))
        except ValueError as exc:
            raise ValueError(f"{rid}: {exc}") from exc
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
