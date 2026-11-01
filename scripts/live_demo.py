#!/usr/bin/env python3
"""Microphone-only live audio_text demonstration; trimodal needs a future camera path."""

from __future__ import annotations

try:  # must run before numpy / torch / ctranslate2 / mrcd are imported (see scripts/_pin_omp.py)
    import _pin_omp  # noqa: F401  (script run as a file)
except ImportError:  # imported as scripts.<module>
    from scripts import _pin_omp  # noqa: F401

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrcd.constants import CONTEXT_S, SAMPLE_RATE, STRIDE_S, WINDOW_S
from mrcd.live.mic_stream import MicrophoneStream
from mrcd.live.streaming_asr import StreamingASR
from mrcd.live.session import CONFIGS, LIVE_CFG_V2, LiveConfig, LiveScorer, get_config
from mrcd.live.window_builder import window_text
from scripts.infer_video import load_ckpt, predict


class _SessionState:
    """Everything one start()..stop() session owns, so a late thread can never touch the next session."""

    def __init__(self, scorer: LiveScorer, generation: int) -> None:
        self.scorer, self.generation = scorer, generation
        self.lock = threading.Lock()  # guards consumed / stopped / tail
        self.wake = threading.Event()  # set by stop() to cut the tick sleep
        self.done = (
            threading.Event()
        )  # set when the worker (including finalize) is over
        self.consumed = 0  # absolute samples already read from the ring
        self.stopped = False
        self.tail: tuple | None = (
            None  # audio left in the ring at stop(), read synchronously by stop()
        )
        self.words: list = []
        self.recent = (
            None  # last CONTEXT_S + WINDOW_S seconds fed to the tracker (for the ASR)
        )
        self.fed = 0
        self.error: BaseException | None = None


class LiveEngine:
    """Threads + microphone + ASR around :class:`mrcd.live.session.LiveScorer`.

    ``config`` selects a named live configuration (``live_cfg_v1`` = behaviour of commit 80a921a,
    ``live_cfg_v2`` = + end-of-session pass and provisional/final rows; see mrcd/live/session.py).
    Each ``start()`` creates a fresh session state: ``session_rows`` / ``gaps`` are new lists, the ASR
    words of the previous session are not shared, and a worker of an older session can only write to
    its own state (no stale rows after a quick stop -> start).
    """

    def __init__(
        self,
        checkpoint: Path,
        model_size: str = "small",
        device: str | None = None,
        mic: MicrophoneStream | None = None,
        asr: StreamingASR | None = None,
        mic_device: int | str | None = None,
        f0_center: str | float = "causal",
        config: str | LiveConfig = LIVE_CFG_V2,
    ):
        import torch

        self.config = get_config(config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net, self.norm, self.labels, _ = load_ckpt(checkpoint, self.device)
        self.ling_dim = self.net.ling.proj[0].in_features
        self.ling_context = self.ling_dim > 10
        self.include_pros = self.net.pros is not None
        self._common_init(
            mic
            or MicrophoneStream(
                device=mic_device, fill_dropped=self.config.fill_dropped_audio
            ),
            asr or StreamingASR(model_size),
            f0_center,
        )
        self.checkpoint = Path(checkpoint)

    @classmethod
    def for_parts(
        cls,
        score,
        labels,
        *,
        mic,
        asr,
        config=LIVE_CFG_V2,
        f0_center="causal",
        ling_context=True,
        include_pros=True,
        ling_dim=None,
        tick_s=STRIDE_S,
    ):
        """Engine around a plain ``score(tensors) -> probabilities`` (tests, no checkpoint)."""
        self = cls.__new__(cls)
        self.config = get_config(config)
        self.labels, self.ling_context, self.include_pros = (
            list(labels),
            ling_context,
            include_pros,
        )
        self.ling_dim = ling_dim
        self._score_fn = score
        self.checkpoint = Path("<parts>")
        self._common_init(mic, asr, f0_center)
        self.tick_s = tick_s
        return self

    def _common_init(self, mic, asr, f0_center) -> None:
        # 'causal': median of the voiced F0 seen so far (live cannot know the whole-file median that the
        # offline path uses; NOT equivalent to offline, divergence measured in PARIDAD_LIVE.md).
        self.f0_center = f0_center
        self.mic, self.asr = mic, asr
        self.tick_s = STRIDE_S
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._state_lock = threading.Lock()
        self._generation = 0
        self._state: _SessionState | None = None
        # Historial completo de la sesion, para descargarlo junto al audio y poder
        # auditar cada ventana contra lo que realmente se dijo. New list per start().
        self.session_rows: list[dict] = []
        self.gaps: list[dict] = []
        self.last_error: BaseException | None = None

    # ----------------------------------------------------------------- scoring
    def _score(self, tensors: dict):
        if hasattr(self, "_score_fn"):
            return self._score_fn(tensors)
        return predict(self.net, self.norm, tensors, self.device)[0]

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        state = self._begin_session(loop)
        if state is None:
            return False
        threading.Thread(target=self._run, args=(state,), daemon=True).start()
        return True

    def _begin_session(self, loop) -> "_SessionState | None":
        """Create the new session state and open the microphone (no thread: tests drive ``_run`` directly)."""
        with self._state_lock:
            if self._running:
                return None
            self._loop = loop
            self.mic.reset()
            scorer = LiveScorer(
                self._score,
                self.labels,
                config=self.config,
                f0_center=self.f0_center,
                context=self.ling_context,
                include_pros=self.include_pros,
                ling_dim=self.ling_dim,
                asr_mark=self.asr.asr_mark() if hasattr(self.asr, "asr_mark") else None,
            )
            self._generation += 1
            state = _SessionState(scorer, self._generation)
            self._state = state
            self.session_rows, self.gaps = scorer.rows, scorer.gaps
            if hasattr(self.asr, "words"):
                self.asr.words = []  # legacy attribute; the session words live in ``state.words``
            self.mic.start()
            self._running = True
        return state

    def stop(self) -> bool:
        """Stop the microphone; the worker then scores the tail (``finalize``). Idempotent."""
        with self._state_lock:
            was_running = self._running
            self._running = False
            self.mic.stop()
            state = self._state
            if state is not None:
                with state.lock:
                    if not state.stopped:
                        # read what is left in the ring NOW, before a start() can reset it
                        new, start, lost = self.mic.buffer.read_since(state.consumed)
                        state.consumed = start + len(new)
                        state.tail = (new, lost)
                        state.stopped = True
        if state is not None:
            state.wake.set()
        return was_running

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the current session's worker (including its end-of-session pass) finished."""
        state = self._state
        return True if state is None else state.done.wait(timeout)

    def _emit(self, message: dict, state: _SessionState) -> None:
        # rows of a session that was replaced by a newer start() stay in that session only
        if self._loop and self._state is state:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, message)

    def _asr(self, state: _SessionState, audio, begin_s: float) -> list:
        merge = getattr(self.asr, "merge_recent", None)
        if (
            merge is not None
        ):  # pure w.r.t. the ASR object: only this session's words are touched
            state.words = merge(state.words, audio, begin_s)
        else:  # legacy ASR objects keep their own state
            state.words = list(self.asr.transcribe_recent(audio, begin_s))
        return state.words

    @staticmethod
    def _remember(state: _SessionState, new, lost: int) -> None:
        """Keep the last CONTEXT_S + WINDOW_S seconds fed to the tracker (what the ASR re-transcribes)."""
        import numpy as np

        chunk = np.concatenate([np.zeros(lost, np.float32), new]) if lost else new
        keep = int(round((CONTEXT_S + WINDOW_S) * SAMPLE_RATE))
        state.recent = (
            chunk if state.recent is None else np.concatenate([state.recent, chunk])
        )
        state.recent = state.recent[-keep:]
        state.fed += len(chunk)

    def _tick(self, state: _SessionState, new, lost: int) -> None:
        scorer = state.scorer
        if lost or len(new):
            scorer.push(new, lost)
            self._remember(state, new, lost)
        if scorer.tracker.latest_start() is None:
            self._emit(
                {
                    "status": "warming",
                    "timestamp": scorer.duration_s,
                    "message": "calentando...",
                },
                state,
            )
            return
        begin = (state.fed - len(state.recent)) / SAMPLE_RATE
        words = self._asr(state, state.recent, begin)
        # The ASR re-transcribes only the last 13 s: older windows lack word context.
        for row in scorer.step(words, asr_floor_s=begin + 5.0) or ():
            self._emit(row, state)

    def _run(self, state: _SessionState) -> None:
        """One tick: feed the session tracker, refresh the ASR, score every newly complete grid window.

        Features come from ``SessionFeatureTracker`` (session-accumulated VAD / F0 / frames, absolute
        word times), not from a re-extraction of the 13 s ring buffer: see
        ``mrcd/live/session_features.py`` and results/paridad_live_20261030/PARIDAD_LIVE.md.
        When the session stops the tail is scored by ``LiveScorer.finalize`` (config-dependent).
        """
        try:
            while True:
                with state.lock:
                    if state.stopped:
                        break
                    new, start, lost = self.mic.buffer.read_since(state.consumed)
                    state.consumed = start + len(new)
                self._tick(state, new, lost)
                state.wake.wait(self.tick_s)
            new, lost = state.tail
            scorer = state.scorer
            if len(new) or lost:
                scorer.push(new, lost)
                self._remember(state, new, lost)
            if (
                self.config.finalize_on_stop
                and state.recent is not None
                and scorer.duration_s >= WINDOW_S
            ):
                begin = (state.fed - len(state.recent)) / SAMPLE_RATE
                words = self._asr(
                    state, state.recent, begin
                )  # last transcription sees the whole tail
                for row in scorer.finalize(words, asr_floor_s=begin + 5.0):
                    self._emit(row, state)
            else:
                scorer.finalize(state.words)
        except BaseException as exc:  # noqa: BLE001 - a dead worker must not be silent
            state.error = self.last_error = exc
            import traceback

            traceback.print_exc()
        finally:
            state.done.set()
# La taxonomía interna (nombres de clase, claves de la API, contrato con
# core/constants.py) se queda en inglés a propósito: es el vocabulario que
# comparten el modelo, el checkpoint y el resto del pipeline. Lo que cambia
# aquí es solo la CAPA DE PRESENTACIÓN — lo que lee un humano en pantalla.
CLASS_ES = {
    "fluent": "fluido",
    "filler_word": "muletilla",
    "prolongation": "prolongación",
    "repetition": "repetición",
    "block": "bloqueo",
    "revision": "reparación",
    "rhetorical_pause": "pausa retórica",
    "neutral_pause": "pausa neutra",
}

PAGE = r"""<!doctype html><meta charset=utf-8><title>MRCD live</title>
<style>
body{font:16px system-ui;max-width:960px;margin:3rem auto;padding:0 1rem;color:#17212b}h1{margin-bottom:.25rem}h2{margin-top:2rem}.card,.panel{padding:1rem;border-radius:8px;background:#eef1f4}.controls{display:flex;gap:.5rem;margin:1rem 0}button{border:0;border-radius:6px;padding:.6rem 1rem;font-weight:600;cursor:pointer}button:disabled{opacity:.5;cursor:not-allowed}#start{background:#2a9d8f;color:white}#stop{background:#e63946;color:white}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.75rem}.metric{background:#f5f7f9;padding:.75rem;border-radius:6px}.metric strong{display:block;font-size:1.35rem}.two{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.bar-row{display:grid;grid-template-columns:130px 1fr 52px;align-items:center;gap:.5rem;margin:.45rem 0}.bar{height:.65rem;background:#dfe5ea;border-radius:99px;overflow:hidden}.bar i{height:100%;display:block;background:#457b9d}.events{max-height:300px;overflow:auto}.event{padding:.55rem 0;border-bottom:1px solid #d9e0e6}.event small{color:#59636c}.filler_word{background:#ffd166}.block{background:#ef476f;color:white}.repetition{background:#f78c6b}.revision{background:#b8de6f}.prolongation{background:#b9a7ff}.neutral_pause,.rhetorical_pause{background:#75d5ff}@media(max-width:650px){.two{grid-template-columns:1fr}.bar-row{grid-template-columns:105px 1fr 48px}}
</style>
<h1>MRCD · audio_text en vivo</h1><p id=s>detenido</p>
<div class=controls><button id=start>Iniciar</button><button id=stop disabled>Parar</button><a id=dlaudio href="/download/audio.wav" download><button type=button>Descargar audio</button></a><a id=dljson href="/download/session.json" download><button type=button>Descargar analisis</button></a></div>
<div id=c class=card>Presiona Iniciar para capturar audio.</div>
<h2>Resumen de la sesión</h2><div id=metrics class=grid></div>
<div class=two><section><h2>Probabilidades — ventana actual</h2><div id=probabilities class=panel>Sin ventana evaluada.</div></section><section><h2>Distribución de clases</h2><div id=distribution class=panel>Sin ventanas evaluadas.</div></section></div>
<h2>Últimas ventanas procesadas</h2><div id=events class="panel events">Sin ventanas evaluadas.</div>
<h2>Transcripción acumulada</h2><p id=t></p>
<script>
const CLASS_ES=__CLASS_ES_JSON__;function tr(clase){return CLASS_ES[clase]||clase}
const s=document.querySelector('#s'),c=document.querySelector('#c'),t=document.querySelector('#t'),start=document.querySelector('#start'),stop=document.querySelector('#stop'),metrics=document.querySelector('#metrics'),probabilities=document.querySelector('#probabilities'),distribution=document.querySelector('#distribution'),events=document.querySelector('#events'),ws=new WebSocket(`ws://${location.host}/ws`);let transcript=[],rows=[];
function state(r){start.disabled=r;stop.disabled=!r}function esc(v){let e=document.createElement('span');e.textContent=v;return e.innerHTML}function seconds(v){v=Math.max(0,Math.round(v));return `${Math.floor(v/60)}:${String(v%60).padStart(2,'0')}`}
// `label` llega en la clave interna (inglés); se traduce SOLO al mostrarlo. El
// orden y el conteo siguen agrupados por la clave original, no por la traducción,
// para que dos clases nunca se confundan por compartir texto en español.
function bars(values,total,pct){return Object.entries(values).sort((a,b)=>b[1]-a[1]).map(([label,value])=>`<div class=bar-row><span>${esc(tr(label))}</span><div class=bar><i style="width:${total?100*value/total:0}%"></i></div><b>${pct?`${(100*value).toFixed(1)}%`:`${value}`}</b></div>`).join('')}
function reset(){transcript=[];rows=[];t.textContent='';metrics.innerHTML='';probabilities.textContent='Sin ventana evaluada.';distribution.textContent='Sin ventanas evaluadas.';events.textContent='Sin ventanas evaluadas.'}
function render(){let count=rows.length,last=rows.at(-1),mean=count?rows.reduce((sum,row)=>sum+row.confianza,0)/count:0,counts={};rows.forEach(row=>counts[row.clase]=(counts[row.clase]||0)+1);metrics.innerHTML=`<div class=metric><span>Ventanas evaluadas</span><strong>${count}</strong></div><div class=metric><span>Tiempo analizado</span><strong>${seconds(last?.timestamp||0)}</strong></div><div class=metric><span>Confianza media</span><strong>${(100*mean).toFixed(1)}%</strong></div><div class=metric><span>Clases detectadas</span><strong>${Object.keys(counts).length}</strong></div>`;probabilities.innerHTML=last?bars(last.todas_las_probabilidades,1,true):'Sin ventana evaluada.';distribution.innerHTML=count?bars(counts,count,false):'Sin ventanas evaluadas.';events.innerHTML=rows.slice(-20).reverse().map(row=>`<div class=event><b>${esc(tr(row.clase))}</b> · ${(100*row.confianza).toFixed(1)}% <small>${seconds(row.timestamp)}</small><br>${esc(row.texto_ventana||'—')}</div>`).join('')}
async function control(action){let r=await fetch(`/control/${action}`,{method:'POST'});if(!r.ok){s.textContent='error al cambiar estado';return}if(action==='start'){reset();s.textContent='calentando...';c.className='card';c.textContent='Esperando contexto suficiente.';state(true)}else{s.textContent='detenido';c.className='card';c.textContent='Captura detenida. El resumen queda disponible.';state(false)}}
start.onclick=()=>control('start');stop.onclick=()=>control('stop');function append(text){let incoming=text.trim().split(/\s+/).filter(Boolean),shared=0;for(let n=Math.min(transcript.length,incoming.length);n;n--){if(transcript.slice(-n).join(' ')===incoming.slice(0,n).join(' ')){shared=n;break}}transcript.push(...incoming.slice(shared));t.textContent=transcript.join(' ')}
// x.clase se conserva en inglés para el className (así engancha con el CSS
// .filler_word, .block, etc. definido arriba); el texto visible pasa por tr().
// UPSERT-BEGIN: una ventana cuenta UNA vez. live_cfg_v2 emite al detener filas `final` con `revises`=timestamp de la fila provisional que
// corrigen; esa fila se reemplaza en su lugar (no se agrega otra). Una fila `final` sin provisional previa se agrega.
function upsert(list,x){let k=(x.status==='final'&&x.revises!=null)?x.revises:x.timestamp,i=list.findIndex(r=>r.timestamp===k);if(i>=0){list[i]=x;return false}list.push(x);return true}
// UPSERT-END
ws.onmessage=e=>{let x=JSON.parse(e.data);if(x.status==='warming'){s.textContent='calentando...';return}s.textContent='activo';state(true);c.className='card '+x.clase;c.textContent=`${tr(x.clase)} (${(x.confianza*100).toFixed(1)}%) — ${x.texto_ventana}`;if(upsert(rows,x))append(x.texto_ventana);render()};render();
</script>"""


# El JSON se genera una sola vez desde CLASS_ES: una sola fuente de verdad para la
# traduccion, en vez de mantenerla escrita dos veces (Python y JS) y que diverjan.
RENDERED_PAGE = PAGE.replace(
    "__CLASS_ES_JSON__", json.dumps(CLASS_ES, ensure_ascii=False)
)


def create_app(engine: LiveEngine):
    app = FastAPI()
    running = False

    @app.get("/")
    async def index():
        # no-store: una pagina vieja cacheada por el navegador puede sobrevivir a
        # un reinicio del servidor y mostrar comportamiento que ya se corrigio en
        # el codigo. Ya paso una vez con el bug de conteo-como-porcentaje.
        return HTMLResponse(RENDERED_PAGE, headers={"Cache-Control": "no-store"})

    @app.post("/control/start")
    async def start_control():
        nonlocal running
        if not running:
            engine.start(asyncio.get_running_loop())
            running = True
        return {"running": True}

    @app.post("/control/stop")
    async def stop_control():
        nonlocal running
        if running:
            engine.stop()
            running = False
        return {"running": False}

    @app.get("/control/status")
    async def control_status():
        return {"running": engine.running}

    @app.get("/download/audio.wav")
    async def download_audio():
        """Audio crudo de la sesion, tal como lo oyo el modelo.

        Es la unica forma de distinguir «el modelo no detecto la disfluencia» de
        «el ASR la borro antes»: sin el audio original ambas se ven igual en la UI.
        """
        rec = getattr(engine.mic, "recorder", None)
        if rec is None or rec.duration_s <= 0:
            return JSONResponse(
                {"error": "sin audio grabado en esta sesion"}, status_code=404
            )
        return Response(
            content=rec.to_wav_bytes(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="mrcd_sesion_{int(time.time())}.wav"'
            },
        )

    @app.get("/download/session.json")
    async def download_session():
        """Cada ventana con su clase, confianza, probabilidades y texto del ASR.

        Alineado por `timestamp` con el WAV, de modo que cada decision se puede
        escuchar en su momento exacto.
        """
        rec = getattr(engine.mic, "recorder", None)
        return JSONResponse(
            {
                "generado": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint": str(getattr(engine, "checkpoint", "")),
                "modo": engine.net.mode,
                "aviso": "Checkpoint entrenado con etiquetas heuristicas (supervision debil). "
                "Estas predicciones muestran comportamiento del pipeline, NO exactitud.",
                "sin_video": "Modo audio_text: no hay rama cinesica. 'block' se define por "
                "tension facial visible y aqui no se observa.",
                "audio_duracion_s": round(rec.duration_s, 2) if rec else 0.0,
                "sample_rate": getattr(rec, "sample_rate", None),
                "config_live": getattr(getattr(engine, "config", None), "name", None),
                "huecos": list(getattr(engine, "gaps", [])),
                "n_ventanas": len(engine.session_rows),
                "etiquetas": list(engine.labels),
                "ventanas": engine.session_rows,
            }
        )

    @app.on_event("shutdown")
    async def shutdown():
        if running:
            engine.stop()

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        receive = asyncio.create_task(ws.receive())
        queued = asyncio.create_task(engine.queue.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    (receive, queued), return_when=asyncio.FIRST_COMPLETED
                )
                if receive in done:
                    if receive.result()["type"] == "websocket.disconnect":
                        break
                    receive = asyncio.create_task(ws.receive())
                if queued in done:
                    await ws.send_json(queued.result())
                    queued = asyncio.create_task(engine.queue.get())
        finally:
            for task in (receive, queued):
                if not task.done():
                    task.cancel()
            await asyncio.gather(receive, queued, return_exceptions=True)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    default_ckpt = ROOT / "models" / "ckpt_audio_text.pt"
    if not default_ckpt.exists():
        default_ckpt = ROOT / "models_stale" / "ckpt_audio_text.pt"
    parser.add_argument("--ckpt", default=str(default_ckpt))
    parser.add_argument("--whisper", default="small")
    parser.add_argument(
        "--no-omp-pin",
        action="store_true",
        help="do not set OMP_NUM_THREADS=4 (processed before numpy is imported; see scripts/_pin_omp.py)",
    )
    parser.add_argument(
        "--live-config",
        default=LIVE_CFG_V2.name,
        choices=sorted(CONFIGS),
        help="named live configuration (live_cfg_v1 = previous behaviour, no end-of-session pass)",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--device",
        default=None,
        help="índice o nombre del micrófono; omitir para el predeterminado del sistema",
    )
    args = parser.parse_args()
    checkpoint = Path(args.ckpt)
    if not checkpoint.exists():
        raise SystemExit(f"No existe checkpoint audio_text: {checkpoint}")
    if "models_stale" in checkpoint.parts:
        print(
            "[live] AVISO: usando checkpoint histórico compatible con su contrato antiguo (ling=10, sin prosodia)."
        )
    import uvicorn

    mic_device = (
        int(args.device) if args.device and args.device.isdigit() else args.device
    )
    uvicorn.run(
        create_app(LiveEngine(checkpoint, args.whisper, mic_device=mic_device, config=args.live_config)),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
