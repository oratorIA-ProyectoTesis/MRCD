#!/usr/bin/env python3
"""Microphone-only live audio_text demonstration; trimodal needs a future camera path."""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import CONTEXT_S, STRIDE_S, WINDOW_S
from core.live.mic_stream import MicrophoneStream
from core.live.streaming_asr import StreamingASR
from core.live.window_builder import build_latest_contextual_window
from scripts.infer_video import load_ckpt, predict


class LiveEngine:
    def __init__(self, checkpoint: Path, model_size: str = "small", device: str | None = None,
                 mic: MicrophoneStream | None = None, asr: StreamingASR | None = None,
                 mic_device: int | str | None = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net, self.norm, self.labels, _ = load_ckpt(checkpoint, self.device)
        self.ling_dim = self.net.ling.proj[0].in_features
        self.ling_context = self.ling_dim > 10
        self.include_pros = self.net.pros is not None
        self.mic, self.asr = mic or MicrophoneStream(device=mic_device), asr or StreamingASR(model_size)
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._state_lock = threading.Lock()
        self._generation = 0

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        with self._state_lock:
            if self._running:
                return False
            self._loop = loop
            self.mic.reset()
            self.asr.words.clear()
            self.mic.start()
            self._running = True
            self._generation += 1
            generation = self._generation
        threading.Thread(target=self._run, args=(generation,), daemon=True).start()
        return True

    def stop(self) -> bool:
        with self._state_lock:
            was_running = self._running
            self._running = False
            self._generation += 1
            self.mic.stop()
        return was_running

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    def _emit(self, message: dict) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, message)

    def _run(self, generation: int) -> None:
        while self.running and generation == self._generation:
            audio, begin = self.mic.latest_audio(CONTEXT_S + WINDOW_S)
            elapsed = self.mic.buffer.duration_s
            if elapsed < CONTEXT_S + WINDOW_S:
                self._emit({"status": "warming", "timestamp": elapsed, "message": "calentando..."})
            else:
                words = self.asr.transcribe_recent(audio, begin)
                tensors, timestamp, text = build_latest_contextual_window(
                    audio, begin, words, context=self.ling_context, include_pros=self.include_pros,
                    ling_dim=self.ling_dim)
                probabilities = predict(self.net, self.norm, tensors, self.device)[0]
                idx = int(probabilities.argmax())
                self._emit({"timestamp": timestamp, "texto_ventana": text, "clase": self.labels[idx],
                            "confianza": float(probabilities[idx]),
                            "todas_las_probabilidades": {label: float(p) for label, p in zip(self.labels, probabilities)}})
            time.sleep(STRIDE_S)


PAGE = r"""<!doctype html><meta charset=utf-8><title>MRCD live</title>
<style>
body{font:16px system-ui;max-width:960px;margin:3rem auto;padding:0 1rem;color:#17212b}h1{margin-bottom:.25rem}h2{margin-top:2rem}.card,.panel{padding:1rem;border-radius:8px;background:#eef1f4}.controls{display:flex;gap:.5rem;margin:1rem 0}button{border:0;border-radius:6px;padding:.6rem 1rem;font-weight:600;cursor:pointer}button:disabled{opacity:.5;cursor:not-allowed}#start{background:#2a9d8f;color:white}#stop{background:#e63946;color:white}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.75rem}.metric{background:#f5f7f9;padding:.75rem;border-radius:6px}.metric strong{display:block;font-size:1.35rem}.two{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.bar-row{display:grid;grid-template-columns:130px 1fr 52px;align-items:center;gap:.5rem;margin:.45rem 0}.bar{height:.65rem;background:#dfe5ea;border-radius:99px;overflow:hidden}.bar i{height:100%;display:block;background:#457b9d}.events{max-height:300px;overflow:auto}.event{padding:.55rem 0;border-bottom:1px solid #d9e0e6}.event small{color:#59636c}.filler_word{background:#ffd166}.block{background:#ef476f;color:white}.repetition{background:#f78c6b}.revision{background:#b8de6f}.prolongation{background:#b9a7ff}.neutral_pause,.rhetorical_pause{background:#75d5ff}@media(max-width:650px){.two{grid-template-columns:1fr}.bar-row{grid-template-columns:105px 1fr 48px}}
</style>
<h1>MRCD · audio_text en vivo</h1><p id=s>detenido</p>
<div class=controls><button id=start>Iniciar</button><button id=stop disabled>Parar</button></div>
<div id=c class=card>Presiona Iniciar para capturar audio.</div>
<h2>Resumen de la sesión</h2><div id=metrics class=grid></div>
<div class=two><section><h2>Probabilidades — ventana actual</h2><div id=probabilities class=panel>Sin ventana evaluada.</div></section><section><h2>Distribución de clases</h2><div id=distribution class=panel>Sin ventanas evaluadas.</div></section></div>
<h2>Últimas ventanas procesadas</h2><div id=events class="panel events">Sin ventanas evaluadas.</div>
<h2>Transcripción acumulada</h2><p id=t></p>
<script>
const s=document.querySelector('#s'),c=document.querySelector('#c'),t=document.querySelector('#t'),start=document.querySelector('#start'),stop=document.querySelector('#stop'),metrics=document.querySelector('#metrics'),probabilities=document.querySelector('#probabilities'),distribution=document.querySelector('#distribution'),events=document.querySelector('#events'),ws=new WebSocket(`ws://${location.host}/ws`);let transcript=[],rows=[];
function state(r){start.disabled=r;stop.disabled=!r}function esc(v){let e=document.createElement('span');e.textContent=v;return e.innerHTML}function seconds(v){v=Math.max(0,Math.round(v));return `${Math.floor(v/60)}:${String(v%60).padStart(2,'0')}`}function bars(values,total){return Object.entries(values).sort((a,b)=>b[1]-a[1]).map(([label,value])=>`<div class=bar-row><span>${esc(label)}</span><div class=bar><i style="width:${total?100*value/total:0}%"></i></div><b>${typeof value==='number'&&value<=1?`${(100*value).toFixed(1)}%`:`${value}`}</b></div>`).join('')}
function reset(){transcript=[];rows=[];t.textContent='';metrics.innerHTML='';probabilities.textContent='Sin ventana evaluada.';distribution.textContent='Sin ventanas evaluadas.';events.textContent='Sin ventanas evaluadas.'}
function render(){let count=rows.length,last=rows.at(-1),mean=count?rows.reduce((sum,row)=>sum+row.confianza,0)/count:0,counts={};rows.forEach(row=>counts[row.clase]=(counts[row.clase]||0)+1);metrics.innerHTML=`<div class=metric><span>Ventanas evaluadas</span><strong>${count}</strong></div><div class=metric><span>Tiempo analizado</span><strong>${seconds(last?.timestamp||0)}</strong></div><div class=metric><span>Confianza media</span><strong>${(100*mean).toFixed(1)}%</strong></div><div class=metric><span>Clases detectadas</span><strong>${Object.keys(counts).length}</strong></div>`;probabilities.innerHTML=last?bars(last.todas_las_probabilidades,1):'Sin ventana evaluada.';distribution.innerHTML=count?bars(counts,count):'Sin ventanas evaluadas.';events.innerHTML=rows.slice(-20).reverse().map(row=>`<div class=event><b>${esc(row.clase)}</b> · ${(100*row.confianza).toFixed(1)}% <small>${seconds(row.timestamp)}</small><br>${esc(row.texto_ventana||'—')}</div>`).join('')}
async function control(action){let r=await fetch(`/control/${action}`,{method:'POST'});if(!r.ok){s.textContent='error al cambiar estado';return}if(action==='start'){reset();s.textContent='calentando...';c.className='card';c.textContent='Esperando contexto suficiente.';state(true)}else{s.textContent='detenido';c.className='card';c.textContent='Captura detenida. El resumen queda disponible.';state(false)}}
start.onclick=()=>control('start');stop.onclick=()=>control('stop');function append(text){let incoming=text.trim().split(/\s+/).filter(Boolean),shared=0;for(let n=Math.min(transcript.length,incoming.length);n;n--){if(transcript.slice(-n).join(' ')===incoming.slice(0,n).join(' ')){shared=n;break}}transcript.push(...incoming.slice(shared));t.textContent=transcript.join(' ')}
ws.onmessage=e=>{let x=JSON.parse(e.data);if(x.status==='warming'){s.textContent='calentando...';return}s.textContent='activo';state(true);c.className='card '+x.clase;c.textContent=`${x.clase} (${(x.confianza*100).toFixed(1)}%) — ${x.texto_ventana}`;rows.push(x);append(x.texto_ventana);render()};render();
</script>"""


def create_app(engine: LiveEngine):
    app = FastAPI()
    @app.get("/")
    async def index(): return HTMLResponse(PAGE)
    @app.post("/control/start")
    async def start_control():
        engine.start(asyncio.get_running_loop())
        return {"running": True}
    @app.post("/control/stop")
    async def stop_control():
        engine.stop()
        return {"running": False}
    @app.get("/control/status")
    async def control_status():
        return {"running": engine.running}
    @app.on_event("shutdown")
    async def shutdown(): engine.stop()
    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        while True: await ws.send_json(await engine.queue.get())
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    default_ckpt = ROOT / "models" / "ckpt_audio_text.pt"
    if not default_ckpt.exists():
        default_ckpt = ROOT / "models_stale" / "ckpt_audio_text.pt"
    parser.add_argument("--ckpt", default=str(default_ckpt))
    parser.add_argument("--whisper", default="small")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default=None,
                        help="índice o nombre del micrófono; omitir para el predeterminado del sistema")
    args = parser.parse_args()
    checkpoint = Path(args.ckpt)
    if not checkpoint.exists(): raise SystemExit(f"No existe checkpoint audio_text: {checkpoint}")
    if "models_stale" in checkpoint.parts:
        print("[live] AVISO: usando checkpoint histórico compatible con su contrato antiguo (ling=10, sin prosodia).")
    import uvicorn
    mic_device = int(args.device) if args.device and args.device.isdigit() else args.device
    uvicorn.run(create_app(LiveEngine(checkpoint, args.whisper, mic_device=mic_device)), host="127.0.0.1", port=args.port)


if __name__ == "__main__": main()
