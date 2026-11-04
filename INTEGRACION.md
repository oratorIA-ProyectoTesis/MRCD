# Guía de integración del servicio MRCD

Esta guía es para quien quiere llamar al motor MRCD desde otra aplicación. El motor se ejecuta como un servicio HTTP local (contrato `mrcd.analysis/v1`) que recibe un WAV y devuelve eventos de muletilla, prolongación y repetición con su intervalo y puntaje.

> Resultados exploratorios: los puntajes **no están calibrados** (no son probabilidades) y las clases no están confirmadas como detectables con evidencia suficiente. Consulte el [README](README.md) antes de mostrar cifras a usuarios finales.
>
> Las notas históricas de integración del motor (octubre de 2026) se conservan en [`docs/INTEGRACION_AVANCE_20261030.md`](docs/INTEGRACION_AVANCE_20261030.md).

## Qué necesita antes de empezar

| Requisito  | Detalle                                                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| Audio      | WAV mono, 16 kHz, PCM de 16 bits, entre 13 s y 600 s. Otro formato se rechaza con un error del contrato. |
| Checkpoint | `ckpt_audio_text.pt`. **No está dentro de la imagen ni del repositorio**: se entrega bajo solicitud, solo para investigación no comercial (ver [`models/README.md`](models/README.md)). Se monta como volumen (ver más abajo). |
| Docker     | Docker Engine 24 o superior (o Docker Desktop). Para GPU, además NVIDIA Container Toolkit.               |

El servicio **no tiene autenticación**. Publique el puerto solo en el loopback del equipo (`127.0.0.1`); si necesita exponerlo, colóquelo detrás de su propio proxy con autenticación.

## Tres formas de instalarlo

> La imagen en `ghcr.io` se publica **solo cuando se crea una etiqueta `v*`** en el repositorio (el primer paquete nace privado: quien publica debe hacerlo público una vez desde los ajustes del paquete en GitHub). Mientras no exista, construya la imagen localmente (`docker build -t mrcd .`) y use `mrcd` en lugar de `ghcr.io/oratoria-proyectotesis/mrcd:latest`.

### 1. `docker run`

```bash
docker run --rm -p 127.0.0.1:8765:8765 \
  -v "$PWD/models:/models:ro" \
  ghcr.io/oratoria-proyectotesis/mrcd:latest
```

`./models` debe contener `ckpt_audio_text.pt`. En PowerShell use `${PWD}` en lugar de `$PWD` (`-v "${PWD}/models:/models:ro"`). Whisper (`small`) ya viene dentro de la imagen y el servicio no descarga nada al ejecutarse (`HF_HUB_OFFLINE=1`). Eso no aísla la red del contenedor; si lo necesita, use reglas de firewall o una red Docker interna.

### 2. Docker Compose

```bash
git clone https://github.com/oratorIA-ProyectoTesis/MRCD.git && cd MRCD
# Coloque el checkpoint en ./models (docker-compose.yml ya monta esa carpeta como solo lectura)
docker compose up --build            # CPU
docker compose --profile gpu up --build mrcd-gpu    # GPU (ver más abajo)
```

### 3. `pip` (sin Docker)

```bash
pip install "mrcd[service] @ git+https://github.com/oratorIA-ProyectoTesis/MRCD@<etiqueta>"
# Whisper debe estar en la caché local antes de arrancar:
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
OMP_NUM_THREADS=4 python -m mrcd.service serve --ckpt ./models/ckpt_audio_text.pt \
  --device cpu --preload p0 --warmup --incremental
```

Python 3.11 o superior. En este modo el servicio queda en `http://127.0.0.1:8765` y `/docs` (Swagger) está activo, porque el servidor escucha solo en loopback.

## Cómo se maneja el checkpoint

Los pesos entrenados no se incluyen en la imagen. El script de arranque del contenedor busca el archivo en `/models/ckpt_audio_text.pt` (variable `MRCD_CKPT` para cambiar la ruta):

- **Con checkpoint montado:** el servicio precarga el perfil `p0`, hace un calentamiento y habilita `/v1/sessions` (modo incremental).
- **Sin checkpoint:** el contenedor arranca igualmente y `GET /v1/health` responde 200, así como `GET /v1/schema`. Cualquier análisis falla con el código `checkpoint_unavailable` (HTTP 503; en un trabajo asíncrono, `status: "failed"` con `error.code: "checkpoint_unavailable"`). Ningún perfil de análisis funciona sin el checkpoint.

Monte solo un checkpoint de origen confiable: el motor lo carga con `torch.load(..., weights_only=False)`, que ejecuta código arbitrario si el archivo está manipulado. Por eso el volumen se monta de solo lectura (`:ro`).

Los pesos entrenados con corpus de terceros siguen las condiciones de esas fuentes (ver [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

## Endpoints

| Método y ruta                                            | Uso                                                                                                                    |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `GET /v1/health`                                         | Estado (`status`, `engine_loaded`, `warmed_up`, `incremental`).                                                        |
| `GET /v1/schema`                                         | JSON Schema del contrato `mrcd.analysis/v1`.                                                                           |
| `POST /v1/jobs`                                          | Cuerpo: WAV (`Content-Type: audio/wav`). Responde `202` con un `job_id`.                                               |
| `GET /v1/jobs/{job_id}`                                  | Estado del trabajo: `queued`, `running`, `completed` o `failed`; el campo `result` contiene el documento del análisis. |
| `POST /v1/sessions` y rutas bajo `/v1/sessions/{id}/...` | Análisis incremental por lotes de audio (solo con checkpoint y CPU). Ver `docs/contracts/DESIGN_B_INCREMENTAL.md`.     |

Parámetros de consulta de `POST /v1/jobs`: `include_text` (`false` por defecto; el texto transcrito no se devuelve salvo que lo pida), `include_out_of_scope`, `asr_profile` (`p0` por defecto), `decision_profile` (`o2026_11` por defecto o `v0`). Los demás son experimentales; consulte `/v1/schema`.

Los trabajos se guardan solo en memoria y el audio se descarta al terminar. Un resultado `completed` trae `events[]` (cada uno con `class`, `start_s`, `end_s`, `score`) y `summary` (por ejemplo `events_per_minute`, con `available` y `unavailable_reason`: un indicador no medido se informa como no disponible, nunca como `0`). Un resultado `failed` trae `error.code` y `error.message`. Todo documento se puede validar con `python -m mrcd.service validate archivo.json`.

## Ejemplos

Necesita un WAV de prueba. Si no tiene uno propio, genere uno sintético (sin voz humana; solo sirve para comprobar que el servicio responde):

```bash
python scripts/docker_smoke.py --write-wav synthetic.wav
```

### curl

```bash
curl -sf http://127.0.0.1:8765/v1/health
curl -sf -X POST --data-binary @grabacion.wav -H "Content-Type: audio/wav" \
  "http://127.0.0.1:8765/v1/jobs"             # devuelve {"job_id": "...", "status": "queued", ...}
curl -sf http://127.0.0.1:8765/v1/jobs/<job_id>
```

Script completo con espera: [`examples/curl.sh`](examples/curl.sh).

### Python (solo biblioteca estándar)

```python
import json, time, urllib.request

BASE = "http://127.0.0.1:8765"

def call(method, path, data=None):
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "audio/wav"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

job = call("POST", "/v1/jobs", open("grabacion.wav", "rb").read())
while job["status"] not in ("completed", "failed"):
    time.sleep(1)
    job = call("GET", f"/v1/jobs/{job['job_id']}")
for ev in job["result"].get("events", []):
    print(ev["class"], ev["start_s"], ev["end_s"], ev["score"]["value"])
```

Versión ejecutable: [`examples/python_client.py`](examples/python_client.py).

### TypeScript (Node 18+ o navegador)

```ts
const base = "http://127.0.0.1:8765";

export async function analyze(wav: Blob) {
  let job = await (
    await fetch(`${base}/v1/jobs`, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: wav,
    })
  ).json();
  while (job.status !== "completed" && job.status !== "failed") {
    await new Promise((r) => setTimeout(r, 1000));
    job = await (await fetch(`${base}/v1/jobs/${job.job_id}`)).json();
  }
  return job.result; // documento mrcd.analysis/v1
}
```

Desde una página web en otro origen, inicie el servicio con `--cors-origin https://su-sitio.example` (en Docker: `docker run ... mrcd --cors-origin https://su-sitio.example`). Versión para Node: [`examples/analyze.mjs`](examples/analyze.mjs).

## Uso de GPU

La imagen `-cuda` (`ghcr.io/oratoria-proyectotesis/mrcd:latest-cuda`, solo `linux/amd64`) requiere el NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all -p 127.0.0.1:8765:8765 -v "$PWD/models:/models:ro" \
  ghcr.io/oratoria-proyectotesis/mrcd:latest-cuda
```

Qué cambia respecto de la imagen de CPU:

- Solo el reconocimiento de voz (Whisper) usa la GPU. Los modelos que clasifican corren en CPU.
- `serve --device` solo admite `cpu`, así que esta imagen arranca sin ese parámetro y precarga el perfil `p0_cuda`. Con `--device` automático, el perfil por defecto `p0` también se ejecuta en CUDA si hay una GPU visible.
- El texto de Whisper en GPU (float16) puede diferir del texto en CPU (int8) con el que se entrenaron los modelos. Los perfiles `p0_cuda` y `p1_cuda` son **experimentales** y el servicio lo advierte en `warnings`. Si necesita el comportamiento validado, use la imagen de CPU.
- El modo incremental (`/v1/sessions`) no está disponible, porque exige `--device cpu`.

## Solución de problemas

| Síntoma                                                          | Causa probable y solución                                                                                                                                                                                |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Los trabajos terminan en `failed` con `checkpoint_unavailable`   | No hay checkpoint en `/models`. Monte `./models` con `ckpt_audio_text.pt` y verifique el nombre exacto.                                                                                                  |
| El primer arranque tarda varios minutos                          | Con checkpoint, el contenedor carga el modelo y hace un calentamiento antes de aceptar tráfico. El `HEALTHCHECK` espera hasta 180 s (240 s en CUDA). Siga el progreso con `docker logs -f <contenedor>`. |
| `address already in use`                                         | El puerto 8765 está ocupado. Use `-p 127.0.0.1:9000:8765` y apunte su cliente a ese puerto.                                                                                                              |
| Contenedor terminado por memoria (`OOMKilled`)                   | Whisper `small` y el modelo necesitan alrededor de 3 a 4 GB libres; ajuste el límite de memoria de Docker Desktop.                                                                                       |
| `audio_too_short` / `unsupported_sample_rate` / `audio_not_mono` | El WAV debe durar al menos 13 s, ser mono, 16 kHz y PCM16. Convierta con `ffmpeg -i entrada.mp3 -ac 1 -ar 16000 -c:a pcm_s16le salida.wav`.                                                              |
| `asr_unavailable` con un perfil `*_cuda`                         | El contenedor no ve la GPU: ejecute con `--gpus all` y compruebe `nvidia-smi` en el host. Los perfiles de GPU nunca recurren a CPU.                                                                      |
| `queue_full` (HTTP 429)                                          | Hay demasiados trabajos pendientes; reintente más tarde. El motor procesa un trabajo a la vez.                                                                                                           |
| `/docs` devuelve 404 en Docker                                   | Es intencional: el servicio desactiva Swagger cuando escucha fuera del loopback. Use `GET /v1/schema`.                                                                                                   |
| El navegador bloquea la llamada (CORS)                           | Agregue `--cors-origin <origen>` al final del comando de `docker run`.                                                                                                                                   |

## Verificación automática

El flujo [`.github/workflows/docker.yml`](.github/workflows/docker.yml) construye la imagen, la ejecuta y comprueba `/v1/health` y un análisis con `scripts/docker_smoke.py`, validando la respuesta contra `mrcd/service/schemas/mrcd.analysis.v1.schema.json`. En etiquetas `v*` publica las imágenes en `ghcr.io`.
