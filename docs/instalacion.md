# Instalación rápida

## Requisitos

- Python 3.12
- FFmpeg en el `PATH` (`ffmpeg -version` debe responder)
- Unos 2 GB libres: dependencias y el modelo Whisper `small`, que se descarga solo la primera vez
- GPU opcional: con CUDA el análisis es más rápido; en CPU funciona

## 1. Descargar

```bash
git clone -b refactor/mrcd-architecture https://github.com/oratorIA-ProyectoTesis/MRCD.git
cd MRCD
pip install -r requirements.txt
```

El repositorio ya incluye el modelo entrenado (`models/ckpt_audio_text.pt`).

## 2. Crear la primera cuenta

```bash
python -m app.cli add-user admin admin
```

Imprime un token **una sola vez**: guárdalo. Las demás cuentas se crean desde la web, en **Administración → Personas**.

## 3. Levantar la aplicación

En dos terminales:

```bash
uvicorn app.api:app --port 8000
```

```bash
python -m app.worker
```

Abre `http://127.0.0.1:8000`, entra con el token y ve a **Prueba del producto**.

## 4. (Opcional) Cargar el piloto de ejemplo

Si tienes la carpeta del piloto con su `manifest.json`:

```bash
python -m app.cli import-pilot RUTA/AL/PILOTO --speaker spk_piloto
```

## Configuración

| Variable          | Para qué                                                | Valor por defecto           |
| ----------------- | ------------------------------------------------------- | --------------------------- |
| `MRCD_DATA`       | Carpeta de la base de datos y los audios                | `data/app`                  |
| `DATABASE_URL`    | Base de datos (SQLite o PostgreSQL)                     | SQLite en `MRCD_DATA`       |
| `MRCD_CHECKPOINT` | Checkpoint que usa el worker                            | `models/ckpt_audio_text.pt` |
| `OPENAI_API_KEY`  | Solo para el comparador GPT, si el proyecto lo autoriza | —                           |

## Compartir la demo con otra persona

El micrófono del navegador exige HTTPS. Para exponer tu instancia local con una URL pública:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Imprime una URL `https://….trycloudflare.com` que cambia en cada ejecución. Comparte solo tokens de rol «Usuario del producto» y cierra el túnel al terminar.

## Verificar la instalación

```bash
python -m pytest -q tests/test_app.py tests/test_evaluation_engine.py
```
