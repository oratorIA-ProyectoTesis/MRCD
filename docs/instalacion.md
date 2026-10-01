# Instalación rápida

Dos formas de levantar MRCD. Las dos dejan la misma aplicación en `http://127.0.0.1:8000`, con la API y el worker funcionando juntos.

|            | Docker                                                  | Python                            |
| ---------- | ------------------------------------------------------- | --------------------------------- |
| Necesitas  | Docker Desktop (o Docker Engine)                        | Python 3.12 y FFmpeg              |
| Comando    | `docker compose up --build`                             | `python -m app`                   |
| Ideal para | Probarlo sin instalar nada, o publicarlo en un servidor | Desarrollar y modificar el código |

## Opción 1: Docker (recomendada)

```bash
git clone -b refactor/mrcd-architecture https://github.com/oratorIA-ProyectoTesis/MRCD.git
cd MRCD
docker compose up --build
```

La primera construcción tarda varios minutos. Cuando termine, busca el token del administrador en el registro:

```bash
docker compose logs mrcd
```

Abre `http://127.0.0.1:8000` y entra con ese token.

Para elegir tú el token en lugar de que se genere uno, créalo antes del primer arranque:

```bash
MRCD_ADMIN_TOKEN=mi-token-secreto docker compose up --build
```

**Datos persistentes:** la base de datos y los audios viven en el volumen `mrcd-data`; el modelo Whisper se descarga una sola vez en `mrcd-cache`. Sobreviven a `docker compose down`; solo `docker compose down -v` los borra.

La imagen usa CPU. El primer análisis tarda más porque descarga Whisper `small` (unos 500 MB).

**Reproducibilidad:** la imagen instala versiones exactas (`requirements-app.lock`), así que reconstruirla da el mismo entorno y los mismos resultados. Con `pip install -r requirements-app.txt` las versiones pueden variar y, con ellas, la transcripción y algunos eventos. Para comparar resultados entre máquinas, usa Docker o instala desde el lock.

## Opción 2: Python

Requisitos: Python 3.12, FFmpeg en el `PATH` (`ffmpeg -version` debe responder) y unos 2 GB libres.

```bash
git clone -b refactor/mrcd-architecture https://github.com/oratorIA-ProyectoTesis/MRCD.git
cd MRCD
pip install -r requirements-app.txt
python -m app
```

En el primer arranque imprime el token del administrador **una sola vez**: guárdalo. El repositorio ya incluye el modelo entrenado (`models/ckpt_audio_text.pt`).

Opciones de `python -m app`:

| Opción           | Para qué                                                            |
| ---------------- | ------------------------------------------------------------------- |
| `--port 9000`    | Otro puerto                                                         |
| `--host 0.0.0.0` | Aceptar conexiones de otras máquinas de tu red                      |
| `--no-worker`    | Solo la API; el worker se levanta aparte con `python -m app.worker` |

`requirements-app.txt` instala solo lo que usa la aplicación (audio y texto). Para el pipeline de investigación completo (extracción de video, minería de datos, entrenamiento), usa `requirements.txt`.

## Cargar el piloto de ejemplo (opcional)

Si tienes la carpeta del piloto con su `manifest.json`:

```bash
python -m app.cli import-pilot RUTA/AL/PILOTO --speaker spk_piloto
```

## Configuración

| Variable           | Para qué                                                | Valor por defecto            |
| ------------------ | ------------------------------------------------------- | ---------------------------- |
| `MRCD_ADMIN_TOKEN` | Token del primer administrador                          | Se genera y se imprime       |
| `MRCD_DATA`        | Carpeta de la base de datos y los audios                | `data/app` (Docker: `/data`) |
| `DATABASE_URL`     | Base de datos (SQLite o PostgreSQL)                     | SQLite en `MRCD_DATA`        |
| `MRCD_CHECKPOINT`  | Checkpoint que usa el worker                            | `models/ckpt_audio_text.pt`  |
| `OPENAI_API_KEY`   | Solo para el comparador GPT, si el proyecto lo autoriza | —                            |

## Compartir la demo con otra persona

El micrófono del navegador exige HTTPS. Para exponer tu instancia local con una URL pública:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Imprime una URL `https://….trycloudflare.com` que cambia en cada ejecución. Comparte solo tokens de rol «Usuario del producto» y cierra el túnel al terminar.

## Verificar la instalación

```bash
pip install pytest
python -m pytest -q tests/test_app.py tests/test_evaluation_engine.py
```
