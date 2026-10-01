# MRCD (API + worker) en CPU. Construir y levantar: docker compose up --build
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# torch CPU primero: evita bajar la build CUDA de PyPI (varios GB que no se usan).
# Versiones exactas: otra versión de Whisper o torch puede cambiar los eventos detectados.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchaudio==2.11.0+cpu
COPY requirements-app.lock .
RUN pip install --no-cache-dir -r requirements-app.lock

COPY core core
COPY app app
COPY config config
COPY docs docs
COPY models models

# Versión del código para la procedencia de cada análisis (la imagen no incluye git)
ARG MRCD_COMMIT=
ENV MRCD_COMMIT=$MRCD_COMMIT \
    MRCD_DATA=/data \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000
VOLUME ["/data", "/cache"]
CMD ["python", "-m", "app", "--host", "0.0.0.0", "--port", "8000"]
