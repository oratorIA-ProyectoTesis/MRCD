# MRCD service image: CPU only, Whisper small baked in, non-root, health-checked.
#
#   docker build -t mrcd .
#   docker run --rm -p 127.0.0.1:8765:8765 -v "$PWD/models:/models:ro" mrcd
#
# The trained checkpoint (ckpt_audio_text.pt) is NOT baked into the image: mount it at /models (read-only).
# Without it the container still starts and /v1/health answers, but analysis jobs fail with `checkpoint_unavailable`.
# Extra service flags go after the image name, e.g. `mrcd --cors-origin https://example.org`.
#
# Base pinned by digest (python:3.12-slim, multi-arch index); Python dependencies pinned in docker/constraints.txt.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ARG MRCD_VERSION=0.3.0
ARG TORCH_VERSION=2.9.1
ARG WHISPER_SIZE=small

LABEL org.opencontainers.image.title="mrcd" \
      org.opencontainers.image.description="MRCD: local detection of filler words, prolongations and repetitions in Spanish speech (HTTP service, contract mrcd.analysis/v1)" \
      org.opencontainers.image.source="https://github.com/oratorIA-ProyectoTesis/MRCD" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${MRCD_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    OMP_NUM_THREADS=4 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MRCD_CKPT=/models/ckpt_audio_text.pt \
    MRCD_DEVICE=cpu \
    MRCD_PRELOAD=p0

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin mrcd

# PyTorch from the CPU index first: the default Linux wheel pulls the CUDA stack (several GB) the service never uses.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

WORKDIR /src
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY docker/constraints.txt ./constraints.txt
COPY mrcd ./mrcd
RUN pip install -c constraints.txt ".[service]" \
 && mkdir -p /usr/share/doc/mrcd \
 && cp LICENSE THIRD_PARTY_NOTICES.md /usr/share/doc/mrcd/ \
 && cd / \
 && rm -r /src

COPY docker/entrypoint.sh /usr/local/bin/mrcd-entrypoint
RUN chmod 0755 /usr/local/bin/mrcd-entrypoint

# Whisper baked into the image: no download at run time; HF_HUB_OFFLINE forbids any network access afterwards.
# (download_model fetches into $HF_HOME/hub without loading ctranslate2, so it also works under QEMU arm64.
# Silero VAD ships inside the silero-vad wheel.)
RUN python -c "from faster_whisper.utils import download_model; download_model('${WHISPER_SIZE}')" \
 && chown -R mrcd:mrcd /opt/hf
ENV HF_HUB_OFFLINE=1

USER mrcd
WORKDIR /home/mrcd
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/v1/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["/usr/local/bin/mrcd-entrypoint"]
CMD []
