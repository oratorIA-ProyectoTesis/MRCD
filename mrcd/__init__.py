"""MRCD — detección de disfluencias del habla en español por fusión multimodal.

Integración mínima
------------------
    from mrcd import DisfluencyEngine

    engine = DisfluencyEngine.load("models/ckpt_audio_text.pt")   # una vez, al arrancar
    result = engine.analyze_file("audio.wav")

    for ev in result:
        print(f"{ev.start:6.2f}s  {ev.label:16s} {ev.confidence:.0%}")

Lo que se exporta aquí es la API estable. Todo lo demás (`mrcd.dataset`,
`mrcd.models`, `mrcd.extractors`, ...) es interno y puede cambiar entre versiones
menores sin aviso: si tu integración lo importa, se romperá.

◆ Antes de mostrar estos resultados a un usuario final
------------------------------------------------------
El checkpoint se entrena con SUPERVISIÓN DÉBIL: las etiquetas vienen de una
cascada heurística, no de anotación humana. Su exactitud NO está establecida —
falta el acuerdo del etiquetador automático con anotadores humanos. `result.warning`
lo declara; propágalo. Presentar estas salidas como detección validada afirma una
precisión que nadie ha medido.

Límites conocidos, medidos sobre el corpus de desarrollo:

- La clase `block` no es medible con este corpus (n=2 en evaluación). El motor
  puede emitirla, pero no hay evidencia de que acierte.
- El aporte del canal de vídeo sobre audio+texto NO es distinguible de cero:
  IC 95 % de la diferencia de macro-F1 = [-0.057, +0.014]. El modo `trimodal`
  existe y funciona, pero afirmar que el vídeo mejora la detección no tiene
  respaldo en estos datos.
- El corpus es habla ensayada (charlas). No hay evidencia sobre habla
  patológica ni sobre conversación espontánea.
"""

# Recorded BEFORE anything below imports numpy. ``OMP_NUM_THREADS`` only reaches the OpenMP runtime
# when it is in the environment before numpy is first imported (measured, see reproducibility.py);
# the engine reports this state in ``AnalysisResult.metadata["environment"]`` and warns when it is
# missing. Nothing is pinned here: importing a library must not change the caller's environment.
import os as _os
import sys as _sys

_IMPORT_STATE = {
    "omp_num_threads_at_import": _os.environ.get("OMP_NUM_THREADS"),
    "numpy_imported_before_mrcd": "numpy" in _sys.modules,
}

from mrcd.constants import BACKGROUND, LABELS, SAMPLE_RATE, TAXONOMY
from mrcd.engine import AnalysisResult, DisfluencyEngine, Event, Window

__version__ = "0.3.0"

__all__ = [
    "DisfluencyEngine",
    "AnalysisResult",
    "Event",
    "Window",
    "TAXONOMY",
    "LABELS",
    "BACKGROUND",
    "SAMPLE_RATE",
    "__version__",
]
