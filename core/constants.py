"""Constantes canónicas del motor MRCD (fuente única de verdad).

Todo módulo debe importar desde aquí: taxonomía, orden del vector 12-D,
parámetros de ventana y frecuencias de muestreo.
"""
from __future__ import annotations

# ---------------------------------------------------------------- audio
SAMPLE_RATE = 16_000            # Hz, PCM 16-bit mono
WINDOW_S = 3.0                  # ventana de análisis
STRIDE_S = 0.5                  # avance
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_S)   # 48 000
ACOUSTIC_HOP_S = 0.010          # 100 fps para F0 / RMS / VAD

# ---------------------------------------------------------------- video
VIDEO_FPS = 10                  # fps de trabajo del extractor cinésico
KINESIC_STEPS = int(WINDOW_S * VIDEO_FPS)      # 30 pasos por ventana
MIN_KINESIC_FRAMES = 24         # mínimo de frames reales para considerar la ventana "visual"

# ------------------------------------------------ vector cinésico 12-D
# Orden canónico e inmutable (ver especificación AsyncAPI / tabla de índices).
KINESIC_FIELDS: tuple[str, ...] = (
    "head_pitch",          # 0  grados
    "head_yaw",            # 1  grados
    "head_roll",           # 2  grados
    "eye_look_in_left",    # 3  [0,1] ARKit
    "eye_look_out_right",  # 4
    "eye_look_up_left",    # 5
    "eye_look_down_right", # 6
    "mouth_press_left",    # 7
    "mouth_press_right",   # 8
    "jaw_open",            # 9
    "mouth_pucker",        # 10
    "brow_down_avg",       # 11 promedio browDownLeft/Right
)
KINESIC_DIM = len(KINESIC_FIELDS)
assert KINESIC_DIM == 12
KIDX = {name: i for i, name in enumerate(KINESIC_FIELDS)}

# ------------------------------------------------------------ taxonomía
TAXONOMY: tuple[str, ...] = (
    "filler_word",
    "prolongation",
    "repetition",
    "block",
    "revision",
    "rhetorical_pause",
    "neutral_pause",
)
BACKGROUND = "fluent"                       # clase de fondo para entrenamiento
LABELS: tuple[str, ...] = (BACKGROUND,) + TAXONOMY
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

CLASSIFICATION_OF = {
    "rhetorical_pause": "RHETORICAL_PAUSE",
    "neutral_pause": "NEUTRAL_PAUSE",
}  # el resto -> "DISFLUENCY"

# ------------------------------------ umbrales heurísticos (preliminares)
# «Umbrales heurísticos de diseño preliminares propuestos por el autor,
#  sujetos a calibración experimental».
H = dict(
    silence_min_s=0.60,          # silencio candidato
    rhetorical_min_s=0.80,
    rhetorical_max_s=2.50,
    prolongation_min_s=0.35,
    flat_f0_max_hz=5.0,          # ΔF0 máximo para considerar "plano"
    repetition_max_gap_s=1.5,
    mouth_press_relaxed=0.15,
    mouth_press_tense=0.40,
    brow_down_tense=0.35,
    head_stable_deg=4.0,         # desviación estándar máx. para "cabeza estable"
    confidence_threshold=0.75,
)

# Léxico base de muletillas (es-PE + panhispánico). Ampliable por variedad.
FILLER_LEXICON: dict[str, set[str]] = {
    "es": {"eh", "ehh", "em", "emm", "mm", "mmm", "este", "esteee", "o sea", "bueno",
           "tipo", "pues", "a ver", "digamos", "ah", "am"},
    "es-PE": {"pe", "ya", "manyas", "osea"},
}
EDIT_TERMS = {"digo", "perdón", "perdon", "es decir", "mejor dicho", "o sea"}
# Palabras funcionales: una pausa tras ellas rompe un constituyente estrecho.
FUNCTION_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del",
    "de", "a", "en", "con", "por", "para", "sin", "sobre", "entre", "hacia", "desde",
    "y", "e", "o", "u", "que", "pero", "mi", "mis", "tu", "tus", "su", "sus",
    "este", "esta", "estos", "estas", "ese", "esa", "muy", "más", "se", "me", "te", "nos",
}

WHISPER_INITIAL_PROMPT = "eh, este, o sea, mmm, bueno, tipo, ah, este..."

# ------------------------------------------- contexto discursivo (v2)
# La ventana de 3 s no alcanza para decidir si una pausa cierra una unidad
# de discurso (retórica) o la interrumpe (bloqueo/duda). El ramal lingüístico
# mira una ventana más amplia centrada en la de análisis.
CONTEXT_S = 10.0                 # ventana lingüística ampliada
CONTEXT_PRE_S = 5.0              # cuánto mira hacia atrás respecto del inicio
MAX_CONTEXT_WORDS = 40           # tokens máx. del ramal lingüístico ampliado

# ------------------------------- rasgos de frontera prosódica (v2)
# Medidas explícitas alrededor del silencio más largo de la ventana.
# El reinicio de tono (pitch reset) es el marcador canónico de frontera
# prosódica frente a una suspensión disfluente.
PROSODY_FIELDS: tuple[str, ...] = (
    "sil_dur",          # 0  duración del silencio más largo (s)
    "sil_pos",          # 1  inicio del silencio, normalizado en la ventana [0,1]
    "f0_slope_pre",     # 2  pendiente de F0 en los 300 ms previos (st/s)
    "f0_range_pre",     # 3  rango de F0 en esos 300 ms (st)
    "f0_level_pre",     # 4  F0 medio previo, centrado por hablante (st)
    "f0_reset",         # 5  F0 al retomar menos F0 previo (st)  <-- clave
    "rms_slope_pre",    # 6  pendiente de energía previa (log10/s)
    "final_lengthen",   # 7  alargamiento de la última sílaba antes del silencio
    "reset_valid",      # 8  1.0 si f0_reset pudo medirse; 0.0 si no había voz a ambos
                        #    lados. Sin esta bandera, un 0.0 en f0_reset significa a
                        #    la vez «no hubo reinicio» (frontera ausente) y «no se
                        #    pudo medir» (falta de frames sonoros), que son lo
                        #    contrario. El modelo no podía distinguirlos.
)
PROSODY_DIM = len(PROSODY_FIELDS)
PROSODY_PRE_S = 0.30             # ventana de análisis pre-silencio
PROSODY_POST_S = 0.30            # ventana de análisis post-silencio

# Encabezado obligatorio para cualquier tabla basada en etiquetas heurísticas
SMOKE_HEADER = "PRUEBA DE HUMO — etiquetas heurísticas, NO válida como evidencia"
RANDOM_SEED = 13
