# Comparacion de extracciones

- **archivo entero**: `dataset_manifest.json`
- **trozos 8s**: `dataset_manifest.json` (asr_chunk_s = 8.0)
- Grabaciones comparadas: 12 · 28.2 min
- Palabras transcritas: 4190 -> 4559 (+8.8 %)

> Estas son ETIQUETAS HEURISTICAS, no exactitud. Mas eventos significa que la
> cascada encontro mas candidatos, no que acierte mas. Confirmarlo exige el
> acuerdo con el juez ciego y con anotacion humana.

| Clase | archivo entero | trozos 8s | cambio | oradores con >=1 |
|---|---|---|---|---|
| filler_word | 45 | 99 | **+54** (+120 %) | 6 |
| prolongation | 30 | 15 | **-15** (-50 %) | 5 |
| repetition | 19 | 42 | **+23** (+121 %) | 6 |
| block | 1 | 0 | **-1** (-100 %) | 0 |
| revision | 2 | 2 | **+0** (+0 %) | 1 |
| rhetorical_pause | 21 | 25 | **+4** (+19 %) | 4 |
| neutral_pause | 115 | 117 | **+2** (+2 %) | 5 |

| **total** | 233 | 300 | **+67** | |

## Como leer los dos efectos

**Repeticiones y muletillas (+23 y +54).** Recuperacion de recall en la
capa ASR: Whisper borra repeticiones y muletillas cuando transcribe mucho
contexto de una vez, y lo que no transcribe no puede etiquetarse porque
`detect_repetitions` y `detect_fillers` operan sobre las palabras del ASR.

**Prolongaciones (-15).** Si baja, es esperable y probablemente una mejora
de PRECISION, no una perdida. `detect_prolongations` (auto_labeler.py:244)
excluye toda palabra que sea muletilla o solape con una. Antes, un «eeeh» que
el ASR borraba dejaba su alargamiento acustico atribuido a la palabra de
contenido vecina como prolongacion. Ahora esa muletilla se transcribe, reclama
el intervalo, y deja de contarse como prolongacion de otra cosa.

**Prediccion falsable:** si esa lectura es correcta, el acuerdo del juez ciego
con la clase `prolongation` deberia SUBIR respecto al 12/95 actual
(results/judge_agreement.md), porque desaparecen falsos positivos que eran
muletillas mal atribuidas. Si no sube, la lectura es incorrecta y el cambio
solo movio ruido.

