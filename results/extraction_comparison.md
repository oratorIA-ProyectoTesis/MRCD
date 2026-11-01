# Comparacion de extracciones

- **archivo entero**: `dataset_manifest.json`
- **trozos 8s**: `dataset_manifest.json` (asr_chunk_s = 8.0)
- Grabaciones comparadas: 87 · 161.5 min
- Palabras transcritas: 24744 -> 26728 (+8.0 %)

> Estas son ETIQUETAS HEURISTICAS, no exactitud. Mas eventos significa que la
> cascada encontro mas candidatos, no que acierte mas. Confirmarlo exige el
> acuerdo con el juez ciego y con anotacion humana.

| Clase | archivo entero | trozos 8s | cambio | oradores con >=1 |
|---|---|---|---|---|
| filler_word | 204 | 421 | **+217** (+106 %) | 17 |
| prolongation | 115 | 71 | **-44** (-38 %) | 15 |
| repetition | 92 | 228 | **+136** (+148 %) | 18 |
| block | 8 | 3 | **-5** (-62 %) | 3 |
| revision | 33 | 33 | **+0** (+0 %) | 8 |
| rhetorical_pause | 89 | 101 | **+12** (+13 %) | 12 |
| neutral_pause | 493 | 492 | **-1** (-0 %) | 17 |

| **total** | 1034 | 1349 | **+315** | |

## Como leer los dos efectos

**Repeticiones y muletillas (+136 y +217).** Recuperacion de recall en la
capa ASR: Whisper borra repeticiones y muletillas cuando transcribe mucho
contexto de una vez, y lo que no transcribe no puede etiquetarse porque
`detect_repetitions` y `detect_fillers` operan sobre las palabras del ASR.

**Prolongaciones (-44).** Si baja, es esperable y probablemente una mejora
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

