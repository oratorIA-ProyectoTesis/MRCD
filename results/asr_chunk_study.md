# Disfluencias que sobreviven al ASR, por tamano de trozo

- Audio: `mrcd_sesion_1790263526.wav` (90.8 s)
- Whisper: `small` (VerbatimASR) | solape 2.0 s

> Whisper se entrena para producir texto legible. Cuanto mas contexto ve de
> una vez, mas corrige hacia prosa fluida y mas disfluencias borra. El tamano
> de trozo no es un detalle: gobierna cuanta evidencia linguistica llega al
> clasificador, y en entrenamiento, que etiquetas produce la cascada.

| trozo | palabras | repeticiones | muletillas | edicion | alargamientos | marcadores |
|---|---|---|---|---|---|---|
| 5s | 115 | 2 | 3 | 0 | 0 | **5** |
| 8s | 115 | 2 | 4 | 0 | 0 | **6** |
| 13s | 112 | 2 | 4 | 0 | 0 | **6** |
| 30s | 111 | 0 | 4 | 0 | 0 | **4** |
| entero | 107 | 0 | 3 | 0 | 0 | **3** |

Transcribir en trozos de **8s** conserva **3 marcadores mas** que transcribir el archivo entero (6 vs 3).

## Transcripciones

**5s**

> [text withheld]

**8s**

> [text withheld]

**13s**

> [text withheld]

**30s**

> [text withheld]

**entero**

> [text withheld]

