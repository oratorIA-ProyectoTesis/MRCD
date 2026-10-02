# Evidencia y benchmark

Esta página separa lo que está **medido** de lo que está **pendiente**. Ninguna cifra de exactitud se publica hasta tener una referencia humana independiente.

## Resumen

| Pregunta                                       | Estado        | Evidencia                                                                          |
| ---------------------------------------------- | ------------- | ---------------------------------------------------------------------------------- |
| ¿El motor procesa audio real de punta a punta? | **Medido**    | Grabación → ASR → rasgos → detección → reporte, con Whisper `small` en CPU         |
| ¿Qué tan rápido es?                            | **Medido**    | RTF 0,21 en CPU (ver abajo)                                                        |
| ¿La optimización cambió los resultados?        | **Medido**    | Diferencia máxima de probabilidades 4,5 × 10⁻⁸ entre el camino anterior y el nuevo |
| ¿Qué tan exacto es frente a personas?          | **Pendiente** | Requiere el conjunto de prueba anotado y adjudicado                                |
| ¿Supera a GPT o a las reglas?                  | **Pendiente** | En el piloto no hay referencia humana; solo se midió acuerdo entre sistemas        |

## Rendimiento

Recorte de 40 s de una entrevista real (piloto, segmento s001), CPU Intel de 13.ª generación, sin GPU.

| Etapa                       | Antes (ventana por ventana)             | Ahora (preparación única) |
| --------------------------- | --------------------------------------- | ------------------------- |
| Transcripción ASR           | —                                       | 5,73 s                    |
| Extracción acústica         | Repetida en cada una de las 75 ventanas | 2,46 s (una vez)          |
| Ventanas + inferencia       | —                                       | 0,38 s                    |
| **Extracción + inferencia** | **78,4 s**                              | **2,84 s**                |
| **Total del análisis**      | —                                       | **8,57 s (RTF 0,21)**     |

RTF es el tiempo de análisis dividido por la duración del audio: 0,21 significa que 40 s de audio se analizan en unos 8,6 s. La medición usó pesos aleatorios con la misma arquitectura (el tiempo no depende de los pesos) y VAD por energía. Se reproduce con `python scripts/profile_engine.py`.

Como referencia histórica, la versión anterior del motor tardó 413,36 s en analizar los tres segmentos del piloto (90 s de audio evaluado).

## Reproducibilidad entre plataformas

El mismo audio analizado en Windows (Python local) y en Linux (Docker) daba **30 y 24 eventos**. Se aisló la causa etapa por etapa:

| Etapa | Windows frente a Linux |
|---|---|
| Archivos del modelo Whisper | Idénticos byte a byte |
| Rasgos acústicos (VAD, F0, energía) | Idénticos |
| Espectrograma de entrada al ASR | Diferencias de redondeo de ~10⁻⁵ (NumPy y plataforma) |
| Transcripción con ASR `int8` | **Distinta**: «eso sería entero» frente a «eso sería el feo» |
| Transcripción con ASR `float32` | **Idéntica**: «eso sería el problema» |

Cada entorno era determinista por sí mismo; ni el número de hilos ni la versión de CTranslate2 explicaban la diferencia. El modo cuantizado `int8` convierte diferencias mínimas del espectrograma en otras palabras, y la búsqueda en haz las amplifica. Esas palabras cambian la rama de texto del modelo: 17 de 75 ventanas cambiaban de clase.

**Corrección:** el ASR corre por defecto en CPU con `float32`. Con eso, Windows y Linux producen las mismas 52 palabras, las mismas probabilidades y los mismos 24 eventos, verificado a través de la aplicación completa. El costo es un ASR unas 1,45 veces más lento (13,7 s frente a 9,4 s para 40 s de audio). La configuración del ASR queda registrada en cada análisis y forma parte de la clave de caché, así que un resultado calculado con otra configuración nunca se reutiliza.

Las mediciones de rendimiento de la sección anterior se tomaron con ASR `int8`.

## Piloto de comparación automática

Tres segmentos de 30 s de una entrevista (90 s evaluados), procesados por tres sistemas sin referencia humana.

| Sistema                     | Eventos | Disfluencias | Pausas | Segmentos completados |
| --------------------------- | ------- | ------------ | ------ | --------------------- |
| MRCD                        | 57      | 27           | 30     | 3 de 3                |
| Reglas (cascada heurística) | 31      | 22           | 9      | 2 de 3                |
| GPT audio (`gpt-audio-1.5`) | 6       | 6            | 0      | 3 de 3                |

Coincidencias de misma clase con IoU ≥ 0,5:

| Par           | Coincidencias       |
| ------------- | ------------------- |
| GPT – MRCD    | 0                   |
| GPT – reglas  | 0                   |
| MRCD – reglas | 0 (2 con IoU ≥ 0,3) |

**Cómo leerlo:** los sistemas no coinciden casi en nada. Eso **no** dice cuál acierta: ninguno es la verdad. Muestra por qué hace falta una referencia humana antes de afirmar exactitud, y que la delimitación temporal de los eventos es un problema abierto.

## Modelo publicado

|                         |                                                                         |
| ----------------------- | ----------------------------------------------------------------------- |
| Modo                    | Audio + texto                                                           |
| Parámetros              | 33 097                                                                  |
| Datos de entrenamiento  | 18 927 ventanas de 3 s                                                  |
| Origen de las etiquetas | **Heurístico** (supervisión débil), no anotación humana                 |
| Uso válido              | Demostrar el funcionamiento del sistema; no como evidencia de exactitud |

## Protocolo del benchmark (fijado antes de evaluar)

Cuando exista el conjunto de prueba adjudicado, todos los sistemas se evaluarán así:

- **Emparejamiento** uno a uno, misma clase, con IoU ≥ 0,50. Sensibilidad adicional a 0,30 y 0,70.
- **Métricas** por clase y agregadas: precisión, exhaustividad y F1. Las disfluencias y las pausas se reportan por separado.
- **Bordes**: error absoluto de inicio y fin en los pares emparejados, junto con la proporción emparejada. Métrica secundaria con tolerancia de 100 ms.
- **Falsas alarmas** de disfluencia por minuto evaluable y sobre usos legítimos.
- **Solo audio revisado**: lo que nadie revisó no cuenta como negativo.
- **Incertidumbre**: intervalos de confianza al 95 % por remuestreo de hablantes y diferencias pareadas entre sistemas.
- **Sin ajustes sobre la prueba**: umbrales y calibración se fijan en desarrollo. Cada evaluación sobre un conjunto queda registrada.

Los sistemas a comparar: reglas (B0), clasificadores simples de texto, acústica y su combinación (B1–B3), MRCD actual y sus variantes (M0, M1) y GPT audio (G).

## Esta tabla se completa al evaluar

| Sistema       | F1 disfluencias [IC 95 %] | F1 pausas | Falsas alarmas/min |
| ------------- | ------------------------- | --------- | ------------------ |
| Reglas (B0)   | pendiente                 | pendiente | pendiente          |
| MRCD (M0)     | pendiente                 | pendiente | pendiente          |
| GPT audio (G) | pendiente                 | pendiente | pendiente          |
