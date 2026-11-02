# Preregistro técnico de las grabaciones propias (Vía B, opcional)

**Versión 1.0, fecha de redacción 2026-11-01.** Congelado con sha256 en `PREREGISTRO_TECNICO.freeze.json` antes de grabar. Pendiente de ratificación del asesor: el congelamiento fija el texto y el código, no sustituye la aprobación. Autorrevisión; sin revisión independiente.

## 0. Declaración principal

**Estas grabaciones no tienen etiquetas humanas y no las tendrán. Por lo tanto, ningún resultado de este estudio es una cifra de exactitud: no se calcula ni se afirma precisión, recall, F1, kappa, IoU ni cobertura/riesgo de clase alguna.** Las salidas del motor no se usan como referencia ni para preanotar nada; no hay personas que anoten. Cualquier texto de la tesis que cite este estudio debe decir que mide propiedades técnicas del flujo audio-video y no la calidad de la detección.

El canal visual nunca infiere estados psicológicos: la compuerta S6 verifica geometría (rostro presente, frontal, único, coherente) y sincronía labios-audio.

## 1. Preguntas

| Id  | Pregunta técnica                                                     | Qué se mide                                                                                         |
| --- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| T1  | ¿La compuerta S6 congelada admite el video de un montaje controlado? | Fracción de segmentos de 3 s admitidos                                                              |
| T2  | ¿La sincronía audio-video es verificable y aceptable?                | Videos cuyo desfase es estimable y está dentro de los límites de la compuerta; desfase por claqueta |
| T3  | ¿El rostro está visible?                                             | Mediana de la fracción de cuadros con rostro presente                                               |
| T4  | ¿La cadena de audio con micrófono real es sana?                      | Grabaciones que pasan nivel, recorte y piso de ruido                                                |
| T5  | ¿El motor termina sin fallar sobre hablantes no vistos?              | Fracción de grabaciones con estado `ok`; factor de tiempo real                                      |

## 2. Diseño

- Hablantes nuevos (no usados para desarrollo, ajuste ni compuerta), mayores de edad, hispanohablantes, sin diagnóstico declarado de tartamudez (autodeclaración de elegibilidad, sin registrar el motivo). Mínimo 10 hablantes y 30 min de habla neta; objetivo 45 min. Protocolo en `PROTOCOLO_GRABACION.md`.
- La unidad estadística es el **hablante**: todo intervalo de confianza remuestrea hablantes con reemplazo (percentil, B = 2000, semilla 20261102, nivel 95 %).
- Segmentos de evaluación: ventanas consecutivas de 3,0 s sin solape; el resto final se conserva solo si dura ≥ 1,5 s (`mrcd/recording_check/stats.py`).

## 3. Compuerta y configuración congeladas

- Compuerta de video S6: `mrcd/av/config.py` y `mrcd/av/gate.py`, versión `s6-gate-1.1-A1`, hash de configuración `2357f15a17599d9293c0d3ed6047926b7b4748861dafb558e99bcd3d53e2a213` (cubre todos los umbrales de la compuerta). **No se modifica ningún umbral de la compuerta con estos datos.**
- Constantes de este estudio: `mrcd/recording_check/config.py`, versión `viab-technical-1.0`.
- Versión del motor y de los perfiles ASR: se registra el commit en la hoja de sesión antes de la primera sesión real. Si el motor cambia después de grabar, se declara y el análisis T5 se repite con las dos versiones, sin sustituir la primera.

## 4. Criterios y reglas de decisión

Cada criterio se resuelve con tres respuestas posibles. Con una proporción `p` y su IC de hablantes `[lo, hi]` frente al umbral `u`:

- **cumple**: `lo ≥ u`;
- **cumple solo en la estimación puntual**: `p ≥ u` pero `lo < u`;
- **no cumple**: `p < u`.

| Criterio                                                  | Umbral `u`                         | Fuente del umbral                                   |
| --------------------------------------------------------- | ---------------------------------- | --------------------------------------------------- |
| T1 fracción de segmentos admitidos                        | 0,80                               | Aceptación de la acción 4.1 del plan (01/11/2026)   |
| T2 videos con sincronía verificable y dentro de límites ( | desfase                            | ≤ 120 ms, ancho de IC ≤ 200 ms, según la compuerta) | 0,80 | Fijado a priori, igual que T1 |
| T3 mediana de rostro presente                             | 0,90 (comparación directa, sin IC) | Fijado a priori                                     |
| T4 grabaciones que pasan todos los chequeos de audio      | 0,90                               | Fijado a priori                                     |
| T5 grabaciones con análisis `ok`                          | 0,95                               | Fijado a priori                                     |

Los umbrales son elecciones de ingeniería declaradas antes de medir; no provienen de la literatura y no se ajustan con estos datos. La desviación de la claqueta (|desfase| ≤ 120 ms y deriva entre la primera y la segunda palmada ≤ 60 ms) se reporta junto con T2 y no lo modifica.

**Sin multiplicidad que corregir:** los cinco criterios son descriptivos y se reportan los cinco; no hay pruebas de hipótesis de diferencias ni declaración de "significancia". El IC se interpreta como incertidumbre, no como prueba.

## 5. Datos faltantes y exclusiones

- Una medición no disponible (sin pista de audio, sin estimación de desfase, compuerta no ejecutada) se reporta como **no disponible con el motivo**, nunca como 0 ni como "cumple".
- Una grabación fallida por causa técnica corregible (sin audio, sin rostro por mala colocación, claqueta ausente) se **repite una vez** (la repetición lleva el número de toma siguiente, p. ej. `PRES2`); el registro de sesión conserva ambas y el informe incluye todas las tomas verificadas. No se descartan tomas por su resultado.
- Se excluye del informe: el piloto (`VB_S00`), grabaciones de hablantes que retiraron su consentimiento, y tomas que incumplen el protocolo de participante (segunda persona visible durante toda la toma). Cada exclusión se lista con su motivo.
- El motor `failed` o `not_run` cuenta como fallo en T5; los eventos de esa grabación no se inventan ni se rellenan.

## 6. Resúmenes de corrida del motor (T5)

Un JSON por grabación, producido al ejecutar el motor fuera de este kit:

```json
{
  "recording_id": "VB_S01_PRES1",
  "status": "ok",
  "wall_s": 41.2,
  "audio_s": 183.0,
  "warnings": ["texto corto"],
  "events_total": 17
}
```

`events_total` es solo una cifra de sensatez (un valor cero o desmesurado indica una corrida rota); **no** es detección ni exactitud, y los eventos individuales no se guardan en el informe.

## 7. Análisis de sensibilidad

1. T1 quitando un hablante cada vez (rango mínimo y máximo de la fracción admitida): influencia de un solo hablante.
2. T1 desglosado por motivo de exclusión de la compuerta (rostro no visible, sincronía no verificada, correlación labios-audio baja), solo descriptivo.
3. Distribución del desfase por claqueta junto a la del desfase estimado por la compuerta (medianas y percentil 95).

No se cambia ningún umbral en estos análisis; solo se muestra cuánto cambia el resultado.

## 8. Prohibiciones

- No ajustar la compuerta, el motor, los perfiles ASR ni este protocolo con estas grabaciones. Una mejora derivada de lo observado exige una nueva versión del software, un nuevo preregistro fechado y grabaciones nuevas para ser medida.
- No usar estas grabaciones como corpus de entrenamiento, validación ni referencia de ninguna clase.
- No usar etiquetas automáticas (reglas, modelos, ASR) como referencia ni mostrarlas a nadie como "correctas".
- No publicar video, audio, capturas ni resultados individuales.
- No presentar T1 como evidencia de que el video mejora la detección: la ablación A+T frente a A+T+V con exactitud queda **sin hacer** por falta de etiquetas.

## 9. Enmiendas

Cualquier cambio a este texto o al código listado en el archivo de congelamiento exige una enmienda con fecha, el motivo y el hash anterior, y un archivo `.freeze.json` nuevo; el anterior se conserva. `tests/test_via_b_recording_check.py` falla si los archivos congelados cambian sin enmienda.

## 10. Verificación del congelamiento

```powershell
cd C:\zArnes-tesis\mrcd
.\.venv\Scripts\python.exe -m mrcd.recording_check verify-freeze docs\via-b\PREREGISTRO_TECNICO.freeze.json
```

Debe imprimir `INTACTO`. Para fijar un sello externo, envíe el sha256 del archivo `.freeze.json` por correo al asesor antes de la primera sesión real.
