> **NOTA 2026-10-31 (revisión 6): documento HISTÓRICO, NO congelado y NO vigente.** Conserva campos `[COMPLETAR]` (fecha, revisión de git, esquema) y trata una pregunta anterior (fuga intra-charla en detección audiovisual), no el motor actual. Debe actualizarse y ratificarse con un commit real antes de citarse como protocolo. El contenido original se conserva sin cambios debajo.

# Pre-registro — Fuga intra-charla en detección audiovisual de disfluencias

> **Este documento se congela con un commit ANTES de ejecutar cualquier experimento.**
> Toda hipótesis no listada aquí como confirmatoria es exploratoria, y se reporta
> como tal aunque el resultado sea favorable.

- Fecha de congelamiento: `[COMPLETAR antes del commit]`
- Revisión de git al congelar: `[COMPLETAR]`
- Datos: `data/windows_auto.npz` (train, etiquetas heurísticas) y
  `data/windows_gold.npz` (test, juez VLM ciego), esquema `[COMPLETAR: FEATURE_SCHEMA_VERSION]`

---

## 1. Antecedente y objeto

El corpus tiene 18 charlas de 18 oradores distintos: **una charla por orador**. Los 87
"segmentos" (`<video_id>_sNN`) son recortes de la misma charla, así que identidad del
hablante, cámara, iluminación, día y tema están perfectamente confundidos. **Ningún
diseño sobre estos datos separa identidad de sesión.**

Por tanto el objeto de estudio es la **fuga intra-charla**: cuánto infla la ganancia
aparente del canal visual el hecho de que material de la misma charla aparezca en
entrenamiento y en prueba.

Medición previa que motiva el estudio (descomposición de varianza por hablante, ICC):

| rama                            | ICC medio |
| ------------------------------- | --------- |
| cinésica (12-D, sin normalizar) | 0.501     |
| prosódica (9-D, F0 ya centrado) | 0.089     |

Canal más extremo: `head_yaw` ICC = 0.760, atribuible al ángulo de cámara de cada charla.

---

## 2. Hipótesis confirmatorias

### H1 — La fuga intra-charla infla la ganancia visual

**Estimando:** Δ = macro-F1(`trimodal`) − macro-F1(`audio_text`).

**Predicción:** Δ(P2) − Δ(P3) > 0 con rasgos sin normalizar.

**Criterio de decisión:** el IC 95 % por bootstrap de clúster (charla) de la
diferencia en diferencias no contiene 0.

### H2 — El canal visual porta información por sí solo

**Predicción:** macro-F1 de `kinesic_only` supera el percentil 95 de la distribución
nula por permutación (etiquetas barajadas dentro de cada charla, 20 repeticiones,
reentrenamiento completo).

**Criterio de decisión:** macro-F1 observado > percentil 95 del nulo.

**Corrección por comparaciones múltiples:** Holm sobre {H1, H2}.

---

## 3. Resultados descriptivos pre-especificados (sin prueba de hipótesis)

No llevan corrección ni criterio de rechazo. Se reportan siempre, salga lo que salga.

- **R1 — Sonda de identidad por rama.** Sonda lineal que predice cuál de las charlas
  es la ventana, entrenada por separado con: solo cinésica, solo prosodia, solo
  lingüística, solo acústica. Métrica: exactitud balanceada. Azar = 1/K.
- **R2 — Ganancia por fuga, modo a modo.** F1(P2) − F1(P3) para cada uno de los
  cuatro modos por separado, no solo su diferencia.
- **R3 — Efecto del remedio.** R2 recalculado con z-score por hablante: cuánto se
  encoge F1(P2) − F1(P3) en cada modo al normalizar.
- **R4 — Identidad residual tras normalizar.** Sonda con rasgos **invariantes a
  transformación afín por canal** (autocorrelación de retardo 1 y correlaciones
  cruzadas entre canales), que sobreviven al z-score. Mide si la dinámica facial
  sigue identificando la charla después del remedio.

> ◆ **Corrección registrada antes de congelar.** Una versión previa definía R3 como
> "R1 recalculado con z-score por hablante". Eso es **tautológico**: el z-score por
> hablante deja la varianza entre hablantes en exactamente 0 para la media y la
> desviación, que son justamente los rasgos que agrega la sonda de R1. Medido:
> las cuatro ramas caen a 1/K exacto, que no es evidencia de remedio sino
> degeneración por construcción. El efecto del remedio se mide sobre la tarea de
> disfluencias (R3), y la identidad residual con rasgos invariantes (R4).

---

## 4. Diseño

### Particiones

| id  | agrupamiento                      | mismo hablante en train y test | ventanas solapadas entre train y test | fuga que conserva                   |
| --- | --------------------------------- | ------------------------------ | ------------------------------------- | ----------------------------------- |
| P1  | aleatoria por ventana             | sí                             | **sí**                                | intra-charla + duplicación temporal |
| P2  | `GroupKFold` por segmento (`rec`) | sí                             | no                                    | **intra-charla**                    |
| P3  | `GroupKFold` por charla (`group`) | no                             | no                                    | ninguna                             |

P1 es cota superior ("lo peor que se puede hacer"). El contraste que sostiene H1 es
**P2 vs P3**.

### Factores

3 particiones × 2 normalizaciones (`raw`, `speaker`) × 4 modos
(`audio_only`, `audio_text`, `trimodal`, `kinesic_only`) × 5 semillas × 5 folds.

### Configuración final única

Fijada aquí y no se toca:

```
d = 64
heads = 4
epochs = 15
lr = 2e-3
batch_size = 64
weight_decay = 1e-3
modality_dropout = 0.0        # simétrico: ningún modo recibe regularización que otro no reciba
pooling = concat(mean, max)
pros_dim = 0 para kinesic_only  # un modo "solo video" no puede recibir F0
semillas = {13, 17, 23, 29, 31}
bootstrap B = 1000, clúster = charla
```

### Normalización `speaker`

z-score por hablante y por canal (nivel **y** escala) sobre el vector cinésico 12-D,
con estadísticas calculadas **solo sobre los segmentos de entrenamiento** del fold.
Simétrico al centrado de F0 que ya existe en `core/dataset.py:186`.

Trabajo relacionado, no implementado aquí: Iterative Feature Normalization
(Busso, ICASSP 2011) detecta habla neutra de forma no supervisada e itera.

---

## 5. Qué dirá el paper si H1 sale nula

**Escrito antes de ver los datos. No se reescribe después.**

- **Caso A — DiD ≈ 0 pero R2 grande en todos los modos.** El paper afirma: _la fuga
  intra-charla infla todas las modalidades de forma comparable; el beneficio visual
  aparente en la literatura es un artefacto general de protocolo, no específico del
  video._ R1 cuantifica cuánto filtra cada rama y sostiene la lectura de forma
  independiente de la DiD.
- **Caso B — DiD ≈ 0 y R2 ≈ 0 en todos los modos.** El protocolo original ya era
  limpio. El paper reporta _"no se detectó fuga intra-charla con 18 charlas"_, con el
  poder estadístico declarado como límite principal.
- **Caso C — H2 nula (`kinesic_only` ≈ nulo por permutación).** El canal visual no
  porta información de disfluencia detectable en habla ensayada en español. Es un
  resultado negativo con mecanismo descartado (no es competencia entre modalidades:
  la rama sola tampoco funciona).

En los tres casos R1 se reporta y el paper conserva su contribución.

---

## 6. Hipótesis exploratorias

Se reportan en sección aparte, marcadas como exploratorias, sin corrección y sin
valor confirmatorio.

- **E3 — Función de la pausa.** `trimodal` mejora la discriminación binaria
  `rhetorical_pause` vs `neutral_pause` frente a `audio_text`.
  **Declarado exploratorio de antemano:** las 20 pausas retóricas del gold provienen
  de 9 charlas; con 9 clústeres el bootstrap subestima la incertidumbre. No se
  calculará efecto mínimo detectable porque el poder es insuficiente por inspección.
- Δ y ΔΔ cinésicas (12 → 36 dimensiones).
- `VIDEO_FPS` 10 → 25.
- Pooling linear-softmax (Wang, Li y Metze, ICASSP 2019).
- Colapso de `block` a `neutral_pause` en entrenamiento.

---

## 7. Validación humana del juez

Bloquea la **publicación**, no la ejecución: la DiD es una comparación relativa y el
ruido de etiqueta diluye diferencias, no las fabrica. E1 y la anotación corren en
paralelo.

- **Anotadores: dos externos. El autor NO anota.** κ(E1, E2) es el techo de acuerdo
  humano, sin contaminación por conocimiento de las reglas heurísticas.
- Guía de anotación de 1–2 páginas, derivada de las definiciones del prompt del juez
  para que humanos y juez compartan criterio. Congelada con commit tras la calibración.
- Ronda de calibración sobre 20 clips fuera de la muestra; luego la guía no se toca.
- Muestra ≈ 150 ítems, estratificada por la etiqueta del juez: **todos** los ítems de
  las clases raras (`block`, `prolongation`, `revision`, `rhetorical_pause`) y ≈ 25 de
  cada clase frecuente. Se guarda la probabilidad de inclusión de cada ítem para
  ponderar por su inverso.
- Clips cortados con 1 s de contexto antes y después, nombre aleatorio, orden
  aleatorio. El anotador no ve la etiqueta del juez ni la de la heurística.
- Anotación independiente y ciega entre anotadores.
- Métricas: κ(E1,E2) como techo; κ(juez, humano) global y **por clase**, con IC por
  bootstrap de charla.
- Adjudicación de desacuerdos **después** de calcular los κ. Se reporta el κ
  pre-discusión.
- Clases donde κ(juez, humano) quede muy por debajo del techo humano se **fusionan**,
  y la fusión se documenta como resultado sobre la taxonomía.

---

## 8. Límites declarados

Se escriben en el paper, no se esperan a la revisión.

| no se puede afirmar                                  | razón                               |
| ---------------------------------------------------- | ----------------------------------- |
| Nada sobre detección de bloqueos                     | n = 2 en el gold                    |
| Separar identidad del hablante de sesión/cámara/tema | una sola charla por orador          |
| Superar el estado del arte en macro-F1               | 18 charlas, oradores TEDx ensayados |
| Generalización a habla patológica                    | corpus de habla ensayada fluida     |

**Requisito de diseño derivado, para el corpus propio:** cada participante grabado en
**≥ 2 sesiones**, en días y encuadres distintos. Es la única forma de separar identidad
de sesión, y este pre-registro documenta por qué.
