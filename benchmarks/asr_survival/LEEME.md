# Benchmark de supervivencia ASR (disfluencias en español, marco DIHANA)

Un marco de evaluación pequeño y reutilizable: **¿cuánta de la evidencia de disfluencia anotada por humanos en un
turno sigue presente en la transcripción que produce un sistema ASR?** Versión en inglés: [README.md](README.md).

Estado: exploratorio, no es una medida calibrada ni validada de la calidad del ASR. La supervivencia es una
heurística sobre la transcripción, no un juicio humano (el kit de validación manual no se ha ejecutado). Un solo
corpus, habla telefónica, español peninsular.

## Qué se mide

Para cada turno del marco, la «supervivencia» la decide `mrcd.context_eval.survival.turn_survives` (el mismo código
de la sección 4.8 del artículo y de `results/metrics.json`):

| Métrica | Población | El turno sobrevive cuando |
|---|---|---|
| `survival_rep` | turnos con repetición literal (subtipo A), n = 214 | el tramo anotado se recupera en orden con >= 80 % de sus tokens **y** queda un duplicado adyacente dentro de la región alineada del ASR |
| `survival_rev` | turnos con revisión (sustitución/inserción/borrado), n = 161 | reparandum, corrección y tramo se recuperan cada uno en >= 80 % |
| `survival_filler` | turnos con pausa llena no léxica anotada, n = 308 | el texto del ASR contiene un token de relleno no léxico (`eh`, `mm`, `em`, `ah`...; `este` nunca cuenta) |

Cada una es una razón de sumas con **bootstrap agrupado por hablante** (122 hablantes, semilla 20261101, 5000
remuestreos, IC 95 % por percentiles). El intervalo se lee con niveles fijos: límite superior < 0,50 «mayoría
borrada»; < 0,75 «el borrado persiste»; límite inferior >= 0,75 «sustancialmente atenuado»; en otro caso
«no concluyente». La ausencia nunca se reporta como cero: una población vacía es `unavailable`.

## El marco

`frame/turn_ids.csv`: 1 025 turnos de usuario de DIHANA, pliegues 1-4 (214 con repetición literal, 161 con
revisión, 308 con relleno; 679 negativos muestreados, 122 hablantes). Columnas:

| Columna | Significado |
|---|---|
| `turn_id` | identificador de turno de DIHANA (`<diálogo>_<turno>`); también el nombre de tu archivo de salida ASR |
| `speaker_id` | código de hablante de DIHANA; el conglomerado del bootstrap |
| `fold` | pliegue disjunto por hablante (1-4) |
| `kind` | `pos` (elegido por repetición/revisión) o `neg` (negativo muestreado) |
| `sampling_weight` | peso inverso de muestreo del diseño del marco |
| `rep_literal` | 1/0, o vacío si el subtipo de repetición es indeterminado |
| `revision` | 1/0 |
| `filler_present` | 1 si el turno tiene una pausa llena no léxica anotada |

No hay texto de transcripciones ni audio. Las marcas de referencia se reconstruyen desde tu propia copia de DIHANA y
`survival_bench.py` se detiene si alguna etiqueta reconstruida no coincide con el CSV.

## Obtener DIHANA

DIHANA (diálogos orientados a tareas en español), registro de Zenodo 5572470, DOI [10.5281/zenodo.5572470](https://doi.org/10.5281/zenodo.5572470) (Benedí, Lleida, Varona et al.; verificado el 04/11/2026 con la API de Zenodo),
licencia **CC BY 4.0** (se exige atribución; cita a los autores del corpus como figuran en el registro). Está listado
en [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md). Extrae el corpus de modo que exista `data/*/*/*.xml` bajo
la raíz que pases en `--dihana-root`. Verifica la licencia y el DOI en la página del registro antes de redistribuir
cualquier cosa.

## Ejecutar

```bash
pip install -e .     # desde la raíz del repositorio, Python >= 3.11
cd benchmarks/asr_survival/scripts

# 1) Decodificar (sirve cualquier ASR; este es el decodificador de referencia). Audio: WAV mono 16 kHz llamado <turn_id>.wav, lo aportas tú.
python decode_faster_whisper.py --wav-dir WAVS --model large-v3 --variant P0 --out-dir out/lv3_P0 \
       --device cuda --compute-type float16        # variantes: P0, P1, C1, p_none

# 2) Puntuar
python survival_bench.py --dihana-root /ruta/a/dihana --asr-dir out/lv3_P0 --out out/lv3_P0.json
```

Para otro sistema ASR escribe un `<turn_id>.json` por turno del marco: `{"words": [{"text": "..."}, ...]}`
(`--kind words`) o `{"text": "..."}` con etiquetas entre corchetes al estilo CrisperWhisper (`--kind cw`; las
etiquetas se eliminan y las de pausa llena cuentan como relleno, PROTOCOL 4.3). Salida: `survival_rep`,
`survival_rev`, `survival_filler`, cada una con `k`, `n`, `value`, `ci`, `reading`.

Prompts (cadenas exactas `P0`, `P1`, `C1`; `p_none` = sin prompt): [prompts/](prompts/).

## Salidas esperadas y resultados de referencia

`results/metrics.json` contiene solo agregados (sin texto ni datos por turno) de los sistemas del protocolo
congelado: Whisper small (CPU int8) y large-v3 (GPU float16) con P0/P1/C1/sin prompt, más un modelo con etiquetas
verbatim. Por ejemplo, `small_P0` da survival_rep 105/214 = 0,491 (IC 0,426-0,550), survival_rev 53/161 = 0,329,
survival_filler 158/308 = 0,513. `survival_bench.py` reproduce esos tres valores a partir de las transcripciones
guardadas de small-P0 (verificado antes de publicar). Decodificar en otro hardware no es idéntico bit a bit. Los hashes `code_sha256` de `metrics.json` se calcularon sobre la copia de trabajo de los autores en Windows (CRLF) y se refieren a `scripts/asr_survival_models/*`, que no se publica; una copia con finales de línea LF no coincidirá byte a byte.

`results/PROTOCOL.md` es el protocolo congelado; su sha256 (`b83fb8fc...357c`) está en
`results/PROTOCOL.sha256.json` y dentro de `metrics.json`, y las pruebas lo verifican. Se publica byte a byte, por lo
que todavía nombra rutas locales del entorno original de los autores; son históricas y no hacen falta para ejecutar
nada aquí.

No incluido: el script completo de contrastes pareados/Holm del artículo (sus cifras están en `metrics.json`), las
transcripciones de cualquier sistema, y el decodificador y las salidas de CrisperWhisper (sus pesos y salidas tienen
una licencia de investigación no comercial; ejecútalo tú bajo esos términos).

## Cómo citar

Cita el software (ver `CITATION.cff` en la raíz, versión 0.3.0) **y** DIHANA. Cuando exista el archivo en Zenodo,
cita su DOI.

## Licencia

Código: MIT (`LICENSE` del repositorio). `frame/turn_ids.csv` deriva de anotaciones de DIHANA (CC BY 4.0), por lo que
usarlo exige atribuir a DIHANA. `results/` contiene agregados calculados a partir de salidas de modelos; no hay texto
de terceros.

## Archivar en Zenodo (mantenedores; tres pasos)

Los autores deben activar la integración ellos mismos; nada aquí lo hace.

1. Inicia sesión en <https://zenodo.org> con GitHub, abre *Account > GitHub* y activa el repositorio
   `oratorIA-ProyectoTesis/MRCD` (puede que el propietario de la organización deba aprobar el acceso de Zenodo).
2. Revisa `CITATION.cff` y `.zenodo.json` (título, autores, versión, licencia) y crea un **release** en GitHub con
   una etiqueta (por ejemplo `v0.3.0`); solo se archivan los releases.
3. Zenodo archiva el release y asigna un DOI. Agrega el DOI a `CITATION.cff` (`doi:`) y a la insignia del README, y
   usa el DOI de concepto («todas las versiones») en los artículos.
