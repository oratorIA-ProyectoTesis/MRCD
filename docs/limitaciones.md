# Limitaciones

Lo que MRCD **no** hace todavía, o hace con reservas. Esta lista se actualiza con cada versión.

## Del modelo

- **Exactitud no medida.** El modelo publicado se entrenó con etiquetas heurísticas (supervisión débil). Sirve para demostrar el sistema; sus detecciones no son evidencia de exactitud.
- **Contratos temporales distintos.** Las nuevas salidas legacy son clasificación con soporte de ventanas de 3 s, no localización precisa de eventos. V2 proyecta tipos/contextos a celdas de 0,5 s, con cola parcial real; no recupera timestamps humanos arbitrarios. HU059 (refinamiento experimental con palabras/silencios) sigue pendiente, sin mejora de exactitud afirmada. Véase [contrato del producto v2](cell-product-v2.md), aún pendiente de revisión independiente.
- **Clases difíciles.** `block` y `rhetorical_pause` dependen de interpretación. Si el acuerdo entre anotadores resulta bajo, se tratarán como exploratorias.
- **Sin video en el producto.** El motor tiene una rama visual en investigación, pero la demo analiza solo audio y texto.
- **Dependencia del ASR.** Whisper puede omitir muletillas o repeticiones; lo que el ASR no transcribe solo se detecta por vía acústica.
- **Configuración del ASR.** Usar GPU o el modo `int8` acelera el ASR, pero cambia la transcripción y, con ella, los eventos. Los scripts de investigación (`extract_and_label.py`, `ingest_single_video.py`) siguen usando `int8` por defecto; el modelo publicado se entrenó con rasgos extraídos sin registrar esa configuración.
- **Corpus.** Las grabaciones de entrenamiento provienen de charlas públicas; hablantes fluidos y ensayados, con pocos bloqueos y prolongaciones.

## De la plataforma

- **Un análisis a la vez** por worker. Varios usuarios simultáneos esperan en cola.
- **SQLite por defecto.** Para la anotación con varias personas a la vez conviene PostgreSQL.
- **Sin transmisión en tiempo real.** Se analiza después de grabar; el análisis continuo está planificado para una etapa posterior.
- **Duración según contrato.** La ruta legacy conserva su mínimo histórico de 13 s. El motor v2 no impone ese mínimo: admite clips cortos/cola parcial y D=0 como ledger vacío; esto no prueba calidad ni operación real con esos medios.
- **Intercambio/evaluación nuevos.** CSV/EAF, revisión asistida y evaluación de referencia para salidas nuevas de celdas/ventanas se rechazan hasta disponer de consumidores por tarea (U5). JSON íntegro y Markdown seguro están disponibles; no se inventa contexto disfluente ni se exporta automáticamente como anotación humana.

## De interpretación

- Una detección **no es un diagnóstico** ni una evaluación de la competencia oral de una persona.
- Las puntuaciones del modelo **no están calibradas**.
- El acuerdo entre sistemas automáticos no es exactitud.
