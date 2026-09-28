# Guía de anotación MRCD — `guia-anotacion-v1`

Estado: **borrador para aprobación del equipo**. Se congela después del piloto de guía. Cualquier cambio de criterio crea `guia-anotacion-v2`; cada tarea conserva la versión con la que se anotó y los casos afectados se vuelven a revisar.

## 1. Qué se anota

Fenómenos audibles del habla dentro de la **región central** de cada tarea (10–20 s). El contexto a cada lado sirve para escuchar y para completar eventos que cruzan el borde. Escucha primero a velocidad normal (1×) y con contexto.

No se evalúa a la persona. No se infiere ansiedad, nerviosismo, intención, tensión muscular ni diagnóstico. Se marca lo que se oye.

## 2. Taxonomía `mrcd-taxonomy-v1`

| Tecla | Clase                               | Qué marcar                                                                                                            | Ejemplo                                   | No marcar como esta clase                                  |
| ----- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------- |
| 1     | Muletilla (`filler_word`)           | Vocalización de relleno (eh, em, mmm) o expresión usada para ganar tiempo                                             | «fui al… **eh**… al centro»               | «**este** libro» (demostrativo): uso legítimo              |
| 2     | Prolongación (`prolongation`)       | Alargamiento perceptible de un sonido dentro de la palabra, sin función de énfasis ni de final de frase               | «la **sss**emana»                         | «¡muuuy bien!» enfático; vocal final de enunciado          |
| 3     | Repetición (`repetition`)           | Repetición de sonido, sílaba, palabra o frase corta asociada a vacilación                                             | «**pero pero** no sé»                     | «no, no, no» usado como énfasis deliberado                 |
| 4     | Bloqueo (`block`)                   | Interrupción audible dentro de una unidad de habla: arranque fallido, corte súbito                                    | «la c— (corte) casa»                      | Pausa entre enunciados completos                           |
| 5     | Revisión (`revision`)               | Abandono o reparación de una formulación. Marca todo el tramo: lo abandonado, la muletilla de edición y la reparación | «estamos llevando… **llevamos** el curso» | Aclaración planificada sin abandono                        |
| 6     | Pausa retórica (`rhetorical_pause`) | Pausa que organiza o enfatiza el discurso, típicamente tras una cláusula completa. **No es disfluencia**              | «Y eso fue todo. (pausa) Ahora…»          | Silencio dentro de un constituyente («de (pausa) la casa») |
| 7     | Pausa neutra (`neutral_pause`)      | Silencio de al menos 0,3 s sin función retórica clara ni interrupción disfluente. **No es disfluencia**               | Pausa para respirar entre ideas           | Cualquier caso que no sabes clasificar: usa Incierto       |

Casos combinados: si una revisión contiene una repetición, anota la **revisión** con su tramo completo y la **repetición** como evento aparte (pueden superponerse). Ambas cuentan en su propia clase.

`block` y `rhetorical_pause` dependen de interpretación. Si en el piloto el acuerdo entre anotadores es bajo, el equipo las declara exploratorias en la siguiente versión; nunca se cambian en silencio.

## 3. Decisiones distintas de «evento»

- **Uso legítimo** (botón): expresión que suena a muletilla pero cumple una función — demostrativo «este», «o sea» explicativo, «bueno» como respuesta. Registra la expresión, si es disfluente (sí / no / indeterminado) y su función. No es un evento de disfluencia.
- **Incierto** (`U`): hay un fenómeno pero no puedes decidir. Elige la causa: _por clase_ (existe, dudas de categoría), _por fronteras_ (existe, no se puede delimitar) o _audio no evaluable_.
- **No evaluable**: ruido, solapamiento de voces o audio cortado. Márcalo con su tramo; se excluye de las métricas.

Nunca uses «Pausa neutra» ni otra clase para rellenar lo que no sabes clasificar.

## 4. Fronteras

- Inicio: primer sonido del fenómeno. Fin: último sonido. En pausas: desde el final del sonido previo hasta el inicio del siguiente.
- Ajusta con ±10 ms o ±100 ms solo si la frontera es perceptible; no persigas precisión aparente.
- Confirma duración y prosodia a 1×; 0,75× y 1,25× sirven para ubicarte.
- Si un evento cruza el borde de la región, márcalo completo. Usa «Ampliar contexto» si hace falta.

## 5. Cobertura

Una zona sin marcas puede estar revisada sin eventos o pendiente. Por eso, cuando termines de escuchar la región y no queden fenómenos, pulsa **«Revisado: sin más eventos»**. «Revisado hasta el cursor» sirve para avanzar por partes. La tarea no se puede finalizar con tramos sin revisar.

## 6. Hablantes y audio deficiente

Anota al participante salvo que la tarea indique otra cosa. Si habla el entrevistador, usa el campo _Hablante_. Con solapamiento o ruido, usa _No evaluable_ o _Incierto — audio_.

## 7. Modos de trabajo

- **Ciego** (referencia y prueba): sin sugerencias ni anotaciones de otras personas. Primera escucha sin transcripción. El ASR se habilita solo tras escuchar la región completa, no trae marcas de disfluencia y su uso queda registrado. Las formas se verifican sobre el audio.
- **Asistido** (solo entrenamiento y desarrollo): aparecen candidatos del modelo sin puntuación. Escucha cada uno antes de pulsar _Correcto_; usa _Cambiar_, _Uso legítimo_, _Descartar_, _Falta otro evento_ o _Tengo dudas_. Al terminar, barre la región completa: los candidatos no muestran lo que todos los sistemas omitieron.
- **Adjudicación**: se ven las pistas A y B sin nombres. Decide con el audio, deja un motivo breve y revisa también algunos acuerdos.

## 8. Atajos

| Tecla            | Acción                                     |
| ---------------- | ------------------------------------------ |
| Espacio          | Reproducir / pausar                        |
| ← / →            | Retroceder / avanzar 1 s                   |
| I / O            | Marcar inicio / fin de la selección        |
| R                | Repetir la selección con 0,5 s de contexto |
| 1–7              | Clase (orden de la tabla)                  |
| U                | Incierto                                   |
| Supr             | Borrar el evento seleccionado              |
| Ctrl/Cmd + Enter | Guardar y finalizar                        |
| Ctrl/Cmd + Z     | Deshacer                                   |

Los atajos de letras no actúan mientras escribes en un campo.

## 9. Qué no hacer

- No copiar el ASR como «forma audible»: ese campo contiene solo lo verificado al escuchar.
- No consultar a otra persona durante la fase ciega.
- No redactar explicaciones largas: el motivo se pide solo en dudas, desacuerdos o cambios de criterio.

## 10. Ejemplos de audio (pendiente del equipo)

Tras el piloto de guía, el equipo agrega 2–3 clips autorizados por clase (ejemplo y contraejemplo), identificados por grabación e intervalo. Los clips del conjunto de prueba nunca se usan como ejemplos.

| Clase       | Clip ejemplo | Clip contraejemplo | Aprobado por |
| ----------- | ------------ | ------------------ | ------------ |
| (completar) |              |                    |              |
