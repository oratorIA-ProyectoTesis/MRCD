# Guía de anotación de disfluencias (versión 1, para calibración)

Usted escuchará fragmentos cortos (6 a 10 s) de habla espontánea en español. Su tarea es marcar **qué se oye**, con su inicio y su final. Trabaje solo con el audio: nadie le mostrará texto ni resultados de ningún sistema, y usted no debe buscarlos.

## 1. Qué marcar

| Código | Evento | Se marca cuando… |
|---|---|---|
| `FP` | pausa llena | hay un sonido sostenido sin contenido léxico: *eh, em, mm, hm*. |
| `LF` | muletilla léxica | una palabra existente (*este, o sea, bueno, pues, entonces*…) se usa como apoyo para pensar y no por su significado. |
| `REP` | repetición | una o más palabras contiguas se dicen dos o más veces. |
| `PRO` | prolongación | un sonido de una palabra se alarga de forma audible (*tooodo, peeero*). |
| `INAUD` | inaudible | no se entiende un tramo; márquelo y no adivine. |

Fuera de esta ronda (no los marque): falsos inicios, autocorrecciones, pausas silenciosas, errores gramaticales, acento o dialecto, palabras en otro idioma.

## 2. Estado de cada evento: tres opciones

Todo evento lleva un estado: `d` (disfluente), `f` (funcional) o `i` (incierto). El estado se decide **solo con lo que se oye**: nunca con lo que la persona quería o sentía.

Criterios observables (úselos en este orden):

1. **Prueba de eliminación:** quite mentalmente el evento (o la copia repetida). ¿La frase queda bien formada y con el mismo sentido?
2. **Marcación prosódica:** ¿el evento tiene un tono propio, un alargamiento o una pausa a cada lado que lo separan del resto de la frase?
3. **Pausa de búsqueda:** ¿hay una pausa audible justo antes o justo después del evento, antes de continuar con otra palabra?
4. **Repetición exacta:** ¿la copia es idéntica a la anterior (mismas palabras, mismo orden)?
5. **Posición:** ¿el evento está pegado a una pausa, o dentro de una frase que fluye sin pausas?

**Repetición**

- `d`: la eliminación deja la frase bien formada y con el mismo sentido; la copia es exacta; no hay tono propio y la copia queda pegada a una pausa o a un corte.
  - «Vamos a a revisar el el informe.»
  - «Yo creo que que sí se puede.»
  - «La casa de de mi tía era grande.»
- `f`: la eliminación cambia el sentido o la frase pierde un rasgo audible (cada copia lleva tono propio, ascendente o más fuerte; o la frase queda distinta, como en una enumeración con elementos distintos); no hay pausa de búsqueda entre copias.
  - «Fue muy, muy, muy lejos» (cada «muy» con tono más alto).
  - «Caminamos y caminamos todo el día» (si se quita una copia, cambia el sentido).
  - «No, no, no, eso no es así» (tres golpes de tono parejos y sin pausa de búsqueda).
- `i`: los criterios se contradicen o no se oyen con claridad.
  - «Entonces, entonces fuimos» (la eliminación deja la frase bien, pero hay tono propio en la segunda copia).
  - «Ya ya» con tono ambiguo y una pausa muy breve.
  - Repetición solapada con ruido de fondo.

**Pausa llena (`FP`):** `d` por defecto (sonido sostenido sin palabra, como *eh, em, mm*). `i` si no puede distinguirla de un «mm-hm» corto con tono ascendente o descendente que responde a otra persona; `f` no se usa para `FP`.

**Muletilla léxica (`LF`):** `d` si la eliminación deja la frase bien formada y con el mismo sentido, y la palabra va pegada a una pausa o sin tono propio (por ejemplo «Entonces este… pasamos a lo siguiente»). `f` si la eliminación cambia o rompe la frase (por ejemplo «Este resultado es mejor», «O sea, ciento diez» como equivalencia). `i` si ambos criterios se cumplen a medias.

**Prolongación (`PRO`):** `d` si el tramo alargado va seguido de una pausa de búsqueda antes de la siguiente palabra y no tiene movimiento de tono propio. `f` si el alargamiento lleva un movimiento claro de tono y la frase continúa sin pausa («¡qué bueeeno!» con tono descendente marcado). `i` si no puede decidir con estos dos criterios.

## 3. Dónde empieza y termina un evento

- **Inicio:** el instante en que empieza el sonido del evento (no la pausa anterior).
- **Final:** el instante en que termina el último sonido del evento (no el silencio posterior).
- **Repetición:** una sola marca desde el inicio de la **primera** copia hasta el final de la **última** copia, sin incluir lo que sigue.
- **Prolongación:** solo el tramo alargado de la palabra, no la palabra completa.
- Un evento cortado por el borde del fragmento se marca igual, con la nota `borde`.
- Los inicios y finales pueden variar unos 250 ms entre personas: no se angustie por milisegundos. Sí importa no marcar sonidos vecinos.

## 4. Qué NO hacer

- No infiera emociones, intención, nervios, seguridad ni estado mental de la persona. Marque el sonido, no la causa.
- No diagnostique tartamudez ni ningún trastorno. Esto no es una evaluación clínica.
- No juzgue la calidad del discurso, el acento, ni la gramática.
- No consulte con la otra persona anotadora hasta que se termine toda la ronda.

## 5. Cómo registrar duda y casos especiales

- Duda de clasificación: estado `i`, y una nota de pocas palabras sobre qué criterio no pudo aplicar (por ejemplo "pausa dudosa"), sin interpretar.
- No se entiende: evento `INAUD`, con inicio y final del tramo.
- Fragmento sin eventos: no marque nada; quedará registrado como "sin eventos".

## 6. Lista de calibración (20 fragmentos)

1. Escuché cada fragmento con audífonos y al menos dos veces antes de marcar.
2. Marqué solo `FP`, `LF`, `REP`, `PRO`, `INAUD`.
3. Todo evento tiene estado `d`, `f` o `i`.
4. No usé `d` o `f` por adivinar: en la duda usé `i`.
5. En repeticiones, una sola marca de la primera a la última copia.
6. En prolongaciones, marqué solo el tramo alargado.
7. No escribí emociones, causas ni diagnósticos en las notas.
8. No leí transcripciones ni usé reconocimiento automático.
9. Anoté sin hablar con la otra persona anotadora.
10. Anoté mis dudas de la guía para la reunión de calibración.
