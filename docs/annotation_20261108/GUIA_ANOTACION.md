# Guía de anotación de disfluencias (versión 1.1, para calibración)

Versión 1.1 (2026-11-08, antes de iniciar cualquier anotación): se agrega la sección 3, función de los marcadores del discurso (lista cerrada: este, o sea, bueno, pues, a ver, tipo; orden de decisión exhaustivo en 3.2). Las secciones y códigos anteriores no cambian.

Usted escuchará fragmentos cortos (6 a 10 s) de habla espontánea en español. Su tarea es marcar **qué se oye**, con su inicio y su final. Trabaje solo con el audio: nadie le mostrará texto ni resultados de ningún sistema, y usted no debe buscarlos.

## 1. Qué marcar

| Código | Evento | Se marca cuando… |
|---|---|---|
| `FP` | pausa llena | hay un sonido sostenido sin contenido léxico: *eh, em, mm, hm*. |
| `LF` | muletilla léxica | una palabra existente (*este, o sea, bueno, pues, entonces*…) se oye sin aportar su significado habitual a la frase (relleno). |
| `REP` | repetición | una o más palabras contiguas se dicen dos o más veces. |
| `PRO` | prolongación | un sonido de una palabra se alarga de forma audible (*tooodo, peeero*). |
| `INAUD` | inaudible | no se entiende un tramo; márquelo y no adivine. |
| `MK` | marcador del discurso | capa adicional (sección 3): una palabra de la lista cerrada, con su función `f`, `d` o `i`. |

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

## 3. Función de los marcadores del discurso (`MK`)

En **las mismas pasadas** marque, además, cada vez que oiga una de las palabras de la lista cerrada, qué función cumple en esa frase. Es una capa aparte: no reemplaza a `LF`, `FP` ni a ningún código anterior, y puede coincidir en el tiempo con ellos.

### 3.1 Lista cerrada

| Código en la etiqueta | Se oye como |
|---|---|
| `este` | *este* |
| `o_sea` | *o sea* |
| `bueno` | *bueno* |
| `pues` | *pues* |
| `a_ver` | *a ver* |
| `tipo` | *tipo* |

- No marque ninguna otra palabra, aunque le parezca una muletilla (*entonces*, *digamos*, etc.): solo se marca lo de la lista.
- *eh*, *em* y *mm* **no** están en la lista: siguen siendo pausa llena (`FP`) y **no** llevan `MK`.
- Un marcador se marca cada vez que aparece, incluso si ya marcó esa misma palabra como `LF`.

### 3.2 Los tres valores (solo criterios que se oyen)

Etiqueta: `MK <palabra> <estado>`, con `f`, `d` o `i`. Se marca el tramo de la palabra (inicio y final como en la sección 4). Escriba `functional`, `disfluent` o `uncertain` en el CSV; en Audacity use `f`, `d`, `i`.

- **`functional` (`f`)**: la palabra tiene un papel dentro de la estructura de la frase. Indicios: *este* es demostrativo y va antes de un sustantivo; *bueno* es una respuesta («¿Vamos? —Bueno.») o un adjetivo evaluativo; *o sea* va seguido de una reformulación real de lo que se dijo antes; *pues* conecta una consecuencia; *a ver* introduce una acción de mirar o comprobar; *tipo* introduce un ejemplo o una categoría.
- **`disfluent` (`d`)**: la palabra se usa como relleno. Se exige **todo** lo siguiente: no cumple ningún papel sintáctico; quitarla no cambia el contenido de la frase (prueba de eliminación, sección 2); y está pegada a una pausa silenciosa o llena, a una prolongación, o a un reinicio o corrección de la frase (basta uno de estos tres).
- **`uncertain` (`i`)**: no se cumple `functional` ni se cumple todo lo de `disfluent` (incluido el caso sin papel sintáctico pero sin pausa, prolongación ni reinicio al lado), los indicios se contradicen, o el audio no permite decidir. **Úselo sin culpa:** es preferible `i` a forzar una elección; un `d` o `f` dudoso empeora la medición, un `i` honesto no.

**Orden de decisión (exhaustivo; aplíquelo en este orden y deténgase en el primer paso que se cumpla):**

1. **`functional` (`f`)** si la palabra tiene un papel sintáctico que se observa en la frase (los indicios de `functional` de arriba).
2. Si no, **`disfluent` (`d`)** solo si se cumplen **todos** estos: (a) no tiene papel sintáctico; (b) se puede quitar sin cambiar el contenido de la frase; (c) hay **al menos un** indicio de adyacencia: pausa silenciosa o llena pegada, prolongación, o reinicio o corrección de la frase.
3. En cualquier otro caso, **`uncertain` (`i`)**. Esto incluye: sin papel sintáctico pero **sin** pausa, prolongación ni reinicio al lado (por ejemplo, un *o sea* o un *tipo* dicho de corrido a mitad de frase, o un *pues* al inicio de la oración como en «Pues, no sé»), y siempre que el audio no permita decidir.

Nunca decida por lo que la persona quería decir o sentía.

**Esta capa no es la misma que `LF`.** La capa `MK` es aparte y más estricta que `LF`. Para `LF` basta con que la palabra se oiga sin aportar su significado habitual («sin tono propio»); eso por sí solo puede justificar `LF` = `d`. Para `MK`, «sin tono propio» por sí solo da `i` mientras no haya un indicio de adyacencia (paso 2c). Es esperable, por diseño, que las dos capas no coincidan en una misma palabra; ambas se analizan por separado y el desacuerdo entre capas no se corrige.

### 3.3 Ejemplos inventados

1. «Compré este libro ayer.» → `este` `f` (demostrativo antes de un sustantivo).
2. «Entonces… este… pasamos al segundo punto.» → `este` `d` (sin papel en la frase, entre pausas).
3. «—¿Empezamos? —Bueno.» → `bueno` `f` (es la respuesta).
4. «Bueno, bueno, eh, ya vamos.» (con pausa antes y después) → `bueno` `d` (relleno junto a pausa llena).
5. «Subió el doble; o sea, de diez a veinte.» → `o_sea` `f` (le sigue una reformulación real).
6. «Es… o sea… es que no sé.» (pausas a ambos lados, nada se reformula) → `o_sea` `d`.
7. «No llegó el tren, pues tuvimos que caminar.» → `pues` `f` (conecta una consecuencia).
8. «Es una fruta, tipo manzana o pera.» → `tipo` `f` (introduce ejemplos).
9. «Y entonces tipo… no sé.» (hay una pausa breve, pero «tipo» podría introducir algo) → `tipo` `i` (indicios en conflicto).
10. «A ver… a ver si me acuerdo» con ruido encima de la primera palabra → `a_ver` `i` (audio poco claro).
11. «Es que o sea no lo sabía» dicho de corrido, sin pausa, alargamiento ni reinicio al lado (no hay nada que reformular) → `o_sea` `i` (sin papel sintáctico, pero sin indicio de adyacencia; aquí `LF` podría ser `d`, y está bien).
12. «Pues, no sé.» al inicio de la oración, sin pausa ni alargamiento a su alrededor → `pues` `i` (no conecta una consecuencia, pero tampoco hay indicio de adyacencia).

## 4. Dónde empieza y termina un evento

- **Inicio:** el instante en que empieza el sonido del evento (no la pausa anterior).
- **Final:** el instante en que termina el último sonido del evento (no el silencio posterior).
- **Repetición:** una sola marca desde el inicio de la **primera** copia hasta el final de la **última** copia, sin incluir lo que sigue.
- **Prolongación:** solo el tramo alargado de la palabra, no la palabra completa.
- Un evento cortado por el borde del fragmento se marca igual, con la nota `borde`.
- Los inicios y finales pueden variar unos 250 ms entre personas: no se angustie por milisegundos. Sí importa no marcar sonidos vecinos.

## 5. Qué NO hacer

- No infiera emociones, intención, nervios, seguridad ni estado mental de la persona. Marque el sonido, no la causa.
- No diagnostique tartamudez ni ningún trastorno. Esto no es una evaluación clínica.
- No juzgue la calidad del discurso, el acento, ni la gramática.
- No consulte con la otra persona anotadora hasta que se termine toda la ronda.

## 6. Cómo registrar duda y casos especiales

- Duda de clasificación: estado `i`, y una nota de pocas palabras sobre qué criterio no pudo aplicar (por ejemplo "pausa dudosa"), sin interpretar.
- No se entiende: evento `INAUD`, con inicio y final del tramo.
- Fragmento sin eventos: no marque nada; quedará registrado como "sin eventos".

## 7. Lista de calibración (20 fragmentos)

1. Escuché cada fragmento con audífonos y al menos dos veces antes de marcar.
2. Marqué solo `FP`, `LF`, `REP`, `PRO`, `INAUD` y, para la lista cerrada de la sección 3, `MK`.
3. Todo evento y todo `MK` tiene estado `d`, `f` o `i`.
4. No usé `d` o `f` por adivinar: en la duda usé `i`.
5. En repeticiones, una sola marca de la primera a la última copia.
6. En prolongaciones, marqué solo el tramo alargado.
7. No escribí emociones, causas ni diagnósticos en las notas.
8. No leí transcripciones ni usé reconocimiento automático.
9. Anoté sin hablar con la otra persona anotadora.
10. Anoté mis dudas de la guía para la reunión de calibración.
11. Marqué `MK` solo para las palabras de la lista cerrada, y no marqué *eh*, *em* ni *mm* como `MK` (son `FP`).
12. Ante indicios en conflicto, sin indicio de adyacencia o audio poco claro, usé `i` en los `MK`.
