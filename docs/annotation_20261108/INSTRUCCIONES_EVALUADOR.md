# Instrucciones para la persona evaluadora

Documento para entregar junto con los clips. Complementa a `GUIA_ANOTACION.md` (versión 1.1), que se entrega en el mismo paquete y manda en caso de duda. Por diseño, este documento no menciona ningún sistema, resultado ni hipótesis del estudio, ni la ubicación de los audios originales.

## 1. Qué vas a hacer

Vas a escuchar fragmentos cortos (de 6 a 10 segundos) de habla espontánea en español y marcar **qué se oye**: pausas llenas, muletillas, repeticiones, prolongaciones y la función de algunas palabras. Trabajas solo con el oído.

## 2. Qué recibes

Una carpeta comprimida con:

| Elemento            | Qué es                                                                  |
| ------------------- | ----------------------------------------------------------------------- |
| `clips\`            | Los fragmentos `.wav` con nombres aleatorios (por ejemplo `k7x2q9.wav`) |
| `etiquetas\`        | Carpeta vacía donde guardarás tus archivos `.txt`                       |
| `GUIA_ANOTACION.md` | La guía completa con definiciones y ejemplos                            |
| Este documento      | Los pasos                                                               |

Descomprímela en una carpeta de tu computadora, por ejemplo `D:\anotacion\`. Así tendrás `D:\anotacion\clips\` y `D:\anotacion\etiquetas\`.

## 3. Reglas que no se rompen

1. **Solo el oído.** No leas transcripciones, no uses dictado ni reconocimiento automático de voz, ni ningún programa o asistente que procese el audio.
2. **No cambies el nombre de ningún clip.**
3. **No hables de los clips con otra persona evaluadora** hasta que ambas hayan entregado.
4. **No escribas emociones, intenciones ni causas** (nervioso, inseguro, ansioso). Marcas el sonido, no la causa. Esto no es una evaluación clínica.
5. **No compartas ni subas los clips** a ningún sitio. Bórralos cuando entregues.

## 4. Antes de empezar (una sola vez)

1. Instala **Audacity** desde `https://www.audacityteam.org` (es gratuito).
2. Usa **audífonos**, no parlantes.
3. Lee completa `GUIA_ANOTACION.md` (unos 20 minutos). Lo más importante está en las secciones 1 a 4.
4. Recomendación: trabaja en sesiones de no más de 45 minutos, con descanso entre ellas.

## 5. Pasos por cada clip

1. En Audacity: `Archivo > Abrir` y elige un clip de `clips\`.
2. **Escúchalo completo al menos dos veces** antes de marcar nada.
3. Por cada evento que oigas:
   1. Arrastra el mouse sobre la onda para **seleccionar el tramo**, desde que empieza el sonido hasta que termina, sin el silencio de alrededor.
   2. Presiona `Espacio` para escuchar solo esa selección y ajústala si hace falta.
   3. Presiona `Ctrl + B` y escribe la etiqueta (sección 6).
   4. Presiona `Enter`.
   - Selecciona **siempre un tramo**. Una etiqueta sin tramo (un punto) no sirve.
   - Una misma palabra puede llevar dos etiquetas superpuestas (por ejemplo `LF d` y `MK este d`).
4. Exporta: `Archivo > Exportar > Exportar etiquetas...` (en versiones recientes: `Archivo > Exportar otros > Exportar etiquetas...`). Guarda en `etiquetas\` con **el mismo nombre del clip** y extensión `.txt`. Ejemplo: `k7x2q9.wav` → `k7x2q9.txt`.
5. **Clip sin ningún evento:** abre el Bloc de notas, guarda un archivo **vacío** con el nombre del clip (`k7x2q9.txt`) en `etiquetas\`. Un archivo que falta cuenta como error, no como "sin eventos".
6. Cierra el clip **sin guardar el proyecto** y pasa al siguiente.

## 6. Qué escribir en cada etiqueta

Formato: `CÓDIGO estado [nota opcional]`, separado por espacios. El estado es `d` (disfluente), `f` (funcional) o `i` (incierto).

| Lo que oyes                                                | Etiqueta                                                                               |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Sonido sostenido sin palabra: _eh, em, mm_                 | `FP d`, o `FP i` si suena como un "mm-hm" de respuesta. **Nunca** `FP f`.              |
| Palabra usada como relleno (_este, entonces, bueno..._)    | `LF d`, `LF f` o `LF i`                                                                |
| Una o más palabras dichas dos o más veces seguidas         | `REP d`, `REP f` o `REP i`. **Una sola marca** desde la primera copia hasta la última. |
| Sonido de una palabra alargado (_peeero, tooodo_)          | `PRO d`, `PRO f` o `PRO i`. Marca **solo** el tramo alargado.                          |
| Un tramo que no se entiende                                | `INAUD` (no adivines)                                                                  |
| Cada vez que oigas _este, o sea, bueno, pues, a ver, tipo_ | `MK este d`, `MK o_sea f`, `MK bueno i`, `MK pues d`, `MK a_ver f`, `MK tipo i`        |
| Evento cortado por el borde del clip                       | Agrega la nota `borde`, por ejemplo `REP d borde`                                      |
| Duda sobre un criterio                                     | Estado `i` y una nota breve, por ejemplo `PRO i pausa dudosa`                          |

- En `MK`, `o_sea` y `a_ver` van con guion bajo.
- `MK` solo para esas seis palabras. _eh, em, mm_ son siempre `FP`, nunca `MK`.
- No marques falsos inicios, autocorrecciones, pausas silenciosas, errores gramaticales, acento ni palabras en otro idioma.

## 7. Cómo decidir el estado

**Para `REP`, `LF` y `PRO`:** quita mentalmente el evento (o la copia repetida).

- Si la frase queda igual y el evento va pegado a una pausa o no tiene un tono propio: `d`.
- Si la frase cambia de sentido o el evento lleva un tono propio claro: `f`.
- Si los criterios se contradicen o no se oyen con claridad: `i`.

**Para `MK`**, aplica estos pasos en orden y detente en el primero que se cumpla:

1. **`f`** si la palabra tiene un papel en la frase: _este libro_ (demostrativo), _—¿Vamos? —Bueno._ (respuesta), _o sea, de diez a veinte_ (reformulación real), _no llegó, pues caminamos_ (consecuencia), _a ver si funciona_ (comprobar), _tipo manzana o pera_ (ejemplo).
2. **`d`** solo si se cumplen **las tres**: no tiene papel en la frase; se puede quitar sin cambiar el contenido; y va pegada a una pausa (silenciosa o _eh_), a un alargamiento o a un reinicio de la frase.
3. **`i`** en cualquier otro caso, incluso si no tiene papel pero se dice de corrido, sin pausa al lado.

Usar `i` no es un error. Forzar un `d` o un `f` dudoso sí empeora la medición.

Los inicios y finales pueden variar unos 250 ms: no te preocupes por milisegundos, pero no incluyas sonidos vecinos.

## 8. Antes de entregar (lista de control)

1. Hay **un `.txt` por cada `.wav`** en `etiquetas\`, con el mismo nombre (incluidos los vacíos).
2. Todas las etiquetas tienen código y estado (salvo `INAUD`).
3. Ninguna nota describe emociones, causas ni diagnósticos.
4. No usé transcripciones ni reconocimiento automático.
5. Anoté mis dudas sobre la guía en un archivo aparte, `dudas.txt` (opcional).

## 9. Entrega

Comprime la carpeta `etiquetas\` (y `dudas.txt`, si lo hiciste) y envíala a quien te entregó los clips. Después, borra los clips de tu computadora. No necesitas ejecutar ningún comando.
