# Cómo funciona

## Del audio a los eventos

```mermaid
flowchart LR
    A["Audio<br/>16 kHz mono"] --> C["ASR Whisper<br/>palabras y tiempos"]
    A --> D["Pista acústica<br/>VAD · F0 · energía"]
    C --> E["Ventanas 3 s<br/>+ 10 s de texto"]
    D --> E
    E --> F["Red de fusión"]
    F --> G["Eventos en ms<br/>o «incierto»"]
```

1. **Preparación única.** La transcripción y la pista acústica se calculan **una sola vez** por grabación y se guardan en caché por hash del audio y de la configuración. Todas las ventanas y el sistema de reglas reutilizan esa preparación.
2. **Ventanas.** Cada 0,5 s se arma una ventana de 3 s con 300 cuadros acústicos, 9 rasgos prosódicos de la pausa más larga (por ejemplo el reinicio tonal) y hasta 40 palabras de una ventana de texto de 10 s.
3. **Fusión.** La red usa las consultas acústicas para atender a los tokens de texto con su posición temporal. El modelo actual tiene 33 097 parámetros.
4. **Decodificación.** Las ventanas contiguas de la misma clase se unen en un evento. Si la probabilidad no supera el umbral de su clase, el tramo queda como «incierto». Opcionalmente, los bordes se ajustan a las palabras del ASR o al silencio detectado.

## La plataforma

```mermaid
flowchart LR
    W["Cliente web<br/>revisión · prueba · experimentos"] --> API["API FastAPI"]
    API --> DB[("Base de datos<br/>documentos versionados")]
    Q["Worker<br/>un análisis a la vez"] --> DB
    Q --> M["Sistemas<br/>MRCD · reglas · GPT"]
```

- La API nunca espera a la inferencia: registra el trabajo y responde. El **worker** lo toma de la base de datos, informa el progreso por etapas y escribe el resultado completo en una sola transacción.
- Si el worker se cae, el trabajo vuelve a la cola con un número limitado de reintentos. Repetir un análisis no duplica eventos.
- Cada sistema (MRCD, reglas, GPT) tiene su propio estado. Si uno falla, los demás no se invalidan.

## El ciclo de revisión humana

```mermaid
flowchart LR
    T["Tarea<br/>región de 10–20 s"] --> A1["Persona A<br/>anota a ciegas"]
    T --> A2["Persona B<br/>anota a ciegas"]
    A1 --> J["Adjudicación<br/>A/B anónimos"]
    A2 --> J
    J --> S["Conjunto congelado<br/>inmutable, con hash"]
    S --> E["Evaluación<br/>protocolo fijo"]
```

- **Ciega:** quien anota no ve predicciones, puntuaciones ni el trabajo de otras personas. El servidor no las envía.
- **Cobertura explícita:** una zona sin marcas solo cuenta como «sin eventos» si alguien la declaró revisada.
- **Sin fugas:** un conjunto no se congela si el mismo hablante o el mismo audio aparece en dos particiones (entrenamiento, desarrollo, prueba).

Detalle completo en el [protocolo técnico](protocolo-mrcd.md) y la [guía de anotación](guia-anotacion-v1.md).
