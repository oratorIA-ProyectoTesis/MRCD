# MRCD

**Motor de Reconocimiento Contextual de Disfluencias del habla en español.**

MRCD escucha una grabación, la transcribe y marca dónde aparecen muletillas, prolongaciones, repeticiones, bloqueos, revisiones y pausas. Cada detección tiene un intervalo en milisegundos que se puede escuchar en su contexto.

Junto al motor hay una plataforma de **revisión humana**: dos personas anotan el mismo audio sin verse, una tercera adjudica las diferencias y el resultado se congela como referencia para evaluar cualquier sistema con un protocolo fijo.

[Probar la demo](demo.md){ .md-button .md-button--primary } [Ver la evidencia](evidencia.md){ .md-button } [Instalar](instalacion.md){ .md-button }

## Qué lo distingue

| Rasgo | Por qué importa |
| --- | --- |
| **Contexto, no conteo**  | Cada tramo se decide mirando 10 s de texto alrededor y la prosodia de la pausa. Por eso distingue una pausa retórica de un bloqueo, y «este» demostrativo de «este» muletilla. |
| **Pausas aparte**        | Las pausas retóricas y neutras se reportan, pero nunca se suman a las disfluencias.                                                                                            |
| **Abstención explícita** | Cuando el modelo no supera el umbral de una clase, devuelve «incierto» en lugar de inventar una etiqueta.                                                                      |
| **Evaluación honesta**   | Emparejamiento uno a uno, solo sobre audio revisado por personas, con intervalos de confianza por hablante.                                                                    |
| **Trazabilidad**         | Cada resultado guarda versión del modelo, configuración, hash del audio y tiempos por etapa.                                                                                   |

## Taxonomía

| Clase          | Qué marca                                               | ¿Disfluencia? |
| -------------- | ------------------------------------------------------- | ------------- |
| Muletilla      | Vocalización o expresión usada para vacilar (eh, mmm)   | Sí            |
| Prolongación   | Alargamiento no explicable por énfasis o final de frase | Sí            |
| Repetición     | Sonido, sílaba o palabra repetida al vacilar            | Sí            |
| Bloqueo        | Interrupción audible dentro de una unidad de habla      | Sí            |
| Revisión       | Abandono o reparación de una formulación                | Sí            |
| Pausa retórica | Pausa que organiza o enfatiza el discurso               | No            |
| Pausa neutra   | Pausa sin función clara                                 | No            |

## Estado del proyecto

MRCD es un proyecto de tesis en curso. La plataforma, el motor y el protocolo de evaluación funcionan y están probados. La **exactitud del modelo todavía no está medida**: el modelo actual se entrenó con etiquetas heurísticas y la referencia humana está en construcción. La página [Evidencia y benchmark](evidencia.md) separa lo medido de lo pendiente.

Una detección automática no es un diagnóstico ni una evaluación de la persona.
