# Comparativa de capacidades

Esta comparación es de **capacidades**, no de exactitud. La exactitud solo se compara con el mismo audio, la misma referencia humana y el mismo protocolo (ver [Evidencia](evidencia.md)).

## Frente a otros detectores

| Capacidad                                     | MRCD                                                             | GPT audio (`gpt-audio-1.5`)                              | Reglas heurísticas             |
| --------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------ |
| Taxonomía fija de 7 clases para español       | Sí                                                               | Por instrucción en el prompt                             | Sí                             |
| Intervalos con resolución fija                | Sí, cuadrícula de 0,5 s; bordes refinables a palabras o silencio | Propuestos por el modelo, sin verificar                  | Sí, según cada regla           |
| Distingue pausas de disfluencias              | Sí                                                               | Por instrucción                                          | Sí                             |
| Abstención explícita («incierto»)             | Sí, por umbral de clase                                          | Sí, por instrucción                                      | No                             |
| Corre sin conexión, en tu equipo              | Sí                                                               | No: el audio se envía a un servicio externo              | Sí                             |
| Costo por uso                                 | Ninguno                                                          | Por tokens                                               | Ninguno                        |
| Resultado reproducible con versión registrada | Sí: checkpoint con hash y configuración                          | Parcial: modelo y prompt registrados, respuesta guardada | Sí                             |
| Puede reentrenarse con anotación propia       | Sí                                                               | No                                                       | No: se ajustan umbrales a mano |
| Usa el contexto de 10 s alrededor             | Sí                                                               | Recibe el segmento completo                              | Parcial                        |

En la plataforma, GPT y las reglas funcionan como **comparadores**: corren sobre el mismo audio y sus resultados se muestran en pistas paralelas.

## Frente a herramientas de anotación

| Capacidad                                                     | Plataforma MRCD                          | ELAN                                   | Label Studio (edición comunitaria)                                 |
| ------------------------------------------------------------- | ---------------------------------------- | -------------------------------------- | ------------------------------------------------------------------ |
| Anotación de intervalos sobre audio                           | Sí, en el navegador                      | Sí, aplicación de escritorio           | Sí, en el navegador                                                |
| Anotación ciega por defecto (sin predicciones del modelo)     | Sí, el servidor no las envía             | Depende de los archivos que se carguen | Depende de la configuración del proyecto                           |
| Doble anotación independiente y adjudicación A/B anónima      | Sí, integrada                            | No integrada                           | Las funciones de consenso y acuerdo están en la edición Enterprise |
| Cobertura explícita («revisado sin eventos»)                  | Sí                                       | No integrada                           | No integrada                                                       |
| Conjuntos congelados con detección de fugas entre particiones | Sí                                       | No                                     | No                                                                 |
| Evaluación por eventos integrada                              | Sí                                       | No                                     | No                                                                 |
| Intercambio con ELAN                                          | Exporta `.eaf` (capa `human_disfluency`) | Formato nativo                         | Requiere conversión                                                |

ELAN y Label Studio son herramientas generales y maduras; MRCD no las reemplaza. Su plataforma está hecha para un flujo concreto: producir una referencia humana confiable de disfluencias y evaluar sistemas contra ella sin contaminar la prueba.

Fuentes: documentación de ELAN (archive.mpi.nl/tla/elan) y de Label Studio sobre predicciones y funciones Enterprise (labelstud.io/guide/predictions, labelstud.io/guide/enterprise_features).
