# Kit de usabilidad SUS (OE4): listo para aplicar, **NO REALIZADO**

**ESTADO: NO REALIZADO.** Nadie ha respondido este cuestionario. No existe ningún valor SUS, ni parcial ni estimado. El indicador original de OE4 ("SUS >= 70 con >= 10 usuarios reales") sigue **sin cumplirse**. Este kit solo prepara la aplicación. Autorrevisión de un agente; sin revisión independiente. Fecha de preparación: 2026-11-01.

## Contenido

| Archivo | Qué es |
|---|---|
| `PROTOCOLO.md` | Protocolo: criterios de inclusión, tareas en la aplicación, procedimiento, qué se registra |
| `CONSENTIMIENTO.md` | Borrador de consentimiento informado (pendiente de revisión del asesor/comité de ética) |
| `CUESTIONARIO_SUS_ES.md` | Cuestionario SUS en español, con la cita de la versión validada **marcada "no verificada"** |
| `responses_template.csv` | Plantilla de respuestas (una fila por participante; vacía) |
| `sus_score.py` | Cálculo SUS (ítems impares/pares, escala 0-100) con IC por bootstrap; sin datos informa "no disponible", nunca 0 |
| `test_sus_score.py` | Pruebas unitarias del cálculo (14 pruebas) |
| `RESULTADOS_PLANTILLA.md` | Plantilla de resultados, **NO REALIZADO** |

## Requisitos previos (ninguno se cumple hoy)

1. **Aplicación operable con la sección de disfluencias.** El parche de integración existe, pasa `tsc`, `eslint`, `vitest` y `git apply --check`, pero **no está aplicado** a `ORATORIA-PROYECTO`; aplicarlo requiere la autorización escrita del usuario.
2. **Decisión de privacidad del audio** (`../contracts/PRIVACY_DECISION.md`, PENDIENTE).
3. **Aprobación ética o aval del asesor/comité** para aplicar con personas; no existe hoy.
4. **Revisión del texto del cuestionario** contra la versión validada (la cita no se pudo verificar localmente).
5. El reporte de la aplicación debe mostrar **solo las clases del alcance evaluado propuesto** (`filler_word`, `prolongation` como alargamiento observable y `repetition` literal; `revision` solo como aproximación exploratoria rotulada), y `block`, `rhetorical_pause` y `neutral_pause` ocultas o rotuladas "fuera de alcance, no validada". Es una propuesta pendiente de ratificación del asesor.

## Uso (cuando existan respuestas)

```bash
.venv/Scripts/python.exe docs/sus-kit/sus_score.py ruta/a/responses.csv            # texto
.venv/Scripts/python.exe docs/sus-kit/sus_score.py ruta/a/responses.csv --json     # JSON
.venv/Scripts/python.exe -m unittest discover -s docs/sus-kit -p "test_*.py"       # pruebas
```

No guarde en el repositorio las respuestas reales ni datos personales: `responses.csv` con datos debe quedar fuera del control de versiones (use solo códigos `P01`, `P02`, ...).

## Qué NO afirma este kit

- No afirma que la aplicación sea usable, ni que se alcance 70.
- Un puntaje SUS no es un porcentaje ni una probabilidad calibrada; no mide la exactitud del motor.
- Con n < 10 el indicador no se puede evaluar; el script lo declara.
- Evaluar usabilidad de la aplicación no valida ninguna clase del motor ni atribuye al video estado psicológico alguno.
