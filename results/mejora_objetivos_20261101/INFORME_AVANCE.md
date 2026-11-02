# Informe de avance — OE1 a OE4 (01/11/2026)

Estado verificado al cierre de la jornada del 01/11/2026. Los porcentajes son estimaciones de planificación, no métricas. Ninguna aprobación del asesor ha ocurrido; todas las hojas están en estado PENDIENTE.

## Resumen

| Objetivo | Inicio del día | Ahora (estimado) | Qué lo movió | Qué falta |
|---|---|---|---|---|
| OE1 Fundamentar | ~85 % | **~92 %** | Taxonomía unificada (`TAXONOMIA_EQUIVALENCIAS.md`), estado del arte v3.1, requisitos (16 RF / 10 RNF), Charter/ACP y Memoria alineados, fuentes verificadas en Crossref | Firma del asesor; cuartiles sin verificar (Scimago bloqueado); 4 PDF de acceso abierto para descargar a mano |
| OE2 Diseñar | ~85 % | **~92 %** | Contratos reales (rev 1.2), C4 de 4 niveles con render local verificado, especificación de desambiguación, decisión de privacidad documentada (pendiente), estado incierto implementado como opción | Firma del asesor; decisión de privacidad del audio |
| OE3 Implementar | ~60 % | **~80 %** | E2E con navegador real (backend de la app simulado), perfiles GPU (RTF ≈ 0,04), perfiles `noloop`, ruta textual P1, alcance de 3 clases en servicio y parche, kit de micrófono, ~914 pruebas | Aplicar el parche en la app real (autorización), prueba con micrófono real, backend real de la app |
| OE4 Validar | ~20 % (original) / ~70 % (reducido) | **~22 % (original) / ~78 % (reducido)** | S2, S3 y S6 medidos con protocolos congelados, Holm y revisión independiente; cobertura/riesgo ahora medible | Ratificación del OE4 reducido; SUS con ≥ 10 usuarios; prueba independiente; video evaluable |

**Total aproximado:** ~70 % (antes ~60 %).

## Resultados nuevos (exploratorios, DIHANA folds 1-4 salvo indicación; fold 5 no usado)

- **S2 — Bucles del ASR:** `p1_noloop` en CPU: 2 clips con bucle (antes 11), WER 0,400 (antes 0,583), supervivencia de repeticiones 143/214 (antes 105). La reducción de la tasa de bucles no es significativa tras Holm (p ≈ 0,42-0,44); la ganancia de WER se concentra en AMERESCO (4 grabaciones) y resiste el análisis dejando fuera una grabación; en DIHANA el WER no cambia.
- **S3 — Clases:** repetición con texto P1, `ctx_LR` 0,699 frente a regla 0,682 (no demostrado); frente a V0 +0,183 y frente a `ctx_LR` sobre P0 +0,118 (tras Holm). La mejora viene del ASR P1. Fusión regla + modelo sin ganancia. Calibración reduce el ECE de V0, no el de los modelos aprendidos.
- **Cobertura/riesgo (indicador original):** pausa llena 0,932 [0,923; 0,940] / 0,097 [0,083; 0,110] — cumple en el punto, pero el límite superior del riesgo supera 0,10: cumplimiento no demostrado. Repetición 0,962 / 0,029, pero el criterio no es informativo (riesgo de "siempre ausente" = 0,043). La abstención reduce el F1 sobre todos los casos. Las marcas de DIHANA no son la doble anotación que exige el indicador.
- **S6 — Video:** no concluyente. Compuerta congelada: 69/1533 oraciones (4,5 %), 0/173 positivos; cara en 31,7 % de los fotogramas; sincronía verificable en 2/74 videos. La ablación post hoc sugiere que añadir video empeora el F1 (−0,13, no significativo). Hace falta video frontal propio con hablantes nuevos.
- **S1 — GPU y E2E:** RTF del ASR 0,04 en GPU frente a 0,31-0,48 en CPU; tick de 0,5 s en GPU con 3,2 % de ticks fuera de presupuesto; recorrido navegador → servicio → reporte verificado.

## Documentos (copias en `<thesis-documents>\<course-code>\`)

Refactorizados y revisados de forma independiente (rondas de revisión documental): Charter, ACP, Plan de Dirección, Cronograma, Plan vigente, Scrum, Capstone (plantilla UPC de 4 capítulos), Memoria (95 pp.), Cumplimiento OE1-OE4, Casos y Plan de Pruebas, Manual, Informe de Evidencias, Solicitud de Conformidad (borrador), Repositorios, Perfil, presentación (15 diapositivas) y paper (compila con la plantilla oficial de Springer Nature). Originales intactos.

Pendiente del equipo: actualizar índices y exportar PDF desde Word (los PDF actuales salieron con LibreOffice); confirmar el código de curso/periodo y la plantilla UPC.

## Publicación

Rama pública `mrcd-oe-closure` en `d05df48` (repositorio `oratorIA-ProyectoTesis/MRCD`), saneada: sin corpus, datos personales, hojas de firma ni documentos de tesis. `main` remoto sin cambios.

## Acciones que solo puede hacer el equipo

1. Entregar las hojas de aprobación OE1, OE2 y OE4 al asesor.
2. Autorizar el parche de la app ("autorizo aplicar el parche a ORATORIA-PROYECTO").
3. Ejecutar el kit de micrófono.
4. Aplicar el SUS con ≥ 10 usuarios (requiere app operable, decisión de privacidad y consentimiento aprobado).
5. Decidir licencia del repositorio y si se retira `case_study_interview.md` del `main` público.
6. Opcional, de mayor impacto en OE4: un conjunto pequeño propio (30-45 min) anotado por dos personas en las 3 clases, con video frontal.

## Limpieza local pendiente

Carpetas temporales en el scratchpad de la sesión (`app_clone`, `app_check`, `clone_*`, `fixmut`, `clone_pub`, `clone_rev2`) y `<thesis-documents>\_work_A`. **Antes de borrar cualquier `clone_*` o `fixmut`, quitar su enlace `data` con `rmdir <carpeta>\data`**: apunta a los 11 GB reales de `<repo>\data`.


## Addendum 01/11/2026 (noche): campaña de optimización anidada y consolidación de OE4

Autorrevisión de los autores. La campaña tiene revisión independiente (`results/mejora_objetivos_20261101/O_campaign/REVISION_O.md`). Todo exploratorio (DIHANA pliegues 1-4; fold 5 solo como reporte histórico, una vez).

| Clase | F1 externo [IC] | V0 | Δ vs V0 (Holm) | Δ vs mejor referencia fija | Decisión |
|---|---|---|---|---|---|
| Pausa llena | 0,697 [0,658; 0,732] | 0,567 | +0,130 (0,0012) | +0,036 sobre la regla léxica (Holm 0,014) | candidato `default: false` |
| Prolongación | 0,619 [0,582; 0,652] | 0,458 | +0,160 (0,0012) | +0,027 sobre la pila de S3 (Holm 0,007) | candidato `default: false` |
| Repetición literal | 0,680 [0,608; 0,741] | 0,516 | +0,165 (0,0012) | −0,002 frente a la regla P1; −0,030 frente a S3 A2 (no demostrado) | candidato; se prefiere `rep_ctx_p1` |

- Criterio de promoción congelado: cumplido frente a V0; la adopción en el servicio no está decidida. El techo parece de características, etiquetas y ASR más que de datos (curva plana desde ~75 % de los hablantes, solo una configuración probada). EPIC +0,006 (sin ganancia); AMERESCO no informativo; fold 5 histórico 0,64 (pausa llena) y 0,61 (prolongación).
- Revisión independiente: sin hallazgos que invaliden; correcciones de redacción (techo, "ganancias tempranas pequeñas", calibración elegida "ninguna" en 12/15 pliegues y no 13/15).
- F1_context 0,49 frente a 0,56 (n = 96): no demostrado. Cobertura/riesgo: pausa llena (S3) 0,932 / 0,097 con límite 0,110, no demostrado; pipeline optimizado 0,938 / 0,059 [0,050; 0,068], cumple numéricamente sobre marcas del corpus (sin doble anotación, sin calibración); repetición no informativa. Video (S6): no evaluable (4,5 %).
- Los porcentajes de avance de OE4 de la tabla superior no cambian: son estimaciones de planificación y los indicadores originales siguen sin cumplirse.
- Consolidado también en `results/cierre_oe4_reducido_20261031/INFORME_OE4_REDUCIDO.md` (addendum 01/11/2026, con la matriz indicador → evidencia → estado), `RESUMEN_EJECUTIVO_OE4.md` y `results/fundamentacion_20261101/SUSTENTACION_ASESOR.md`.
