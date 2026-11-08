# Lista de verificación: preregistro del protocolo G en OSF (gratuito), en dos etapas

Uso: la sigue el autor. El protocolo G se congela en **dos etapas** (`PROTOCOL_G_DRAFT.md`, encabezado y sección 14):

- **Etapa 1 (registro previo a la anotación):** **antes** de empezar el pase 1 de anotación. Aquí se registra todo lo que
  define la prueba: hipótesis, umbrales, pisos de evaluabilidad, compuertas de la referencia, estadística, reglas de
  emparejamiento, método de tiempos de `R_T` (OD-2) y todos los valores por defecto.
- **Etapa 2:** **después** de que las anotaciones estén en commit y **antes** de que cualquier sistema o el alineador toque
  el audio de las 19 grabaciones. Solo se agregan el código del arnés, los valores fijados de `SYSTEM_FREEZE_G.json` y el
  registro de OD-4.

Basada en la ayuda de OSF consultada el 2026-11-08 (help.osf.io, "Welcome to Registrations & Preregistrations"); los
nombres de botones pueden cambiar en la interfaz.

## A. Etapa 1: antes de entrar a OSF (en el repositorio)

- [ ] **OD-1 cerrada por escrito** (la única decisión abierta; por defecto: conservar a `maria` con análisis de
      sensibilidad). Si decide excluirla, el manifiesto se vuelve a sortear con `sample_items.py --exclude-speaker maria`
      y se hace commit del manifiesto nuevo **antes** del pase 1.
- [ ] Calibración terminada y guía congelada por commit, con su hash anotado. Tenga presentes los conteos de calibración
      que ya vio (son ventanas de las mismas 19 grabaciones): los declarará en el paso B6.
- [ ] `PROTOCOL_G.md` (versión final, sin "DRAFT", con OD-1 cerrada) y `PROTOCOL_G.stage1.sha256.json` están en commit.
      Si algún valor cambió respecto del borrador después de la calibración, figura en `PROTOCOL_G.md` como "cambiado
      después de la calibración". Anote el hash del commit y el sha256 de `PROTOCOL_G.md`.
- [ ] **El pase 1 no ha empezado.** Ningún sistema (MRCD, Whisper, CrisperWhisper, el modelo afinado) ni el alineador CTC
      se ha ejecutado sobre estas grabaciones, y nadie ha abierto sus transcripciones (solo se calcularon sus sha256 con un
      script).
- [ ] Licencia de NOCANDO confirmada en la página de origen (OD-5). Mientras no esté confirmada, a OSF solo se suben
      hashes y agregados: **los manifiestos se suben solo como sha256**, nunca su contenido, y nunca las etiquetas.
- [ ] Licencia del alineador (`jonatasgrosman/wav2vec2-large-xlsr-53-spanish`, Apache-2.0 según su ficha) releída en la
      ficha del modelo.

## B. Etapa 1: en osf.io

1. [ ] Cree una cuenta gratuita en https://osf.io (puede usar ORCID). Active la verificación en dos pasos.
2. [ ] Cree un proyecto **privado** (por ejemplo, "MRCD - protocolo G").
3. [ ] Suba al proyecto: `PROTOCOL_G.md`, `PROTOCOL_G.stage1.sha256.json`, `GUIA_ANOTACION.md`, `PROTOCOL_ANNOTATION.md`,
       `ANNEX_20261108_SELF_ANNOTATION.md` y, para los manifiestos, **solo su sha256** hasta que OD-5 esté resuelta (si la
       licencia está confirmada, puede subir los manifiestos: nombres de archivo, hashes y tiempos).
4. [ ] **No suba nunca:** audio, fragmentos, transcripciones, la clave de cegado, salidas de CrisperWhisper ni rutas
       personales.
5. [ ] Abra "Registrations" y cree un registro nuevo desde el proyecto. Plantilla recomendada: **"Open-Ended
       Registration"** (la más flexible). Indique en el título o en la primera línea que es un **registro previo a la
       anotación (etapa 1)**. En el resumen pegue el título de la tesis, el commit y el sha256 de `PROTOCOL_G.md`, y las
       tres hipótesis primarias, una línea cada una: - **PG1:** la F1 de la regla léxica sobre `p0` es mayor contra la referencia derivada de la transcripción (`R_T`,
       con tiempos de un alineador CTC independiente) que contra la referencia humana de repeticiones en cualquier
       estado (`H_lit`), en los mismos ítems. - **PG2:** la supervivencia en el texto de `p0` de las repeticiones disfluentes oídas por la persona anotadora es
       menor que 0,80. - **PG3:** el estado binario de los marcadores del perfil `o2026_12` concuerda con la función humana mejor que el
       azar (límite inferior del IC 95 % de κ mayor que 0); es solo una verificación "mejor que el azar".
       Indique que las familias secundarias (PGS1-PGS6, PGC1-PGC3, PGF1-PGF3) son exploratorias, sin control global del
       error por familia, y remita al archivo adjunto para todo lo demás. Si prefiere un formulario guiado, la plantilla de
       datos secundarios ("Secondary Data Preregistration") también encaja, porque el corpus ya existe.
6. [ ] Declare en el texto, sin suavizar: anotación por el propio autor (un solo evaluador principal, ciego al sistema),
       subconjunto del evaluador 2 con su tamaño previsto, contacto previo con el corpus (la celda gastada de 14
       grabaciones del protocolo F, los conteos publicados de `hm`/`eh`, la hablante `maria` compartida), conocimiento de
       los resultados del protocolo F y del plan A, y los conteos de calibración ya vistos.
7. [ ] Embargo: opcional, hasta 4 años. Recomendación: público, o embargo hasta la sustentación. Para retirar un registro
       embargado primero hay que terminar el embargo.
8. [ ] Envíe el registro. Mientras tenga menos de 48 h y no lo hayan confirmado los colaboradores, todavía puede
       cancelarse; después queda fijo con fecha y hora.
9. [ ] Copie la URL (y el DOI, si OSF lo asigna) y la fecha y hora del registro.

## C. Etapa 1: de vuelta en el repositorio

- [ ] Cree `results/protocol_g/OSF_REGISTRATION_STAGE1.json` con la URL, el DOI si existe, la fecha y hora UTC, el commit
      de la etapa 1 y el sha256 de `PROTOCOL_G.md` registrado; haga commit.
- [ ] Recién entonces empiece el pase 1 de anotación.

## D. Etapa 2: después de las anotaciones y antes de cualquier ejecución

- [ ] Los CSV del pase 1, pase 2, evaluador 2 y adjudicado, y los informes de acuerdo (calculados **antes** de adjudicar),
      están en commit con sus `.sha256`.
- [ ] Solo se agregó lo permitido: el código del arnés y sus pruebas, `SYSTEM_FREEZE_G.json` y el registro de OD-4
      (incluido solo si la rama de `o2026_13` está fusionada y el hash de su `model.bin` está fijado; ninguna otra
      consideración). `REFERENCE_GATES_G.json` se calculó solo con las anotaciones.
- [ ] Cualquier otro cambio posterior al inicio del pase 1 está en un anexo fechado, declarado como posterior a la
      anotación, que dice qué conteos, cifras de acuerdo o valores de compuertas ya había visto.
- [ ] `PROTOCOL_G.stage2.sha256.json` está en commit y repite sin cambios los hashes de la etapa 1.
- [ ] Suba `PROTOCOL_G.stage2.sha256.json` (y los anexos, si los hay) al proyecto de OSF como documento fechado. El
      registro de la etapa 1 no se edita.
- [ ] Cree `results/protocol_g/OSF_REGISTRATION.json` con los datos de la etapa 1 y la fecha y hora UTC de la carga de la
      etapa 2; haga commit. El arnés no ejecuta `run` sin este archivo ni sin el de la etapa 1.
- [ ] Recién entonces: `freeze-system` ya hecho, `verify` y **una sola** ejecución de `run`.

## E. Si algo cambia después del registro

- Nunca edite el registro ni `PROTOCOL_G.md`. Escriba un anexo fechado en el repositorio **antes** del paso afectado y
  súbalo al proyecto de OSF como documento de cambios transparentes. Si el anexo se escribe después de iniciado el pase 1,
  debe declarar qué datos de anotación había visto.
- Si el cambio fuera de fondo: registro nuevo con la información actualizada y retiro del original indicando el motivo y
  la URL del nuevo. El retiro deja un aviso público; no borra el rastro.
