// MRCD web client: review (blind/assisted/adjudication), product playground,
// experiments and administration. No build step; WaveSurfer pinned to an exact version.
import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7.12.12/dist/wavesurfer.esm.js'
import Regions from 'https://unpkg.com/wavesurfer.js@7.12.12/dist/plugins/regions.esm.js'
import Timeline from 'https://unpkg.com/wavesurfer.js@7.12.12/dist/plugins/timeline.esm.js'
import Spectrogram from 'https://unpkg.com/wavesurfer.js@7.12.12/dist/plugins/spectrogram.esm.js'

// Order = keys 1..7. Name + letter badge + color, never color alone.
const CLASSES = [
  ['filler_word', 'Muletilla', 'M', '#c2410c', 'Vocalización o expresión usada para vacilar (eh, mmm).'],
  ['prolongation', 'Prolongación', 'P', '#6d28d9', 'Alargamiento no explicable por énfasis o final de frase.'],
  ['repetition', 'Repetición', 'R', '#1d4ed8', 'Repetición de sonido, sílaba o palabra asociada a vacilación.'],
  ['block', 'Bloqueo', 'B', '#be123c', 'Interrupción audible dentro de una unidad de habla.'],
  ['revision', 'Revisión', 'V', '#15803d', 'Abandono o reparación de una formulación.'],
  ['rhetorical_pause', 'Pausa retórica', 'Pr', '#57534e', 'Pausa que organiza o enfatiza. No es disfluencia.'],
  ['neutral_pause', 'Pausa neutra', 'Pn', '#78716c', 'Pausa sin función clara. No es disfluencia.']
]
const CLS = Object.fromEntries(CLASSES.map(([id, name, short, color, def]) => [id, { name, short, color, def }]))
const PAUSES = new Set(['rhetorical_pause', 'neutral_pause'])
const DECISIONS = { event: 'Evento', uncertain: 'Incierto', not_evaluable: 'No evaluable' }
const STATUS = {
  queued: ['En cola', 'info'],
  preparing: ['Preparando', 'info'],
  ready: ['Lista', 'ok'],
  running: ['Analizando', 'info'],
  succeeded: ['Completado', 'ok'],
  partial: ['Parcial', 'warn'],
  failed: ['Falló', 'bad'],
  cancelled: ['Cancelado', 'muted'],
  cancel_requested: ['Cancelando', 'warn'],
  assigned: ['Asignada', 'info'],
  in_progress: ['En curso', 'warn'],
  submitted: ['Finalizada', 'ok'],
  pool: ['Sin asignar', 'muted'],
  open: ['Abierto', 'warn'],
  closed: ['Cerrado', 'ok'],
  not_run: ['Sin ejecutar', 'muted'],
  frozen: ['Congelado', 'ok'],
  active: ['Activa', 'ok']
}
const ROLE = {
  admin: 'Administración de datos',
  annotator: 'Anotación',
  adjudicator: 'Adjudicación',
  researcher: 'Investigación',
  user: 'Usuario del producto'
}
const SYSTEM = { mrcd: 'MRCD', rules: 'Reglas', gpt: 'GPT (audio)' }
const MODE = { blind: 'Ciega', assisted: 'Asistida' }
const SPLIT = { pilot: 'Piloto', train: 'Entrenamiento', dev: 'Desarrollo', test: 'Prueba' }
const STAGES = {
  load_audio: 'Preparando el audio',
  prepare_features: 'Transcribiendo y extrayendo rasgos',
  detect: 'Detectando eventos',
  save: 'Guardando el reporte',
  done: 'Listo'
}
const TERMINAL = ['succeeded', 'partial', 'failed', 'cancelled']
const WAVE = { height: 120, waveColor: '#8a94a6', progressColor: '#46536b', cursorColor: '#c2410c', normalize: true }

const $ = (s, el = document) => el.querySelector(s)
const view = $('#view')
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`)
const fmt = ms => {
  const s = Math.max(0, ms) / 1000
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, '0')}`
}
const pct = v => (v == null ? '—' : `${Math.round(v * 100)} %`)
const num = v => (v == null ? '—' : v.toFixed(2))
const uid = () => crypto.randomUUID()
const local = {
  get: k => {
    try {
      return JSON.parse(localStorage.getItem(k))
    } catch {
      return null
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem(k, JSON.stringify(v))
    } catch {}
  },
  del: k => {
    try {
      localStorage.removeItem(k)
    } catch {}
  },
  clear: () => {
    try {
      localStorage.clear()
    } catch {}
  }
}
const token = () => local.get('token') || ''
const mediaUrl = p => `/api${p}${p.includes('?') ? '&' : '?'}token=${encodeURIComponent(token())}`
const badge = label =>
  label
    ? `<span class="badge" style="background:${CLS[label].color}">${CLS[label].short}</span> ${CLS[label].name}`
    : '<span class="muted">sin clase</span>'
const chip = s => {
  const [t, tone] = STATUS[s] || [String(s).startsWith('failed') ? 'Falló' : s, String(s).startsWith('failed') ? 'bad' : 'muted']
  return `<span class="chip ${tone}">${esc(t)}</span>`
}
const tiles = items =>
  `<div class="tiles">${items.map(([label, value, hint]) => `<div class="tile"><b>${value}</b><span>${label}</span>${hint ? `<small>${hint}</small>` : ''}</div>`).join('')}</div>`
const empty = (msg, extra = '') => `<div class="empty"><p>${msg}</p>${extra}</div>`
const legend = () => `<div class="legend">${CLASSES.map(([c]) => `<span title="${esc(CLS[c].def)}">${badge(c)}</span>`).join('')}</div>`

// Header of every view: title plus a collapsible "what is this / how to use it" guide.
function intro(key, title, what, steps = []) {
  const closed = local.get(`intro:${key}`)
  return `<h1>${title}</h1><details class="intro" data-intro="${key}" ${closed ? '' : 'open'}><summary>¿Qué es esta vista y cómo se usa?</summary>
    <p>${what}</p>${steps.length ? `<ol>${steps.map(s => `<li>${s}</li>`).join('')}</ol>` : ''}</details>`
}
document.addEventListener(
  'toggle',
  e => {
    if (e.target.dataset?.intro) local.set(`intro:${e.target.dataset.intro}`, !e.target.open)
  },
  true
)

function toast(msg, kind = 'ok') {
  const t = document.createElement('div')
  t.className = `toast ${kind}`
  t.innerHTML = msg
  $('#toasts').append(t)
  setTimeout(() => t.remove(), 6000)
}

async function api(path, { method = 'GET', body, headers = {} } = {}) {
  const r = await fetch('/api' + path, {
    method,
    body: body === undefined ? undefined : JSON.stringify(body),
    headers: { Authorization: `Bearer ${token()}`, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...headers }
  })
  const data = (r.headers.get('content-type') || '').includes('json') ? await r.json() : await r.text()
  if (!r.ok) {
    const d = data?.detail
    throw Object.assign(new Error(typeof d === 'string' ? d : typeof data === 'string' ? data : JSON.stringify(d)), {
      status: r.status,
      detail: d
    })
  }
  return data
}

// ------------------------------------------------------------------ routing
let me = null
let cleanup = () => {}
const routes = [
  [/^#\/?$/, home],
  [/^#\/review$/, inbox],
  [/^#\/review\/tasks\/([\w-]+)$/, id => workbench('task', id)],
  [/^#\/review\/adjudication\/([\w-]+)$/, id => workbench('case', id)],
  [/^#\/playground$/, playground],
  [/^#\/playground\/([\w-]+)$/, results],
  [/^#\/experiments$/, experiments],
  [/^#\/experiments\/runs\/([\w-]+)$/, comparison],
  [/^#\/admin$/, admin],
  [/^#\/sus$/, sus]
]
const NAV = {
  user: [
    ['#/', 'Inicio'],
    ['#/playground', 'Prueba del producto'],
    ['#/sus', 'Encuesta']
  ],
  annotator: [
    ['#/', 'Inicio'],
    ['#/review', 'Mis tareas']
  ],
  adjudicator: [
    ['#/', 'Inicio'],
    ['#/review', 'Mis casos']
  ],
  researcher: [
    ['#/', 'Inicio'],
    ['#/playground', 'Prueba del producto'],
    ['#/review', 'Revisión'],
    ['#/experiments', 'Experimentos'],
    ['#/sus', 'Encuesta']
  ]
}
NAV.admin = [...NAV.researcher.slice(0, 4), ['#/admin', 'Administración'], ['#/sus', 'Encuesta']]

async function route() {
  cleanup()
  cleanup = () => {}
  view.onclick = view.onchange = null
  if (!me) {
    if (!token()) return login('')
    try {
      me = await api('/me')
    } catch (e) {
      return login('El token no es válido o ya no está activo.')
    }
  }
  nav()
  const hit = routes.find(([re]) => re.test(location.hash || '#/'))
  if (!hit) {
    location.hash = '#/'
    return
  }
  view.innerHTML = '<p class="muted">Cargando…</p>'
  scrollTo(0, 0)
  try {
    await hit[1](...(location.hash || '#/').match(hit[0]).slice(1))
  } catch (e) {
    view.innerHTML = `<div class="banner bad"><b>No se pudo cargar esta vista.</b> ${esc(e.message)}</div>`
  }
  view.focus({ preventScroll: true })
}
addEventListener('hashchange', route)

function login(msg) {
  $('#nav').innerHTML = ''
  view.innerHTML = `<form id="login" class="card narrow"><h1>MRCD</h1>
    <p>Plataforma para revisar disfluencias del habla en español y probar el motor MRCD.</p>
    <label>Token de acceso <input name="t" type="password" autocomplete="off" required autofocus></label>
    <button class="primary">Entrar</button>${msg ? `<p class="error">${esc(msg)}</p>` : ''}
    <p class="muted">El token lo entrega el administrador de datos (web: Administración → Personas, o <code>python -m app.cli add-user</code>). Al salir se borran los datos guardados en este navegador.</p></form>`
  $('#login').onsubmit = e => {
    e.preventDefault()
    local.set('token', e.target.t.value.trim())
    me = null
    route()
  }
}

function nav() {
  const here = (location.hash || '#/').split('/').slice(0, 2).join('/')
  $('#nav').innerHTML =
    (NAV[me.role] || [])
      .map(([h, t]) => `<a href="${h}" ${h === here || (h === '#/' && here === '#') ? 'aria-current="page"' : ''}>${t}</a>`)
      .join('') + `<span class="who">${esc(me.name)} · ${ROLE[me.role] || me.role}</span><button id="logout">Salir</button>`
  $('#logout').onclick = () => {
    local.clear()
    me = null
    location.hash = '#/'
    route()
  }
}

// -------------------------------------------------------------------- home
function home() {
  const cards = [
    [
      '#/playground',
      'Prueba del producto',
      'Graba o sube un audio, ejecuta MRCD y escucha cada detección en su contexto. Sirve para demostrar el funcionamiento, no para medir exactitud.',
      ['user', 'researcher', 'admin']
    ],
    [
      '#/review',
      me.role === 'adjudicator' ? 'Mis casos de adjudicación' : 'Revisión humana',
      me.role === 'adjudicator'
        ? 'Compara dos anotaciones independientes (A y B) de una misma región y decide la versión final.'
        : 'Escucha fragmentos asignados y marca muletillas, repeticiones, pausas y demás fenómenos según la guía. Así se construye la referencia humana.',
      ['annotator', 'adjudicator', 'researcher', 'admin']
    ],
    [
      '#/experiments',
      'Experimentos',
      'Compara sistemas (MRCD, reglas, GPT) sobre el mismo audio y evalúalos contra un conjunto humano congelado con un protocolo fijo.',
      ['researcher', 'admin']
    ],
    [
      '#/admin',
      'Administración',
      'Crea personas y tokens, asigna tareas de anotación, arma casos de adjudicación y congela conjuntos de datos.',
      ['admin']
    ],
    [
      '#/sus',
      'Encuesta de usabilidad',
      'Cuestionario SUS de 10 preguntas sobre tu experiencia con la plataforma.',
      ['user', 'researcher', 'admin']
    ]
  ].filter(c => c[3].includes(me.role))
  view.innerHTML = `<h1>Hola, ${esc(me.name)}</h1>
    <p class="lead">Tu rol es <b>${ROLE[me.role]}</b>. MRCD tiene dos espacios que comparten el mismo reproductor y los mismos datos:
    la <b>revisión humana</b>, que produce referencias confiables, y la <b>prueba del producto</b>, que muestra lo que el motor detecta.
    Una detección automática no es un diagnóstico ni una evaluación de la persona.</p>
    <div class="cards">${cards.map(([h, t, d]) => `<a class="card link" href="${h}"><h2>${t}</h2><p>${d}</p><span class="go">Abrir →</span></a>`).join('')}</div>`
}

// ------------------------------------------------------------------ inbox
async function inbox() {
  const d = await api('/review/tasks')
  const count = s => d.tasks.filter(t => t.status === s).length
  const adj = me.role === 'adjudicator'
  const rows = filter =>
    d.tasks
      .filter(t => !filter || t.status === filter)
      .map(
        t => `<tr>
    <td>${fmt(t.start_ms)}–${fmt(t.end_ms)}</td><td>${MODE[t.mode]}</td><td>${chip(t.status)}</td><td>${esc(t.guideline_version)}</td>
    <td><a class="button ${t.status === 'submitted' ? '' : 'primary'}" href="#/review/tasks/${t.id}">${t.status === 'submitted' ? 'Ver' : 'Abrir'}</a></td></tr>`
      )
      .join('')
  view.innerHTML =
    intro(
      adj ? 'adj-inbox' : 'inbox',
      adj ? 'Mis casos de adjudicación' : 'Mis tareas de anotación',
      adj
        ? 'Cada caso reúne dos anotaciones independientes (A y B) de la misma región, hechas por personas distintas. No verás quién hizo cada una ni ninguna predicción del modelo.'
        : 'Aquí están los fragmentos de audio que te asignaron. Cada tarea es una región de 10 a 20 s con 5 s de contexto a cada lado. Tu trabajo se guarda solo en el servidor mientras avanzas.',
      adj
        ? [
            'Abre un caso abierto.',
            'Escucha las diferencias entre A y B.',
            'Copia la versión correcta o dibuja una nueva y deja un motivo breve.',
            'Guarda la adjudicación.'
          ]
        : [
            'Pulsa <b>Continuar</b> para abrir la siguiente tarea pendiente.',
            'Escucha la región completa y marca cada fenómeno según la guía.',
            'Marca la región como revisada y pulsa <b>Finalizar</b>.'
          ]
    ) +
    (adj
      ? ''
      : tiles([
          ['Asignadas', count('assigned')],
          ['En curso', count('in_progress')],
          ['Finalizadas', count('submitted')],
          ['Min. de audio pendientes', d.pending_minutes]
        ])) +
    (adj
      ? ''
      : `<div class="row">${d.tasks.some(t => t.status !== 'submitted') || me.role === 'annotator' ? '<button class="primary" id="next">Continuar con la siguiente tarea</button>' : ''}
      <label class="inline">Mostrar <select id="flt"><option value="">todas</option><option value="assigned">asignadas</option><option value="in_progress">en curso</option><option value="submitted">finalizadas</option></select></label></div>
      ${
        d.tasks.length
          ? `<div class="scroll"><table><thead><tr><th>Región</th><th>Modo</th><th>Estado</th><th>Guía</th><th></th></tr></thead><tbody id="rows">${rows()}</tbody></table></div>`
          : empty(
              'No tienes tareas asignadas.',
              ['admin', 'researcher'].includes(me.role)
                ? '<p>Las tareas se crean en <a href="#/admin">Administración → Asignar tareas</a>.</p>'
                : '<p>Cuando el equipo te asigne fragmentos aparecerán aquí.</p>'
            )
      }`) +
    (adj || d.cases.length
      ? `<h2>Casos de adjudicación</h2>${d.cases.length ? `<table><tbody>${d.cases.map(c => `<tr><td>${c.id}</td><td>${chip(c.status)}</td><td><a class="button ${c.status === 'open' ? 'primary' : ''}" href="#/review/adjudication/${c.id}">${c.status === 'open' ? 'Adjudicar' : 'Ver'}</a></td></tr>`).join('')}</tbody></table>` : empty('No tienes casos asignados.')}`
      : '')
  if ($('#flt'))
    $('#flt').onchange = e => {
      $('#rows').innerHTML = rows(e.target.value)
    }
  if ($('#next'))
    $('#next').onclick = async () => {
      const t = await api('/review/tasks/next')
      t ? (location.hash = `#/review/tasks/${t.id}`) : toast('No hay tareas pendientes.', 'warn')
    }
}

// ------------------------------------------------ editor + adjudication view
async function workbench(kind, id) {
  const opened = performance.now()
  const isCase = kind === 'case'
  const d = await api(isCase ? `/review/adjudication/${id}` : `/review/tasks/${id}`)
  const t = isCase ? d.case : d.task
  const ann = d.annotation
  const locked = isCase ? t.status !== 'open' : ann.status === 'submitted'
  const blank = { events: [], contextual: [], coverage: [], actions: [], active_ms: 0, asr_used: false, telemetry: {} }
  let state = isCase ? blank : { ...blank, ...ann.data, telemetry: ann.data.telemetry || {} }
  let rev = ann?.rev
  let extra = 0
  let offset = Math.max(0, t.start_ms - t.context_ms)
  let sel = {}
  let current = null
  let op = null
  const reasons = {}
  const undo = []
  const pendingKey = ann && `pending:${ann.id}`
  const title = isCase ? 'Adjudicación' : t.mode === 'blind' ? 'Anotación independiente (ciega)' : 'Revisión asistida'
  const what = isCase
    ? 'Decides la versión final de una región a partir de dos anotaciones independientes. Las pistas A y B aparecen como franjas tenues sobre la onda; su orden es aleatorio y no muestra quién las hizo. Tu decisión no reemplaza las anotaciones originales.'
    : t.mode === 'blind'
      ? 'Anotas sin ver predicciones del modelo ni el trabajo de otras personas. La zona sombreada es la <b>región central</b> que debes cubrir; lo demás es contexto para escuchar. Todo cambio se guarda automáticamente.'
      : 'Ves candidatos propuestos por el modelo (sin puntuación). Escucha cada uno antes de aceptarlo y, al terminar, recorre toda la región: los candidatos no muestran lo que el modelo omitió.'
  const steps = isCase
    ? [
        'Escucha la región (<kbd>Espacio</kbd>).',
        'En «Diferencias A/B» pulsa <b>Usar A</b> o <b>Usar B</b>, o dibuja un evento nuevo.',
        'Escribe un motivo breve cuando haya desacuerdo.',
        'Pulsa <b>Guardar adjudicación</b>.'
      ]
    : [
        'Escucha la región completa a velocidad normal.',
        'Arrastra sobre la onda para seleccionar un tramo (o usa <kbd>I</kbd> y <kbd>O</kbd>).',
        'Elige la clase con el panel derecho o las teclas <kbd>1</kbd>–<kbd>7</kbd>. Ajusta los bordes arrastrando o con ±10/±100 ms.',
        'Si no hay más fenómenos, pulsa <b>Revisado: sin más eventos</b>.',
        'Pulsa <b>Finalizar</b> (<kbd>Ctrl</kbd>+<kbd>Enter</kbd>).'
      ]

  view.innerHTML = `${intro(isCase ? 'case' : `task-${t.mode}`, title, what, steps)}
    <div class="row meta"><span>Región central <b>${fmt(t.start_ms)}–${fmt(t.end_ms)}</b></span><span>Guía ${esc(t.guideline_version)}</span>
      <span class="chip ${locked ? 'ok' : 'info'}" id="save">${locked ? 'Finalizada · solo lectura' : isCase ? 'Se guarda al confirmar' : 'Guardado'}</span></div>
    <div id="alert"></div>
    <div class="grid"><section>
      <div id="wave"></div>
      <div class="row toolbar">
        <button id="play" class="primary">▶ Reproducir / pausar</button>
        <label class="inline">Velocidad <select id="rate"><option value="0.75">0,75×</option><option value="1" selected>1×</option><option value="1.25">1,25×</option></select></label>
        <label class="inline">Zoom <input id="zoom" type="range" min="20" max="600" value="100"></label>
        <button id="ctx">Ampliar contexto +5 s</button><button id="spec">Espectrograma</button>
        ${isCase ? '' : '<button id="asr" disabled title="Se habilita después de escuchar la región completa">Mostrar transcripción automática</button>'}
      </div>
      <div class="words" id="words"></div>
      ${isCase ? '<h2>Diferencias A/B</h2><p class="muted">Cada fila compara un evento de A con el de B que más se superpone.</p><div class="scroll list"><table id="diff"></table></div>' : ''}
      ${d.suggestions?.length ? '<h2>Candidatos del modelo</h2><p class="muted">Escucha cada candidato antes de decidir. «Falta otro evento» te lleva al lugar para marcar una omisión.</p><div class="scroll list"><table id="sugs"></table></div>' : ''}
      <h2>${isCase ? 'Decisión final' : 'Mis eventos'}</h2><div class="scroll list"><table id="events"></table></div>
      <h2>Usos legítimos</h2><p class="muted">Expresiones que parecen muletillas pero cumplen una función (por ejemplo «este» como demostrativo). No cuentan como disfluencias.</p><div class="scroll"><table id="ctxs"></table></div>
      <details><summary>Historial de versiones y guía completa</summary><div id="hist" class="muted"></div><pre id="guide"></pre></details>
    </section>
    <aside class="card">
      <h2>1. Clase del tramo</h2>
      <div class="classes" id="classes">${CLASSES.map(([c], i) => `<button data-c="${c}" title="${esc(CLS[c].def)}" ${locked ? 'disabled' : ''}><kbd>${i + 1}</kbd>${badge(c)}</button>`).join('')}</div>
      <div class="row"><button id="unc" ${locked ? 'disabled' : ''}>Incierto <kbd>U</kbd></button><button id="legit" ${locked ? 'disabled' : ''}>Uso legítimo</button><button id="noeval" ${locked ? 'disabled' : ''}>No evaluable</button></div>
      <div id="panel"></div>
      <h2>2. Cobertura</h2><div class="meter"><i id="covbar"></i></div><p id="cov" class="muted"></p>
      <div class="row"><button id="cov-all" ${locked ? 'disabled' : ''}>Revisado: sin más eventos</button><button id="cov-cur" ${locked ? 'disabled' : ''}>Revisado hasta el cursor</button></div>
      <h2>3. Terminar</h2>
      <div class="row"><button id="submit" class="primary" ${locked ? 'disabled' : ''}>${isCase ? 'Guardar adjudicación' : 'Finalizar tarea'} <kbd>Ctrl+Enter</kbd></button><button id="undo" ${locked ? 'disabled' : ''}>Deshacer <kbd>Ctrl+Z</kbd></button></div>
      <p class="muted small"><kbd>Espacio</kbd> reproducir · <kbd>←</kbd><kbd>→</kbd> ±1 s · <kbd>I</kbd>/<kbd>O</kbd> inicio/fin · <kbd>R</kbd> repetir selección · <kbd>Supr</kbd> borrar evento</p>
    </aside></div>`

  const ws = WaveSurfer.create({
    container: '#wave',
    ...WAVE,
    minPxPerSec: 100,
    url: audioUrl(),
    plugins: [Timeline.create({ formatTimeCallback: s => fmt(s * 1000 + offset) })]
  })
  const regions = ws.registerPlugin(Regions.create())
  if (!locked) regions.enableDragSelection({ color: 'rgba(29,78,216,.22)' })
  let spec = null
  let drawing = false
  let heardUntil = 0
  const L = ms => (ms - offset) / 1000
  const G = s => Math.round(s * 1000) + offset
  function audioUrl() {
    return mediaUrl(isCase ? `/review/adjudication/${id}/audio?extra_ms=${extra}` : `/review/tasks/${id}/audio?extra_ms=${extra}`)
  }

  // Reference tracks: A/B annotations when adjudicating, model candidates when assisted.
  const refs = isCase
    ? [...d.tracks.A.events.map(e => ({ ...e, side: 'A' })), ...d.tracks.B.events.map(e => ({ ...e, side: 'B' }))]
    : (d.suggestions || []).map(e => ({ ...e, side: 'S' }))

  function draw() {
    drawing = true
    regions.clearRegions()
    regions.addRegion({ id: 'central', start: L(t.start_ms), end: L(t.end_ms), color: 'rgba(120,130,150,.12)', drag: false, resize: false })
    for (const r of refs)
      if (r.label)
        regions.addRegion({
          start: L(r.start_ms),
          end: L(r.end_ms),
          color: CLS[r.label].color + '26',
          drag: false,
          resize: false,
          content: `${r.side}:${CLS[r.label].short}`
        })
    for (const e of state.events) {
      const color = e.label ? CLS[e.label].color : '#6b7280'
      regions.addRegion({
        id: e.event_id,
        start: L(e.start_ms),
        end: L(e.end_ms),
        color: color + (e.event_id === current ? '88' : '55'),
        drag: !locked,
        resize: !locked,
        content: e.decision === 'event' ? CLS[e.label].short : DECISIONS[e.decision]
      })
    }
    if (sel.start_ms != null && sel.end_ms != null)
      regions.addRegion({ id: 'sel', start: L(sel.start_ms), end: L(sel.end_ms), color: 'rgba(29,78,216,.22)' })
    drawing = false
    renderLists()
  }

  regions.on('region-created', r => {
    if (drawing || locked) return
    sel = { start_ms: G(r.start), end_ms: G(r.end) }
    current = null
    say(
      `Tramo seleccionado ${fmt(sel.start_ms)}–${fmt(sel.end_ms)}: elige una clase (1–7), «Incierto», «Uso legítimo» o «No evaluable».`,
      'info'
    )
    queueMicrotask(draw)
  })
  regions.on('region-updated', r => {
    if (r.id === 'sel') {
      sel = { start_ms: G(r.start), end_ms: G(r.end) }
      return
    }
    const e = state.events.find(x => x.event_id === r.id)
    if (e)
      commit(
        () => {
          e.start_ms = G(r.start)
          e.end_ms = G(r.end)
        },
        { type: 'adjust', event_id: e.event_id }
      )
  })
  regions.on('region-clicked', (r, ev) => {
    if (state.events.some(e => e.event_id === r.id)) {
      ev.stopPropagation()
      current = r.id
      draw()
    }
  })
  ws.on('decode', draw)
  // Client instrumentation for the interaction targets: task open, play response, save confirmation.
  const tel = (key, ms) => (state.telemetry[key] = [...(state.telemetry[key] || []).slice(-49), Math.round(ms)])
  ws.once('decode', () => tel('open_ms', performance.now() - opened))
  let playAsked = 0
  ws.on('play', () => playAsked && tel('play_ms', performance.now() - playAsked))
  state.telemetry.agent = navigator.userAgent
  ws.on('timeupdate', s => {
    heardUntil = Math.max(heardUntil, s)
    if ($('#asr') && heardUntil >= ws.getDuration() - 0.5) $('#asr').disabled = false
  })

  function say(msg, cls = 'warn') {
    $('#alert').innerHTML = msg ? `<div class="banner ${cls}">${msg}</div>` : ''
  }
  const status = (s, tone = 'info') => {
    const el = $('#save')
    el.textContent = s
    el.className = `chip ${tone}`
  }
  let lastAct = Date.now()
  const activity = () => {
    const now = Date.now()
    if (now - lastAct < 30000) state.active_ms += now - lastAct
    lastAct = now
  }

  function commit(mutate, action) {
    if (locked) return
    undo.push(JSON.stringify(state))
    mutate()
    if (action && !isCase) state.actions.push({ ...action, at: new Date().toISOString() })
    activity()
    draw()
    scheduleSave()
  }

  function classify(label, decision = 'event') {
    const e = state.events.find(x => x.event_id === current)
    if (e)
      return commit(
        () => {
          e.label = label ?? e.label
          e.decision = decision
          e.uncertainty = decision === 'uncertain' ? e.uncertainty || 'class' : decision === 'not_evaluable' ? 'audio' : null
        },
        { type: 'relabel', event_id: e.event_id }
      )
    if (sel.start_ms == null || sel.end_ms == null || sel.end_ms <= sel.start_ms)
      return say(
        'Primero selecciona un tramo: arrastra sobre la onda o marca <kbd>I</kbd> (inicio) y <kbd>O</kbd> (fin) mientras escuchas.'
      )
    const ev = {
      event_id: uid(),
      start_ms: sel.start_ms,
      end_ms: sel.end_ms,
      label,
      decision,
      uncertainty: decision === 'uncertain' ? 'class' : decision === 'not_evaluable' ? 'audio' : null,
      source_kind: isCase ? 'human_adjudicated' : 'human_independent',
      verbatim_text: null,
      speaker_id: null,
      note: null
    }
    commit(
      () => {
        state.events.push(ev)
        state.events.sort((a, b) => a.start_ms - b.start_ms)
      },
      { type: 'add', event_id: ev.event_id }
    )
    sel = {}
    current = ev.event_id
    say('')
    draw()
  }

  function renderLists() {
    const e = state.events.find(x => x.event_id === current)
    $('#events').innerHTML = state.events.length
      ? `<thead><tr><th>Tramo</th><th>Clase / decisión</th><th>Forma audible</th><th></th></tr></thead>` +
        state.events
          .map(
            x => `<tr data-id="${x.event_id}" class="${x.event_id === current ? 'sel' : ''}">
      <td>${fmt(x.start_ms)}–${fmt(x.end_ms)}</td><td>${x.decision === 'event' ? badge(x.label) : DECISIONS[x.decision] + (x.label ? ' · ' + badge(x.label) : '')}</td>
      <td>${esc(x.verbatim_text || '')}</td><td class="row"><button data-play="${x.event_id}" aria-label="Escuchar">▶</button><button data-pick="${x.event_id}">Editar</button>
      ${locked ? '' : `<button data-split="${x.event_id}">Dividir en cursor</button><button data-merge="${x.event_id}">Unir con siguiente</button><button data-del="${x.event_id}" class="danger">Borrar</button>`}</td></tr>`
          )
          .join('')
      : `<tr><td>${empty('Aún no hay eventos. Si la región no tiene fenómenos, márcala como revisada en el panel derecho.')}</td></tr>`
    $('#ctxs').innerHTML =
      state.contextual
        .map(
          (c, i) => `<tr><td>${fmt(c.start_ms)}–${fmt(c.end_ms)}</td>
      <td><input data-ci="${i}" data-f="expression" value="${esc(c.expression)}" placeholder="expresión (p. ej. «este»)" ${locked ? 'disabled' : ''}></td>
      <td><select data-ci="${i}" data-f="is_disfluent" ${locked ? 'disabled' : ''}>${[
        ['false', 'No es disfluencia'],
        ['true', 'Sí es disfluencia'],
        ['null', 'Indeterminado']
      ]
        .map(([v, n]) => `<option value="${v}" ${String(c.is_disfluent) === v ? 'selected' : ''}>${n}</option>`)
        .join('')}</select></td>
      <td><input data-ci="${i}" data-f="function" value="${esc(c.function || '')}" placeholder="función (p. ej. demostrativo)" ${locked ? 'disabled' : ''}></td>
      <td>${locked ? '' : `<button data-cdel="${i}" class="danger">Borrar</button>`}</td></tr>`
        )
        .join('') || `<tr><td class="muted">Sin usos registrados. Selecciona un tramo y pulsa «Uso legítimo».</td></tr>`
    $('#panel').innerHTML = e
      ? `<h2>Evento seleccionado</h2><div class="form">
      <label>Decisión <select id="p-dec" ${locked ? 'disabled' : ''}>${Object.entries(DECISIONS)
        .map(([k, v]) => `<option value="${k}" ${e.decision === k ? 'selected' : ''}>${v}</option>`)
        .join('')}</select></label>
      ${
        e.decision === 'uncertain'
          ? `<label>Motivo de la duda <select id="p-unc" ${locked ? 'disabled' : ''}>${[
              ['class', 'no sé qué clase es'],
              ['boundaries', 'no puedo delimitarlo'],
              ['audio', 'el audio no permite decidir']
            ]
              .map(([k, v]) => `<option value="${k}" ${e.uncertainty === k ? 'selected' : ''}>${v}</option>`)
              .join('')}</select></label>`
          : ''
      }
      <label>Forma audible <input id="p-text" value="${esc(e.verbatim_text || '')}" placeholder="solo lo que escuchaste" ${locked ? 'disabled' : ''}></label>
      <label>Hablante <select id="p-spk" ${locked ? 'disabled' : ''}>${[
        ['', '—'],
        ['participant', 'Participante'],
        ['interviewer', 'Entrevistador/a'],
        ['other', 'Otra persona']
      ]
        .map(([k, v]) => `<option value="${k}" ${(e.speaker_id || '') === k ? 'selected' : ''}>${v}</option>`)
        .join('')}</select></label>
      <label>Nota breve <input id="p-note" value="${esc(e.note || '')}" ${locked ? 'disabled' : ''}></label>
      ${['start_ms', 'end_ms']
        .map(
          side => `<label>${side === 'start_ms' ? 'Inicio' : 'Fin'} (ms) <input type="number" data-side="${side}" value="${e[side]}" step="10" ${locked ? 'disabled' : ''}></label>
        ${locked ? '' : `<div class="row nudge">${[-100, -10, 10, 100].map(dx => `<button data-nudge="${side}:${dx}">${dx > 0 ? '+' : ''}${dx}</button>`).join('')}</div>`}`
        )
        .join('')}</div>`
      : ''
    const covered = coverage()
    $('#cov').textContent =
      `Región central revisada: ${Math.round(100 * covered)} %${covered < 1 ? ' — escucha lo que falta y márcalo como revisado.' : ' ✓'}`
    $('#covbar').style.width = `${Math.round(100 * covered)}%`
    if (isCase) renderDiff()
    if (d.suggestions?.length) renderSuggestions()
  }

  function coverage() {
    const spans = [...state.coverage, ...state.events.map(e => [e.start_ms, e.end_ms])]
      .map(([a, b]) => [Math.max(a, t.start_ms), Math.min(b, t.end_ms)])
      .filter(([a, b]) => b > a)
      .sort((x, y) => x[0] - y[0])
    let total = 0
    let cur = t.start_ms
    for (const [a, b] of spans)
      if (b > cur) {
        total += b - Math.max(a, cur)
        cur = b
      }
    return total / (t.end_ms - t.start_ms)
  }

  const iou = (a, b) => {
    const i = Math.max(0, Math.min(a.end_ms, b.end_ms) - Math.max(a.start_ms, b.start_ms))
    return i / (a.end_ms - a.start_ms + (b.end_ms - b.start_ms) - i || 1)
  }
  function renderDiff() {
    const A = d.tracks.A.events
    const B = d.tracks.B.events
    const usedB = new Set()
    const rows = []
    for (const a of A) {
      const b = B.filter(x => !usedB.has(x)).sort((x, y) => iou(a, y) - iou(a, x))[0]
      if (b && iou(a, b) >= 0.5) {
        usedB.add(b)
        rows.push([
          a,
          b,
          a.label === b.label && iou(a, b) >= 0.9
            ? ['Acuerdo', 'ok']
            : a.label !== b.label
              ? ['Clase distinta', 'bad']
              : ['Bordes distintos', 'warn']
        ])
      } else rows.push([a, null, ['Solo en A', 'warn']])
    }
    for (const b of B) if (!usedB.has(b)) rows.push([null, b, ['Solo en B', 'warn']])
    rows.sort((x, y) => (x[0] || x[1]).start_ms - (y[0] || y[1]).start_ms)
    const cell = e => (e ? `${fmt(e.start_ms)}–${fmt(e.end_ms)} ${e.decision === 'event' ? badge(e.label) : DECISIONS[e.decision]}` : '—')
    $('#diff').innerHTML = rows.length
      ? `<thead><tr><th>Diferencia</th><th>A</th><th>B</th><th>Decisión</th></tr></thead>` +
        rows
          .map(
            ([a, b, [k, tone]], i) => `<tr><td><span class="chip ${tone}">${k}</span></td><td>${cell(a)}</td><td>${cell(b)}</td>
      <td class="row">${a ? `<button data-take="${i}:A">Usar A</button>` : ''}${b ? `<button data-take="${i}:B">Usar B</button>` : ''}
      <input data-reason="${i}" placeholder="motivo breve" value="${esc(reasons[`row${i}`] || '')}"></td></tr>`
          )
          .join('')
      : `<tr><td>${empty('A y B no marcaron eventos en esta región. Escúchala y confirma si falta algo.')}</td></tr>`
    renderDiff.rows = rows
  }

  function renderSuggestions() {
    const done = new Set(state.actions.map(a => a.suggestion_id).filter(Boolean))
    $('#sugs').innerHTML = d.suggestions
      .map(
        (s, i) => `<tr><td>${fmt(s.start_ms)}–${fmt(s.end_ms)}</td><td>${s.label ? badge(s.label) : ''}</td>
      <td>${esc(s.text || '')}</td><td class="row">${done.has(s.event_id) ? '<span class="chip ok">Revisado</span>' : `<button data-sugplay="${i}">▶</button>` + ['Correcto', 'Cambiar', 'Uso legítimo', 'Descartar', 'Falta otro evento', 'Tengo dudas'].map(a => `<button data-sug="${i}:${a}">${a}</button>`).join('')}</td></tr>`
      )
      .join('')
  }

  function suggestion(s, act) {
    const ev = {
      event_id: uid(),
      start_ms: s.start_ms,
      end_ms: s.end_ms,
      label: s.label,
      decision: 'event',
      uncertainty: null,
      suggestion_id: s.event_id,
      source_kind: 'human_verified_model_suggestion',
      verbatim_text: null,
      speaker_id: null,
      note: null
    }
    const log = { suggestion_id: s.event_id, before: s }
    if (act === 'Correcto') commit(() => state.events.push(ev), { ...log, type: 'accept', event_id: ev.event_id })
    if (act === 'Cambiar') {
      commit(() => state.events.push(ev), { ...log, type: 'correct', event_id: ev.event_id })
      current = ev.event_id
      draw()
    }
    if (act === 'Uso legítimo')
      commit(
        () =>
          state.contextual.push({ start_ms: s.start_ms, end_ms: s.end_ms, expression: s.text || '', is_disfluent: false, function: '' }),
        { ...log, type: 'reject', reason: 'legitimate_use' }
      )
    if (act === 'Descartar') commit(() => {}, { ...log, type: 'reject', reason: 'not_a_phenomenon' })
    if (act === 'Falta otro evento') {
      ws.setTime(L(s.start_ms))
      commit(() => {}, { ...log, type: 'missing_nearby' })
      say('Marca el evento omitido seleccionando su tramo y eligiendo la clase.', 'info')
    }
    if (act === 'Tengo dudas')
      commit(() => state.events.push({ ...ev, decision: 'uncertain', uncertainty: 'class' }), {
        ...log,
        type: 'doubt',
        event_id: ev.event_id
      })
  }

  view.onclick = e => {
    const b = e.target.closest('button')
    if (!b) return
    const ds = b.dataset
    const ev = id => state.events.find(x => x.event_id === id)
    if (ds.c) classify(ds.c)
    else if (ds.play) {
      const x = ev(ds.play)
      ws.play(Math.max(0, L(x.start_ms) - 1), L(x.end_ms) + 0.5)
    } else if (ds.pick) {
      current = ds.pick
      draw()
    } else if (ds.sugplay) {
      const s = d.suggestions[+ds.sugplay]
      ws.play(Math.max(0, L(s.start_ms) - 1), L(s.end_ms) + 0.5)
    } else if (ds.del)
      commit(
        () => {
          state.events = state.events.filter(x => x.event_id !== ds.del)
        },
        { type: 'delete', event_id: ds.del }
      )
    else if (ds.split) {
      const x = ev(ds.split)
      const at = G(ws.getCurrentTime())
      if (at <= x.start_ms || at >= x.end_ms) return say('Coloca el cursor dentro del evento para dividirlo.')
      const second = { ...x, event_id: uid(), start_ms: at, parent_event_id: x.event_id }
      commit(
        () => {
          x.end_ms = at
          state.events.push(second)
          state.events.sort((a, b) => a.start_ms - b.start_ms)
        },
        { type: 'split', event_id: x.event_id }
      )
    } else if (ds.merge) {
      const i = state.events.findIndex(x => x.event_id === ds.merge)
      const x = state.events[i]
      const y = state.events[i + 1]
      if (!y || y.label !== x.label) return say('Solo se pueden unir eventos consecutivos de la misma clase.')
      commit(
        () => {
          x.end_ms = Math.max(x.end_ms, y.end_ms)
          state.events.splice(i + 1, 1)
        },
        { type: 'merge', event_id: x.event_id, merged: y.event_id }
      )
    } else if (ds.cdel) commit(() => state.contextual.splice(+ds.cdel, 1))
    else if (ds.nudge) {
      const [side, dx] = ds.nudge.split(':')
      const x = ev(current)
      commit(
        () => {
          x[side] += +dx
        },
        { type: 'adjust', event_id: x.event_id }
      )
    } else if (ds.take) {
      const [i, side] = ds.take.split(':')
      const src = renderDiff.rows[+i][side === 'A' ? 0 : 1]
      commit(() => state.events.push({ ...src, event_id: uid(), parent_event_id: src.event_id, source_kind: 'human_adjudicated' }))
      reasons[`row${i}`] = reasons[`row${i}`] || `usar ${side}`
      toast(`Evento de ${side} copiado a la decisión final.`)
    } else if (ds.sug) {
      const [i, act] = ds.sug.split(/:(.+)/)
      suggestion(d.suggestions[+i], act)
    }
  }
  view.onchange = e => {
    const x = state.events.find(y => y.event_id === current)
    const tgt = e.target
    if (tgt.dataset.ci) {
      const c = state.contextual[+tgt.dataset.ci]
      commit(() => {
        c[tgt.dataset.f] = tgt.dataset.f === 'is_disfluent' ? JSON.parse(tgt.value) : tgt.value
      })
    } else if (tgt.dataset.reason) reasons[`row${tgt.dataset.reason}`] = tgt.value
    else if (tgt.dataset.side && x)
      commit(
        () => {
          x[tgt.dataset.side] = Math.round(+tgt.value)
        },
        { type: 'adjust', event_id: x.event_id }
      )
    else if (x && tgt.id?.startsWith('p-')) {
      const field = { 'p-dec': 'decision', 'p-unc': 'uncertainty', 'p-text': 'verbatim_text', 'p-spk': 'speaker_id', 'p-note': 'note' }[
        tgt.id
      ]
      commit(() => {
        x[field] = tgt.value || null
        if (field === 'decision' && x.decision === 'event' && !x.label) x.decision = 'uncertain'
      })
    }
  }
  $('#play').onclick = () => {
    playAsked = performance.now()
    ws.playPause()
  }
  $('#rate').onchange = e => ws.setPlaybackRate(+e.target.value, true)
  $('#zoom').oninput = e => ws.zoom(+e.target.value)
  $('#ctx').onclick = () => {
    extra += 5000
    offset = Math.max(0, t.start_ms - t.context_ms - extra)
    ws.load(audioUrl())
  }
  $('#spec').onclick = () => {
    if (spec) {
      spec.destroy()
      spec = null
    } else spec = ws.registerPlugin(Spectrogram.create({ labels: true, height: 140 }))
  }
  $('#unc').onclick = () => classify(null, 'uncertain')
  $('#noeval').onclick = () => classify(null, 'not_evaluable')
  $('#legit').onclick = () => {
    if (sel.start_ms == null || sel.end_ms == null) return say('Selecciona primero el tramo donde está la expresión.')
    commit(() => state.contextual.push({ start_ms: sel.start_ms, end_ms: sel.end_ms, expression: '', is_disfluent: false, function: '' }))
    sel = {}
    draw()
    toast('Uso legítimo agregado: completa la expresión y su función abajo.')
  }
  $('#cov-all').onclick = () => commit(() => state.coverage.push([t.start_ms, t.end_ms]), { type: 'coverage' })
  $('#cov-cur').onclick = () => commit(() => state.coverage.push([t.start_ms, G(ws.getCurrentTime())]), { type: 'coverage' })
  $('#undo').onclick = () => {
    if (undo.length && !locked) {
      state = JSON.parse(undo.pop())
      draw()
      scheduleSave()
    }
  }
  $('#submit').onclick = submit
  if ($('#asr')) {
    $('#asr').onclick = async () => {
      const words = await api(`/review/tasks/${id}/asr`)
      commit(
        () => {
          state.asr_used = true
        },
        { type: 'asr_shown' }
      )
      $('#words').innerHTML = words.length
        ? '<span class="muted">Transcripción automática sin marcas de disfluencia (su uso queda registrado). Haz clic en una palabra para ir a ese punto:</span> ' +
          words.map(w => `<span data-t="${w.start_ms}">${esc(w.text)}</span>`).join(' ')
        : '<span class="muted">No hay transcripción automática para esta grabación.</span>'
    }
    $('#words').onclick = e => {
      if (e.target.dataset.t) ws.setTime(L(+e.target.dataset.t))
    }
  }
  api(`/guideline/${t.guideline_version}`)
    .then(g => {
      $('#guide').textContent = g.markdown
    })
    .catch(() => {
      $('#guide').textContent = 'Guía no disponible.'
    })
  if (ann)
    api(`/annotations/${ann.id}/history`).then(h => {
      $('#hist').innerHTML = h
        .map(
          x => `Versión ${x.rev} · ${STATUS[x.status]?.[0] || x.status} · ${x.n_events} eventos · ${new Date(x.created).toLocaleString()}`
        )
        .join('<br>')
    })

  function onKey(e) {
    if (e.target.closest('input,textarea,select')) return
    const k = e.key.toLowerCase()
    const mod = e.ctrlKey || e.metaKey
    if (k === ' ') {
      e.preventDefault()
      playAsked = performance.now()
      ws.playPause()
    } else if (k === 'arrowleft' || k === 'arrowright') {
      e.preventDefault()
      ws.setTime(Math.max(0, ws.getCurrentTime() + (k === 'arrowleft' ? -1 : 1)))
    } else if (mod && k === 'enter') submit()
    else if (mod && k === 'z') {
      e.preventDefault()
      $('#undo').click()
    } else if (mod || locked) return
    else if (k === 'i') {
      sel = { start_ms: G(ws.getCurrentTime()), end_ms: sel.end_ms > G(ws.getCurrentTime()) ? sel.end_ms : null }
      current = null
      say('Inicio marcado. Avanza y pulsa <kbd>O</kbd> para marcar el fin.', 'info')
      draw()
    } else if (k === 'o') {
      if (sel.start_ms != null) {
        sel.end_ms = G(ws.getCurrentTime())
        current = null
        say(`Tramo ${fmt(sel.start_ms)}–${fmt(sel.end_ms)}: elige una clase.`, 'info')
        draw()
      }
    } else if (k === 'r') {
      if (sel.end_ms) ws.play(Math.max(0, L(sel.start_ms) - 0.5), L(sel.end_ms) + 0.5)
    } else if (/^[1-7]$/.test(k)) classify(CLASSES[+k - 1][0])
    else if (k === 'u') classify(null, 'uncertain')
    else if (k === 'delete' && current) $(`[data-del="${current}"]`)?.click()
    else return
    activity()
  }
  document.addEventListener('keydown', onKey)

  // Autosave: confirmed by the server only; unsynced edits survive reloads locally.
  let timer = null
  let retry = null
  let saving = false
  let again = false
  function scheduleSave() {
    if (isCase || locked) return
    op = uid()
    local.set(pendingKey, { rev, state, op })
    status('Pendiente de sincronizar', 'warn')
    clearTimeout(timer)
    timer = setTimeout(save, 500)
  }
  async function save() {
    if (saving) {
      again = true
      return
    }
    saving = true
    clearTimeout(retry)
    status('Guardando…', 'info')
    const sentOp = op
    const began = performance.now()
    try {
      const r = await api(`/annotations/${ann.id}`, {
        method: 'PATCH',
        body: { ...state, client_op: op },
        headers: { 'If-Match': String(rev) }
      })
      rev = r.rev
      tel('save_ms', performance.now() - began)
      if (sentOp === op) {
        local.del(pendingKey)
        status('Guardado ✓', 'ok')
      }
    } catch (e) {
      if (e.status === 409) {
        status('Conflicto', 'bad')
        say(
          'Otra pestaña o sesión guardó cambios en esta tarea. <button id="c-reload">Usar la versión del servidor</button> <button id="c-keep">Conservar mis cambios</button>',
          'bad'
        )
        $('#c-reload').onclick = () => {
          local.del(pendingKey)
          route()
        }
        $('#c-keep').onclick = async () => {
          rev = (await api(`/review/tasks/${id}`)).annotation.rev
          say('')
          save()
        }
      } else {
        status('Sin conexión: reintentando', 'warn')
        retry = setTimeout(save, 5000)
      }
    }
    saving = false
    if (again) {
      again = false
      save()
    }
  }
  const pending = ann && local.get(pendingKey)
  if (pending && !locked) {
    if (pending.rev === rev) {
      state = pending.state
      op = pending.op
      save()
    } else
      say(
        'Hay cambios guardados en este navegador que se hicieron sobre una versión anterior. <button id="p-apply">Aplicarlos sobre la versión actual</button> <button id="p-drop">Descartarlos</button>',
        'bad'
      )
    $('#p-apply') &&
      ($('#p-apply').onclick = () => {
        state = pending.state
        say('')
        draw()
        scheduleSave()
      })
    $('#p-drop') &&
      ($('#p-drop').onclick = () => {
        local.del(pendingKey)
        say('')
      })
  }

  async function submit() {
    if (locked) return
    activity()
    try {
      if (isCase) {
        await api('/adjudications', {
          method: 'POST',
          body: {
            case_id: id,
            events: state.events,
            contextual: state.contextual,
            reasons,
            based_on: { A: d.tracks.A.rev, B: d.tracks.B.rev }
          }
        })
        toast('Adjudicación guardada.')
      } else {
        clearTimeout(timer)
        await save()
        if (local.get(pendingKey))
          return say('Aún hay cambios sin confirmar por el servidor; espera a ver «Guardado ✓» y vuelve a intentarlo.')
        await api(`/review/tasks/${id}/submit`, { method: 'POST' })
        toast('Tarea finalizada. ¡Gracias!')
      }
      location.hash = '#/review'
    } catch (e) {
      const gaps = e.detail?.unreviewed_gaps_ms
      say(
        gaps
          ? `Aún no se puede finalizar: faltan tramos de la región central por revisar (${gaps.map(([a, b]) => `${fmt(a)}–${fmt(b)}`).join(', ')}). Escúchalos y pulsa «Revisado: sin más eventos».${e.detail.outside_context.length ? ' También hay eventos fuera del audio de contexto.' : ''}`
          : esc(e.message),
        'bad'
      )
    }
  }

  // Prefetch the next assigned fragment without autoplay.
  if (!isCase && !locked)
    api('/review/tasks').then(x => {
      const n = x.tasks.find(y => y.id !== id && y.status !== 'submitted')
      if (n) fetch(mediaUrl(`/review/tasks/${n.id}/audio`))
    })
  cleanup = () => {
    document.removeEventListener('keydown', onKey)
    clearTimeout(timer)
    clearTimeout(retry)
    ws.destroy()
  }
}

// ----------------------------------------------------------------- product
async function playground() {
  const list = await api('/recordings')
  view.innerHTML = `${intro(
    'playground',
    'Prueba del producto',
    'Aquí pruebas el motor MRCD con audio nuevo. El análisis corre en segundo plano: puedes salir de la página y volver al resultado. Lo que ves es la salida automática del modelo; no reemplaza la revisión humana ni mide exactitud.',
    [
      'Elige de dónde viene el audio: grabarlo, subir un archivo o usar un ejemplo.',
      'Espera a que se prepare el audio (conversión a 16 kHz mono).',
      'Pulsa <b>Analizar con MRCD</b> y sigue el progreso.',
      'Escucha cada detección, filtra por clase, exporta y deja tu comentario.'
    ]
  )}
    <div class="cards three">
      <button class="card choice" id="rec"><h2>🎙 Grabar</h2><p>Usa el micrófono de este equipo. Recomendado: 20–60 s hablando con naturalidad, con auriculares.</p></button>
      <label class="card choice" id="drop"><h2>⤒ Subir archivo</h2><p>Arrastra aquí o haz clic. WAV, MP3, M4A, WebM, OGG o MP4 (se usa solo el audio).</p><input type="file" id="file" accept="audio/*,video/*" hidden></label>
      <button class="card choice" id="ex"><h2>▤ Usar un ejemplo</h2><p>Abre una grabación ya cargada, como el piloto, con resultados precalculados.</p></button>
    </div>
    <p class="muted">MRCD necesita al menos 13 s de audio porque mira 10 s de contexto para decidir cada tramo.</p>
    <section id="stage"></section>`
  $('#file').onchange = e => e.target.files[0] && upload(e.target.files[0], e.target.files[0].name)
  const drop = $('#drop')
  drop.ondragover = e => {
    e.preventDefault()
    drop.classList.add('over')
  }
  drop.ondragleave = () => drop.classList.remove('over')
  drop.ondrop = e => {
    e.preventDefault()
    drop.classList.remove('over')
    const f = e.dataTransfer.files[0]
    if (f) upload(f, f.name)
  }
  const showList = () => {
    $('#stage').innerHTML = `<h2>Grabaciones disponibles</h2>${
      list.length
        ? `<div class="scroll"><table><thead><tr><th>Nombre</th><th>Origen</th><th>Duración</th><th>Estado</th><th></th></tr></thead><tbody>
      ${list
        .map(
          r => `<tr><td>${esc(r.filename)}</td><td>${r.example ? 'Ejemplo' : 'Subida'}</td><td>${r.duration_ms ? fmt(r.duration_ms) : '—'}</td><td>${chip(r.status)}</td>
        <td><a class="button primary" href="#/playground/${r.id}">Abrir</a></td></tr>`
        )
        .join('')}</tbody></table></div>`
        : empty('Todavía no hay grabaciones. Graba o sube un archivo para empezar.')
    }`
  }
  $('#ex').onclick = showList
  $('#rec').onclick = recorder
  if (list.length) showList()
}

async function upload(blob, filename, capture) {
  $('#stage').innerHTML = `<div class="banner info">Subiendo «${esc(filename)}»…</div>`
  const q = new URLSearchParams({ filename, ...(capture ? { capture: JSON.stringify(capture) } : {}) })
  const r = await fetch(`/api/recordings?${q}`, { method: 'POST', body: blob, headers: { Authorization: `Bearer ${token()}` } })
  const d = await r.json()
  if (!r.ok)
    return ($('#stage').innerHTML =
      `<div class="banner bad">No se pudo subir: ${esc(typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail))}</div>`)
  if (d.duplicate) toast('Ese archivo ya estaba cargado; se abre la grabación existente.', 'warn')
  location.hash = `#/playground/${d.id}`
}

async function recorder() {
  const stage = $('#stage')
  const constraints = id => ({
    audio: {
      deviceId: id ? { exact: id } : undefined,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1
    }
  })
  let stream
  let rec
  let chunks = []
  let started = 0
  let raf
  let blob
  let ctx
  let low = 0
  let clip = 0
  let frames = 0
  stage.innerHTML = `<div class="card"><h2>Grabar con el micrófono</h2>
    <p id="rec-state" class="banner info">Solicitando permiso para usar el micrófono…</p>
    <div class="form two"><label>Micrófono <select id="dev"><option>Esperando permiso…</option></select></label>
      <div><span class="muted">Nivel de entrada</span><div class="meter big"><i id="lvl"></i></div></div></div>
    <p class="clock" id="clock">0:00.00</p>
    <div class="row"><button id="start" class="primary" disabled>● Iniciar grabación</button><button id="stop" disabled>■ Detener</button></div>
    <div id="after" hidden><h2>Escucha tu grabación</h2><audio id="preview" controls></audio><p id="rec-msg"></p>
      <div class="row"><button id="use" class="primary">Analizar esta grabación</button><button id="again">Grabar de nuevo</button></div></div>
    <details><summary>Ajustes de captura</summary><p id="settings" class="muted"></p>
      <p class="muted">Se pide desactivar la cancelación de eco, la reducción de ruido y la ganancia automática para no alterar la señal; el navegador puede ignorar el pedido y aquí se muestra lo que aplicó realmente. Usa auriculares para evitar que el altavoz se grabe.</p></details></div>`
  const open = async id => {
    stream?.getTracks().forEach(tr => tr.stop())
    ctx?.close()
    cancelAnimationFrame(raf)
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints(id))
    } catch (e) {
      $('#rec-state').className = 'banner bad'
      $('#rec-state').innerHTML =
        `No se pudo abrir el micrófono (${esc(e.message)}). Revisa el permiso del navegador; la grabación requiere <code>localhost</code> o HTTPS.`
      return
    }
    const s = stream.getAudioTracks()[0].getSettings()
    $('#settings').textContent =
      `Aplicado: cancelación de eco ${s.echoCancellation ? 'sí' : 'no'}, reducción de ruido ${s.noiseSuppression ? 'sí' : 'no'}, ganancia automática ${s.autoGainControl ? 'sí' : 'no'}, ${s.sampleRate || '?'} Hz, ${s.channelCount || '?'} canal(es).`
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(x => x.kind === 'audioinput')
    $('#dev').innerHTML = devices
      .map(x => `<option value="${x.deviceId}" ${x.deviceId === s.deviceId ? 'selected' : ''}>${esc(x.label || 'Micrófono')}</option>`)
      .join('')
    $('#rec-state').className = 'banner ok'
    $('#rec-state').textContent = 'Micrófono listo. Habla un momento para ver el nivel y pulsa «Iniciar grabación».'
    $('#start').disabled = false
    ctx = new AudioContext()
    const an = ctx.createAnalyser()
    an.fftSize = 2048
    ctx.createMediaStreamSource(stream).connect(an)
    const buf = new Float32Array(an.fftSize)
    const tick = () => {
      an.getFloatTimeDomainData(buf)
      let peak = 0
      let sum = 0
      for (const x of buf) {
        peak = Math.max(peak, Math.abs(x))
        sum += x * x
      }
      const db = 20 * Math.log10(Math.sqrt(sum / buf.length) || 1e-9)
      $('#lvl').style.width = `${Math.min(100, Math.max(0, ((db + 60) * 100) / 60))}%`
      $('#lvl').className = peak > 0.99 ? 'bad' : db < -45 ? 'warn' : ''
      if (rec?.state === 'recording') {
        frames++
        if (db < -45) low++
        if (peak > 0.99) clip++
        $('#clock').textContent = fmt(Date.now() - started)
      }
      raf = requestAnimationFrame(tick)
    }
    tick()
  }
  cleanup = () => {
    cancelAnimationFrame(raf)
    stream?.getTracks().forEach(tr => tr.stop())
    ctx?.close()
  }
  $('#dev').onchange = e => open(e.target.value)
  $('#start').onclick = () => {
    chunks = []
    low = clip = frames = 0
    rec = new MediaRecorder(stream)
    rec.ondataavailable = e => chunks.push(e.data)
    rec.onstop = () => {
      blob = new Blob(chunks, { type: rec.mimeType })
      const secs = (Date.now() - started) / 1000
      $('#preview').src = URL.createObjectURL(blob)
      $('#after').hidden = false
      const notes = []
      if (secs < 13)
        notes.push(`<span class="error">Duración ${secs.toFixed(1)} s: MRCD necesita al menos 13 s. Graba de nuevo un poco más.</span>`)
      else if (secs < 20) notes.push('<span class="warn">Funciona, pero se recomiendan 20–60 s para tener contexto suficiente.</span>')
      if (clip > frames * 0.01)
        notes.push('<span class="warn">La señal se saturó: baja el volumen o aléjate del micrófono y repite.</span>')
      if (low > frames * 0.8) notes.push('<span class="warn">La señal es muy débil: acércate al micrófono y repite.</span>')
      $('#rec-msg').innerHTML = notes.join('<br>') || '<span class="ok">La grabación se ve bien.</span>'
      $('#use').disabled = secs < 13
      $('#rec-state').className = 'banner ok'
      $('#rec-state').textContent = `Grabación lista (${secs.toFixed(1)} s).`
    }
    rec.start()
    started = Date.now()
    $('#rec-state').className = 'banner bad'
    $('#rec-state').textContent = '● Grabando… habla con naturalidad. Pulsa «Detener» al terminar.'
    $('#start').disabled = true
    $('#stop').disabled = false
    $('#after').hidden = true
  }
  $('#stop').onclick = () => {
    rec.stop()
    $('#start').disabled = false
    $('#stop').disabled = true
  }
  $('#again').onclick = () => {
    $('#after').hidden = true
    $('#clock').textContent = '0:00.00'
    $('#start').click()
  }
  $('#use').onclick = () => {
    const s = stream.getAudioTracks()[0].getSettings()
    upload(blob, `grabacion.${rec.mimeType.includes('mp4') ? 'mp4' : rec.mimeType.includes('ogg') ? 'ogg' : 'webm'}`, {
      settings: s,
      requested: constraints().audio,
      mimeType: rec.mimeType,
      userAgent: navigator.userAgent
    })
  }
  open()
}

async function player(recId, rec) {
  const pps = rec.data.media.analysis.duration_ms > 600000 ? 10 : 100
  const p = await api(`/recordings/${recId}/peaks?pps=${pps}`)
  const ws = WaveSurfer.create({
    container: '#wave',
    ...WAVE,
    url: mediaUrl(`/recordings/${recId}/media`),
    peaks: p.peaks,
    duration: p.duration_ms / 1000,
    minPxPerSec: 40,
    plugins: [Timeline.create()]
  })
  return { ws, regions: ws.registerPlugin(Regions.create()) }
}

async function results(recId) {
  let rec = await api(`/recordings/${recId}`)
  while (['queued', 'preparing'].includes(rec.status)) {
    view.innerHTML = `<h1>Preparando el audio…</h1><div class="banner info">Se está convirtiendo a 16 kHz mono y calculando la forma de onda. Esto toma unos segundos; el original se conserva sin cambios. <b>¿Tarda mucho?</b> Verifica que el worker esté corriendo (<code>python -m app.worker</code>).</div>`
    await new Promise(r => setTimeout(r, 1000))
    if (!location.hash.endsWith(recId)) return
    rec = await api(`/recordings/${recId}`)
  }
  if (rec.status === 'failed') {
    view.innerHTML = `<div class="banner bad"><b>No se pudo preparar el audio.</b> ${esc(rec.data.error)}</div><a class="button" href="#/playground">Volver</a>`
    return
  }
  const researcher = ['researcher', 'admin'].includes(me.role)
  const segs = rec.data.segments
  view.innerHTML = `${intro(
    'results',
    esc(rec.data.filename),
    'Resultado del análisis automático de esta grabación. Cada franja de color sobre la onda es una detección; haz clic en ella para escucharla con 1 s de contexto. Las pausas se cuentan aparte porque no son disfluencias.',
    [
      'Pulsa <b>Analizar con MRCD</b> si aún no hay resultados (o para repetirlo).',
      'Revisa el resumen y filtra por clase o tipo.',
      'Pulsa «Detalle» para escuchar y ver la evidencia de un evento.',
      'Marca «Está bien», «No corresponde» o «Falta algo en el cursor»: tus comentarios se revisan después, no cambian el modelo al instante.'
    ]
  )}
    <p class="muted">Duración ${fmt(rec.data.media.analysis.duration_ms)} · ${segs.length} segmento(s) de análisis de 30 s</p>
    <div class="card"><h2>Analizar</h2><div class="row"><button id="run" class="primary">Analizar con MRCD</button>
      ${researcher ? '<label class="inline"><input type="checkbox" id="w-rules"> Comparar con reglas</label><label class="inline"><input type="checkbox" id="w-gpt"> Comparar con GPT (servicio externo)</label>' : ''}</div>
      <p id="gpt-note" class="muted"></p><div id="progress"></div></div>
    ${legend()}
    <div id="wave"></div><div class="row"><button id="play" class="primary">▶ Reproducir / pausar</button><button id="missing">Falta algo en el cursor</button></div>
    <div id="out"></div>`
  const { ws, regions } = await player(recId, rec)
  $('#play').onclick = () => ws.playPause()
  if ($('#w-gpt'))
    $('#w-gpt').onchange = e => {
      $('#gpt-note').textContent = e.target.checked
        ? `Se enviarán ${segs.length} segmento(s) (${fmt(segs.reduce((a, s) => a + s.audio_end_ms - s.audio_start_ms, 0))} de audio con contexto) a OpenAI. El costo se registra por tokens al terminar. Requiere que el proyecto autorice el envío externo.`
        : ''
    }
  let runs = await api(`/analysis-runs?recording_id=${recId}`)
  let shown = null
  const latest = () => Object.values(Object.fromEntries(runs.map(r => [r.data.system, r])))

  function render(run) {
    shown = run
    const res = run.data.result || {}
    const events = res.events || []
    const f = { label: $('#f-label')?.value || '', kind: $('#f-kind')?.value || '' }
    const visible = events.filter(
      e =>
        (!f.label || e.label === f.label) &&
        (!f.kind ||
          (f.kind === 'pause'
            ? PAUSES.has(e.label)
            : f.kind === 'uncertain'
              ? e.decision === 'uncertain'
              : !PAUSES.has(e.label) && e.decision === 'event'))
    )
    const words = res.words || []
    const n = {
      dis: events.filter(e => e.decision === 'event' && !PAUSES.has(e.label)).length,
      pau: events.filter(e => e.decision === 'event' && PAUSES.has(e.label)).length,
      unc: events.filter(e => e.decision === 'uncertain').length
    }
    regions.clearRegions()
    visible.forEach(
      e =>
        e.label &&
        regions.addRegion({
          id: e.event_id,
          start: e.start_ms / 1000,
          end: e.end_ms / 1000,
          color: CLS[e.label].color + '44',
          drag: false,
          resize: false,
          content: CLS[e.label].short
        })
    )
    $('#out').innerHTML = `<h2>Resultados por sistema</h2><div class="tabs">${latest()
      .map(
        r =>
          `<button data-show="${r.id}" class="${r.id === run.id ? 'active' : ''}">${SYSTEM[r.data.system] || r.data.system} ${chip(r.status)}</button>`
      )
      .join('')}
      ${researcher ? `<a class="button" href="#/experiments/runs/${run.data.comparison}">Comparar sistemas →</a>` : ''}</div>
      ${run.data.importer ? `<div class="banner info">Resultado precalculado e importado (${esc(run.data.importer)}, ${new Date(run.created).toLocaleString()}); no es inferencia en vivo.</div>` : ''}
      ${res.warning ? `<div class="banner warn">${esc(res.warning)}</div>` : ''}
      ${run.data.error ? `<div class="banner bad"><b>El análisis falló</b> en la etapa «${esc(STAGES[run.data.error.stage] || run.data.error.stage)}»: ${esc(run.data.error.message)}</div>` : ''}
      ${tiles([
        ['Disfluencias', n.dis, 'muletillas, repeticiones, etc.'],
        ['Pausas', n.pau, 'no son disfluencias'],
        ['Inciertos', n.unc, 'pendientes de revisión'],
        [
          'Tiempo de análisis',
          res.timings?.total_s != null ? `${res.timings.total_s.toFixed(1)} s` : '—',
          res.rtf ? `${res.rtf.toFixed(2)} × la duración` : ''
        ]
      ])}
      <div class="row"><label class="inline">Clase <select id="f-label"><option value="">todas</option>${CLASSES.map(([c, nm]) => `<option value="${c}" ${f.label === c ? 'selected' : ''}>${nm}</option>`).join('')}</select></label>
        <label class="inline">Tipo <select id="f-kind">${[
          ['', 'todo'],
          ['dis', 'solo disfluencias'],
          ['pause', 'solo pausas'],
          ['uncertain', 'solo inciertos']
        ]
          .map(([k, v]) => `<option value="${k}" ${f.kind === k ? 'selected' : ''}>${v}</option>`)
          .join('')}</select></label>
        <span class="spacer"></span><span class="muted">Descargar:</span>
        <a class="button" href="${mediaUrl(`/exports/${run.id}?format=json`)}" download>JSON</a><a class="button" href="${mediaUrl(`/exports/${run.id}?format=csv`)}" download>CSV</a><a class="button" href="${mediaUrl(`/exports/${run.id}?format=md`)}" download>Reporte</a></div>
      ${
        words.length
          ? `<details open><summary>Transcripción automática (ASR, no verificada)</summary><p class="words">${words
              .map(w => {
                const hit = events.find(e => e.decision === 'event' && e.start_ms <= w.start * 1000 && w.start * 1000 < e.end_ms)
                return `<span data-t="${w.start * 1000}" class="${hit ? 'hit' : ''}" style="${hit ? `text-decoration-color:${CLS[hit.label].color}` : ''}" title="${hit ? CLS[hit.label].name : ''}">${esc(w.text)}</span>`
              })
              .join(' ')}</p></details>`
          : ''
      }
      <p class="muted small">Las puntuaciones no están calibradas: no son probabilidades de acierto.</p>
      ${
        visible.length
          ? `<div class="scroll list"><table><thead><tr><th>Tramo</th><th>Clase</th><th>Decisión</th><th>Texto disponible</th><th></th></tr></thead><tbody>
      ${visible
        .map(
          e => `<tr><td>${fmt(e.start_ms)}–${fmt(e.end_ms)}</td><td>${e.label ? badge(e.label) : '—'}</td><td>${DECISIONS[e.decision] || e.decision}</td>
        <td>${esc(e.text || '')} <span class="muted">${e.text_scope === 'window_context' ? '(texto de la ventana)' : e.text_scope === 'model_proposal' ? '(propuesta del modelo)' : ''}</span></td>
        <td class="row"><button data-ev="${e.event_id}">▶ Detalle</button><button data-fb="${e.event_id}:ok">Está bien</button><button data-fb="${e.event_id}:wrong">No corresponde</button></td></tr>`
        )
        .join('')}</tbody></table></div>`
          : empty(
              events.length
                ? 'Ningún evento coincide con el filtro.'
                : run.status === 'succeeded'
                  ? 'El sistema no detectó eventos en este audio.'
                  : 'Sin eventos para mostrar.'
            )
      }
      <div id="detail"></div>`
  }

  view.onclick = async e => {
    const b = e.target.closest('button,[data-t]')
    if (!b) return
    const ds = b.dataset
    if (ds.show) render(runs.find(r => r.id === ds.show))
    if (ds.t) ws.setTime(+ds.t / 1000)
    if (ds.ev) {
      const x = shown.data.result.events.find(y => y.event_id === ds.ev)
      ws.play(Math.max(0, x.start_ms / 1000 - 1), x.end_ms / 1000 + 0.5)
      $('#detail').innerHTML = `<div class="card"><h2>${x.label ? badge(x.label) : ''} ${fmt(x.start_ms)}–${fmt(x.end_ms)}</h2>
        <dl><dt>Evidencia observable</dt><dd>${esc(x.evidence == null ? 'sin detalle' : typeof x.evidence === 'object' ? JSON.stringify(x.evidence) : x.evidence)}</dd>
        <dt>Ventanas de 3 s que lo sustentan</dt><dd>${x.n_windows ?? '—'}</dd><dt>Puntuación</dt><dd>${x.score?.toFixed?.(2) ?? '—'} (${esc(x.score_kind || 'sin puntuación')})</dd><dt>Segmento</dt><dd>${esc(x.segment_id)}</dd></dl>
        <p class="muted">Se describe evidencia acústica o textual, no intención ni estado psicológico.</p>
        ${researcher ? `<button data-send="${x.event_id}">Enviar este caso a revisión (desarrollo)</button>` : ''}</div>`
      $('#detail').scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
    if (ds.fb) {
      const [eid, kind] = ds.fb.split(':')
      await api('/feedback', { method: 'POST', body: { run_id: shown.id, event_id: eid, kind } })
      toast('Comentario registrado para revisión.')
    }
    if (ds.send)
      sendToReview(
        recId,
        shown,
        shown.data.result.events.find(y => y.event_id === ds.send)
      )
  }
  view.onchange = e => {
    if (e.target.id?.startsWith('f-') && shown) render(shown)
  }
  regions.on('region-clicked', (r, ev) => {
    ev.stopPropagation()
    $(`[data-ev="${r.id}"]`)?.click()
  })
  $('#missing').onclick = async () => {
    if (!shown) return toast('Primero analiza o abre un resultado.', 'warn')
    await api('/feedback', { method: 'POST', body: { run_id: shown.id, kind: 'missing', time_ms: Math.round(ws.getCurrentTime() * 1000) } })
    toast(`Omisión registrada en ${fmt(ws.getCurrentTime() * 1000)}.`)
  }
  $('#run').onclick = async () => {
    const systems = ['mrcd', ...($('#w-rules')?.checked ? ['rules'] : []), ...($('#w-gpt')?.checked ? ['gpt'] : [])]
    $('#run').disabled = true
    try {
      const r = await api('/analysis-runs', { method: 'POST', body: { recording_id: recId, systems } })
      await Promise.all(r.runs.map(follow))
      runs = await api(`/analysis-runs?recording_id=${recId}`)
      render(runs.find(x => x.id === r.runs[0].id))
    } finally {
      $('#run').disabled = false
    }
  }
  async function follow(run) {
    const began = Date.now()
    return new Promise(resolve => {
      const line = document.createElement('div')
      line.className = 'progress-line'
      $('#progress').append(line)
      let last = { status: run.status }
      const show = () => {
        line.innerHTML = `<b>${SYSTEM[run.system]}</b> ${chip(last.status)} ${TERMINAL.includes(last.status) ? '' : `${STAGES[last.stage] || (last.stage?.startsWith('gpt:') ? `GPT, segmento ${last.stage.slice(4)}` : 'esperando al worker')} · ${Math.round((Date.now() - began) / 1000)} s <button data-cancel="${run.id}">Cancelar</button>`}`
      }
      const done = () => {
        es.close()
        clearInterval(clock)
        show()
        resolve()
      }
      const clock = setInterval(show, 1000)
      const es = new EventSource(mediaUrl(`/analysis-runs/${run.id}/stream`))
      es.onmessage = m => {
        last = JSON.parse(m.data)
        show()
        if (TERMINAL.includes(last.status)) done()
      }
      es.onerror = async () => {
        const r = await api(`/analysis-runs/${run.id}`)
        last = { status: r.status, stage: r.data.stage }
        if (TERMINAL.includes(r.status)) done()
      }
      line.onclick = async e => {
        if (e.target.dataset.cancel) {
          const r = await api(`/analysis-runs/${run.id}/cancel`, { method: 'POST' })
          toast(`Cancelación: ${STATUS[r.status]?.[0] || r.status}.`, 'warn')
        }
      }
      show()
    })
  }
  const first = runs.find(r => r.data.system === 'mrcd' && r.status !== 'queued') || runs[0]
  if (first) render(first)
  else $('#out').innerHTML = empty('Esta grabación aún no tiene análisis. Pulsa «Analizar con MRCD».')
  cleanup = () => ws.destroy()
}

async function sendToReview(recId, run, e) {
  try {
    const mid = (e.start_ms + e.end_ms) / 2
    await api('/review/tasks', {
      method: 'POST',
      body: {
        recording_id: recId,
        start_ms: Math.max(0, Math.round(mid - 7500)),
        end_ms: Math.round(mid + 7500),
        mode: 'assisted',
        split: 'dev',
        run_id: run.id
      }
    })
    toast('Caso enviado a la cola de revisión asistida (desarrollo). Una persona anotadora lo tomará con «Continuar».')
  } catch (err) {
    toast(esc(err.message), 'bad')
  }
}

// --------------------------------------------------------------- research
async function experiments() {
  const d = await api('/experiments')
  const byCmp = {}
  for (const r of d.runs) (byCmp[r.comparison] ||= []).push(r)
  const usable = d.runs.filter(r => ['succeeded', 'partial'].includes(r.status))
  view.innerHTML = `${intro(
    'experiments',
    'Experimentos',
    'Espacio del investigador. Una <b>comparación</b> agrupa sistemas ejecutados sobre el mismo audio con el mismo protocolo. Un <b>conjunto congelado</b> (snapshot) es una versión inmutable de las anotaciones humanas adjudicadas. La <b>evaluación</b> mide cada sistema contra ese conjunto: solo cuenta el audio revisado, empareja eventos uno a uno (misma clase, IoU ≥ 0,5) y reporta las pausas aparte.',
    [
      'Abre una comparación para escuchar dónde discrepan los sistemas.',
      'Congela un conjunto humano (Administración de datos).',
      'Elige el conjunto y las ejecuciones y pulsa <b>Evaluar</b>. Evalúa el conjunto de prueba solo al final: cada evaluación queda registrada.'
    ]
  )}
    <h2>Comparaciones</h2>${
      Object.keys(byCmp).length
        ? `<div class="scroll"><table><thead><tr><th>Grabación</th><th>Sistemas</th><th>Fecha</th><th></th></tr></thead><tbody>
      ${Object.entries(byCmp)
        .map(
          ([
            cid,
            rs
          ]) => `<tr><td>${esc(rs[0].filename)}</td><td>${rs.map(r => `${SYSTEM[r.system] || r.system} ${chip(r.status)}`).join(' ')}</td><td>${new Date(rs[0].created).toLocaleString()}</td>
        <td><a class="button primary" href="#/experiments/runs/${cid}">Abrir</a></td></tr>`
        )
        .join('')}</tbody></table></div>`
        : empty('Aún no hay ejecuciones. Analiza una grabación en <a href="#/playground">Prueba del producto</a>.')
    }
    <h2>Conjuntos congelados</h2>
    ${
      me.role === 'admin'
        ? `<form id="snap" class="row card"><label class="inline">Nombre <input name="name" required placeholder="p. ej. desarrollo-v1"></label>
      <label class="inline">Particiones ${Object.entries(SPLIT)
        .map(
          ([k, v]) =>
            `<label class="inline"><input type="checkbox" name="split" value="${k}" ${k !== 'test' ? 'checked' : ''}> ${v}</label>`
        )
        .join('')}</label>
      <label class="inline">Origen <select name="include"><option value="adjudicated">anotaciones adjudicadas</option><option value="submitted">anotaciones individuales finalizadas</option></select></label>
      <button class="primary">Congelar conjunto</button></form>`
        : ''
    }
    ${d.snapshots.length ? `<table><thead><tr><th>Nombre</th><th>Regiones</th><th>Particiones</th><th>Fecha</th></tr></thead><tbody>${d.snapshots.map(s => `<tr><td>${esc(s.name)} <span class="muted small">${s.id}</span></td><td>${s.items}</td><td>${s.splits.map(x => SPLIT[x] || x).join(', ')}</td><td>${new Date(s.created).toLocaleString()}</td></tr>`).join('')}</tbody></table>` : empty('No hay conjuntos congelados. Se crean después de adjudicar anotaciones.')}
    <h2>Evaluar</h2>
    ${
      d.snapshots.length && usable.length
        ? `<form id="eval" class="card form"><label>Conjunto de referencia <select name="snapshot">${d.snapshots.map(s => `<option value="${s.id}">${esc(s.name)}</option>`).join('')}</select></label>
      <fieldset><legend>Ejecuciones a evaluar (se agrupan por sistema)</legend>${usable.map(r => `<label class="inline"><input type="checkbox" name="run" value="${r.id}" data-system="${r.system}"> ${SYSTEM[r.system] || r.system} · ${esc(r.filename)} ${chip(r.status)}</label>`).join('<br>')}</fieldset>
      <button class="primary">Evaluar</button></form>`
        : empty('Para evaluar necesitas al menos un conjunto congelado y una ejecución completada.')
    }
    <div id="eval-out"></div>
    <h2>Evaluaciones anteriores</h2>${d.evaluations.length ? d.evaluations.slice().reverse().map(evalTable).join('') : empty('Todavía no hay evaluaciones.')}`
  $('#snap') &&
    ($('#snap').onsubmit = async e => {
      e.preventDefault()
      const f = new FormData(e.target)
      try {
        await api('/dataset-snapshots', {
          method: 'POST',
          body: { name: f.get('name'), include: f.get('include'), splits: f.getAll('split') }
        })
        toast('Conjunto congelado.')
        route()
      } catch (err) {
        toast(
          err.detail?.leakage
            ? `Fuga detectada: ${esc(Object.keys(err.detail.leakage).join(', '))} aparece en más de una partición.`
            : esc(err.message),
          'bad'
        )
      }
    })
  $('#eval') &&
    ($('#eval').onsubmit = async e => {
      e.preventDefault()
      const systems = {}
      for (const c of e.target.querySelectorAll('input[name=run]:checked')) (systems[c.dataset.system] ||= []).push(c.value)
      if (!Object.keys(systems).length) return toast('Elige al menos una ejecución.', 'warn')
      $('#eval-out').innerHTML = '<div class="banner info">Evaluando (incluye intervalos de confianza por remuestreo de hablantes)…</div>'
      try {
        const r = await api('/evaluation-runs', {
          method: 'POST',
          body: { snapshot_id: new FormData(e.target).get('snapshot'), systems, bootstrap: 300 }
        })
        $('#eval-out').innerHTML = evalTable({ ...r.data, created: r.created, prior: r.data.prior_evaluations_on_snapshot })
      } catch (err) {
        $('#eval-out').innerHTML = `<div class="banner bad">${esc(err.message)}</div>`
      }
    })
}

function evalTable(x) {
  const rows = Object.entries(x.results)
    .map(([name, r]) => {
      const dis = r.micro.disfluency || {}
      const ci = x.bootstrap_ci?.[name]
      return `<tr><td><b>${SYSTEM[name] || name}</b></td><td>${pct(dis.precision)}</td><td>${pct(dis.recall)}</td><td>${num(dis.f1)}${ci?.low != null ? ` <span class="muted small">[${num(ci.low)}–${num(ci.high)}]</span>` : ''}</td>
      <td>${num(r.micro.pause?.f1)}</td><td>${r.n_ref}</td><td>${r.n_hyp}</td><td>${r.n_uncertain}</td><td>${r.false_alarms_per_min == null ? '—' : r.false_alarms_per_min.toFixed(1)}</td></tr>`
    })
    .join('')
  const perClass = Object.entries(x.results)
    .map(
      ([name, r]) =>
        `<tr><td>${SYSTEM[name] || name}</td>${CLASSES.map(([c]) => `<td>${num(r.per_class[c]?.f1)} <span class="muted small">n=${r.per_class[c]?.support ?? 0}</span></td>`).join('')}</tr>`
    )
    .join('')
  return `<div class="card"><p class="muted">${new Date(x.created).toLocaleString()} · conjunto ${esc(x.snapshot_id)}${x.prior ? ` · <span class="warn">este conjunto ya tenía ${x.prior} evaluación(es) previas</span>` : ''}</p>
    <div class="scroll"><table><thead><tr><th>Sistema</th><th>Precisión disfl.</th><th>Exhaustividad disfl.</th><th>F1 disfl. [IC 95 %]</th><th>F1 pausas</th><th>Eventos ref.</th><th>Eventos sistema</th><th>Inciertos</th><th>Falsas alarmas/min</th></tr></thead><tbody>${rows}</tbody></table></div>
    <details><summary>F1 por clase</summary><div class="scroll"><table><thead><tr><th></th>${CLASSES.map(([c]) => `<th>${badge(c)}</th>`).join('')}</tr></thead><tbody>${perClass}</tbody></table></div></details></div>`
}

async function comparison(cid) {
  const c = await api(`/experiments/comparisons/${cid}`)
  const recId = c.runs[0].parent
  const rec = await api(`/recordings/${recId}`)
  const lanes = [
    ...c.runs.map(r => ({
      name: `${SYSTEM[r.data.system] || r.data.system}`,
      status: r.status,
      events: r.data.result?.events || [],
      run: r
    })),
    ...c.reference.map(a => ({ name: 'Referencia humana', status: 'closed', events: a.data.events }))
  ]
  const dur = rec.data.media.analysis.duration_ms
  const counts = lanes.map(l =>
    Object.fromEntries(CLASSES.map(([k]) => [k, l.events.filter(e => e.label === k && e.decision !== 'uncertain').length]))
  )
  view.innerHTML = `${intro(
    'comparison',
    `Comparación · ${esc(rec.data.filename)}`,
    'Cada pista muestra lo que detectó un sistema sobre el mismo audio. Sirve para escuchar dónde coinciden y dónde discrepan. El acuerdo entre sistemas <b>no es exactitud</b>: ninguno de ellos es la verdad; para medir exactitud usa la evaluación contra la referencia humana.',
    [
      'Mueve «Desde» y elige el ancho de la ventana.',
      'Haz clic en un bloque para escucharlo con contexto.',
      'Si un caso es interesante, envíalo a revisión asistida de desarrollo.'
    ]
  )}
    ${legend()}<div id="wave"></div>
    <div class="row"><label class="inline">Desde <input id="t0" type="range" min="0" max="${Math.max(0, dur - 30000)}" step="1000" value="${rec.data.segments.find(s => s.pilot)?.core_start_ms ?? 0}"></label><b id="t0v"></b>
    <label class="inline">Ventana <select id="len"><option value="15000">15 s</option><option value="30000" selected>30 s</option><option value="120000">2 min</option></select></label></div>
    <div id="lanes"></div><div id="pick"></div>
    <h2>Eventos por clase</h2><div class="scroll"><table><thead><tr><th>Pista</th>${CLASSES.map(([k]) => `<th>${badge(k)}</th>`).join('')}</tr></thead><tbody>
    ${lanes.map((l, i) => `<tr><td>${esc(l.name)}</td>${CLASSES.map(([k]) => `<td>${counts[i][k]}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
    <h2>Estado, segmentos y costo</h2><div class="scroll"><table><thead><tr><th>Sistema</th><th>Estado</th><th>Segmentos</th><th>Tiempo</th><th>Versión</th></tr></thead><tbody>${c.runs
      .map(
        r => `<tr><td>${SYSTEM[r.data.system] || r.data.system}</td><td>${chip(r.status)}</td>
      <td>${
        Object.entries(r.data.result?.segments || {})
          .map(([s, v]) => `${s} ${chip(v)}`)
          .join(' ') || '—'
      }</td>
      <td>${r.data.result?.timings?.total_s?.toFixed?.(1) ?? '—'} s${r.data.result?.rtf ? ` · ${r.data.result.rtf.toFixed(2)}× la duración` : ''}${r.data.result?.timings?.initialization_s ? ` · carga ${r.data.result.timings.initialization_s.toFixed(1)} s` : ''}</td>
      <td class="muted small">${esc(JSON.stringify(r.data.result?.model_version || {}).slice(0, 160))}</td></tr>`
      )
      .join('')}</tbody></table></div>
    <h2>Acuerdo entre sistemas por clase</h2><p class="muted">F1 simétrico (misma clase, IoU ≥ 0,5), solo en segmentos que ambos completaron. 1 = coinciden en todo; — = ninguno marcó esa clase.</p>
    <div class="scroll"><table><thead><tr><th>Par</th>${CLASSES.map(([k]) => `<th>${badge(k)}</th>`).join('')}</tr></thead><tbody>
    ${Object.entries(c.symmetric_agreement)
      .map(
        ([pair, v]) =>
          `<tr><td>${pair
            .split('~')
            .map(s => SYSTEM[s] || s)
            .join(' vs. ')}</td>${CLASSES.map(([k]) => `<td>${num(v[k])}</td>`).join('')}</tr>`
      )
      .join('')}</tbody></table></div>`
  const { ws } = await player(recId, rec)
  const inWindow = () => {
    const t0 = +$('#t0').value
    const len = +$('#len').value
    return [t0, len, e => e.end_ms > t0 && e.start_ms < t0 + len && e.label]
  }
  const drawLanes = () => {
    const [t0, len, keep] = inWindow()
    $('#t0v').textContent = `${fmt(t0)}–${fmt(t0 + len)}`
    $('#lanes').innerHTML = lanes
      .map(
        (l, li) =>
          `<div class="lane-name">${esc(l.name)} ${chip(l.status)}</div><div class="lane">${l.events
            .filter(keep)
            .map(
              (e, ei) =>
                `<b data-l="${li}:${ei}" title="${CLS[e.label].name} ${fmt(e.start_ms)}–${fmt(e.end_ms)}" style="left:${Math.max(0, ((e.start_ms - t0) / len) * 100)}%;width:${((Math.min(e.end_ms, t0 + len) - Math.max(e.start_ms, t0)) / len) * 100}%;background:${CLS[e.label].color}${e.decision === 'uncertain' ? '66' : ''}"></b>`
            )
            .join('')}</div>`
      )
      .join('')
  }
  $('#t0').oninput = drawLanes
  $('#len').onchange = drawLanes
  drawLanes()
  $('#lanes').onclick = e => {
    if (!e.target.dataset.l) return
    const [li, ei] = e.target.dataset.l.split(':').map(Number)
    const lane = lanes[li]
    const ev = lane.events.filter(inWindow()[2])[ei]
    ws.play(Math.max(0, ev.start_ms / 1000 - 1), ev.end_ms / 1000 + 0.5)
    $('#pick').innerHTML =
      `<div class="card row">${esc(lane.name)}: ${badge(ev.label)} ${fmt(ev.start_ms)}–${fmt(ev.end_ms)} <span class="muted">${esc(ev.text || '')}</span> ${lane.run ? '<button id="send">Enviar caso a revisión</button>' : ''}</div>`
    if (lane.run) $('#send').onclick = () => sendToReview(recId, lane.run, ev)
  }
  cleanup = () => ws.destroy()
}

// ---------------------------------------------------------- administration
async function admin() {
  const [o, recs] = await Promise.all([api('/admin/overview'), api('/recordings')])
  const people = role => o.users.filter(u => u.role === role || u.role === 'admin').sort((a, b) => (a.role !== role) - (b.role !== role))
  const ready = recs.filter(r => r.status === 'ready')
  const count = s => o.tasks.filter(t => t.status === s).length
  view.innerHTML = `${intro(
    'admin',
    'Administración de datos',
    'Aquí se organiza el trabajo humano: quién participa, qué fragmentos anota cada persona y qué pares de anotaciones pasan a adjudicación. Todo queda registrado con autor y versión.',
    [
      'Crea una cuenta por persona y entrégale su token (se muestra una sola vez).',
      'Asigna tareas: cada región se reparte a dos personas que anotan sin verse (modo ciego).',
      'Cuando ambas terminen, crea los casos de adjudicación.',
      'Luego congela el conjunto en <a href="#/experiments">Experimentos</a>.'
    ]
  )}
    ${tiles([
      ['Personas', o.users.length],
      ['Tareas asignadas', count('assigned') + count('in_progress')],
      ['Tareas finalizadas', count('submitted')],
      ['Casos abiertos', o.cases.filter(c => c.status === 'open').length]
    ])}
    <div class="cards two">
    <section class="card"><h2>1. Personas</h2>
      <form id="u-form" class="row"><input name="name" placeholder="Nombre" required><select name="role">${Object.entries(ROLE)
        .map(([k, v]) => `<option value="${k}" ${k === 'annotator' ? 'selected' : ''}>${v}</option>`)
        .join('')}</select><button class="primary">Crear cuenta</button></form>
      <div id="u-token"></div>
      <div class="scroll list"><table><thead><tr><th>Nombre</th><th>Rol</th><th>Estado</th></tr></thead><tbody>${o.users.map(u => `<tr><td>${esc(u.name)}</td><td>${ROLE[u.role]}</td><td>${chip(u.status)}</td></tr>`).join('')}</tbody></table></div></section>
    <section class="card"><h2>2. Asignar tareas de anotación</h2>
      ${
        ready.length
          ? `<form id="t-form" class="form">
        <label>Grabación <select name="recording">${ready.map(r => `<option value="${r.id}">${esc(r.filename)} (${fmt(r.duration_ms)}${r.speaker_id ? ` · ${esc(r.speaker_id)}` : ''})</option>`).join('')}</select></label>
        <fieldset><legend>Personas anotadoras (elige dos para anotación doble)</legend>${
          people('annotator')
            .map(u => `<label class="inline"><input type="checkbox" name="who" value="${u.id}"> ${esc(u.name)}</label>`)
            .join('') || '<span class="muted">Primero crea cuentas con rol Anotación.</span>'
        }</fieldset>
        <div class="form two"><label>Modo <select name="mode"><option value="blind">Ciega (sin ver el modelo)</option><option value="assisted">Asistida (con candidatos; no en prueba)</option></select></label>
        <label>Partición <select name="split">${Object.entries(SPLIT)
          .map(([k, v]) => `<option value="${k}">${v}</option>`)
          .join('')}</select></label>
        <label>Qué parte <select name="segments"><option value="pilot">solo segmentos piloto (si existen)</option><option value="">toda la grabación</option></select></label>
        <label>Duración de cada región <select name="region"><option value="10">10 s</option><option value="15" selected>15 s</option><option value="20">20 s</option></select></label></div>
        <button class="primary">Crear tareas</button></form>`
          : empty('No hay grabaciones listas. Súbelas en <a href="#/playground">Prueba del producto</a> o importa el piloto con el CLI.')
      }</section>
    </div>
    <section class="card"><h2>3. Adjudicación</h2><p class="muted">Crea un caso por cada región que ya tenga dos anotaciones finalizadas por personas distintas. La persona adjudicadora no puede ser una de ellas.</p>
      <form id="c-form" class="row"><select name="adj">${people('adjudicator')
        .map(u => `<option value="${u.id}">${esc(u.name)}</option>`)
        .join('')}</select><button class="primary">Crear casos pendientes</button></form>
      ${o.cases.length ? `<div class="scroll list"><table><thead><tr><th>Caso</th><th>Región</th><th>Adjudica</th><th>Estado</th></tr></thead><tbody>${o.cases.map(c => `<tr><td>${c.id}</td><td>${fmt(c.start_ms)}–${fmt(c.end_ms)}</td><td>${esc(c.adjudicator)}</td><td>${chip(c.status)}</td></tr>`).join('')}</tbody></table></div>` : ''}</section>
    <section class="card"><h2>Todas las tareas</h2><label class="inline">Mostrar <select id="t-flt"><option value="">todas</option>${['assigned', 'in_progress', 'submitted', 'pool'].map(s => `<option value="${s}">${STATUS[s][0]}</option>`).join('')}</select></label>
      <div class="scroll list"><table><thead><tr><th>Persona</th><th>Región</th><th>Modo</th><th>Partición</th><th>Estado</th></tr></thead><tbody id="t-rows"></tbody></table></div></section>`
  const taskRows = f =>
    o.tasks
      .filter(t => !f || t.status === f)
      .map(
        t =>
          `<tr><td>${esc(t.owner || '—')}</td><td>${fmt(t.start_ms)}–${fmt(t.end_ms)}</td><td>${MODE[t.mode]}</td><td>${SPLIT[t.split] || t.split}</td><td>${chip(t.status)}</td></tr>`
      )
      .join('') || '<tr><td class="muted">Sin tareas.</td></tr>'
  $('#t-rows').innerHTML = taskRows('')
  $('#t-flt').onchange = e => {
    $('#t-rows').innerHTML = taskRows(e.target.value)
  }
  $('#u-form').onsubmit = async e => {
    e.preventDefault()
    const f = new FormData(e.target)
    try {
      const r = await api('/admin/users', { method: 'POST', body: { name: f.get('name'), role: f.get('role') } })
      $('#u-token').innerHTML =
        `<div class="banner ok">Cuenta creada para <b>${esc(f.get('name'))}</b>. Copia y entrega este token ahora: no se volverá a mostrar.<div class="row"><code id="tok">${esc(r.token)}</code><button id="copy">Copiar</button></div></div>`
      $('#copy').onclick = () => navigator.clipboard.writeText(r.token).then(() => toast('Token copiado.'))
      e.target.reset()
    } catch (err) {
      toast(esc(err.message), 'bad')
    }
  }
  $('#t-form') &&
    ($('#t-form').onsubmit = async e => {
      e.preventDefault()
      const f = new FormData(e.target)
      const rec = recs.find(r => r.id === f.get('recording'))
      const pilot = rec.segments.filter(s => s.pilot).map(s => s.id)
      try {
        const r = await api('/admin/tasks', {
          method: 'POST',
          body: {
            recording_id: rec.id,
            annotators: f.getAll('who'),
            mode: f.get('mode'),
            split: f.get('split'),
            region_s: +f.get('region'),
            segments: f.get('segments') === 'pilot' && pilot.length ? pilot : null
          }
        })
        toast(`${r.created} tareas creadas.`)
        route()
      } catch (err) {
        toast(esc(err.message), 'bad')
      }
    })
  $('#c-form').onsubmit = async e => {
    e.preventDefault()
    try {
      const r = await api('/admin/cases', { method: 'POST', body: { adjudicator: new FormData(e.target).get('adj') } })
      toast(
        r.created.length ? `${r.created.length} caso(s) creados.` : 'No hay regiones con dos anotaciones finalizadas sin caso.',
        r.created.length ? 'ok' : 'warn'
      )
      if (r.skipped.length) toast(`Omitidos: ${esc(r.skipped.join('; '))}`, 'warn')
      if (r.created.length) route()
    } catch (err) {
      toast(esc(err.message), 'bad')
    }
  }
}

// --------------------------------------------------------------- usability
async function sus() {
  const items = [
    'Creo que me gustaría usar este sistema con frecuencia.',
    'Encontré el sistema innecesariamente complejo.',
    'Pensé que el sistema era fácil de usar.',
    'Creo que necesitaría el apoyo de una persona técnica para poder usar este sistema.',
    'Encontré que las distintas funciones del sistema estaban bien integradas.',
    'Pensé que había demasiada inconsistencia en este sistema.',
    'Imagino que la mayoría de las personas aprendería a usar este sistema muy rápidamente.',
    'Encontré el sistema muy engorroso de usar.',
    'Me sentí con mucha confianza al usar el sistema.',
    'Necesité aprender muchas cosas antes de poder empezar a usar este sistema.'
  ]
  view.innerHTML = `${intro('sus', 'Encuesta de usabilidad', 'Cuestionario SUS (System Usability Scale): 10 afirmaciones sobre tu experiencia con la plataforma. Toma unos dos minutos. Mide facilidad de uso, no la exactitud del motor.')}
    <form id="sus" class="card"><p class="muted">Para cada afirmación elige de 1 (totalmente en desacuerdo) a 5 (totalmente de acuerdo).</p>
    ${items.map((q, i) => `<fieldset class="likert"><legend>${i + 1}. ${q}</legend><span class="muted small">En desacuerdo</span>${[1, 2, 3, 4, 5].map(v => `<label><input type="radio" name="q${i}" value="${v}" required> ${v}</label>`).join('')}<span class="muted small">De acuerdo</span></fieldset>`).join('')}
    <button class="primary">Enviar respuestas</button><p id="sus-out"></p></form>`
  $('#sus').onsubmit = async e => {
    e.preventDefault()
    const r = await api('/sus', { method: 'POST', body: { answers: items.map((_, i) => +new FormData(e.target).get(`q${i}`)) } })
    $('#sus-out').innerHTML = `<div class="banner ok">Gracias. Puntuación registrada: <b>${r.score}</b> de 100.</div>`
  }
}

route()
