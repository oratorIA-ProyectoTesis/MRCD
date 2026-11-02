// S7 E2E: full session through the REAL app (frontend + backend + Postgres) and the REAL local MRCD service.
// Nothing of the app is stubbed. The browser microphone is a Chromium fake device fed with a synthetic WAV.
// usage: node e2e_session.mjs <playwright-module-dir> <chromium-exe> <wav> <outDir> [appUrl] [recordSeconds]
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const [pwDir, exe, wav, outDir, appUrl = "http://localhost:5173", recSec = "47", mode = "normal"] = process.argv.slice(2);
// mode: normal (explicit close) | abrupt (browser page closed mid-recording, no close_session)
const { chromium } = createRequire(import.meta.url)(pwDir);
fs.mkdirSync(outDir, { recursive: true });
const shot = (page, name) => page.screenshot({ path: path.join(outDir, name), fullPage: true });

const browser = await chromium.launch({
  executablePath: exe, headless: true,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${wav}%noloop`],
});
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ["microphone", "camera"] });
const page = await ctx.newPage();

const log = { console: [], ws: { url: null, sent_binary_frames: 0, sent_binary_bytes: 0, sent_text: [], received: [] }, detail_polls: [], timeline: [] };
const t0 = Date.now();
const mark = (what) => log.timeline.push({ t_s: +((Date.now() - t0) / 1000).toFixed(1), what });
page.on("console", (m) => { if (["error", "warning"].includes(m.type())) log.console.push(`${m.type()}: ${m.text().slice(0, 300)}`); });
page.on("pageerror", (e) => log.console.push(`pageerror: ${e.message.slice(0, 300)}`));
page.on("websocket", (ws) => {
  if (!ws.url().includes("/ws/session")) return;
  log.ws.url = ws.url();
  ws.on("framesent", (f) => {
    if (typeof f.payload === "string") { try { log.ws.sent_text.push(JSON.parse(f.payload).tipo); } catch {} }
    else { log.ws.sent_binary_frames++; log.ws.sent_binary_bytes += f.payload.length; }
  });
  ws.on("framereceived", (f) => { try { const m = JSON.parse(f.payload); log.ws.received.push(m.tipo); if (m.tipo === "session_started") log.session_id = m.sesion_id; } catch {} });
  ws.on("close", () => mark("ws closed"));
});
let lastDetail = null;
page.on("response", async (r) => {
  const u = r.url();
  if (/\/api\/v1\/sessions\/[0-9a-f-]{36}$/.test(u) && r.request().method() === "GET") {
    try { lastDetail = await r.json(); log.detail_polls.push({ t_s: +((Date.now() - t0) / 1000).toFixed(1), status: r.status(), mrcd_status: lastDetail?.mrcd_analysis?.status ?? null }); } catch {}
  }
});

// 1. register through the UI
const email = `e2e${Date.now()}@example.com`;
await page.goto(`${appUrl}/registro`, { waitUntil: "networkidle" });
await page.fill("#full_name", "Participante Prueba"); await page.fill("#email", email); await page.fill("#password", "Prueba-Segura-2026");
await page.check("input[type=checkbox]");
await page.getByRole("button", { name: /Crear cuenta/ }).click();
await page.waitForURL(/dashboard/, { timeout: 30000 }); mark("registered");

// 2. configure and start
await page.goto(`${appUrl}/practice/new`, { waitUntil: "networkidle" });
await page.getByRole("button", { name: /Sustentación de tesis/ }).click().catch(() => {});
await page.getByRole("button", { name: /^Informar$/ }).click().catch(() => {});
await page.getByPlaceholder(/solución es técnicamente viable/).fill("que el sistema es útil para practicar").catch(() => {});
await shot(page, "01_configure.png");
await page.getByRole("button", { name: /Empezar práctica/ }).click(); mark("clicked start");
await page.waitForURL(/practice\/sesion/, { timeout: 30000 });
const rec = page.getByRole("button", { name: /Comenzar a grabar/ });
await rec.waitFor({ timeout: 60000 });
await page.waitForFunction(() => { const b = [...document.querySelectorAll("button")].find((x) => /Comenzar a grabar/.test(x.textContent || "")); return b && !b.disabled; }, null, { timeout: 60000 });
await shot(page, "02_ready.png");

// 3. record with the injected audio
await rec.click(); mark("recording");
await page.waitForTimeout(8000); await shot(page, "03_recording.png");
await page.waitForTimeout((Number(recSec) - 8) * 1000);
if (mode === "abrupt") {
  // Abrupt disconnection: close the page without Finalizar/Cerrar sesión; verify through the HTTP API afterwards.
  const sid = log.session_id;
  await page.close(); mark("page closed abruptly");
  const base = appUrl.replace(":5173", ":8000");
  const body = new URLSearchParams({ username: email, password: "Prueba-Segura-2026" });
  const tok = (await (await fetch(`${base}/api/v1/auth/login`, { method: "POST", body })).json()).access_token;
  let det = null;
  for (let i = 0; i < 300; i++) {
    det = await (await fetch(`${base}/api/v1/sessions/${sid}`, { headers: { Authorization: `Bearer ${tok}` } })).json();
    log.detail_polls.push({ t_s: +((Date.now() - t0) / 1000).toFixed(1), mrcd_status: det?.mrcd_analysis?.status ?? null });
    if (det?.mrcd_analysis && det.mrcd_analysis.status !== "pending") break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  fs.writeFileSync(path.join(outDir, "session_detail.json"), JSON.stringify(det, null, 1));
  const checks = { mode, outcome: det?.mrcd_analysis?.status ?? "missing", session_id: sid, ws_sent_binary_frames: log.ws.sent_binary_frames, buffered: det?.mrcd_analysis?.binding?.buffered_duration_s ?? null };
  fs.writeFileSync(path.join(outDir, "checks.json"), JSON.stringify({ checks, log }, null, 1));
  console.log(JSON.stringify(checks, null, 1));
  await browser.close(); process.exit(0);
}
await page.getByRole("button", { name: /Finalizar grabación/ }).click(); mark("stopped recording");
await shot(page, "04_after_stop.png");

// 4. close the session
const closeBtn = page.getByRole("button", { name: /Cerrar sesión/ });
await closeBtn.waitFor({ timeout: 120000 });
await shot(page, "05_review.png");
await closeBtn.click(); mark("clicked close session");
await page.waitForURL(/\/reports\//, { timeout: 180000 }); mark("report page");
const reportUrl = page.url();

// 5. report: wait until the disfluency analysis leaves "pending"
const heading = page.getByText("Disfluencias por evento").first();
await heading.waitFor({ timeout: 120000 }); mark("disfluency section visible");
await shot(page, "06_report_first_paint.png");
let outcome = "timeout";
for (let i = 0; i < 300; i++) {
  const status = lastDetail?.mrcd_analysis?.status;
  if (status && status !== "pending") { outcome = status; break; }
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(1500);
mark(`analysis ${outcome}`);
await shot(page, "07_report_final.png");
const section = page.locator("div.mt-12", { has: page.getByText("Disfluencias por evento") }).last();
const sectionText = await section.innerText().catch(() => "");
fs.writeFileSync(path.join(outDir, "report_section_text.txt"), sectionText);
fs.writeFileSync(path.join(outDir, "session_detail.json"), JSON.stringify(lastDetail, null, 1));
const checks = {
  outcome, report_url: reportUrl, session_id: log.session_id ?? null,
  ws_sent_binary_frames: log.ws.sent_binary_frames, ws_sent_binary_bytes: log.ws.sent_binary_bytes,
  ws_sent_audio_seconds_estimate: +(log.ws.sent_binary_bytes / 32000).toFixed(1),
  ws_received_types: [...new Set(log.ws.received)], ws_sent_text_types: [...new Set(log.ws.sent_text)],
  percent_sign_in_section: sectionText.includes("%"),
  has_softmax_label: sectionText.toLowerCase().includes("score (softmax, no calibrado)"),
  has_unavailable_marker: sectionText.includes("No disponible"),
};
fs.writeFileSync(path.join(outDir, "checks.json"), JSON.stringify({ checks, log }, null, 1));
console.log(JSON.stringify(checks, null, 1));
await browser.close();
