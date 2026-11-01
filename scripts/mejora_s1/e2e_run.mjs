// S1 task 1: browser E2E of the patched app panel against the REAL local MRCD service.
// The app backend (auth + session) is stubbed at the network layer; the MRCD service is NOT stubbed.
// usage: node e2e_run.mjs <playwright-module-dir> <chromium-exe> <wav> <outDir> [appUrl] [serviceUrl]
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const [pwDir, exe, wav, outDir, appUrl = "http://localhost:5173", svcUrl = "http://127.0.0.1:8765"] = process.argv.slice(2);
const { chromium } = createRequire(import.meta.url)(pwDir);
fs.mkdirSync(outDir, { recursive: true });

const SESSION_ID = "11111111-1111-4111-8111-111111111111";
const user = { id: "22222222-2222-4222-8222-222222222222", email: "e2e@example.invalid", is_active: true, is_superuser: false,
  is_verified: true, full_name: "E2E stub", locale: "es", plan: "free", segment: null, codigo_anonimo: "E2E-STUB" };
const session = { id: SESSION_ID, type: "speech", status: "completed", started_at: "2026-11-01T00:00:00Z", ended_at: "2026-11-01T00:01:00Z",
  duration_seconds: 60, context: {}, transcript: null, created_at: "2026-11-01T00:00:00Z",
  report: { score: 50, summary: "Stub report for the E2E run.", strengths: [], improvements: [], paraverbal_metrics: {}, metrics: [], next_steps: [], structured_pitch: null } };

const browser = await chromium.launch({ executablePath: exe, headless: true });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, recordHar: { path: path.join(outDir, "network.har"), content: "omit" } });
await ctx.addInitScript(() => { try { localStorage.setItem("oratoria.auth.token", "e2e-stub-token"); } catch {} });
const stubbed = [];
await ctx.route(/localhost:8000|127\.0\.0\.1:8000/, async (route) => {
  const u = route.request().url();
  stubbed.push(`${route.request().method()} ${u}`);
  const body = u.includes("/users/me") ? user : u.includes(`/sessions/${SESSION_ID}`) ? session : null;
  await route.fulfill({ status: body ? 200 : 404, contentType: "application/json", headers: { "access-control-allow-origin": appUrl, "access-control-allow-credentials": "true" },
    body: JSON.stringify(body ?? { detail: "stubbed: not part of the E2E" }) });
});
const page = await ctx.newPage();
const log = { console: [], svc: [] };
page.on("console", (m) => log.console.push(`${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => log.console.push(`pageerror: ${e.message}`));
page.on("response", async (r) => {
  if (r.url().startsWith(svcUrl)) {
    let body = null; try { body = await r.json(); } catch {}
    log.svc.push({ t: Date.now(), method: r.request().method(), url: r.url(), status: r.status(), body_keys: body ? Object.keys(body) : null,
      job_status: body?.status ?? null, bytes_sent: r.request().postDataBuffer()?.length ?? null });
    if (body?.result || body?.schema_version || body?.events) fs.writeFileSync(path.join(outDir, "service_result.json"), JSON.stringify(body?.result ?? body, null, 1));
  }
});
await page.goto(`${appUrl}/reports/${SESSION_ID}`, { waitUntil: "networkidle" });
await page.getByText("Disfluencias por evento").first().waitFor({ timeout: 60000 });
await page.screenshot({ path: path.join(outDir, "01_report_panel_before.png"), fullPage: true });
const t0 = Date.now();
await page.getByLabel("Archivo WAV para el análisis de disfluencias").setInputFiles(wav);
await page.screenshot({ path: path.join(outDir, "02_after_select.png"), fullPage: true });
let outcome = "timeout";
for (let i = 0; i < 240; i++) {
  await page.waitForTimeout(1000);
  if (await page.getByRole("alert").count()) { outcome = "error"; break; }
  if (await page.getByRole("button", { name: "Limpiar" }).count() && !(await page.getByRole("status").count())) { outcome = "ok"; break; }
}
const elapsed = (Date.now() - t0) / 1000;
await page.screenshot({ path: path.join(outDir, "03_result.png"), fullPage: true });
const section = page.locator("div.mt-12", { has: page.getByText("Disfluencias por evento") }).last();
const text = await section.innerText().catch(() => "");
fs.writeFileSync(path.join(outDir, "panel_text.txt"), text);
const checks = {
  outcome, elapsed_s: elapsed,
  has_no_calibrado: /no calibrado/i.test(text), has_no_disponible: /No disponible/i.test(text), has_proveniencia: /proveniencia|checkpoint|perfil/i.test(text),
  has_seconds: /\d+(\.\d+)?\s*(s\b|–|-)/.test(text),
  classes_shown: ["filler_word", "prolongation", "repetition", "block", "rhetorical_pause", "neutral_pause", "revision"].filter((c) => text.includes(c)),
  has_percent_sign: /%/.test(text.replace(/\bevents?_per_minute\b/g, "")),
};
fs.writeFileSync(path.join(outDir, "checks.json"), JSON.stringify({ checks, stubbed_app_requests: stubbed, service_requests: log.svc, console: log.console }, null, 1));
await ctx.close();
await browser.close();
console.log(JSON.stringify(checks));
