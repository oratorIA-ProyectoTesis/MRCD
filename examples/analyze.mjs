// Node 18+, no dependencies:  node examples/analyze.mjs recording.wav   (MRCD_URL overrides the address)
// The WAV must be mono, 16 kHz, PCM16, at least 13 s long.
import { readFile } from "node:fs/promises";

const base = (process.env.MRCD_URL ?? "http://127.0.0.1:8765").replace(
  /\/$/,
  "",
);
const file = process.argv[2];
if (!file) {
  console.error("usage: node analyze.mjs file.wav");
  process.exit(2);
}

let job = await (
  await fetch(`${base}/v1/jobs?include_text=false`, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: await readFile(file),
  })
).json();
while (job.status !== "completed" && job.status !== "failed") {
  await new Promise((r) => setTimeout(r, 1000));
  job = await (await fetch(`${base}/v1/jobs/${job.job_id}`)).json();
}
const doc = job.result;
if (doc.status !== "completed") {
  console.error("failed:", doc.error.code, doc.error.message);
  process.exit(1);
}
for (const ev of doc.events) {
  console.log(
    `${ev.start_s.toFixed(2)}-${ev.end_s.toFixed(2)} s  ${ev.class}  score ${ev.score.value.toFixed(2)} (uncalibrated)`,
  );
}
