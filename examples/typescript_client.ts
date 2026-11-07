// Typed client example. Node 22.6+ runs it directly:
//   node --experimental-strip-types examples/typescript_client.ts [file.wav]      (MRCD_URL overrides the address)
// In a bundler project, copy clients/typescript/mrcd.ts and import it as "./mrcd".
import { readFile } from "node:fs/promises";
import { MrcdClient, MrcdError } from "../clients/typescript/mrcd.ts";

const client = new MrcdClient(process.env.MRCD_URL ?? "http://127.0.0.1:8765");
const wav = await readFile(process.argv[2] ?? "sample_es.wav");

try {
  const health = await client.health();
  console.log("incremental sessions:", health.incremental);
  const doc = await client.analyze(new Uint8Array(wav), { include_text: true });
  for (const ev of doc.events) {
    console.log(ev.class, ev.start_s, ev.end_s, ev.score.value, "(uncalibrated)");
  }
  const epm = doc.summary?.events_per_minute;
  console.log("events per minute:", epm?.available ? epm.value : `not available: ${epm?.unavailable_reason}`);
} catch (e) {
  if (e instanceof MrcdError) console.error(e.status, e.code, e.message);
  else throw e;
}
