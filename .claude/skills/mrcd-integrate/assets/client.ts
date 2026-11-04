// Node 18+ or browser. A browser on another origin needs the service started with --cors-origin <origin>.
const base = "http://127.0.0.1:8765";

export async function analyze(wav: Blob | Uint8Array, attempts = 3) {
  let job: any;
  for (let i = 0; i < attempts; i++) {
    const res = await fetch(`${base}/v1/jobs?include_text=false`, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: wav,
    });
    job = await res.json();
    if (res.status === 202) break;
    const err = job.error ?? {};
    if (!err.retryable || i === attempts - 1)
      throw new Error(
        `submit failed: HTTP ${res.status} ${err.code}: ${err.message}`,
      );
    await new Promise((r) => setTimeout(r, 2 ** i * 1000));
  }
  while (job.status !== "completed" && job.status !== "failed") {
    await new Promise((r) => setTimeout(r, 1000));
    job = await (await fetch(`${base}/v1/jobs/${job.job_id}`)).json();
  }
  const doc = job.result;
  if (doc.status === "failed")
    throw new Error(
      `analysis failed: ${doc.error.code} (retryable=${doc.error.retryable})`,
    );
  return doc; // mrcd.analysis/v1; scores are uncalibrated
}
