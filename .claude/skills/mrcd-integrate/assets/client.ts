// Node 18+ or browser. A browser on another origin needs the service started with --cors-origin <origin>.
// Retries (bounded, backoff) the whole submit + poll cycle on retryable errors (queue_full, analysis_failed).
const base = "http://127.0.0.1:8765";

class MrcdError extends Error {
  constructor(public code: string, message: string, public retryable = false) {
    super(`${code}: ${message}`);
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const toError = (res: Response, doc: any) =>
  new MrcdError(doc?.error?.code ?? `http_${res.status}`, doc?.error?.message ?? "", !!doc?.error?.retryable);

async function runOnce(wav: Blob | Uint8Array) {
  let res = await fetch(`${base}/v1/jobs?include_text=false`, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: wav,
  });
  let job = await res.json();
  if (res.status !== 202) throw toError(res, job);
  while (job.status !== "completed" && job.status !== "failed") {
    await sleep(1000);
    res = await fetch(`${base}/v1/jobs/${job.job_id}`);
    job = await res.json();
    // unknown or evicted job: error document, no result
    if (!res.ok || !job.result) throw new MrcdError("job_not_found", "job unknown or evicted; resubmit");
  }
  if (job.result.status === "failed") throw toError(res, job.result);
  return job.result; // mrcd.analysis/v1; scores are uncalibrated
}

export async function analyze(wav: Blob | Uint8Array, attempts = 3) {
  for (let i = 0; ; i++) {
    try {
      return await runOnce(wav);
    } catch (e) {
      if (!(e instanceof MrcdError) || !e.retryable || i >= attempts - 1) throw e;
      await sleep(2 ** i * 1000);
    }
  }
}
