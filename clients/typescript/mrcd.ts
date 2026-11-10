/**
 * MRCD service client: one file, no dependencies (fetch + Web Crypto; browsers, Node >= 18, Deno, Bun).
 *
 * Types written from mrcd/service/schemas/mrcd.analysis.v1.schema.json
 * (contract mrcd.analysis/v1, revision 1.4). tests/test_typescript_client.py checks that the enums below match it.
 *
 *   import { MrcdClient } from "./mrcd";
 *   const client = new MrcdClient("http://127.0.0.1:8765");
 *   const doc = await client.analyze(wavBytes, { include_text: true });
 *   for (const ev of doc.events) console.log(ev.class, ev.start_s, ev.end_s, ev.score.value);
 *
 * The service takes a 16 kHz mono PCM16 WAV of at least 13 s. Scores are uncalibrated: never show them as
 * probabilities. A browser page on another origin needs the service started with `--cors-origin <origin>`.
 */

export const CONTRACT = "mrcd.analysis/v1";
export const CONTRACT_REVISION = "1.4";
export const SAMPLE_RATE = 16000;

/** `{available, value, unavailable_reason}`: absence is never encoded as 0. */
export type Measure =
  | { available: true; value: number; unavailable_reason: null }
  | { available: false; value: null; unavailable_reason: string };

export type Modality = { available: true; reason: null } | { available: false; reason: string };

export type EvaluatedClass = "filler_word" | "prolongation" | "repetition";
/** Out-of-scope classes appear only with include_out_of_scope=true and are never counted in the totals. */
export type OutOfScopeClass = "block" | "revision" | "rhetorical_pause" | "neutral_pause";

export type ErrorCode =
  | "analysis_failed"
  | "asr_unavailable"
  | "audio_gap"
  | "audio_not_finite"
  | "audio_not_mono"
  | "audio_too_long"
  | "audio_too_short"
  | "batch_too_large"
  | "checkpoint_unavailable"
  | "close_mismatch"
  | "engine_load_failed"
  | "head_unavailable"
  | "incremental_disabled"
  | "invalid_audio_file"
  | "invalid_request"
  | "job_not_found"
  | "offset_mismatch"
  | "payload_too_large"
  | "profile_not_supported_incremental"
  | "queue_full"
  | "rate_limited"
  | "sequence_conflict"
  | "session_closed"
  | "session_expired"
  | "session_limit"
  | "session_not_found"
  | "unsupported_audio_format"
  | "unauthorized"
  | "unsupported_sample_rate";

export interface ServiceErrorBody {
  code: ErrorCode;
  message: string;
  retryable: boolean;
  details: Record<string, unknown>;
}

export interface Decision {
  state: "present" | "absent" | "uncertain";
  estimate: number;
  lo: number;
  hi: number;
  basis: string;
  calibration_claim: "none";
}

export interface DisfluencyEvent {
  id: string;
  class: EvaluatedClass | OutOfScopeClass | string;
  start_s: number;
  end_s: number;
  interval_semantics: "decision_stride_not_human_boundary";
  score: { value: number; kind: string; calibrated: false; label: string };
  n_windows: number;
  modalities_used: string[];
  text: string | null;
  validation: string | null;
  experimental: boolean;
  in_scope?: boolean;
  decision?: Decision;
  /** Only decision profiles o2026_12 and o2026_13 (opt-in): contextual state of the candidate layer and the filler subclass. */
  context?: {
    state: "disfluent";
    subclass: "filled_pause" | "lexical_filler" | null;
    basis?: string;
  };
}

export interface AnalysisRequestEcho {
  audio_sha256: string | null;
  audio_duration_s: Measure;
  sample_rate_hz: Measure;
  channels: Measure;
  asr_profile: string;
  head: string | null;
  include_text: boolean;
  output_scope?: "evaluated" | "all";
  decision?: boolean;
  text_route?: string | null;
  decision_profile?: DecisionProfileName;
}

export type DecisionProfileName = "v0" | "o2026_11" | "o2026_11b" | "o2026_12" | "o2026_13";

export interface Summary {
  n_events: number;
  events_per_minute: Measure;
  counts_by_class: Record<string, number>;
  out_of_scope?: { included: boolean; n_events: number; counts_by_class: Record<string, number> };
}

export interface Analysis {
  schema_version: "mrcd.analysis/v1";
  status: "completed" | "failed";
  generated_at: string;
  error: ServiceErrorBody | null;
  request: AnalysisRequestEcho;
  events: DisfluencyEvent[];
  summary: Summary | null;
  modalities: { acoustic: Modality; linguistic: Modality; kinesic_video: Modality };
  provenance: {
    service: Record<string, unknown>;
    checkpoint: Record<string, unknown> | null;
    asr: Record<string, unknown> | null;
    head: Record<string, unknown> | null;
    config: Record<string, unknown> | null;
    environment: Record<string, unknown> | null;
    text_route?: Record<string, unknown> | null;
    decision_profile?: {
      name: DecisionProfileName;
      status: "historical" | "exploratory";
      classes: Record<string, Record<string, unknown>>;
      calibration_claim: "none";
      pipelines_verified?: boolean | null;
      note?: string;
    };
  };
  experimental: { active: boolean; components: string[] };
  head_decision: Record<string, unknown> | null;
  transcript: string | null;
  warnings: { code: string; message: string }[];
}

export interface Job {
  schema_version: "mrcd.job/v1";
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  result: Analysis | null;
}

export interface Health {
  status: "ok";
  contract: string;
  engine_loaded: boolean;
  engines_loaded: string[];
  warmed_up: boolean;
  incremental: boolean;
  torch_threads: number | null;
  active_sessions: number;
}

export interface AnalysisOptions {
  include_text?: boolean;
  decision?: boolean;
  include_out_of_scope?: boolean;
  decision_profile?: DecisionProfileName;
  asr_profile?: string;
  text_route?: string;
}

/** Non-2xx answer; `document` is the service's structured error document when it sent one. */
export class MrcdError extends Error {
  readonly status: number;
  readonly document: unknown;
  constructor(status: number, document: unknown) {
    const err = (document as { error?: ServiceErrorBody } | null)?.error;
    super(`HTTP ${status}: ${err?.message ?? JSON.stringify(document)}`);
    this.name = "MrcdError";
    this.status = status;
    this.document = document;
  }
  get code(): ErrorCode | undefined {
    return (this.document as { error?: ServiceErrorBody } | null)?.error?.code;
  }
}

type Body = ArrayBuffer | Uint8Array | Blob;

function query(options: AnalysisOptions): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(options)) if (v !== undefined && v !== null) q.set(k, String(v));
  const s = q.toString();
  return s ? `?${s}` : "";
}

const sleep = (ms: number) => new Promise((ok) => setTimeout(ok, ms));

/** Mono PCM16 LE -> canonical 44-byte-header WAV (the bytes a session close hashes). */
export function pcm16Wav(pcm: Int16Array, sampleRate = SAMPLE_RATE): Uint8Array {
  const n = pcm.length * 2;
  const out = new Uint8Array(44 + n);
  const v = new DataView(out.buffer);
  const tag = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  tag(0, "RIFF"); v.setUint32(4, 36 + n, true); tag(8, "WAVEfmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  tag(36, "data"); v.setUint32(40, n, true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44 + 2 * i, pcm[i], true);
  return out;
}

async function sha256Hex(data: Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export class MrcdClient {
  readonly baseUrl: string;
  constructor(baseUrl = "http://127.0.0.1:8765") {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  async request<T>(method: string, path: string, body?: BodyInit, headers: Record<string, string> = {}): Promise<T> {
    const r = await fetch(this.baseUrl + path, { method, body, headers, cache: "no-store" });
    const text = await r.text();
    let doc: unknown = null;
    try { doc = text ? JSON.parse(text) : null; } catch { doc = text; }
    if (!r.ok) throw new MrcdError(r.status, doc);
    return doc as T;
  }

  health(): Promise<Health> { return this.request("GET", "/v1/health"); }
  schema(): Promise<Record<string, unknown>> { return this.request("GET", "/v1/schema"); }
  job(jobId: string): Promise<Job> { return this.request("GET", `/v1/jobs/${encodeURIComponent(jobId)}`); }

  /** POST /v1/jobs with a 16 kHz mono PCM16 WAV. */
  submit(wav: Body, options: AnalysisOptions = {}): Promise<Job> {
    return this.request("POST", "/v1/jobs" + query(options), wav as BodyInit, { "Content-Type": "audio/wav" });
  }

  /** Submit and wait; resolves to the analysis document (status completed or failed). */
  async analyze(wav: Body, options: AnalysisOptions = {}, pollMs = 500, timeoutMs = 900_000): Promise<Analysis> {
    let job = await this.submit(wav, options);
    const deadline = Date.now() + timeoutMs;
    while (job.status === "queued" || job.status === "running") {
      if (Date.now() > deadline) throw new Error(`job ${job.job_id} did not finish in time`);
      await sleep(pollMs);
      job = await this.job(job.job_id);
    }
    return job.result as Analysis;
  }

  /** Incremental session (service started with --incremental). */
  async openSession(options: AnalysisOptions = {}): Promise<IncrementalSession> {
    const opened = await this.request<{ session_id: string; subscription_token: string; limits: { max_batch_bytes: number } }>(
      "POST", "/v1/sessions" + query(options));
    return new IncrementalSession(this, opened.session_id, opened.subscription_token, opened.limits?.max_batch_bytes);
  }
}

export class IncrementalSession {
  private seq = 0;
  private offset = 0;
  private parts: Int16Array[] = [];
  private chain: Promise<unknown> = Promise.resolve();

  private client: MrcdClient;
  private token: string;
  private maxBatchBytes: number;
  readonly id: string;

  constructor(client: MrcdClient, id: string, token: string, maxBatchBytes = 1 << 20) {
    this.client = client;
    this.id = id;
    this.token = token;
    this.maxBatchBytes = maxBatchBytes;
  }

  private headers(extra: Record<string, string> = {}) { return { "X-Session-Token": this.token, ...extra }; }

  /** Queue one PCM16 batch (16 kHz mono); batches are sent in order. */
  send(pcm: Int16Array): Promise<unknown> {
    const step = Math.max(1, Math.floor(this.maxBatchBytes / 2));
    for (let i = 0; i < pcm.length; i += step) {
      const part = pcm.slice(i, i + step);
      const seq = this.seq++, offset = this.offset;
      this.offset += part.length;
      this.parts.push(part);
      this.chain = this.chain.then(() => this.client.request("PUT", `/v1/sessions/${this.id}/audio/${seq}`,
        new Uint8Array(part.buffer) as unknown as BodyInit,
        this.headers({ "Content-Type": "application/octet-stream", "X-Offset-Samples": String(offset) })));
    }
    return this.chain;
  }

  /** Close and wait for the full-audio analysis. */
  async close(pollMs = 250, timeoutMs = 900_000): Promise<Analysis> {
    await this.chain;
    const all = new Int16Array(this.offset);
    let o = 0;
    for (const p of this.parts) { all.set(p, o); o += p.length; }
    const body = JSON.stringify({ total_samples: this.offset, wav_sha256: await sha256Hex(pcm16Wav(all)), last_seq: this.seq - 1 });
    await this.client.request("POST", `/v1/sessions/${this.id}/close`, body, this.headers({ "Content-Type": "application/json" }));
    this.parts = [];
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const s = await this.client.request<{ status: string; result?: Analysis }>("GET", `/v1/sessions/${this.id}`, undefined, this.headers());
      if (s.status === "completed" || s.status === "failed") return s.result as Analysis;
      if (Date.now() > deadline) throw new Error(`session ${this.id} did not finish in time`);
      await sleep(pollMs);
    }
  }

  delete(): Promise<unknown> { return this.client.request("DELETE", `/v1/sessions/${this.id}`, undefined, this.headers()); }
}
