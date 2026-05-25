# JavaScript and TypeScript Recipes for the FunASR OpenAI-Compatible API

Use this guide when `funasr-server` is already running and you want to connect Node.js, TypeScript services, Next.js route handlers, or other JavaScript agent workflows to local speech recognition.

## Preflight

Start the API server first:

```bash
cd examples/openai_api
python server.py --model sensevoice --device cuda --port 8000
```

Then verify the service from another terminal:

```bash
python smoke_test.py --base-url http://localhost:8000
```

SDK base URLs include `/v1`; direct health checks do not:

```text
OpenAI SDK baseURL: http://localhost:8000/v1
Health endpoint:     http://localhost:8000/health
Transcription URL:   http://localhost:8000/v1/audio/transcriptions
```

## OpenAI JavaScript SDK

Install the official JavaScript SDK:

```bash
npm install openai
```

Create `transcribe.mjs`:

```javascript
import OpenAI from "openai";
import { createReadStream } from "node:fs";

const audioPath = process.argv[2] ?? "sample.wav";

const client = new OpenAI({
  baseURL: process.env.FUNASR_OPENAI_BASE_URL ?? "http://localhost:8000/v1",
  apiKey: process.env.OPENAI_API_KEY ?? "local-development",
});

const result = await client.audio.transcriptions.create({
  model: process.env.FUNASR_MODEL ?? "sensevoice",
  file: createReadStream(audioPath),
  response_format: "verbose_json",
});

console.log(result.text);
for (const segment of result.segments ?? []) {
  console.log(`${segment.start}s-${segment.end}s`, segment.text);
}
```

Run it:

```bash
node transcribe.mjs meeting.wav
```

Most OpenAI-compatible SDKs require an API key value even when the local FunASR server does not check it. Use any placeholder for local development, then add real authentication at your gateway if the service is shared.

## Built-in fetch without an SDK

Node.js 18+ includes `fetch`, `FormData`, and `Blob`, so you can call the API without third-party dependencies:

```javascript
import { readFile } from "node:fs/promises";
import { basename } from "node:path";

const baseUrl = process.env.FUNASR_BASE_URL ?? "http://localhost:8000";
const audioPath = process.argv[2] ?? "sample.wav";
const audio = await readFile(audioPath);

const form = new FormData();
form.append("file", new Blob([audio], { type: "audio/wav" }), basename(audioPath));
form.append("model", process.env.FUNASR_MODEL ?? "sensevoice");
form.append("response_format", "verbose_json");

const response = await fetch(`${baseUrl}/v1/audio/transcriptions`, {
  method: "POST",
  body: form,
});

if (!response.ok) {
  throw new Error(`FunASR request failed: ${response.status} ${await response.text()}`);
}

const result = await response.json();
console.log(result.text);
```

Use this pattern for queue workers, webhook workers, scheduled jobs, and small internal services.

## TypeScript helper

```typescript
import OpenAI from "openai";
import { createReadStream } from "node:fs";

export interface FunASRTranscript {
  text: string;
  segments?: Array<{ start: number; end: number; text: string; speaker?: number }>;
  language?: string;
  duration?: number;
  model?: string;
}

const client = new OpenAI({
  baseURL: process.env.FUNASR_OPENAI_BASE_URL ?? "http://localhost:8000/v1",
  apiKey: process.env.OPENAI_API_KEY ?? "local-development",
});

export async function transcribeWithFunASR(audioPath: string): Promise<FunASRTranscript> {
  const result = await client.audio.transcriptions.create({
    model: process.env.FUNASR_MODEL ?? "sensevoice",
    file: createReadStream(audioPath),
    response_format: "verbose_json",
  });

  return result as FunASRTranscript;
}
```

Keep the return type small and application-owned. FunASR can return richer metadata over time, and your application can opt into only the fields it needs.

## Next.js route handler

Proxy browser uploads through your backend so you can enforce authentication, file-size limits, and audit logs before audio reaches FunASR.

```typescript
export async function POST(request: Request) {
  const incoming = await request.formData();
  const file = incoming.get("file");

  if (!(file instanceof File)) {
    return Response.json({ error: "missing file" }, { status: 400 });
  }

  const upstream = new FormData();
  upstream.append("file", file, file.name || "audio.wav");
  upstream.append("model", "sensevoice");
  upstream.append("response_format", "verbose_json");

  const response = await fetch("http://funasr-api:8000/v1/audio/transcriptions", {
    method: "POST",
    body: upstream,
  });

  const body = await response.json();
  return Response.json(body, { status: response.status });
}
```

In Docker Compose or Kubernetes, replace `funasr-api` with the service name reachable from your web backend. Avoid sending browser traffic directly to an unauthenticated FunASR endpoint on a public network.

## Production checklist

- Put TLS, authentication, upload-size limits, and rate limits in front of the API.
- Set request timeouts based on maximum audio duration; long recordings need longer HTTP timeouts.
- Log audio duration, model alias, response format, latency, and upstream error text.
- Run `GET /health` and `GET /v1/models` during readiness checks before accepting user uploads.
- Keep audio upload handling on the server side for browser applications.
- Pin `openai` package versions in production services and retest after SDK upgrades.

## Troubleshooting

| Symptom | Fix |
|---|---|
| SDK reports a missing API key | Pass any placeholder `apiKey` for local development. |
| 404 from SDK calls | Use `baseURL=http://localhost:8000/v1`; direct endpoint calls use `http://localhost:8000`. |
| `unknown model` | Call `/v1/models` and use one of the returned aliases. |
| Browser upload fails with CORS or auth errors | Send uploads to your backend first, then proxy to FunASR. |
| Request times out | Increase SDK or fetch timeouts, or split very long audio. |
