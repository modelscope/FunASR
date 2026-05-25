# FunASR OpenAI 兼容 API JavaScript/TypeScript 接入配方

当 `funasr-server` 已经启动，而你希望把 Node.js、TypeScript 服务、Next.js route handler 或 JavaScript Agent 工作流接入本地语音识别时，可以按这份指南复制代码。

## 预检查

先启动 API 服务：

```bash
cd examples/openai_api
python server.py --model sensevoice --device cuda --port 8000
```

在另一个终端验证服务：

```bash
python smoke_test.py --base-url http://localhost:8000
```

SDK base URL 需要包含 `/v1`，直接健康检查不需要：

```text
OpenAI SDK baseURL: http://localhost:8000/v1
健康检查:            http://localhost:8000/health
转写接口:            http://localhost:8000/v1/audio/transcriptions
```

## OpenAI JavaScript SDK

安装官方 JavaScript SDK：

```bash
npm install openai
```

创建 `transcribe.mjs`：

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

运行：

```bash
node transcribe.mjs meeting.wav
```

多数 OpenAI 兼容 SDK 即使在本地服务不校验密钥时，也要求传入一个 API key。开发环境可以使用任意占位值；如果服务被多人共享，请在网关层增加真实鉴权。

## 不依赖 SDK 的内置 fetch 写法

Node.js 18+ 内置 `fetch`、`FormData` 和 `Blob`，可以不安装第三方依赖直接调用接口：

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

这个模式适合队列 worker、webhook worker、定时任务和小型内部服务。

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

建议在应用侧维护一个小而稳定的返回类型。FunASR 后续可能返回更丰富的元数据，业务代码只需要消费自己关心的字段。

## Next.js route handler

浏览器上传建议先进入自己的后端，再由后端转发到 FunASR，这样可以统一做鉴权、文件大小限制和审计日志。

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

在 Docker Compose 或 Kubernetes 中，把 `funasr-api` 换成 Web 后端能访问到的 service name。不要把未鉴权的 FunASR 接口直接暴露给公网浏览器。

## 生产检查清单

- 在 API 前增加 TLS、鉴权、上传大小限制和限流。
- 根据最大音频时长设置请求超时；长录音需要更长的 HTTP timeout。
- 记录音频时长、模型别名、响应格式、延迟和上游错误文本。
- 接收用户上传前，先用 `GET /health` 和 `GET /v1/models` 做就绪检查。
- 浏览器应用应把音频上传处理留在服务端。
- 生产服务固定 `openai` 包版本，并在 SDK 升级后重新测试。

## 故障排查

| 现象 | 处理方式 |
|---|---|
| SDK 提示缺少 API key | 本地开发传入任意占位 `apiKey`。 |
| SDK 调用返回 404 | SDK 使用 `baseURL=http://localhost:8000/v1`；直接端点调用使用 `http://localhost:8000`。 |
| `unknown model` | 调用 `/v1/models`，使用返回的模型别名。 |
| 浏览器上传遇到 CORS 或鉴权错误 | 先上传到自己的后端，再由后端代理到 FunASR。 |
| 请求超时 | 增加 SDK 或 fetch 超时，或切分超长音频。 |
