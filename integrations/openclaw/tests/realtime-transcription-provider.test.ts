// FunASR tests cover the official realtime WebSocket protocol adapter.
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import type { OpenClawConfig } from "openclaw/plugin-sdk/config-contracts";
import { afterEach, describe, expect, it, vi } from "vitest";
import type WebSocket from "ws";
import type { RawData } from "ws";
import { WebSocketServer } from "ws";
import { buildFunAsrRealtimeTranscriptionProvider } from "../src/realtime-transcription-provider.js";

let cleanup: (() => Promise<void>) | undefined;

function rawDataToBuffer(data: RawData): Buffer {
  if (Buffer.isBuffer(data)) {
    return data;
  }
  if (Array.isArray(data)) {
    return Buffer.concat(data);
  }
  return Buffer.from(data);
}

function parseClientMessage(data: RawData): Record<string, unknown> {
  return JSON.parse(rawDataToBuffer(data).toString("utf8")) as Record<string, unknown>;
}

async function createFunAsrServer(params: {
  onConnection?: (ws: WebSocket) => void;
  onRequest?: (url: URL, headers: Record<string, string | string[] | undefined>) => void;
}) {
  const server = createServer();
  const wss = new WebSocketServer({ noServer: true, maxPayload: 1024 * 1024 });
  const clients = new Set<WebSocket>();

  server.on("upgrade", (request, socket, head) => {
    params.onRequest?.(new URL(request.url ?? "/", "http://127.0.0.1"), request.headers);
    wss.handleUpgrade(request, socket, head, (ws) => {
      clients.add(ws);
      ws.on("close", () => clients.delete(ws));
      params.onConnection?.(ws);
    });
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const port = (server.address() as AddressInfo).port;
  cleanup = async () => {
    for (const ws of clients) {
      ws.terminate();
    }
    await new Promise<void>((resolve) => {
      wss.close(() => resolve());
    });
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
    });
  };
  return { baseUrl: `http://127.0.0.1:${port}/asr` };
}

describe("buildFunAsrRealtimeTranscriptionProvider", () => {
  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
    vi.unstubAllEnvs();
  });

  it("normalizes nested provider config", () => {
    const provider = buildFunAsrRealtimeTranscriptionProvider();
    const resolved = provider.resolveConfig?.({
      cfg: {} as OpenClawConfig,
      rawConfig: {
        providers: {
          funasr: {
            base_url: "ws://127.0.0.1:10095/asr",
            api_key: "proxy-token",
            mode: "online",
            hotwords: "OpenClaw,FunASR",
            use_itn: "false",
          },
        },
      },
    });

    expect(resolved).toEqual({
      apiKey: "proxy-token",
      baseUrl: "ws://127.0.0.1:10095/asr",
      hotwords: "OpenClaw,FunASR",
      mode: "online",
      useItn: false,
    });
  });

  it("uses environment fallbacks without requiring an API key", () => {
    vi.stubEnv("FUNASR_WS_URL", "wss://asr.example.test/socket");
    vi.stubEnv("FUNASR_API_KEY", "env-token");
    const provider = buildFunAsrRealtimeTranscriptionProvider();

    expect(provider.isConfigured({ providerConfig: {} })).toBe(true);
    expect(() => provider.createSession({ providerConfig: {} })).not.toThrow();
  });

  it("requires an explicit FunASR endpoint", () => {
    vi.stubEnv("FUNASR_WS_URL", "");
    const provider = buildFunAsrRealtimeTranscriptionProvider();

    expect(provider.isConfigured({ providerConfig: {} })).toBe(false);
    expect(() => provider.createSession({ providerConfig: {} })).toThrow(
      "FunASR WebSocket URL missing",
    );
  });

  it.each(["not a url", "ftp://files.example.test/private-marker"])(
    "rejects and redacts invalid endpoint %s",
    (baseUrl) => {
      const provider = buildFunAsrRealtimeTranscriptionProvider();
      try {
        provider.createSession({ providerConfig: { baseUrl } });
        throw new Error("expected rejection");
      } catch (error) {
        const message = (error as Error).message;
        expect(message).toMatch(/^Invalid FunASR WebSocket URL:/);
        expect(message).not.toContain("private-marker");
      }
    },
  );

  it("sends protocol setup before converted 16 kHz PCM audio", async () => {
    const frames: Array<{ binary: boolean; data: Buffer }> = [];
    let authorization: string | string[] | undefined;
    let subprotocol: string | string[] | undefined;
    const server = await createFunAsrServer({
      onRequest: (_url, headers) => {
        authorization = headers.authorization;
        subprotocol = headers["sec-websocket-protocol"];
      },
      onConnection: (ws) => {
        ws.on("message", (data, binary) => frames.push({ binary, data: rawDataToBuffer(data) }));
      },
    });
    const session = buildFunAsrRealtimeTranscriptionProvider().createSession({
      providerConfig: {
        apiKey: "proxy-token",
        baseUrl: server.baseUrl,
        hotwords: "OpenClaw,FunASR",
      },
    });

    await session.connect();
    session.sendAudio(Buffer.alloc(160, 0xff));
    session.sendAudio(Buffer.alloc(160, 0xff));
    session.sendAudio(Buffer.alloc(160, 0xff));
    await vi.waitFor(() => expect(frames).toHaveLength(2));
    session.close();

    expect(authorization).toBe("Bearer proxy-token");
    expect(subprotocol).toBe("binary");
    expect(frames[0]?.binary).toBe(false);
    expect(parseClientMessage(frames[0]!.data)).toEqual({
      audio_fs: 16000,
      chunk_interval: 10,
      chunk_size: [5, 10, 5],
      decoder_chunk_look_back: 1,
      encoder_chunk_look_back: 4,
      hotwords: "OpenClaw,FunASR",
      is_speaking: true,
      itn: true,
      mode: "2pass",
      wav_format: "pcm",
      wav_name: "openclaw",
    });
    expect(frames[1]?.binary).toBe(true);
    expect(frames[1]?.data).toHaveLength(1_920);
  });

  it("flushes a short PCM tail before the end-of-input control", async () => {
    const frames: Array<{ binary: boolean; data: Buffer }> = [];
    const server = await createFunAsrServer({
      onConnection: (ws) => {
        ws.on("message", (data, binary) => {
          frames.push({ binary, data: rawDataToBuffer(data) });
          if (!binary && parseClientMessage(data).is_end === true) {
            ws.send(JSON.stringify({ mode: "2pass", is_final: true, is_end: true }));
          }
        });
      },
    });
    const session = buildFunAsrRealtimeTranscriptionProvider().createSession({
      providerConfig: { baseUrl: server.baseUrl },
    });

    await session.connect();
    session.sendAudio(Buffer.alloc(160, 0xff));
    session.sendAudio(Buffer.alloc(160, 0xff));
    session.close();
    await vi.waitFor(() => expect(frames).toHaveLength(3));

    expect(frames[1]?.binary).toBe(true);
    expect(frames[1]?.data).toHaveLength(1_280);
    expect(frames[2]?.binary).toBe(false);
    expect(parseClientMessage(frames[2]!.data)).toEqual({ is_end: true, is_speaking: false });
  });

  it("accumulates 2pass partials and replaces them with the offline final", async () => {
    const onPartial = vi.fn();
    const onSpeechStart = vi.fn();
    const onTranscript = vi.fn();
    const server = await createFunAsrServer({
      onConnection: (ws) => {
        ws.once("message", () => {
          ws.send(JSON.stringify({ mode: "2pass-online", text: "hello " }));
          ws.send(JSON.stringify({ mode: "2pass-online", text: "world" }));
          ws.send(JSON.stringify({ mode: "2pass-offline", text: "hello world", is_final: true }));
        });
      },
    });
    const session = buildFunAsrRealtimeTranscriptionProvider().createSession({
      providerConfig: { baseUrl: server.baseUrl },
      onPartial,
      onSpeechStart,
      onTranscript,
    });

    await session.connect();
    await vi.waitFor(() => expect(onTranscript).toHaveBeenCalledWith("hello world"));
    session.close();

    expect(onPartial.mock.calls.map(([text]) => text)).toEqual(["hello", "hello world"]);
    expect(onSpeechStart).toHaveBeenCalledTimes(1);
    expect(onTranscript).toHaveBeenCalledTimes(1);
  });

  it("commits accumulated online text when the server marks it final", async () => {
    const onTranscript = vi.fn();
    const server = await createFunAsrServer({
      onConnection: (ws) => {
        ws.once("message", () => {
          ws.send(JSON.stringify({ mode: "online", text: "ni " }));
          ws.send(JSON.stringify({ mode: "online", text: "hao", is_final: true }));
        });
      },
    });
    const session = buildFunAsrRealtimeTranscriptionProvider().createSession({
      providerConfig: { baseUrl: server.baseUrl, mode: "online" },
      onTranscript,
    });

    await session.connect();
    await vi.waitFor(() => expect(onTranscript).toHaveBeenCalledWith("ni hao"));
    session.close();
  });

  it("requests explicit end-of-input and closes on the server acknowledgement", async () => {
    const controls: Array<Record<string, unknown>> = [];
    const server = await createFunAsrServer({
      onConnection: (ws) => {
        ws.on("message", (data, binary) => {
          if (binary) {
            return;
          }
          const message = parseClientMessage(data);
          controls.push(message);
          if (message.is_end === true) {
            ws.send(JSON.stringify({ mode: "2pass", is_final: true, is_end: true }));
          }
        });
      },
    });
    const session = buildFunAsrRealtimeTranscriptionProvider().createSession({
      providerConfig: { baseUrl: server.baseUrl },
    });

    await session.connect();
    session.close();
    await vi.waitFor(() => expect(controls.at(-1)?.is_end).toBe(true));

    expect(controls.at(-1)).toEqual({ is_end: true, is_speaking: false });
    expect(session.isConnected()).toBe(false);
  });

  it("surfaces a failed final acknowledgement", async () => {
    const onError = vi.fn();
    const server = await createFunAsrServer({
      onConnection: (ws) => {
        ws.on("message", (data, binary) => {
          if (!binary && parseClientMessage(data).is_end === true) {
            ws.send(
              JSON.stringify({
                mode: "2pass",
                is_final: false,
                is_end: true,
                error: "offline inference failed",
              }),
            );
          }
        });
      },
    });
    const session = buildFunAsrRealtimeTranscriptionProvider().createSession({
      providerConfig: { baseUrl: server.baseUrl },
      onError,
    });

    await session.connect();
    session.close();
    await vi.waitFor(() =>
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "FunASR server failed to finalize input: offline inference failed",
        }),
      ),
    );
  });

  it("closes a stream that exceeds the retained transcript bound", async () => {
    const onError = vi.fn();
    const server = await createFunAsrServer({
      onConnection: (ws) => {
        ws.once("message", () => {
          ws.send(JSON.stringify({ mode: "online", text: "x".repeat(300 * 1024) }));
        });
      },
    });
    const session = buildFunAsrRealtimeTranscriptionProvider().createSession({
      providerConfig: { baseUrl: server.baseUrl, mode: "online" },
      onError,
    });

    await session.connect();
    await vi.waitFor(() =>
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining("retained transcript exceeded"),
        }),
      ),
    );
  });
});
