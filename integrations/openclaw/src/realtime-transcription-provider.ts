// FunASR provider adapts OpenClaw telephony audio to the official WebSocket API.
import {
  createRealtimeTranscriptionWebSocketSession,
  type RealtimeTranscriptionProviderConfig,
  type RealtimeTranscriptionProviderPlugin,
  type RealtimeTranscriptionSession,
  type RealtimeTranscriptionSessionCreateRequest,
  type RealtimeTranscriptionWebSocketTransport,
} from "openclaw/plugin-sdk/realtime-transcription";
import { mulawToPcm, resamplePcm } from "openclaw/plugin-sdk/realtime-voice";
import { normalizeResolvedSecretInputString } from "openclaw/plugin-sdk/secret-input";
import {
  asOptionalRecord as readRecord,
  normalizeOptionalString,
  parseBooleanValue as readBoolean,
} from "openclaw/plugin-sdk/string-coerce-runtime";

type FunAsrMode = "online" | "offline" | "2pass";

type FunAsrRealtimeTranscriptionProviderConfig = {
  apiKey?: string;
  baseUrl?: string;
  hotwords?: string;
  mode?: FunAsrMode;
  useItn?: boolean;
};

type FunAsrRealtimeTranscriptionSessionConfig = RealtimeTranscriptionSessionCreateRequest & {
  baseUrl: string;
  mode: FunAsrMode;
  useItn: boolean;
  apiKey?: string;
  hotwords?: string;
};

type FunAsrRealtimeTranscriptionEvent = {
  error?: unknown;
  is_end?: boolean;
  is_final?: boolean;
  mode?: string;
  text?: string;
};

const FUNASR_INPUT_SAMPLE_RATE = 8_000;
const FUNASR_OUTPUT_SAMPLE_RATE = 16_000;
const FUNASR_PCM_FRAME_BYTES = 1_920;
const FUNASR_DEFAULT_MODE: FunAsrMode = "2pass";
const FUNASR_CONNECT_TIMEOUT_MS = 10_000;
const FUNASR_CLOSE_TIMEOUT_MS = 5_000;
const FUNASR_MAX_RECONNECT_ATTEMPTS = 3;
const FUNASR_RECONNECT_DELAY_MS = 1_000;
const FUNASR_MAX_QUEUED_BYTES = 2 * 1024 * 1024;
const FUNASR_MAX_RETAINED_TRANSCRIPT_BYTES = 256 * 1024;

function readNestedFunAsrConfig(rawConfig: RealtimeTranscriptionProviderConfig) {
  const raw = readRecord(rawConfig);
  const providers = readRecord(raw?.providers);
  return readRecord(providers?.funasr ?? raw?.funasr ?? raw) ?? {};
}

function normalizeFunAsrMode(value: unknown): FunAsrMode | undefined {
  const normalized = normalizeOptionalString(value)?.toLowerCase();
  if (!normalized) {
    return undefined;
  }
  if (normalized === "online" || normalized === "offline" || normalized === "2pass") {
    return normalized;
  }
  throw new Error(
    `Invalid FunASR mode: "${normalized}" (expected online, offline, or 2pass)`,
  );
}

function normalizeProviderConfig(
  config: RealtimeTranscriptionProviderConfig,
): FunAsrRealtimeTranscriptionProviderConfig {
  const raw = readNestedFunAsrConfig(config);
  return {
    apiKey: normalizeResolvedSecretInputString({
      value: raw.apiKey ?? raw.api_key,
      path: "plugins.entries.voice-call.config.streaming.providers.funasr.apiKey",
    }),
    baseUrl: normalizeOptionalString(raw.baseUrl ?? raw.base_url),
    hotwords: normalizeOptionalString(raw.hotwords),
    mode: normalizeFunAsrMode(raw.mode),
    useItn: readBoolean(raw.useItn ?? raw.use_itn ?? raw.itn),
  };
}

function normalizeFunAsrWebSocketUrl(value?: string): string {
  const resolved = normalizeOptionalString(value ?? process.env.FUNASR_WS_URL);
  if (!resolved) {
    throw new Error("FunASR WebSocket URL missing");
  }
  let parsed: URL;
  try {
    parsed = new URL(resolved);
  } catch {
    throw new Error("Invalid FunASR WebSocket URL: value is not a valid URL");
  }
  if (parsed.protocol === "http:") {
    parsed.protocol = "ws:";
  } else if (parsed.protocol === "https:") {
    parsed.protocol = "wss:";
  } else if (parsed.protocol !== "ws:" && parsed.protocol !== "wss:") {
    throw new Error(
      `Invalid FunASR WebSocket URL: unsupported scheme "${parsed.protocol}" (expected http, https, ws, or wss)`,
    );
  }
  if (parsed.username || parsed.password) {
    throw new Error(
      "Invalid FunASR WebSocket URL: embedded credentials are not allowed; configure apiKey instead",
    );
  }
  return parsed.toString();
}

function readErrorDetail(value: unknown): string {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  const record = readRecord(value);
  return (
    normalizeOptionalString(record?.message) ??
    normalizeOptionalString(record?.code) ??
    "unknown server error"
  );
}

function createFunAsrRealtimeTranscriptionSession(
  config: FunAsrRealtimeTranscriptionSessionConfig,
): RealtimeTranscriptionSession {
  let currentPartial = "";
  let finalizeRequested = false;
  let openedOnce = false;
  let pendingPcm = Buffer.alloc(0);
  let speechStarted = false;

  const queuePcm = (
    pcm: Buffer,
    transport: RealtimeTranscriptionWebSocketTransport,
  ) => {
    const combined = pendingPcm.length > 0 ? Buffer.concat([pendingPcm, pcm]) : pcm;
    let offset = 0;
    while (combined.length - offset >= FUNASR_PCM_FRAME_BYTES) {
      transport.sendBinary(combined.subarray(offset, offset + FUNASR_PCM_FRAME_BYTES));
      offset += FUNASR_PCM_FRAME_BYTES;
    }
    pendingPcm = Buffer.from(combined.subarray(offset));
  };

  const flushPendingPcm = (transport: RealtimeTranscriptionWebSocketTransport) => {
    if (pendingPcm.length > 0) {
      transport.sendBinary(pendingPcm);
      pendingPcm = Buffer.alloc(0);
    }
  };

  const clearTurn = () => {
    currentPartial = "";
    speechStarted = false;
  };

  const emitSpeechStart = () => {
    if (!speechStarted) {
      speechStarted = true;
      config.onSpeechStart?.();
    }
  };

  const failRetainedTranscript = (transport: RealtimeTranscriptionWebSocketTransport) => {
    clearTurn();
    config.onError?.(
      new Error(
        `FunASR realtime retained transcript exceeded ${FUNASR_MAX_RETAINED_TRANSCRIPT_BYTES} bytes`,
      ),
    );
    transport.closeNow();
  };

  const appendPartial = (
    text: string,
    transport: RealtimeTranscriptionWebSocketTransport,
  ): string | undefined => {
    const next = currentPartial + text;
    if (Buffer.byteLength(next, "utf8") > FUNASR_MAX_RETAINED_TRANSCRIPT_BYTES) {
      failRetainedTranscript(transport);
      return undefined;
    }
    currentPartial = next;
    return currentPartial.trim();
  };

  const emitFinal = (
    text: string,
    transport: RealtimeTranscriptionWebSocketTransport,
  ) => {
    const normalized = text.trim();
    if (Buffer.byteLength(normalized, "utf8") > FUNASR_MAX_RETAINED_TRANSCRIPT_BYTES) {
      failRetainedTranscript(transport);
      return;
    }
    clearTurn();
    if (normalized) {
      config.onTranscript?.(normalized);
    }
    if (finalizeRequested) {
      transport.closeNow();
    }
  };

  const handleEvent = (
    event: FunAsrRealtimeTranscriptionEvent,
    transport: RealtimeTranscriptionWebSocketTransport,
  ) => {
    if (event.is_end) {
      if (event.is_final === false || event.error !== undefined) {
        config.onError?.(
          new Error(`FunASR server failed to finalize input: ${readErrorDetail(event.error)}`),
        );
      }
      transport.closeNow();
      return;
    }

    const text = typeof event.text === "string" ? event.text : "";
    if (!text.trim()) {
      return;
    }
    emitSpeechStart();

    if (event.mode === "2pass-online") {
      const partial = appendPartial(text, transport);
      if (partial) {
        config.onPartial?.(partial);
      }
      return;
    }

    if (event.mode === "online") {
      const partial = appendPartial(text, transport);
      if (!partial) {
        return;
      }
      if (event.is_final) {
        emitFinal(currentPartial, transport);
      } else {
        config.onPartial?.(partial);
      }
      return;
    }

    if (event.mode === "offline" || event.mode === "2pass-offline" || event.is_final) {
      emitFinal(text, transport);
    }
  };

  return createRealtimeTranscriptionWebSocketSession<FunAsrRealtimeTranscriptionEvent>({
    providerId: "funasr",
    callbacks: config,
    url: config.baseUrl,
    protocols: ["binary"],
    headers: config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : undefined,
    readyOnOpen: true,
    connectTimeoutMs: FUNASR_CONNECT_TIMEOUT_MS,
    closeTimeoutMs: FUNASR_CLOSE_TIMEOUT_MS,
    maxReconnectAttempts: FUNASR_MAX_RECONNECT_ATTEMPTS,
    reconnectDelayMs: FUNASR_RECONNECT_DELAY_MS,
    maxQueuedBytes: FUNASR_MAX_QUEUED_BYTES,
    connectTimeoutMessage: "FunASR realtime transcription connection timeout",
    connectClosedBeforeReadyMessage:
      "FunASR realtime transcription connection closed before ready",
    reconnectLimitMessage: "FunASR realtime transcription reconnect limit reached",
    onOpen: (transport) => {
      if (openedOnce && currentPartial.trim()) {
        config.onError?.(
          new Error("FunASR connection interrupted; the current partial transcript was discarded"),
        );
      }
      openedOnce = true;
      finalizeRequested = false;
      pendingPcm = Buffer.alloc(0);
      clearTurn();
      transport.sendJson({
        mode: config.mode,
        chunk_size: [5, 10, 5],
        chunk_interval: 10,
        encoder_chunk_look_back: 4,
        decoder_chunk_look_back: 1,
        audio_fs: FUNASR_OUTPUT_SAMPLE_RATE,
        wav_name: "openclaw",
        wav_format: "pcm",
        is_speaking: true,
        hotwords: config.hotwords ?? "",
        itn: config.useItn,
      });
    },
    sendAudio: (audio, transport) => {
      const pcm8k = mulawToPcm(audio);
      queuePcm(
        resamplePcm(pcm8k, FUNASR_INPUT_SAMPLE_RATE, FUNASR_OUTPUT_SAMPLE_RATE),
        transport,
      );
    },
    onClose: (transport) => {
      if (finalizeRequested) {
        return;
      }
      finalizeRequested = true;
      flushPendingPcm(transport);
      transport.sendJson({ is_speaking: false, is_end: true });
    },
    onMessage: (event, transport) => handleEvent(event, transport),
  });
}

export function buildFunAsrRealtimeTranscriptionProvider(): RealtimeTranscriptionProviderPlugin {
  return {
    id: "funasr",
    label: "FunASR Realtime Transcription",
    aliases: ["fun-asr", "funasr-realtime"],
    autoSelectOrder: 45,
    resolveConfig: ({ rawConfig }) => normalizeProviderConfig(rawConfig),
    isConfigured: ({ providerConfig }) =>
      Boolean(normalizeProviderConfig(providerConfig).baseUrl || process.env.FUNASR_WS_URL),
    createSession: (req) => {
      const providerConfig = normalizeProviderConfig(req.providerConfig);
      return createFunAsrRealtimeTranscriptionSession({
        ...req,
        apiKey: providerConfig.apiKey || process.env.FUNASR_API_KEY,
        baseUrl: normalizeFunAsrWebSocketUrl(providerConfig.baseUrl),
        hotwords: providerConfig.hotwords,
        mode: providerConfig.mode ?? FUNASR_DEFAULT_MODE,
        useItn: providerConfig.useItn ?? true,
      });
    },
  };
}
