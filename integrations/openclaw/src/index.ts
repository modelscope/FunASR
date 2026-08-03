// FunASR plugin entrypoint registers realtime transcription support.
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { buildFunAsrRealtimeTranscriptionProvider } from "./realtime-transcription-provider.js";

export default definePluginEntry({
  id: "funasr",
  name: "FunASR Realtime Transcription",
  description: "Self-hosted realtime speech recognition through FunASR",
  register(api) {
    api.registerRealtimeTranscriptionProvider(buildFunAsrRealtimeTranscriptionProvider());
  },
});
