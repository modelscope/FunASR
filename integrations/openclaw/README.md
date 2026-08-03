# FunASR for OpenClaw

`openclaw-funasr` connects OpenClaw Talk and Voice Call to a self-hosted
[FunASR](https://github.com/modelscope/FunASR) WebSocket server. Audio stays on
infrastructure you control.

## Capabilities

- FunASR `online`, `offline`, and `2pass` recognition modes
- OpenClaw's 8 kHz G.711 mu-law input converted to 16 kHz PCM
- Official 60 ms FunASR audio frames and the required `binary` WebSocket subprotocol
- Partial and final transcripts, hotwords, inverse text normalization, and optional Bearer auth
- Bounded audio queues and transcript retention inherited from the OpenClaw session helper

## Compatibility

The first release requires OpenClaw `>=2026.7.2` with
[openclaw/openclaw#118977](https://github.com/openclaw/openclaw/pull/118977).
Publishing to npm and ClawHub is intentionally gated on that SDK change landing
in an OpenClaw release; the compatibility floor will be pinned to the exact
release before publication.

## Start FunASR

Run the official FunASR WebSocket server from this repository. The server must
be reachable from the OpenClaw Gateway and negotiate the `binary` subprotocol.
See [runtime/python/websocket](../../runtime/python/websocket) for Docker,
model, TLS, and client examples.

The default server endpoint is commonly `ws://127.0.0.1:10095`, but the plugin
does not assume a URL. Set it explicitly in OpenClaw or through
`FUNASR_WS_URL`.

## Build and test

```bash
cd integrations/openclaw
npm ci
OPENCLAW_ROOT=/absolute/path/to/openclaw npm test
npm pack
```

`OPENCLAW_ROOT` must point to an OpenClaw checkout containing #118977. The test
suite exercises the real shared WebSocket session helper, including protocol
negotiation, audio conversion, 60 ms framing, finalization, reconnect handling,
bounded transcript state, and package metadata.

Install the packed artifact before publication:

```bash
openclaw plugins install npm-pack:/absolute/path/to/openclaw-funasr-0.1.0.tgz --force
openclaw plugins inspect funasr --runtime --json
```

## Configure OpenClaw

FunASR provider options live under Voice Call's generic streaming provider map.
The same provider map is currently used by Talk transcription sessions.

```json5
{
  plugins: {
    entries: {
      "voice-call": {
        config: {
          streaming: {
            enabled: true,
            provider: "funasr",
            providers: {
              funasr: {
                baseUrl: "ws://127.0.0.1:10095",
                mode: "2pass",
                hotwords: "OpenClaw,FunASR",
                useItn: true
              }
            }
          }
        }
      }
    }
  }
}
```

Options:

| Option | Default | Description |
| --- | --- | --- |
| `baseUrl` | `FUNASR_WS_URL` | `http`, `https`, `ws`, or `wss` endpoint |
| `apiKey` | `FUNASR_API_KEY` | Optional Bearer token for a reverse proxy |
| `mode` | `2pass` | `online`, `offline`, or `2pass` |
| `hotwords` | empty | FunASR hotword string |
| `useItn` | `true` | Enable inverse text normalization |

Embedded credentials in `baseUrl` are rejected. Use `apiKey` or
`FUNASR_API_KEY` instead.

## Verified behavior

The end-to-end proof used the official FunASR server, a public 5.55 second
16 kHz sample converted to the same 8 kHz mu-law frames emitted by OpenClaw,
and the plugin's production adapter. It streamed 278 audio frames, produced
seven partial transcripts, returned the exact final transcript, and completed
without client errors.

## License

MIT
