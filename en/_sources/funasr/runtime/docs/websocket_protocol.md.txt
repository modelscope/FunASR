([简体中文](./websocket_protocol_zh.md)|English)

# WebSocket/gRPC Communication Protocol
## Offline File Transcription
### Sending Data from Client to Server
#### Message Format
Configuration parameters and meta information are in JSON format, while audio data is in bytes.
#### Initial Communication
The message (which needs to be serialized in JSON) is:
```text
{"mode": "offline", "wav_name": "wav_name", "is_speaking": True,"wav_format":"pcm"}
```
Parameter explanation:
```text
`mode`: `offline`, indicating the inference mode for offline file transcription
`wav_name`: the name of the audio file to be transcribed
`wav_format`: the audio and video file extension, such as pcm, mp3, mp4, etc.
`is_speaking`: False indicates the end of a sentence, such as a VAD segmentation point or the end of a WAV file
`audio_fs`: when the input audio is in PCM format, the audio sampling rate parameter needs to be added
```

#### Sending Audio Data
For PCM format, directly send the audio data. For other audio formats, send the header information and audio and video bytes data together. Multiple sampling rates and audio and video formats are supported.

#### Sending End of Audio Flag
After sending the audio data, an end-of-audio flag needs to be sent (which needs to be serialized in JSON):
```text
{"is_speaking": False}
```

### Sending Data from Server to Client
#### Sending Recognition Results
The message (serialized in JSON) is:
```text
{"mode": "offline", "wav_name": "wav_name", "text": "asr ouputs", "is_final": True}
```
Parameter explanation:
```text
`mode`: `offline`, indicating the inference mode for offline file transcription
`wav_name`: the name of the audio file to be transcribed
`text`: the text output of speech recognition
`is_final`: indicating the end of recognition
```

## Real-time Speech Recognition
### System Architecture Diagram

<div align="left"><img src="images/2pass.jpg" width="400"/></div>

### Sending Data from Client to Server
#### Message Format
Configuration parameters and meta information are in JSON format, while audio data is in bytes.

#### Initial Communication
The message (which needs to be serialized in JSON) is:
```text
{"mode": "2pass", "wav_name": "wav_name", "is_speaking": True, "wav_format":"pcm", "chunk_size":[5,10,5]
```
Parameter explanation:
```text
`mode`: `offline` indicates the inference mode for single-sentence recognition; `online` indicates the inference mode for real-time speech recognition; `2pass` indicates real-time speech recognition and offline model correction for sentence endings.
`wav_name`: the name of the audio file to be transcribed
`wav_format`: the audio and video file extension, such as pcm, mp3, mp4, etc. (Note: only PCM audio streams are supported in version 1.0)
`is_speaking`: False indicates the end of a sentence, such as a VAD segmentation point or the end of a WAV file
`chunk_size`: indicates the latency configuration of the streaming model, `[5,10,5]` indicates that the current audio is 600ms long, with a 300ms look-ahead and look-back time.
`audio_fs`: when the input audio is in PCM format, the audio sampling rate parameter needs to be added
```
#### Sending Audio Data
Directly send the audio data, removing the header information and sending only the bytes data. Supported audio sampling rates are 8000 (which needs to be specified as audio_fs in message), and 16000.
#### Sending End of Audio Flag
After sending the audio data, an end-of-audio flag needs to be sent (which needs to be serialized in JSON):
```text
{"is_speaking": False}
```
### Sending Data from Server to Client
#### Sending Recognition Results
The message (serialized in JSON) is:

```text
{"mode": "2pass-online", "wav_name": "wav_name", "text": "asr ouputs", "is_final": True}
```
Parameter explanation:
```text
`mode`: indicates the inference mode, divided into `2pass-online` for real-time recognition results and `2pass-offline` for 2-pass corrected recognition results.
`wav_name`: the name of the audio file to be transcribed
`text`: the text output of speech recognition
`is_final`: indicating the end of recognition
```