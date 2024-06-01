# python funasr_api

This is the api for python to use funasr engine 

## For install

### Install websocket-client and ffmpeg

```shell
pip install websocket-client
apt install ffmpeg -y
```


#### recognizer examples
suport many audio type as ffmpeg support
```shell
from funasr_api import FunasrApi
    rcg = FunasrApi(
        uri="wss://www.funasr.com:10096/"
    )
    text=rcg.rec_file("asr_example.mp3")
    print("asr_example.mp3 text=",text)
```

#### streaming recognizer examples, only support pcm or wav

```shell
    #define call_back function for msg 
    def on_msg(msg):
       print("stream_example msg=",msg)
    rcg = FunasrApi(
        uri="wss://www.funasr.com:10096/",msg_callback=on_msg
    )
    rcg.create_connection()
    
    wav_path = "asr_example.wav"

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
        
    # use FunasrApi's audio2wav to covert other audio to PCM if needed
    #import os
    #file_ext=os.path.splitext(wav_path)[-1].upper()
    #if not file_ext =="PCM" and not file_ext =="WAV":
    #       audio_bytes=rcg.audio2wav(audio_bytes)
    
    stride = int(60 * 10 / 10 / 1000 * 16000 * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1

    for i in range(chunk_num):

        beg = i * stride
        data = audio_bytes[beg : beg + stride]

        rcg.feed_chunk(data)
    msg=rcg.get_result()
    print("asr_example.wav text=",msg)
```

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/fix_bug_for_python_websocket) for contributing the websocket service.
3. We acknowledge [cgisky1980](https://github.com/cgisky1980/FunASR) for contributing the websocket service of offline model.
