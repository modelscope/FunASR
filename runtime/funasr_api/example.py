"""
  Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
  Reserved. MIT License  (https://opensource.org/licenses/MIT)
  
  2023-2024 by zhaomingwork@qq.com  
"""

from funasr_api import FunasrApi
import wave

def recognizer_example():
    # create an recognizer
    rcg = FunasrApi(
        uri="wss://www.funasr.com:10096/"
    )
    # recognizer by filepath
    text=rcg.rec_file("asr_example.mp3")
    print("recognizer by filepath result=",text)
    
    
    # recognizer by buffer
    # rec_buf(audio_buf,ffmpeg_decode=False),set ffmpeg_decode=True if audio is not PCM or WAV type
    with open("asr_example.wav", "rb") as f:
        audio_bytes = f.read()
    text=rcg.rec_buf(audio_bytes)
    print("recognizer by buffer result=",text)

def recognizer_stream_example():

    rcg = FunasrApi(
        uri="wss://www.funasr.com:10096/"
    )
    #define call_back function for msg 
    def on_msg(msg):
       print("stream msg=",msg)
    stream=rcg.create_stream(msg_callback=on_msg)
    
    wav_path = "asr_example.wav"

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
        
    # use FunasrApi's audio2wav to covert other audio to PCM if needed
    #import os
    #from funasr_tools import FunasrTools
    #file_ext=os.path.splitext(wav_path)[-1].upper()
    #if not file_ext =="PCM" and not file_ext =="WAV":
    #       audio_bytes=FunasrTools.audio2wav(audio_bytes)
    
    stride = int(60 * 10 / 10 / 1000 * 16000 * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1

    for i in range(chunk_num):
        beg = i * stride
        data = audio_bytes[beg : beg + stride]
        stream.feed_chunk(data)
    final_result=stream.wait_for_end()
    print("asr_example.wav stream_result=",final_result)
    
    
if __name__ == "__main__":

    print("example for Funasr_websocket_recognizer")
 
    recognizer_stream_example()
   
    recognizer_example()
    
    

