"""
  Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
  Reserved. MIT License  (https://opensource.org/licenses/MIT)
  
  2023-2024 by zhaomingwork@qq.com  
"""

# pip install websocket-client
# apt install ffmpeg
import ssl
from websocket import ABNF
from websocket import create_connection
from queue import Queue
import threading
import traceback
import json
import time
import numpy as np


# class for recognizer in websocket
class FunasrApi:
    """
    python asr recognizer lib

    """

    def __init__(
        self,
        uri="wss://www.funasr.com:10096/",
        msg_callback=None,
        timeout=1000,
        
    ):
        """
        uri: ws or wss server uri
        msg_callback: for message received
        timeout: timeout for get result
        """
        try:
            if uri.find("wss://"):
                       is_ssl=True
            elif uri.find("ws://"):
                 is_ssl=False
            else:
                print("not support uri",uri)
                exit(0)
                
            if is_ssl == True:
                ssl_context = ssl.SSLContext()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                uri = uri  
                ssl_opt = {"cert_reqs": ssl.CERT_NONE}
            else:
                uri = uri  
                ssl_context = None
                ssl_opt = None
 
            self.ssl_opt=ssl_opt
            self.ssl_context=ssl_context
            self.uri = uri

 

            print("connect to url", uri)
 
 



 
            self.msg_callback=msg_callback
            self.is_final=False
            self.rec_text=""
            self.timeout=timeout
            self.rec_file_len=0
            self.connect_state=0

        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()
    
    def create_connection(self):
       try:
         self.websocket = create_connection(self.uri, ssl=self.ssl_context, sslopt=self.ssl_opt)
 
         self.is_final=False
         self.rec_text=""
         self.rec_file_len=0
         self.connect_state=0
         
         message = json.dumps(
                {
                    "mode": "2pass",
                    "chunk_size": [int(x) for x in "0,10,5".split(",")],
                    "encoder_chunk_look_back": 4,
                    "decoder_chunk_look_back": 1,
                    "chunk_interval": 10,
                    "wav_name": "funasr_api",
                    "is_speaking": True,
                }
            )

         self.websocket.send(message)
         self.connect_state=1
         # thread for receive message
         self.thread_msg = threading.Thread(
                target=FunasrApi.thread_rec_msg, args=(self,)
            )
         self.thread_msg.start()
         
         print("create_connection: ",message)
       except Exception as e:
            print("create_connection",e)
    # check ffmpeg installed
    def check_ffmpeg(self):
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            
            return False
    # use ffmpeg to convert audio to wav
    def audio2wav(self,audiobuf):
     try:
      import os
      import subprocess
      if self.check_ffmpeg()==False:
         print("pls instal ffmpeg firest, in ubuntu, you can type apt install -y ffmpeg")
         exit(0)
         return
 
      ffmpeg_target_to_outwav = ["ffmpeg", "-i", '-',  "-ac", "1", "-ar", "16000",  "-f", "wav", "pipe:1"]
      pipe_to = subprocess.Popen(ffmpeg_target_to_outwav,
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
      wavbuf, err = pipe_to.communicate(audiobuf)
      if str(err).find("Error")>=0 or str(err).find("Unknown")>=0 or str(err).find("Invalid")>=0:
            print("ffmpeg err",err)
            return None
      return wavbuf
     except Exception as e:
            print("audio2wav",e)
            return None
 
    # threads for rev msg
    def thread_rec_msg(self):
        try:
            while True:
 
                if  self.connect_state==0:
                    time.sleep(0.1)
                    continue
                if self.connect_state==2:
                    break
                msg = self.websocket.recv()
 
                if msg is None or len(msg) == 0:
                    continue
                msg = json.loads(msg)
                
                if msg['is_final']==True:
                    self.is_final=True
                    
                    
                if msg['mode']=='2pass-offline':
                   self.rec_text=self.rec_text+msg['text']
                if not self.msg_callback is None:
                   self.msg_callback(msg)
 
        except Exception as e:
            print("client closed")

    # feed data to asr engine in stream way
    def feed_chunk(self, chunk):
        try:
            self.websocket.send(chunk, ABNF.OPCODE_BINARY)
 
            return  
        except:
            print("feed chunk error")
            return 
    
    # rec file 
    def rec_file(self,file_path):
       try:
        self.create_connection()
        import os
        file_ext=os.path.splitext(file_path)[-1].upper()
        
        with  open(file_path, "rb") as f:
            
           audio_bytes = f.read()
        if not file_ext =="PCM" and not file_ext =="WAV":
           audio_bytes=self.audio2wav(audio_bytes)
        if audio_bytes==None:
           print("error, ffmpeg can not decode such file!")
           exit(0)

        self.rec_file_len=len(audio_bytes)
        stride = int(60 * 10 / 10 / 1000 * 16000 * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1

        for i in range(chunk_num):

            beg = i * stride
            data = audio_bytes[beg : beg + stride]
            self.feed_chunk(data)
        return self.get_result()
       except  Exception  as e:
            print("rec_file",e)
            return
    def wait_for_result(self):
       try:
        timeout=self.timeout
         
        file_dur=self.rec_file_len/16000/2*100
        if file_dur>timeout:
           timeout=file_dur
        print("wait_for_result timeout=",timeout)
        
        while(self.is_final==False and timeout>0):
            time.sleep(0.01)
            timeout=timeout-1
        if timeout<=0:
           self.rec_text="time out!"
       except Exception  as e:
            print("wait_for_result",e)
            return
    def get_result(self):
       try:
        message = json.dumps({"is_speaking": False})
        self.websocket.send(message)
 
        self.wait_for_result()
 
 
        self.connect_state=2
        self.websocket.close()
        # return the  msg
        return self.rec_text
       except Exception  as e:
            print("get_result ",e)
            return ""

 

