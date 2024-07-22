"""
  Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
  Reserved. MIT License  (https://opensource.org/licenses/MIT)
  
  2023-2024 by zhaomingwork@qq.com  
"""

# pip install websocket-client
# apt install ffmpeg 

import threading
import traceback
import json
import time
import numpy as np
from funasr_stream import FunasrStream
from funasr_tools import FunasrTools
from funasr_core import FunasrCore
# class for recognizer in websocket
class FunasrApi:
    """
    python asr recognizer lib

    """

    def __init__(
        self,
        uri="wss://www.funasr.com:10096/",
        timeout=1000,
        msg_callback=None,
        
    ):
        """
        uri: ws or wss server uri
        msg_callback: for message received
        timeout: timeout for get result
        """
        try:
             
            
            self.uri=uri
            self.timeout=timeout
            self.msg_callback=msg_callback
            self.funasr_core=None
            
        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()
    def create_stream(self,msg_callback=None):
        if self.funasr_core is not None:
            self.funasr_core.close()
        funasr_core=self.new_core(msg_callback=msg_callback)
        return FunasrStream(funasr_core)
         
            
            
        
    def new_core(self,msg_callback=None):
     try:
         if self.funasr_core is not None:
            self.funasr_core.close()
            
         if msg_callback==None:
            msg_callback=self.msg_callback
         funasr_core=FunasrCore(self.uri,msg_callback=msg_callback,timeout=self.timeout)
         funasr_core.new_connection()
         self.funasr_core=funasr_core
         return funasr_core
         
     except Exception as e:
            print("init_core",e)
            exit(0)
    
    # rec buffer, set ffmpeg_decode=True if audio is not PCM or WAV type
    def rec_buf(self,audio_buf,ffmpeg_decode=False):
       try:
           funasr_core=self.new_core()
           funasr_core.rec_buf(audio_buf,ffmpeg_decode=ffmpeg_decode)
           return funasr_core.get_result()
       except  Exception  as e:
            print("rec_file",e)
            return   
    # rec file 
    def rec_file(self,file_path):
       try:
           funasr_core=self.new_core()
           funasr_core.rec_file(file_path)
           return funasr_core.get_result()
       except  Exception  as e:
            print("rec_file",e)
            return 
 
 

 

