"""
  Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
  Reserved. MIT License  (https://opensource.org/licenses/MIT)
  
  2023-2024 by zhaomingwork@qq.com  
"""

# pip install websocket-client
# apt install ffmpeg
 

import threading
import traceback

import time



# class for recognizer in websocket
class FunasrTools:
    """
    python asr recognizer lib

    """

    def __init__(
        self
      
        
    ):
        """
 
        """
        try:
             
              if FunasrTools.check_ffmpeg()==False:
                 print("pls instal ffmpeg firest, in ubuntu, you can type apt install -y ffmpeg")
                 exit(0)
 
             
        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()
    
 
    # check ffmpeg installed
    @staticmethod
    def check_ffmpeg():
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            
            return False
    # use ffmpeg to convert audio to wav
    @staticmethod
    def audio2wav(audiobuf):
     try:
      import os
      import subprocess
      if FunasrTools.check_ffmpeg()==False:
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
 
 

 

