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


# class for recognizer in websocket
class FunasrStream:
    """
    python asr recognizer lib

    """

    def __init__(
        self,
        funasr_core
        
    ):
        """
        uri: ws or wss server uri
        msg_callback: for message received
        timeout: timeout for get result
        """
        try:
            self.funasr_core=funasr_core

        except Exception as e:
            print("FunasrStream init Exception:", e)
            traceback.print_exc()
    

    # feed data to asr engine in stream way
    def feed_chunk(self, chunk):
        try:
            if self.funasr_core is None:
                print("error in stream, funasr_core is None")
                exit(0)
            self.funasr_core.feed_chunk(chunk)
            return  
        except:
            print("feed chunk error")
            return 
         
         
 
    # return all result for this stream
    def wait_for_end(self):
       try:

        message = json.dumps({"is_speaking": False})
        self.funasr_core.websocket.send(message)
        self.funasr_core.wait_for_result()
        self.funasr_core.close()
 
        # return the  msg
        return self.funasr_core.rec_text
       except Exception  as e:
            print("error get_final_result ",e)
            return ""

 

