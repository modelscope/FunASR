import pyaudio
import scipy.io.wavfile as wav
import grpc_client
import grpc
import json
from grpc_client import transcribe_audio_bytes
from paraformer_pb2_grpc import ASRStub
import webrtcvad
import numpy as np
import time
import asyncio
import datetime
import argparse

#SPEAKING = False
#stub = None
#asr_user = None
#language = None

async def deal_chunk(sig_mic):
    
    global stub,SPEAKING,asr_user,language,sample_rate
    sig = np.frombuffer(sig_mic, 'int16')
    if vad.is_speech(sig.tobytes(), sample_rate): #speaking
        SPEAKING = True
        response = transcribe_audio_bytes(stub, sig, user=asr_user, language=language, speaking = True, isEnd = False) #speaking, send audio to server.
        #print("response")
        #print (response.next())
    else: #silence   
        begin_time = 0
        if SPEAKING: #means we have some audio recorded, send recognize order to server.
            SPEAKING = False
            begin_time = int(round(time.time() * 1000))            
            response = transcribe_audio_bytes(stub, None, user=asr_user, language=language, speaking = False, isEnd = False) #speak end, call server for recognize one sentence
            resp = response.next()           
            if "decoding" == resp.action:   
                print(resp.action)
                print(json.loads(resp.sentence))
                resp = response.next() #TODO, blocking operation may leads to miss some audio clips. C++ multi-threading is preferred.
                if "finish" == resp.action:        
                    end_time = int(round(time.time() * 1000))
                    print(resp.action)
                    print (json.loads(resp.sentence))
                    #print ("silence, end_time: %d " % end_time)
                    print ("delay in ms: %d " % (end_time - begin_time))
                else:
                    #debug
                    print (resp.action + " " + str(json.loads(resp.sentence)))
                    pass
        

async def record(host,port,sample_rate,mic_chunk,record_seconds,asr_user,language):
    with grpc.insecure_channel('{}:{}'.format(host, port)) as channel:
        global stub
        stub = ASRStub(channel)
        for i in range(0, int(sample_rate / mic_chunk * record_seconds)):
     
            sig_mic = stream.read(mic_chunk,exception_on_overflow = False) 
            await asyncio.create_task(deal_chunk(sig_mic))

        #end grpc
        response = transcribe_audio_bytes(stub, None, user=asr_user, language=language, speaking = False, isEnd = True)
        #print (response.next())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",
                        type=str,
                        default="127.0.0.1",
                        required=True,
                        help="grpc server host ip")
                        
    parser.add_argument("--port",
                        type=int,
                        default=10095,
                        required=True,
                        help="grpc server port")              
                        
    parser.add_argument("--user_allowed",
                        type=str,
                        default="project1_user1",
                        help="allowed user for grpc client")
                        
    parser.add_argument("--sample_rate",
                        type=int,
                        default=16000,
                        help="audio sample_rate from client")    

    parser.add_argument("--mic_chunk",
                        type=int,
                        default=160,
                        help="chunk size for mic")  

    parser.add_argument("--record_seconds",
                        type=int,
                        default=120,
                        help="run specified seconds then exit ")                       

    args = parser.parse_args()
    

    SPEAKING = False
    asr_user = args.user_allowed
    sample_rate = args.sample_rate
    language = 'zh-CN'  
    

    vad = webrtcvad.Vad()
    vad.set_mode(1)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=args.sample_rate,
                input=True,
                frames_per_buffer=args.mic_chunk)
                
    print("* recording")
    asyncio.run(record(args.host,args.port,args.sample_rate,args.mic_chunk,args.record_seconds,args.user_allowed,language))
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("recording stop")

    

