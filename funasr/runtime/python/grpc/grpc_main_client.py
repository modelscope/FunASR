import grpc
import json
import time
import asyncio
import soundfile as sf
import argparse

from grpc_client import transcribe_audio_bytes
from paraformer_pb2_grpc import ASRStub

# send the audio data once
async def grpc_rec(wav_scp, grpc_uri, asr_user, language):
    with grpc.insecure_channel(grpc_uri) as channel:
        stub = ASRStub(channel)
        for line in wav_scp:
            wav_file = line.split()[1]
            wav, _ = sf.read(wav_file, dtype='int16')
            
            b = time.time()
            response = transcribe_audio_bytes(stub, wav.tobytes(), user=asr_user, language=language, speaking=False, isEnd=False)
            resp = response.next()
            text = ''
            if 'decoding' == resp.action:
                resp = response.next()
                if 'finish' == resp.action:
                    text = json.loads(resp.sentence)['text']
            response = transcribe_audio_bytes(stub, None, user=asr_user, language=language, speaking=False, isEnd=True)
            res= {'text': text, 'time': time.time() - b}
            print(res)

async def test(args):
    wav_scp = open(args.wav_scp, "r").readlines()
    uri = '{}:{}'.format(args.host, args.port)
    res = await grpc_rec(wav_scp, uri, args.user_allowed, language = 'zh-CN')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",
                        type=str,
                        default="127.0.0.1",
                        required=False,
                        help="grpc server host ip")
    parser.add_argument("--port",
                        type=int,
                        default=10108,
                        required=False,
                        help="grpc server port")              
    parser.add_argument("--user_allowed",
                        type=str,
                        default="project1_user1",
                        help="allowed user for grpc client")
    parser.add_argument("--sample_rate",
                        type=int,
                        default=16000,
                        help="audio sample_rate from client") 
    parser.add_argument("--wav_scp",
                        type=str,
                        required=True,
                        help="audio wav scp")                    
    args = parser.parse_args()
    
    asyncio.run(test(args))
