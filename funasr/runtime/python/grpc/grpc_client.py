import paraformer_pb2
import paraformer_pb2_grpc
import grpc
import queue

def transcribe_audio_bytes(stub, chunk, user='zksz', language='zh-CN', speaking = True, isEnd = False):
    req = paraformer_pb2.Request()
    if chunk is not None:
        req.audio_data = chunk.tostring()
    req.user = user
    req.language = language
    req.speaking = speaking
    req.isEnd = isEnd
    my_queue = queue.SimpleQueue()
    my_queue.put(req) 
    return  stub.Recognize(iter(my_queue.get, None))



