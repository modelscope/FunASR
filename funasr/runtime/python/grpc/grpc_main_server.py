import grpc
from concurrent import futures
import argparse

import paraformer_pb2_grpc
from grpc_server import ASRServicer

def serve(args):
      server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                        # interceptors=(AuthInterceptor('Bearer mysecrettoken'),)
                           )
      paraformer_pb2_grpc.add_ASRServicer_to_server(
          ASRServicer(args.user_allowed, args.model, args.sample_rate, args.backend, args.onnx_dir, vad_model=args.vad_model, punc_model=args.punc_model), server)
      port = "[::]:" + str(args.port)
      server.add_insecure_port(port)
      server.start()
      print("grpc server started!")
      server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",
                        type=int,
                        default=10095,
                        required=True,
                        help="grpc server port")
                        
    parser.add_argument("--user_allowed",
                        type=str,
                        default="project1_user1|project1_user2|project2_user3",
                        help="allowed user for grpc client")
                        
    parser.add_argument("--model",
                        type=str,
                        default="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                        help="model from modelscope")
    parser.add_argument("--vad_model",
                        type=str,
                        default="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        help="model from modelscope")
    
    parser.add_argument("--punc_model",
                        type=str,
                        default="",
                        help="model from modelscope")
    
    parser.add_argument("--sample_rate",
                        type=int,
                        default=16000,
                        help="audio sample_rate from client")

    parser.add_argument("--backend",
                        type=str,
                        default="pipeline",
                        choices=("pipeline", "onnxruntime"),
                        help="backend, optional modelscope pipeline or onnxruntime")

    parser.add_argument("--onnx_dir",
                        type=str,
                        default="/nfs/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                        help="onnx model dir")
    
                        


    args = parser.parse_args()

    serve(args)
