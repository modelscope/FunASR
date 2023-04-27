# -*- encoding: utf-8 -*-
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="0.0.0.0",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10095,
                    required=False,
                    help="grpc server port")
parser.add_argument("--asr_model",
                    type=str,
                    default="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    help="model from modelscope")
parser.add_argument("--asr_model_online",
                    type=str,
                    default="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                    help="model from modelscope")
parser.add_argument("--vad_model",
                    type=str,
                    default="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    help="model from modelscope")
parser.add_argument("--punc_model",
                    type=str,
                    default="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
                    help="model from modelscope")
parser.add_argument("--ngpu",
                    type=int,
                    default=1,
                    help="0 for cpu, 1 for gpu")

args = parser.parse_args()