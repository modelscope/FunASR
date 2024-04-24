"""
  Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
  Reserved. MIT License  (https://opensource.org/licenses/MIT)
  2023 by burkliu(刘柏基) liubaiji@xverse.cn
"""

import logging
import argparse
import soundfile as sf
import time

import grpc
import paraformer_pb2_grpc
from paraformer_pb2 import Request, WavFormat, DecodeMode


class GrpcClient:
    def __init__(self, wav_path, uri, mode):
        self.wav, self.sampling_rate = sf.read(wav_path, dtype="int16")
        self.wav_format = WavFormat.pcm
        self.audio_chunk_duration = 1000  # ms
        self.audio_chunk_size = int(self.sampling_rate * self.audio_chunk_duration / 1000)
        self.send_interval = 100  # ms
        self.mode = mode

        # connect to grpc server
        channel = grpc.insecure_channel(uri)
        self.stub = paraformer_pb2_grpc.ASRStub(channel)

        # start request
        for respond in self.stub.Recognize(self.request_iterator()):
            logging.info(
                "[receive] mode {}, text {}, is final {}".format(
                    DecodeMode.Name(respond.mode), respond.text, respond.is_final
                )
            )

    def request_iterator(self, mode=DecodeMode.two_pass):
        is_first_pack = True
        is_final = False
        for start in range(0, len(self.wav), self.audio_chunk_size):
            request = Request()
            audio_chunk = self.wav[start : start + self.audio_chunk_size]

            if is_first_pack:
                is_first_pack = False
                request.sampling_rate = self.sampling_rate
                request.mode = self.mode
                request.wav_format = self.wav_format
                if request.mode == DecodeMode.two_pass or request.mode == DecodeMode.online:
                    request.chunk_size.extend([5, 10, 5])

            if start + self.audio_chunk_size >= len(self.wav):
                is_final = True
            request.is_final = is_final
            request.audio_data = audio_chunk.tobytes()
            logging.info(
                "[request] audio_data len {}, is final {}".format(
                    len(request.audio_data), request.is_final
                )
            )  # int16 = 2bytes
            time.sleep(self.send_interval / 1000)
            yield request


if __name__ == "__main__":
    logging.basicConfig(filename="", format="%(asctime)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", required=False, help="grpc server host ip"
    )
    parser.add_argument("--port", type=int, default=10100, required=False, help="grpc server port")
    parser.add_argument("--wav_path", type=str, required=True, help="audio wav path")
    args = parser.parse_args()

    for mode in [DecodeMode.offline, DecodeMode.online, DecodeMode.two_pass]:
        mode_name = DecodeMode.Name(mode)
        logging.info("[request] start requesting with mode {}".format(mode_name))

        st = time.time()
        uri = "{}:{}".format(args.host, args.port)
        client = GrpcClient(args.wav_path, uri, mode)
        logging.info("mode {}, time pass: {}".format(mode_name, time.time() - st))
