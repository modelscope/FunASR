#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#                2023  Nvidia              (authors: Yuekai Zhang)
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script loads a manifest in lhotse format and sends it to the server
for decoding, in parallel.

Usage:
# For offline wenet server
./decode_manifest_triton.py \
    --server-addr localhost \
    --compute-cer \
    --model-name attention_rescoring \
    --num-tasks 300 \
    --manifest-filename ./aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz # noqa

# For streaming wenet server
./decode_manifest_triton.py \
    --server-addr localhost \
    --streaming \
    --compute-cer \
    --context 7 \
    --model-name streaming_wenet \
    --num-tasks 300 \
    --manifest-filename ./aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz # noqa

# For simulate streaming mode wenet server
./decode_manifest_triton.py \
    --server-addr localhost \
    --simulate-streaming \
    --compute-cer \
    --context 7 \
    --model-name streaming_wenet \
    --num-tasks 300 \
    --manifest-filename ./aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz # noqa

# For test container:
docker run -it --rm --name "wenet_client_test" --net host --gpus all soar97/triton-k2:22.12.1 # noqa

# For aishell manifests:
apt-get install git-lfs
git-lfs install
git clone https://huggingface.co/csukuangfj/aishell-test-dev-manifests
sudo mkdir -p /root/fangjun/open-source/icefall-aishell/egs/aishell/ASR/download/aishell
tar xf ./aishell-test-dev-manifests/data_aishell.tar.gz -C /root/fangjun/open-source/icefall-aishell/egs/aishell/ASR/download/aishell/ # noqa

"""

import argparse
import asyncio
import math
import time
import types
from pathlib import Path
import json
import numpy as np
import tritonclient
import tritonclient.grpc.aio as grpcclient
from lhotse import CutSet, load_manifest
from tritonclient.utils import np_to_triton_dtype

from icefall.utils import store_transcripts, write_error_stats

DEFAULT_MANIFEST_FILENAME = "/mnt/samsung-t7/yuekai/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz"  # noqa


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Port of the server",
    )

    parser.add_argument(
        "--manifest-filename",
        type=str,
        default=DEFAULT_MANIFEST_FILENAME,
        help="Path to the manifest for decoding",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="transducer",
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks to use for sending",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Controls how frequently we print the log.",
    )

    parser.add_argument(
        "--compute-cer",
        action="store_true",
        default=False,
        help="""True to compute CER, e.g., for Chinese.
        False to compute WER, e.g., for English words.
        """,
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="""True for streaming ASR.
        """,
    )

    parser.add_argument(
        "--simulate-streaming",
        action="store_true",
        default=False,
        help="""True for strictly simulate streaming ASR.
        Threads will sleep to simulate the real speaking scene.
        """,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        default=16,
        help="chunk size default is 16",
    )

    parser.add_argument(
        "--context",
        type=int,
        required=False,
        default=-1,
        help="subsampling context for wenet",
    )

    parser.add_argument(
        "--encoder_right_context",
        type=int,
        required=False,
        default=2,
        help="encoder right context",
    )

    parser.add_argument(
        "--subsampling",
        type=int,
        required=False,
        default=4,
        help="subsampling rate",
    )

    parser.add_argument(
        "--stats_file",
        type=str,
        required=False,
        default="./stats.json",
        help="output of stats anaylasis",
    )

    return parser.parse_args()


async def send(
    cuts: CutSet,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    compute_cer: bool,
    model_name: str,
):
    total_duration = 0.0
    results = []

    for i, c in enumerate(cuts):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(cuts)}")

        waveform = c.load_audio().reshape(-1).astype(np.float32)
        sample_rate = 16000

        # padding to nearset 10 seconds
        samples = np.zeros(
            (
                1,
                10 * sample_rate * (int(len(waveform) / sample_rate // 10) + 1),
            ),
            dtype=np.float32,
        )
        samples[0, : len(waveform)] = waveform

        lengths = np.array([[len(waveform)]], dtype=np.int32)

        inputs = [
            protocol_client.InferInput(
                "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
            ),
            protocol_client.InferInput(
                "WAV_LENS", lengths.shape, np_to_triton_dtype(lengths.dtype)
            ),
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        sequence_id = 10086 + i

        response = await triton_client.infer(
            model_name, inputs, request_id=str(sequence_id), outputs=outputs
        )

        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        if type(decoding_results) == np.ndarray:
            decoding_results = b" ".join(decoding_results).decode("utf-8")
        else:
            # For wenet
            decoding_results = decoding_results.decode("utf-8")

        total_duration += c.duration

        if compute_cer:
            ref = c.supervisions[0].text.split()
            hyp = decoding_results.split()
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results.append((c.id, ref, hyp))
        else:
            results.append(
                (
                    c.id,
                    c.supervisions[0].text.split(),
                    decoding_results.split(),
                )
            )  # noqa

    return total_duration, results


async def send_streaming(
    cuts: CutSet,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    compute_cer: bool,
    model_name: str,
    first_chunk_in_secs: float,
    other_chunk_in_secs: float,
    task_index: int,
    simulate_mode: bool = False,
):
    total_duration = 0.0
    results = []
    latency_data = []

    for i, c in enumerate(cuts):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(cuts)}")

        waveform = c.load_audio().reshape(-1).astype(np.float32)
        sample_rate = 16000

        wav_segs = []

        j = 0
        while j < len(waveform):
            if j == 0:
                stride = int(first_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[j : j + stride])
            else:
                stride = int(other_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[j : j + stride])
            j += len(wav_segs[-1])

        sequence_id = task_index + 10086

        for idx, seg in enumerate(wav_segs):
            chunk_len = len(seg)

            if simulate_mode:
                await asyncio.sleep(chunk_len / sample_rate)

            chunk_start = time.time()
            if idx == 0:
                chunk_samples = int(first_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)
            else:
                chunk_samples = int(other_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)

            expect_input[0][0:chunk_len] = seg
            input0_data = expect_input
            input1_data = np.array([[chunk_len]], dtype=np.int32)

            inputs = [
                protocol_client.InferInput(
                    "WAV",
                    input0_data.shape,
                    np_to_triton_dtype(input0_data.dtype),
                ),
                protocol_client.InferInput(
                    "WAV_LENS",
                    input1_data.shape,
                    np_to_triton_dtype(input1_data.dtype),
                ),
            ]

            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
            end = False
            if idx == len(wav_segs) - 1:
                end = True

            response = await triton_client.infer(
                model_name,
                inputs,
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=idx == 0,
                sequence_end=end,
            )
            idx += 1

            decoding_results = response.as_numpy("TRANSCRIPTS")
            if type(decoding_results) == np.ndarray:
                decoding_results = b" ".join(decoding_results).decode("utf-8")
            else:
                # For wenet
                decoding_results = response.as_numpy("TRANSCRIPTS")[0].decode(
                    "utf-8"
                )
            chunk_end = time.time() - chunk_start
            latency_data.append((chunk_end, chunk_len / sample_rate))

        total_duration += c.duration

        if compute_cer:
            ref = c.supervisions[0].text.split()
            hyp = decoding_results.split()
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results.append((c.id, ref, hyp))
        else:
            results.append(
                (
                    c.id,
                    c.supervisions[0].text.split(),
                    decoding_results.split(),
                )
            )  # noqa

    return total_duration, results, latency_data


async def main():
    args = get_args()
    filename = args.manifest_filename
    server_addr = args.server_addr
    server_port = args.server_port
    url = f"{server_addr}:{server_port}"
    num_tasks = args.num_tasks
    log_interval = args.log_interval
    compute_cer = args.compute_cer

    cuts = load_manifest(filename)
    cuts_list = cuts.split(num_tasks)
    tasks = []

    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    protocol_client = grpcclient

    if args.streaming or args.simulate_streaming:
        frame_shift_ms = 10
        frame_length_ms = 25
        add_frames = math.ceil(
            (frame_length_ms - frame_shift_ms) / frame_shift_ms
        )
        # decode_window_length: input sequence length of streaming encoder
        if args.context > 0:
            # decode window length calculation for wenet
            decode_window_length = (
                args.chunk_size - 1
            ) * args.subsampling + args.context
        else:
            # decode window length calculation for icefall
            decode_window_length = (
                args.chunk_size + 2 + args.encoder_right_context
            ) * args.subsampling + 3

        first_chunk_ms = (decode_window_length + add_frames) * frame_shift_ms

    start_time = time.time()
    for i in range(num_tasks):
        if args.streaming:
            assert not args.simulate_streaming
            task = asyncio.create_task(
                send_streaming(
                    cuts=cuts_list[i],
                    name=f"task-{i}",
                    triton_client=triton_client,
                    protocol_client=protocol_client,
                    log_interval=log_interval,
                    compute_cer=compute_cer,
                    model_name=args.model_name,
                    first_chunk_in_secs=first_chunk_ms / 1000,
                    other_chunk_in_secs=args.chunk_size
                    * args.subsampling
                    * frame_shift_ms
                    / 1000,
                    task_index=i,
                )
            )
        elif args.simulate_streaming:
            task = asyncio.create_task(
                send_streaming(
                    cuts=cuts_list[i],
                    name=f"task-{i}",
                    triton_client=triton_client,
                    protocol_client=protocol_client,
                    log_interval=log_interval,
                    compute_cer=compute_cer,
                    model_name=args.model_name,
                    first_chunk_in_secs=first_chunk_ms / 1000,
                    other_chunk_in_secs=args.chunk_size
                    * args.subsampling
                    * frame_shift_ms
                    / 1000,
                    task_index=i,
                    simulate_mode=True,
                )
            )
        else:
            task = asyncio.create_task(
                send(
                    cuts=cuts_list[i],
                    name=f"task-{i}",
                    triton_client=triton_client,
                    protocol_client=protocol_client,
                    log_interval=log_interval,
                    compute_cer=compute_cer,
                    model_name=args.model_name,
                )
            )
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    results = []
    total_duration = 0.0
    latency_data = []
    for ans in ans_list:
        total_duration += ans[0]
        results += ans[1]
        if args.streaming or args.simulate_streaming:
            latency_data += ans[2]

    rtf = elapsed / total_duration

    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += (
        f"processing time: {elapsed:.3f} seconds "
        f"({elapsed/3600:.2f} hours)\n"
    )

    if args.streaming or args.simulate_streaming:
        latency_list = [
            chunk_end for (chunk_end, chunk_duration) in latency_data
        ]
        latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
        latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
        s += f"latency_variance: {latency_variance:.2f}\n"
        s += f"latency_50_percentile: {np.percentile(latency_list, 50) * 1000.0:.2f}\n"
        s += f"latency_90_percentile: {np.percentile(latency_list, 90) * 1000.0:.2f}\n"
        s += f"latency_99_percentile: {np.percentile(latency_list, 99) * 1000.0:.2f}\n"
        s += f"average_latency_ms: {latency_ms:.2f}\n"

    print(s)

    with open("rtf.txt", "w") as f:
        f.write(s)

    name = Path(filename).stem.split(".")[0]
    results = sorted(results)
    store_transcripts(filename=f"recogs-{name}.txt", texts=results)

    with open(f"errs-{name}.txt", "w") as f:
        write_error_stats(f, "test-set", results, enable_log=True)

    with open(f"errs-{name}.txt", "r") as f:
        print(f.readline())  # WER
        print(f.readline())  # Detailed errors

    if args.stats_file:
        stats = await triton_client.get_inference_statistics(
            model_name="", as_json=True
        )
        with open(args.stats_file, "w") as f:
            json.dump(stats, f)


if __name__ == "__main__":
    asyncio.run(main())
