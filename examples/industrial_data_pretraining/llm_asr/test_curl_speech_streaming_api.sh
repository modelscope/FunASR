#!/bin/bash

curl -v -N -G "http://localhost:8091/speech_qwen2_stream/" --data-urlencode "query=[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"<|startofspeech|>!/mnt/workspace/workgroup/hupao/project/FunASR/tests/sft.wav<|endofspeech|>\"}]"

