([简体中文](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md)|English)

FunASR has open-sourced a large number of pre-trained models on industrial data. You are free to use, copy, modify, and share FunASR models under the [Model License Agreement](https://github.com/alibaba-damo-academy/FunASR/blob/main/MODEL_LICENSE). Below, we list some representative models. For a comprehensive list, please refer to our [Model Zoo](https://github.com/alibaba-damo-academy/FunASR/tree/main/model_zoo).

<div align="center">  
<h4>
 <a href="#Inference"> Model Inference </a>   
｜<a href="#Training"> Model Training and Testing </a>
｜<a href="#Export"> Model Export and Testing </a>
｜<a href="#new-model-registration-tutorial"> New Model Registration Tutorial </a>
</h4>
</div>

<a name="Inference"></a>
## Model Inference

### Quick Start

For command-line invocation:
```shell
funasr ++model=paraformer-zh ++vad_model="fsmn-vad" ++punc_model="ct-punc" ++input=asr_example_zh.wav
```

For python code invocation (recommended): 

```python
from funasr import AutoModel

model = AutoModel(model="paraformer-zh")

res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav")
print(res)
```

### API Description 
#### AutoModel Definition
```python
model = AutoModel(model=[str], device=[str], ncpu=[int], output_dir=[str], batch_size=[int], hub=[str], **kwargs)
```
- `model`(str): model name in the [Model Repository](https://github.com/alibaba-damo-academy/FunASR/tree/main/model_zoo), or a model path on local disk.
- `device`(str): `cuda:0` (default gpu0) for using GPU for inference, specify `cpu` for using CPU.
- `ncpu`(int): `4` (default), sets the number of threads for CPU internal operations.
- `output_dir`(str): `None` (default), set this to specify the output path for the results.
- `batch_size`(int): `1` (default), the number of samples per batch during decoding.
- `hub`(str)：`ms` (default) to download models from ModelScope. Use `hf` to download models from Hugging Face.
- `**kwargs`(dict): Any parameters found in config.yaml can be directly specified here, for instance, the maximum segmentation length in the vad model max_single_segment_time=6000 (milliseconds).

#### AutoModel Inference
```python
res = model.generate(input=[str], output_dir=[str])
```
- `input`: The input to be decoded, which could be:
  - A wav file path, e.g., asr_example.wav
  - A pcm file path, e.g., asr_example.pcm, in this case, specify the audio sampling rate fs (default is 16000)
  - An audio byte stream, e.g., byte data from a microphone
  - A wav.scp, a Kaldi-style wav list (wav_id \t wav_path), for example:
  ```text
  asr_example1  ./audios/asr_example1.wav
  asr_example2  ./audios/asr_example2.wav
  ```
  When using wav.scp as input, you must set output_dir to save the output results.
  - Audio samples, `e.g.`: `audio, rate = soundfile.read("asr_example_zh.wav")`, data type is numpy.ndarray. Supports batch inputs, type is list：
  ```[audio_sample1, audio_sample2, ..., audio_sampleN]```
  - fbank input, supports batch grouping. Shape is [batch, frames, dim], type is torch.Tensor.
- `output_dir`: None (default), if set, specifies the output path for the results.
- `**kwargs`(dict): Inference parameters related to the model, for example,`beam_size=10`，`decoding_ctc_weight=0.1`.


### More Usage Introduction


#### Speech Recognition (Non-streaming)
```python
from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh",  
                  vad_model="fsmn-vad", 
                  vad_kwargs={"max_single_segment_time": 60000},
                  punc_model="ct-punc", 
                  # spk_model="cam++"
                  )
wav_file = f"{model.model_path}/example/asr_example.wav"
res = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='魔搭')
print(res)
```
Notes:
- Typically, the input duration for models is limited to under 30 seconds. However, when combined with `vad_model`, support for audio input of any length is enabled, not limited to the paraformer model—any audio input model can be used.
- Parameters related to model can be directly specified in the definition of AutoModel; parameters related to `vad_model` can be set through `vad_kwargs`, which is a dict; similar parameters include `punc_kwargs` and `spk_kwargs`.
- `max_single_segment_time`: Denotes the maximum audio segmentation length for `vad_model`, measured in milliseconds (ms).
- `batch_size_s` represents the use of dynamic batching, where the total audio duration within a batch is measured in seconds (s).
- `batch_size_threshold_s`: Indicates that when the duration of an audio segment post-VAD segmentation exceeds the batch_size_threshold_s threshold, the batch size is set to 1, measured in seconds (s).

Recommendations: 

When you input long audio and encounter Out Of Memory (OOM) issues, since memory usage tends to increase quadratically with audio length, consider the following three scenarios:

a) At the beginning of inference, memory usage primarily depends on `batch_size_s`. Appropriately reducing this value can decrease memory usage.

b) During the middle of inference, when encountering long audio segments cut by VAD and the total token count is less than `batch_size_s`, yet still facing OOM, you can appropriately reduce `batch_size_threshold_s`. If the threshold is exceeded, the batch size is forced to 1.

c) Towards the end of inference, if long audio segments cut by VAD have a total token count less than `batch_size_s` and exceed the `threshold` batch_size_threshold_s, forcing the batch size to 1 and still facing OOM, you may reduce `max_single_segment_time` to shorten the VAD audio segment length.

#### Speech Recognition (Streaming)
```python
from funasr import AutoModel

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming")

import soundfile
import os

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960 # 600ms

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
    print(res)
```
Note: `chunk_size` is the configuration for streaming latency.` [0,10,5]` indicates that the real-time display granularity is `10*60=600ms`, and the lookahead information is `5*60=300ms`. Each inference input is `600ms` (sample points are `16000*0.6=960`), and the output is the corresponding text. For the last speech segment input, `is_final=True` needs to be set to force the output of the last word.

#### Voice Activity Detection (Non-Streaming)
```python
from funasr import AutoModel

model = AutoModel(model="fsmn-vad")
wav_file = f"{model.model_path}/example/vad_example.wav"
res = model.generate(input=wav_file)
print(res)
```
Note: The output format of the VAD model is: `[[beg1, end1], [beg2, end2], ..., [begN, endN]]`, where `begN/endN` indicates the starting/ending point of the `N-th` valid audio segment, measured in milliseconds.

#### Voice Activity Detection (Streaming)
```python
from funasr import AutoModel

chunk_size = 200 # ms
model = AutoModel(model="fsmn-vad")

import soundfile

wav_file = f"{model.model_path}/example/vad_example.wav"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = int(chunk_size * sample_rate / 1000)

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)
    if len(res[0]["value"]):
        print(res)
```
Note: The output format for the streaming VAD model can be one of four scenarios:
- `[[beg1, end1], [beg2, end2], .., [begN, endN]]`：The same as the offline VAD output result mentioned above.
- `[[beg, -1]]`：Indicates that only a starting point has been detected.
- `[[-1, end]]`：Indicates that only an ending point has been detected.
- `[]`：Indicates that neither a starting point nor an ending point has been detected. 

The output is measured in milliseconds and represents the absolute time from the starting point.
#### Punctuation Restoration
```python
from funasr import AutoModel

model = AutoModel(model="ct-punc")
res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
print(res)
```
#### Timestamp Prediction
```python
from funasr import AutoModel

model = AutoModel(model="fa-zh")
wav_file = f"{model.model_path}/example/asr_example.wav"
text_file = f"{model.model_path}/example/text.txt"
res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
print(res)
```

More examples ref to [docs](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining)

<a name="Training"></a>
## Model Training and Testing

### Quick Start

Execute via command line (for quick testing, not recommended):
```shell
funasr-train ++model=paraformer-zh ++train_data_set_list=data/list/train.jsonl ++valid_data_set_list=data/list/val.jsonl ++output_dir="./outputs" &> log.txt &
```

Execute with Python code (supports multi-node and multi-GPU, recommended):

```shell
cd examples/industrial_data_pretraining/paraformer
bash finetune.sh
# "log_file: ./outputs/log.txt"
```
Full code ref to [finetune.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/finetune.sh)

### Detailed Parameter Description:

```shell
funasr/bin/train.py \
++model="${model_name_or_model_dir}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.batch_size=20000 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=50 \
++train_conf.log_interval=1 \
++train_conf.resume=false \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.keep_nbest_models=20 \
++train_conf.avg_nbest_model=10 \
++optim_conf.lr=0.0002 \
++output_dir="${output_dir}" &> ${log_file}
```

- `model`（str）: The name of the model (the ID in the model repository), at which point the script will automatically download the model to local storage; alternatively, the path to a model already downloaded locally.
- `train_data_set_list`（str）: The path to the training data, typically in jsonl format, for specific details refer to [examples](https://github.com/alibaba-damo-academy/FunASR/blob/main/data/list).
- `valid_data_set_list`（str）：The path to the validation data, also generally in jsonl format, for specific details refer to examples](https://github.com/alibaba-damo-academy/FunASR/blob/main/data/list).
- `dataset_conf.batch_type`（str）：example (default), the type of batch. example means batches are formed with a fixed number of batch_size samples; length or token means dynamic batching, with total length or number of tokens of the batch equalling batch_size.
- `dataset_conf.batch_size`（int）：Used in conjunction with batch_type. When batch_type=example, it represents the number of samples; when batch_type=length, it represents the length of the samples, measured in fbank frames (1 frame = 10 ms) or the number of text tokens.
- `train_conf.max_epoch`（int）：The total number of epochs for training.
- `train_conf.log_interval`（int）：The number of steps between logging.
- `train_conf.resume`（int）：Whether to enable checkpoint resuming for training.
- `train_conf.validate_interval`（int）：The interval in steps to run validation tests during training.
- `train_conf.save_checkpoint_interval`（int）：The interval in steps for saving the model during training.
- `train_conf.keep_nbest_models`（int）：The maximum number of model parameters to retain, sorted by validation set accuracy, from highest to lowest.
- `train_conf.avg_nbest_model`（int）：Average over the top n models with the highest accuracy.
- `optim_conf.lr`（float）：The learning rate.
- `output_dir`（str）：The path for saving the model.
- `**kwargs`(dict): Any parameters in config.yaml can be specified directly here, for example, to filter out audio longer than 20s: dataset_conf.max_token_length=2000, measured in fbank frames (1 frame = 10 ms) or the number of text tokens.

#### Multi-GPU Training
##### Single-Machine Multi-GPU Training
```shell
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

torchrun --nnodes 1 --nproc_per_node ${gpu_num} \
../../../funasr/bin/train.py ${train_args}
```
--nnodes represents the total number of participating nodes, while --nproc_per_node indicates the number of processes running on each node.

##### Multi-Machine Multi-GPU Training

On the master node, assuming the IP is 192.168.1.1 and the port is 12345, and you're using 2 GPUs, you would run the following command:
```shell
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

torchrun --nnodes 2 --node_rank 0 --nproc_per_node ${gpu_num} --master_addr=192.168.1.1 --master_port=12345 \
../../../funasr/bin/train.py ${train_args}
```
On the worker node (assuming the IP is 192.168.1.2), you need to ensure that the MASTER_ADDR and MASTER_PORT environment variables are set to match those of the master node, and then run the same command:

```shell
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

torchrun --nnodes 2 --node_rank 1 --nproc_per_node ${gpu_num} --master_addr=192.168.1.1 --master_port=12345 \
../../../funasr/bin/train.py ${train_args}
```

--nnodes indicates the total number of nodes participating in the training, --node_rank represents the ID of the current node, and --nproc_per_node specifies the number of processes running on each node (usually corresponds to the number of GPUs).

#### Data prepare

`jsonl` ref to（[demo](https://github.com/alibaba-damo-academy/FunASR/blob/main/data/list)）.
The instruction scp2jsonl can be used to generate from wav.scp and text.txt. The preparation process for wav.scp and text.txt is as follows:

`train_text.txt`

```bash
ID0012W0013 当客户风险承受能力评估依据发生变化时
ID0012W0014 所有只要处理 data 不管你是做 machine learning 做 deep learning
ID0012W0015 he tried to think how it could be
```


`train_wav.scp`


```bash
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
ID0012W0015 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cn_en.wav
```

`Command`

```shell
# generate train.jsonl and val.jsonl from wav.scp and text.txt
scp2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="../../../data/list/train.jsonl"
```

(Optional, not required) If you need to parse from jsonl back to wav.scp and text.txt, you can use the following command:

```shell
# generate wav.scp and text.txt from train.jsonl and val.jsonl
jsonl2scp \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_in="../../../data/list/train.jsonl"
```

#### Training log

##### log.txt
```shell
tail log.txt
[2024-03-21 15:55:52,137][root][INFO] - train, rank: 3, epoch: 0/50, step: 6990/1, total step: 6990, (loss_avg_rank: 0.327), (loss_avg_epoch: 0.409), (ppl_avg_epoch: 1.506), (acc_avg_epoch: 0.795), (lr: 1.165e-04), [('loss_att', 0.259), ('acc', 0.825), ('loss_pre', 0.04), ('loss', 0.299), ('batch_size', 40)], {'data_load': '0.000', 'forward_time': '0.315', 'backward_time': '0.555', 'optim_time': '0.076', 'total_time': '0.947'}, GPU, memory: usage: 3.830 GB, peak: 18.357 GB, cache: 20.910 GB, cache_peak: 20.910 GB
[2024-03-21 15:55:52,139][root][INFO] - train, rank: 1, epoch: 0/50, step: 6990/1, total step: 6990, (loss_avg_rank: 0.334), (loss_avg_epoch: 0.409), (ppl_avg_epoch: 1.506), (acc_avg_epoch: 0.795), (lr: 1.165e-04), [('loss_att', 0.285), ('acc', 0.823), ('loss_pre', 0.046), ('loss', 0.331), ('batch_size', 36)], {'data_load': '0.000', 'forward_time': '0.334', 'backward_time': '0.536', 'optim_time': '0.077', 'total_time': '0.948'}, GPU, memory: usage: 3.943 GB, peak: 18.291 GB, cache: 19.619 GB, cache_peak: 19.619 GB
```


- `rank`：gpu id。
- `epoch`,`step`,`total step`：the current epoch, step, and total steps.
- `loss_avg_rank`：the average loss across all GPUs for the current step.
- `loss/ppl/acc_avg_epoch`：the overall average loss/perplexity/accuracy for the current epoch, up to the current step count. The last step of the epoch when it ends represents the total average loss/perplexity/accuracy for that epoch; it is recommended to use the accuracy metric.
- `lr`：the learning rate for the current step.
- `[('loss_att', 0.259), ('acc', 0.825), ('loss_pre', 0.04), ('loss', 0.299), ('batch_size', 40)]`：the specific data for the current GPU ID.
- `total_time`：the total time taken for a single step.
- `GPU, memory`：the model-used/peak memory and the model+cache-used/peak memory.

##### tensorboard
```bash
tensorboard --logdir /xxxx/FunASR/examples/industrial_data_pretraining/paraformer/outputs/log/tensorboard
```
http://localhost:6006/

### 训练后模型测试


#### With `configuration.json` file

Assuming the training model path is: ./model_dir, if a configuration.json file has been generated in this directory, you only need to change the model name to the model path in the above model inference method. 

For example, for shell inference:
```shell
python -m funasr.bin.inference ++model="./model_dir" ++input=="${input}" ++output_dir="${output_dir}"
```

Python inference

```python
from funasr import AutoModel

model = AutoModel(model="./model_dir")

res = model.generate(input=wav_file)
print(res)
```

#### Without `configuration.json` file

If there is no configuration.json in the model path, you need to manually specify the exact configuration file path and the model path.

```shell
python -m funasr.bin.inference \
--config-path "${local_path}" \
--config-name "${config}" \
++init_param="${init_param}" \
++tokenizer_conf.token_list="${tokens}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++input="${input}" \
++output_dir="${output_dir}" \
++device="${device}"
```

Parameter Introduction
- `config-path`：This is the path to the config.yaml saved during the experiment, which can be found in the experiment's output directory.
- `config-name`：The name of the configuration file, usually config.yaml. It supports both YAML and JSON formats, for example config.json.
- `init_param`：The model parameters that need to be tested, usually model.pt. You can choose a specific model file as needed.
- `tokenizer_conf.token_list`：The path to the vocabulary file, which is normally specified in config.yaml. There is no need to manually specify it again unless the path in config.yaml is incorrect, in which case the correct path must be manually specified here.
- `frontend_conf.cmvn_file`：The CMVN (Cepstral Mean and Variance Normalization) file used when extracting fbank features from WAV files, which is usually specified in config.yaml. There is no need to manually specify it again unless the path in config.yaml is incorrect, in which case the correct path must be manually specified here.

Other parameters are the same as mentioned above. A complete [example](https://github.com/alibaba-damo-academy/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/infer_from_local.sh) can be found here.

<a name="Export"></a>
## Export ONNX

### Command-line usage
```shell
funasr-export ++model=paraformer ++quantize=false ++device=cpu
```

### Python
```python
from funasr import AutoModel

model = AutoModel(model="paraformer", device="cpu")

res = model.export(quantize=False)
```

### Test ONNX
```python
# pip3 install -U funasr-onnx
from funasr_onnx import Paraformer
model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = Paraformer(model_dir, batch_size=1, quantize=True)

wav_path = ['~/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav']

result = model(wav_path)
print(result)
```

More examples ref to [demo](https://github.com/alibaba-damo-academy/FunASR/tree/main/runtime/python/onnxruntime)


<a name="new-model-registration-tutorial"></a>
## New Model Registration Tutorial

### Viewing the Registry

```plaintext
from funasr.register import tables

tables.print()
```

Supports viewing the registry of a specified type: `tables.print("model")`

### Registering Models

```python
from funasr.register import tables

@tables.register("model_classes", "SenseVoiceSmall")
class SenseVoiceSmall(nn.Module):
  def __init__(*args, **kwargs):
    ...

  def forward(
      self,
      **kwargs,
  ):  

  def inference(
      self,
      data_in,
      data_lengths=None,
      key: list = None,
      tokenizer=None,
      frontend=None,
      **kwargs,
  ):
    ...

```

Add `@tables.register("model_classes","SenseVoiceSmall")` before the class name that needs to be registered to complete the registration. The class needs to implement the methods: __init__, forward, and inference.

Complete code: [https://github.com/modelscope/FunASR/blob/main/funasr/models/sense_voice/model.py#L443](https://github.com/modelscope/FunASR/blob/main/funasr/models/sense_voice/model.py#L443)

After registration, specify the newly registered model in config.yaml to define the model

```python
model: SenseVoiceSmall
model_conf:
  ...
```

[More detailed tutorial documents](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/Tables.md)
