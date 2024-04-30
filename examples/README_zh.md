(简体中文|[English](./README.md))

FunASR开源了大量在工业数据上预训练模型，您可以在 [模型许可协议](https://github.com/alibaba-damo-academy/FunASR/blob/main/MODEL_LICENSE)下自由使用、复制、修改和分享FunASR模型，下面列举代表性的模型，更多模型请参考 [模型仓库](https://github.com/alibaba-damo-academy/FunASR/tree/main/model_zoo)。

<div align="center">  
<h4>
 <a href="#模型推理"> 模型推理 </a>   
｜<a href="#模型训练与测试"> 模型训练与测试 </a>
｜<a href="#模型导出与测试"> 模型导出与测试 </a>
</h4>
</div>

<a name="模型推理"></a>
## 模型推理

### 快速使用

命令行方式调用：
```shell
funasr ++model=paraformer-zh ++vad_model="fsmn-vad" ++punc_model="ct-punc" ++input=asr_example_zh.wav
```

python代码调用（推荐）

```python
from funasr import AutoModel

model = AutoModel(model="paraformer-zh")

res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav")
print(res)
```

### 接口说明

#### AutoModel 定义
```python
model = AutoModel(model=[str], device=[str], ncpu=[int], output_dir=[str], batch_size=[int], hub=[str], **kwargs)
```
- `model`(str): [模型仓库](https://github.com/alibaba-damo-academy/FunASR/tree/main/model_zoo) 中的模型名称，或本地磁盘中的模型路径
- `device`(str): `cuda:0`（默认gpu0），使用 GPU 进行推理，指定。如果为`cpu`，则使用 CPU 进行推理
- `ncpu`(int): `4` （默认），设置用于 CPU 内部操作并行性的线程数
- `output_dir`(str): `None` （默认），如果设置，输出结果的输出路径
- `batch_size`(int): `1` （默认），解码时的批处理，样本个数
- `hub`(str)：`ms`（默认），从modelscope下载模型。如果为`hf`，从huggingface下载模型。
- `**kwargs`(dict): 所有在`config.yaml`中参数，均可以直接在此处指定，例如，vad模型中最大切割长度 `max_single_segment_time=6000` （毫秒）。

#### AutoModel 推理
```python
res = model.generate(input=[str], output_dir=[str])
```
- `input`: 要解码的输入，可以是：
  - wav文件路径, 例如: asr_example.wav
  - pcm文件路径, 例如: asr_example.pcm，此时需要指定音频采样率fs（默认为16000）
  - 音频字节数流，例如：麦克风的字节数数据
  - wav.scp，kaldi 样式的 wav 列表 (`wav_id \t wav_path`), 例如:
  ```text
  asr_example1  ./audios/asr_example1.wav
  asr_example2  ./audios/asr_example2.wav
  ```
  在这种输入 `wav.scp` 的情况下，必须设置 `output_dir` 以保存输出结果
  - 音频采样点，例如：`audio, rate = soundfile.read("asr_example_zh.wav")`, 数据类型为 numpy.ndarray。支持batch输入，类型为list：
  ```[audio_sample1, audio_sample2, ..., audio_sampleN]```
  - fbank输入，支持组batch。shape为[batch, frames, dim]，类型为torch.Tensor，例如
- `output_dir`: None （默认），如果设置，输出结果的输出路径
- `**kwargs`(dict): 与模型相关的推理参数，例如，`beam_size=10`，`decoding_ctc_weight=0.1`。


### 更多用法介绍


#### 非实时语音识别
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
注意：
- 通常模型输入限制时长30s以下，组合`vad_model`后，支持任意时长音频输入，不局限于paraformer模型，所有音频输入模型均可以。
- `model`相关的参数可以直接在`AutoModel`定义中直接指定；与`vad_model`相关参数可以通过`vad_kwargs`来指定，类型为dict；类似的有`punc_kwargs`，`spk_kwargs`；
- `max_single_segment_time`: 表示`vad_model`最大切割音频时长, 单位是毫秒ms.
- `batch_size_s` 表示采用动态batch，batch中总音频时长，单位为秒s。
- `batch_size_threshold_s`: 表示`vad_model`切割后音频片段时长超过 `batch_size_threshold_s`阈值时，将batch_size数设置为1, 单位为秒s.

建议：当您输入为长音频，遇到OOM问题时，因为显存占用与音频时长呈平方关系增加，分为3种情况：
- a)推理起始阶段，显存主要取决于`batch_size_s`，适当减小该值，可以减少显存占用；
- b)推理中间阶段，遇到VAD切割的长音频片段，总token数小于`batch_size_s`，仍然出现OOM，可以适当减小`batch_size_threshold_s`，超过阈值，强制batch为1; 
- c)推理快结束阶段，遇到VAD切割的长音频片段，总token数小于`batch_size_s`，且超过阈值`batch_size_threshold_s`，强制batch为1，仍然出现OOM，可以适当减小`max_single_segment_time`，使得VAD切割音频时长变短。

#### 实时语音识别

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

注：`chunk_size`为流式延时配置，`[0,10,5]`表示上屏实时出字粒度为`10*60=600ms`，未来信息为`5*60=300ms`。每次推理输入为`600ms`（采样点数为`16000*0.6=960`），输出为对应文字，最后一个语音片段输入需要设置`is_final=True`来强制输出最后一个字。

#### 语音端点检测（非实时）
```python
from funasr import AutoModel

model = AutoModel(model="fsmn-vad")

wav_file = f"{model.model_path}/example/vad_example.wav"
res = model.generate(input=wav_file)
print(res)
```
注：VAD模型输出格式为：`[[beg1, end1], [beg2, end2], .., [begN, endN]]`，其中`begN/endN`表示第`N`个有效音频片段的起始点/结束点，
单位为毫秒。

#### 语音端点检测（实时）
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
注：流式VAD模型输出格式为4种情况：
- `[[beg1, end1], [beg2, end2], .., [begN, endN]]`：同上离线VAD输出结果。
- `[[beg, -1]]`：表示只检测到起始点。
- `[[-1, end]]`：表示只检测到结束点。
- `[]`：表示既没有检测到起始点，也没有检测到结束点
输出结果单位为毫秒，从起始点开始的绝对时间。

#### 标点恢复
```python
from funasr import AutoModel

model = AutoModel(model="ct-punc")

res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
print(res)
```

#### 时间戳预测
```python
from funasr import AutoModel

model = AutoModel(model="fa-zh")

wav_file = f"{model.model_path}/example/asr_example.wav"
text_file = f"{model.model_path}/example/text.txt"
res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
print(res)
```
更多（[示例](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining)）

<a name="核心功能"></a>
## 模型训练与测试

### 快速开始

命令行执行（用于快速测试，不推荐）：
```shell
funasr-train ++model=paraformer-zh ++train_data_set_list=data/list/train.jsonl ++valid_data_set_list=data/list/val.jsonl ++output_dir="./outputs" &> log.txt &
```

python代码执行（可以多机多卡，推荐）

```shell
cd examples/industrial_data_pretraining/paraformer
bash finetune.sh
# "log_file: ./outputs/log.txt"
```
详细完整的脚本参考 [finetune.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/finetune.sh)

### 详细参数介绍

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

- `model`（str）：模型名字（模型仓库中的ID），此时脚本会自动下载模型到本读；或者本地已经下载好的模型路径。
- `train_data_set_list`（str）：训练数据路径，默认为jsonl格式，具体参考（[例子](https://github.com/alibaba-damo-academy/FunASR/blob/main/data/list)）。
- `valid_data_set_list`（str）：验证数据路径，默认为jsonl格式，具体参考（[例子](https://github.com/alibaba-damo-academy/FunASR/blob/main/data/list)）。
- `dataset_conf.batch_type`（str）：`example`（默认），batch的类型。`example`表示按照固定数目batch_size个样本组batch；`length` or `token` 表示动态组batch，batch总长度或者token数为batch_size。
- `dataset_conf.batch_size`（int）：与 `batch_type` 搭配使用，当 `batch_type=example` 时，表示样本个数；当 `batch_type=length` 时，表示样本中长度，单位为fbank帧数（1帧10ms）或者文字token个数。
- `train_conf.max_epoch`（int）：`100`（默认），训练总epoch数。
- `train_conf.log_interval`（int）：`50`（默认），打印日志间隔step数。
- `train_conf.resume`（int）：`True`（默认），是否开启断点重训。
- `train_conf.validate_interval`（int）：`5000`（默认），训练中做验证测试的间隔step数。
- `train_conf.save_checkpoint_interval`（int）：`5000`（默认），训练中模型保存间隔step数。
- `train_conf.avg_keep_nbest_models_type`（str）：`acc`（默认），保留nbest的标准为acc（越大越好）。`loss`表示，保留nbest的标准为loss（越小越好）。
- `train_conf.keep_nbest_models`（int）：`500`（默认），保留最大多少个模型参数，配合 `avg_keep_nbest_models_type` 按照验证集 acc/loss 保留最佳的n个模型，其他删除，节约存储空间。
- `train_conf.avg_nbest_model`（int）：`10`（默认），保留最大多少个模型参数，配合 `avg_keep_nbest_models_type` 按照验证集 acc/loss 对最佳的n个模型平均。
- `train_conf.accum_grad`（int）：`1`（默认），梯度累积功能。
- `train_conf.grad_clip`（float）：`10.0`（默认），梯度截断功能。
- `train_conf.use_fp16`（bool）：`False`（默认），开启fp16训练，加快训练速度。
- `optim_conf.lr`（float）：学习率。
- `output_dir`（str）：模型保存路径。
- `**kwargs`(dict): 所有在`config.yaml`中参数，均可以直接在此处指定，例如，过滤20s以上长音频：`dataset_conf.max_token_length=2000`，单位为音频fbank帧数（1帧10ms）或者文字token个数。

#### 多gpu训练
##### 单机多gpu训练
```shell
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

torchrun --nnodes 1 --nproc_per_node ${gpu_num} --master_port 12345 \
../../../funasr/bin/train.py ${train_args}
```
--nnodes 表示参与的节点总数，--nproc_per_node 表示每个节点上运行的进程数，--master_port 表示端口号

##### 多机多gpu训练

在主节点上，假设IP为192.168.1.1，端口为12345，使用的是2个GPU，则运行如下命令：
```shell
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

torchrun --nnodes 2 --node_rank 0 --nproc_per_node ${gpu_num} --master_addr 192.168.1.1 --master_port 12345 \
../../../funasr/bin/train.py ${train_args}
```
在从节点上（假设IP为192.168.1.2），你需要确保MASTER_ADDR和MASTER_PORT环境变量与主节点设置的一致，并运行同样的命令：
```shell
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

torchrun --nnodes 2 --node_rank 1 --nproc_per_node ${gpu_num} --master_addr 192.168.1.1 --master_port 12345 \
../../../funasr/bin/train.py ${train_args}
```

--nnodes 表示参与的节点总数，--node_rank 表示当前节点id，--nproc_per_node 表示每个节点上运行的进程数（通常为gpu个数），--master_port 表示端口号

#### 准备数据

`jsonl`格式可以参考（[例子](https://github.com/alibaba-damo-academy/FunASR/blob/main/data/list)）。
可以用指令 `scp2jsonl` 从wav.scp与text.txt生成。wav.scp与text.txt准备过程如下：

`train_text.txt`

左边为数据唯一ID，需与`train_wav.scp`中的`ID`一一对应
右边为音频文件标注文本，格式如下：

```bash
ID0012W0013 当客户风险承受能力评估依据发生变化时
ID0012W0014 所有只要处理 data 不管你是做 machine learning 做 deep learning
ID0012W0015 he tried to think how it could be
```


`train_wav.scp`

左边为数据唯一ID，需与`train_text.txt`中的`ID`一一对应
右边为音频文件的路径，格式如下

```bash
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
ID0012W0015 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cn_en.wav
```

`生成指令`

```shell
# generate train.jsonl and val.jsonl from wav.scp and text.txt
scp2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="../../../data/list/train.jsonl"
```

（可选，非必需）如果需要从jsonl解析成wav.scp与text.txt，可以使用指令：

```shell
# generate wav.scp and text.txt from train.jsonl and val.jsonl
jsonl2scp \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_in="../../../data/list/train.jsonl"
```

#### 查看训练日志

##### 查看实验log
```shell
tail log.txt
[2024-03-21 15:55:52,137][root][INFO] - train, rank: 3, epoch: 0/50, step: 6990/1, total step: 6990, (loss_avg_rank: 0.327), (loss_avg_epoch: 0.409), (ppl_avg_epoch: 1.506), (acc_avg_epoch: 0.795), (lr: 1.165e-04), [('loss_att', 0.259), ('acc', 0.825), ('loss_pre', 0.04), ('loss', 0.299), ('batch_size', 40)], {'data_load': '0.000', 'forward_time': '0.315', 'backward_time': '0.555', 'optim_time': '0.076', 'total_time': '0.947'}, GPU, memory: usage: 3.830 GB, peak: 18.357 GB, cache: 20.910 GB, cache_peak: 20.910 GB
[2024-03-21 15:55:52,139][root][INFO] - train, rank: 1, epoch: 0/50, step: 6990/1, total step: 6990, (loss_avg_rank: 0.334), (loss_avg_epoch: 0.409), (ppl_avg_epoch: 1.506), (acc_avg_epoch: 0.795), (lr: 1.165e-04), [('loss_att', 0.285), ('acc', 0.823), ('loss_pre', 0.046), ('loss', 0.331), ('batch_size', 36)], {'data_load': '0.000', 'forward_time': '0.334', 'backward_time': '0.536', 'optim_time': '0.077', 'total_time': '0.948'}, GPU, memory: usage: 3.943 GB, peak: 18.291 GB, cache: 19.619 GB, cache_peak: 19.619 GB
```
指标解释：
- `rank`：表示gpu id。
- `epoch`,`step`,`total step`：表示当前epoch，step，总step。
- `loss_avg_rank`：表示当前step，所有gpu平均loss。
- `loss/ppl/acc_avg_epoch`：表示当前epoch周期，截止当前step数时，总平均loss/ppl/acc。epoch结束时的最后一个step表示epoch总平均loss/ppl/acc，推荐使用acc指标。
- `lr`：当前step的学习率。
- `[('loss_att', 0.259), ('acc', 0.825), ('loss_pre', 0.04), ('loss', 0.299), ('batch_size', 40)]`：表示当前gpu id的具体数据。
- `total_time`：表示单个step总耗时。
- `GPU, memory`：分别表示，模型使用/峰值显存，模型+缓存使用/峰值显存。

##### tensorboard可视化
```bash
tensorboard --logdir /xxxx/FunASR/examples/industrial_data_pretraining/paraformer/outputs/log/tensorboard
```
浏览器中打开：http://localhost:6006/

### 训练后模型测试


#### 有configuration.json

假定，训练模型路径为：./model_dir，如果改目录下有生成configuration.json，只需要将 [上述模型推理方法](https://github.com/alibaba-damo-academy/FunASR/blob/main/examples/README_zh.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86) 中模型名字修改为模型路径即可

例如：

从shell推理
```shell
python -m funasr.bin.inference ++model="./model_dir" ++input=="${input}" ++output_dir="${output_dir}"
```
从python推理

```python
from funasr import AutoModel

model = AutoModel(model="./model_dir")

res = model.generate(input=wav_file)
print(res)
```

#### 无configuration.json时

如果模型路径中无configuration.json时，需要手动指定具体配置文件路径与模型路径

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

参数介绍
- `config-path`：为实验中保存的 `config.yaml`，可以从实验输出目录中查找。
- `config-name`：配置文件名，一般为 `config.yaml`，支持yaml格式与json格式，例如 `config.json`
- `init_param`：需要测试的模型参数，一般为`model.pt`，可以自己选择具体的模型文件
- `tokenizer_conf.token_list`：词表文件路径，一般在 `config.yaml` 有指定，无需再手动指定，当 `config.yaml` 中路径不正确时，需要在此处手动指定。
- `frontend_conf.cmvn_file`：wav提取fbank中用到的cmvn文件，一般在 `config.yaml` 有指定，无需再手动指定，当 `config.yaml` 中路径不正确时，需要在此处手动指定。

其他参数同上，完整 [示例](https://github.com/alibaba-damo-academy/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/infer_from_local.sh)


<a name="模型导出与测试"></a>
## 模型导出与测试
### 从命令行导出
```shell
funasr-export ++model=paraformer ++quantize=false
```

### 从Python导出
```python
from funasr import AutoModel

model = AutoModel(model="paraformer")

res = model.export(quantize=False)
```

### 测试ONNX
```python
# pip3 install -U funasr-onnx
from funasr_onnx import Paraformer
model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = Paraformer(model_dir, batch_size=1, quantize=True)

wav_path = ['~/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav']

result = model(wav_path)
print(result)
```

更多例子请参考 [样例](https://github.com/alibaba-damo-academy/FunASR/tree/main/runtime/python/onnxruntime)