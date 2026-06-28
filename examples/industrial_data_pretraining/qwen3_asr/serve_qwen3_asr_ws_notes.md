# serve_qwen3_asr_ws.py 说明与已知问题

本文档记录 `serve_qwen3_asr_ws.py` 的几个容易困惑/踩坑的点。代码里对应位置只留一行
指向本文的精简注释，细节看这里。

---

## 1. 关于 VAD

> 在开源仓库 github.com/QwenLM/Qwen3-ASR 没有关于 `VAD` 内容。但商用 [Qwen3-ASR的文档/示例](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-interaction-process)里**是有 VAD 设置**，例如：
> ```json
> "turn_detection": { "type": "server_vad", "threshold": 0.2, "silence_duration_ms": 800 }
> ```

VAD 在 ASR 里其实有两种完全不同的用途，容易混为一谈：

- **A. 切段用的 VAD（给非流式 encoder 喂分段）**：像 Fun-ASR-Nano 这类模型 encoder 是
  **非流式**的（一次要看完整一段），必须靠 VAD 把连续音频切成一句句、每句整体编码解码。
  这种 VAD 对 Fun-ASR-Nano 这类模型 **技术必需**——不切就没法编码。

- **B. 端点/轮次检测用的 VAD（判断"这一轮说完了没"）**：检测说话人停顿（如静音 800ms）
  来判定"一句话/一轮结束"，从而触发"锁定文本 / 发 is_final / 该回应了"。这是**产品行为**
  层面的需求，和 encoder 能不能流式无关。

对 **Qwen3-ASR 的开源流式 API**（本服务用的 `qwen-asr[vllm]` 的
`init_streaming_state` / `streaming_transcribe`）：

- **不需要 A 类（切段）VAD**：它是**增量式**流式——每次只吃新增的一小段音频、状态向前
  滚动，连续转写，不存在"先切句再解码"。"哪些字定了、哪些会变"由 `unfixed_chunk_num` /
  `unfixed_token_num` 表示（尾部 N 个 chunk/token 算"未定"、会被后续音频修正，其余视为
  已确认）——这相当于内置的 partial/锁定机制，取代了 A 类 VAD 的切段职责。所以在开源
  流式 API 的代码里搜不到 vad，是因为它**这一层不做切段**。

- **仍然需要 B 类（端点）VAD —— 只是开源 API 自己不带**：自动判断"用户停顿=这一轮结束"
  这件事，`streaming_transcribe` 本身不管。商用服务在 ASR **之外**包了一层 `server_vad`
  来做（就是上面那段 `turn_detection`）。本服务目前是用客户端显式发 **`STOP`** 来代替
  这个端点判断（bench 里音频放完即发 STOP）。**若要在真实场景自动断句/断轮，需要自己在
  本服务之外接一个 VAD / 端点检测**（角色等同商用的 server_vad），而不是去 Qwen3-ASR
  内部找——它的开源流式 API 不含这一层。

**一句话**：Qwen3-ASR 增量流式**省掉了"切段 VAD"（A）**，但**"端点/轮次 VAD"（B）这个
职责依然存在**，商用版用 `server_vad` 实现、本服务用手动 `STOP` 代替。两者不矛盾。

### 1.1 官方佐证：商用 Qwen-ASR-Realtime 的"VAD 模式 / Manual 模式"

阿里云百炼的实时语音识别（Qwen-ASR-Realtime）文档明确把"断句由谁做"分成两种模式，
本质就是 `session.turn_detection` 开还是关：

- **VAD 模式（默认，`turn_detection` 配置为 server_vad）**：服务端自动检测语音起点/终点
  来断句，客户端只管持续发音频流，服务端在"检测到一句话结束"时自动返回最终结果。流程中
  服务端会发 `input_audio_buffer.speech_started` / `speech_stopped` 等事件——这就是上面说的
  **B 类端点 VAD**，由服务端那一层（server_vad）实现，**不是** ASR 内核在切段。

- **Manual 模式（`turn_detection` 设为 null）**：由**客户端**控制断句——发完一整句音频后，
  客户端发 `input_audio_buffer.commit` 通知服务端边界。适用于客户端能明确判断语句边界的
  场景（如"按住说话"、聊天发语音）。

> 对应关系：本服务 `serve_qwen3_asr_ws.py` 用客户端显式发 **`STOP`** 来标记一轮结束，
> 等价于商用的 **Manual 模式**（`turn_detection=null`，由客户端控制边界）。若要做成"服务端
> 自动断句"，就是去实现商用 **VAD 模式** 的那一层端点检测（server_vad），加在本服务的
> 增量转写之外，而不是在 Qwen3-ASR 转写内核里找。
>
> 文档：实时语音识别（Qwen-ASR-Realtime）交互流程
> （help.aliyun.com/zh/model-studio/qwen-asr-realtime-interaction-process）:   ”服务端自动检测语音的起点和终点（断句）。开发者只需持续发送音频流，服务端会在检测到一句话结束时自动返回最终识别结果。此模式适用于实时对话、会议记录等场景。“

### 1.2   `chunk-size-sec` 控制流式块大小

`chunk-size-sec` 控制流式块大小（默认 2.0）。值越小出字越快/越勤，但并发开销越大（实测 1.0 比 2.0 在L20 上，29秒音频 48路并发，1.0 全部失败，2.0 全部通过）。

---

## 2 必须用 vllm 0.14，不要用 0.19（rope_scaling / thinker_config 警告）

本服务需要vllm加速， `qwen-asr[vllm]`，[它锁定 `vllm==0.14.0`](https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml)。若换成更新的vllm版本, 比如 0.19.x，启动会**出现**：

```
Unrecognized keys in `rope_scaling` for 'rope_type'='default':
    {'mrope_section', 'mrope_interleaved', 'interleaved'}
thinker_config is None. Initializing thinker model with default values
```

**根因**：vllm 在 0.14 → 0.19 之间 （transformers 两版都是 4.57.6），config 解析里的
`patch_rope_scaling_dict` 会把 `rope_type` 从 `'mrope'` 改写成 `'default'`（它把 mrope
当 legacy、假设由 vllm 内部消化 `mrope_section` 等字段）：

```python
elif rope_scaling["rope_type"] == "mrope":
    assert "mrope_section" in rope_scaling
    rope_scaling["rope_type"] = "default"   # ← 改写
```

但 Qwen3-ASR 自带的 `Qwen3ASRThinkerTextRotaryEmbedding` 期望从 `rope_scaling` 里读到
`"mrope"` 才走多模态 RoPE 分支：

```python
self.rope_type = config.rope_scaling.get("rope_type", "default")
```

被 vllm 改写成 `"default"` 后，它走了普通 RoPE，`mrope_section` / `mrope_interleaved` /
`interleaved` 这几个键没人认领 → 打印 "Unrecognized keys" 警告，且音频/文本的多模态
位置编码退化。`thinker_config is None` 那条同源：0.19 的加载路径没正确解析 Qwen3-ASR 的
thinker 子配置，回退到默认参数。

**影响与抉择**：在 0.19 上服务"能起、也能出字"（两条是 WARNING/INFO 不是 ERROR），抽查
几条转写也"看着正常"；但位置编码退化对长音频/复杂内容可能有害，且**未做 CER 量化对比**，
无法判定等价。保守起见固定用 `qwen-asr[vllm]` 自带的 `vllm==0.14.0`。

### 2.1 vllm 加速需要用 Qwen3ASRModel

funasr 的`AutoModelVLLM` 不能加速 Qwen3-ASR , 必须要用 `from qwen_asr import Qwen3ASRModel`

>  这解答 `#3026`  的问题

### 2.2 关于模型下载

**官方推荐做法（[Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR#released-models-description-and-download)）**：如果运行环境不能在线下载，或无法访问 Hugging Face，先手动把权重下载到本地目录，再把本地路径传给 `--model`：

```bash
# ModelScope（国内推荐）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B

# 启动时指定本地模型目录
python serve_qwen3_asr_ws.py --model ./Qwen3-ASR-1.7B ...
```

注意：仅设置 `VLLM_USE_MODELSCOPE=True` 并安装 `modelscope` 只能接管一部分下载流程。vLLM 会从 ModelScope 拉取 config、tokenizer、merges、vocab、`model.safetensors.index.json` 等文件，但真正的权重 `model.safetensors` 仍可能回到 `huggingface.co` 拉取：

这看起来是 Qwen-ASR 当前下载链路的一个问题：

```
Downloading Model from https://www.modelscope.cn to directory: /home/vllm/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-1.7B
2026-06-28 10:15:56,301 - modelscope - INFO - Got 10 files, start to download ...
...
2026-06-28 10:15:57,023 - modelscope - INFO - Finish downloading 10 files for repo 'Qwen/Qwen3-ASR-1.7B'███████████| 12.2k/12.2k [00:00<00:00, 38.6kB/s]
...
INFO 06-28 10:15:59 [model.py:530] Resolved architecture: Qwen3ASRForConditionalGeneration
'(MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /Qwen/Qwen3-ASR-1.7B/resolve/main/model.safetensors (Caused by NewConnectionError("HTTPSConnection(host=\'huggingface.co\', port=443): Failed to establish a new connection: [Errno 101] Network is unreachable"))'), '(Request ID: f4f58ad5-e161-42ec-baad-cb2b8dbb63b9)')' thrown while requesting HEAD https://huggingface.co/Qwen/Qwen3-ASR-1.7B/resolve/main/model.safetensors
```

---

## 3. tokenizer 的 `fix_mistral_regex` 警告（无害，可忽略）

启动时可能出现：

```
The tokenizer you are loading from '.../Qwen3-ASR-1.7B' with an incorrect regex
pattern ... This will lead to incorrect tokenization. You should set the
`fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
```

**原因**：Qwen3-ASR 的 tokenizer 沿用了一类带已知 regex 问题的分词器实现，底层库检测到
该 regex 模式后给出提醒，建议加 `fix_mistral_regex=True` 修正切分。本服务通过 qwen-asr
的高层 API 加载模型、并不直接构造 tokenizer，没有暴露这个开关，所以这条提醒按原样打印。

**影响**：实测对中文 ASR 转写结果无可见影响（**抽查**多条转写正常，未做 CER 量化）。该
regex 修正主要影响某些特殊 token 的边界切分，对语音转写路径未观察到差异。属于"提醒级"
噪音，可忽略。若要彻底消除，需在更底层自行加载 tokenizer 时传 `fix_mistral_regex=True`，
但 qwen-asr 高层 API 当前不直接支持，且无实测必要。

---

### 顺带：另外两条启动日志（均无害）

- `Error retrieving safetensors: Repo id must be in the form ...`：把本地模型路径当成 HF
  仓库 id 去查线上元数据，失败后重试 2 次、回退本地加载，不影响功能。可设环境变量
  `HF_HUB_OFFLINE=1` 消除。

- `Downcasting torch.float32 to torch.bfloat16`：权重以 fp32 存、按 bf16 加载，正常省显存 /
  提速，bf16 与 fp32 指数位同宽，精度几乎无损。这是 INFO 不是错误。
