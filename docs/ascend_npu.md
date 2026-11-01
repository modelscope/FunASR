# 在昇腾 NPU（Ascend）上运行 FunASR

**验证范围**：本页记录 [#3708 原始贡献者的运行报告](https://github.com/modelscope/FunASR/pull/3708)，
环境为 Atlas 300I Pro（Ascend 310P3）、CANN 9.1.0-beta.1、torch 2.10.0+cpu、
torch_npu 2.10.0、funasr 1.3.14，模型为 fsmn-vad / paraformer / ct-transformer / cam++。
该报告中的 cam++ 使用了当时的池化规避方案；下文修订版 `seg_pooling` 的回归仅为 CPU
语义检查，不代表修订版已重新通过 NPU 验证。

上述报告不覆盖 Nano/Qwen3，也不构成对所有 Atlas 卡、CANN/torch_npu 组合或 FunASR
模型的兼容承诺。更换模型、模型版本或软件栈后，应另行验证设备执行、输出正确性与性能。

[#3708 后续报告](https://github.com/modelscope/FunASR/pull/3708#issuecomment-6558464421)
中的 Nano/Qwen3 `AutoConfig` 加载失败，与后来的 SenseVoiceSmall NPU 耗时报告是不同问题。
CAM++ 池化规避不能解决模型配置或 Transformers/Torch 的依赖兼容性；不要仅为识别
Qwen3 配置而单独升级 torch，破坏与 torch_npu/CANN 的配对。后续耗时报告也尚不能
证明 NPU 性能或识别质量已通过验证。

## 1. 安装注意

昇腾环境要求 torch 为 CPU 版并与 torch_npu 严格配对（如 torch 2.10.0+cpu + torch_npu 2.10.0）。
funasr 元数据不声明 torch（由用户自备环境）。在 #3707 记录的 dry-run 环境（已装
torch 2.10.0+cpu + torch_npu 2.10.0 + funasr 1.3.14，对默认安装命令做解析）中，
已满足的 torch 配对未被解析计划触碰；但在全新无 torch 的环境里，依赖解析可能安装
最新通用（CUDA）构建，与预装 torch_npu 不配对。
注意 `--no-deps` 只作用于下面第一条命令，后续普通 pip 安装仍会解析传递依赖；
执行这些安装前建议先跑 `pip install --dry-run --report` 检查解析计划，
或用 constraint 固定当前 torch（`pip install -c torch-constraint.txt ...`，
constraint 文件内容如 `torch==2.10.0+cpu`），不要默认已有配对必然保持不变。

本节清单对应 funasr **1.3.14** 的声明依赖（取自 PyPI 1.3.14 元数据），作为依赖完整性的基准：

```bash
pip install funasr==1.3.14 --no-deps

# funasr 1.3.14 完整声明依赖（install_requires）
pip install "scipy>=1.4.1" librosa "soundfile>=0.12.1" numpy "PyYAML>=5.1.2" tqdm requests \
    "omegaconf>=2.0" "hydra-core>=1.3.2" modelscope huggingface_hub safetensors transformers \
    tiktoken sentencepiece "kaldiio>=2.17.0" jieba jamo jaconv umap_learn "editdistance>=0.5.2" \
    torch_complex tensorboardX oss2

# 部分模型运行时还会用到（按需）
pip install kaldi-native-fbank zhconv tgt praat-parselmouth onnxruntime

# 安装后自检
pip check
```

其中 `hydra-core`、`modelscope`、`huggingface_hub`、`safetensors`、`transformers`、`tiktoken`
等在常见 NPU 基础镜像中通常已预装，是否缺项以 `pip check` 为准；
预装镜像可运行不代表清单本身完整，补包时仍应对照上面的声明依赖全集。

注意：在某个已配好环境 dry-run 中曾解析出 7 个缺失包（jaconv、jamo、oss2、
aliyun-python-sdk-core、aliyun-python-sdk-kms、crcmod、pycryptodome），
那只是该环境与镜像预装的当次差集，不是所有环境通用的依赖表；
其中 oss2 会带入 aliyun-python-sdk-*、crcmod、pycryptodome 等传递依赖。

## 2. 设备指定

```python
import torch_npu  # 必须先导入，注册 npu 设备
from funasr import AutoModel

model = AutoModel(
    model=<模型路径>,
    device="npu:0",
    disable_update=True,
    disable_log=True,
    disable_pbar=True,
)
```

## 3. CAM++（说话人模型）已知问题与等价规避

`funasr/models/campplus/components.py` 的 `seg_pooling` 使用
`F.avg_pool1d` / `F.max_pool1d`(kernel_size=100, stride=100, ceil_mode=True)。原始报告环境中该调用被
lower 为 `AvgPoolV2`，其融合算子仅支持 stride ∈ [1,63]，图编译失败并崩溃。

下面是拆分为基础归约算子的运行时规避方案，需在首次前向之前执行。
其 CPU 语义等价检查见下文；目标 NPU 上的算子支持、精度和性能仍需另行验证。
实现要点：完整段正常池化；**不完整尾段仅对其真实帧归约（显式补零会污染尾段均值/最大值，
与 ceil_mode 语义不一致）**；未知 stype 与原版一致抛出 ValueError：

```python
import torch
import torch.nn.functional as F
from funasr.models.campplus import components as cp

def seg_pooling(self, x, seg_len=100, stype="avg"):
    # numerically equivalent to avg_pool1d/max_pool1d(kernel_size=seg_len,
    # stride=seg_len, ceil_mode=True): the incomplete tail segment is
    # reduced over its REAL frames only (no zero padding).
    if stype not in ("avg", "max"):
        raise ValueError("Wrong segment pooling type.")
    B, C, T = x.shape
    n_full = T // seg_len
    tail = T - n_full * seg_len
    segs = []
    if n_full > 0:
        full = x[:, :, : n_full * seg_len].reshape(B, C, n_full, seg_len)
        segs.append(full.mean(dim=-1) if stype == "avg" else full.max(dim=-1).values)
    if tail > 0:
        tail_x = x[:, :, n_full * seg_len:]
        segs.append(tail_x.mean(dim=-1, keepdim=True) if stype == "avg"
                    else tail_x.max(dim=-1, keepdim=True).values)
    seg = torch.cat(segs, dim=-1) if len(segs) > 1 else segs[0]  # [B, C, nseg]
    shape = seg.shape
    seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
    return seg[..., :T]

cp.CAMLayer.seg_pooling = seg_pooling
```

**等价性回归**（仅 CPU 语义检查，torch 2.10.0+cpu，对照 `avg_pool1d`/`max_pool1d` + `ceil_mode=True`）：
长度 {50, 99, 100, 101, 110, 150, 199, 200, 201, 250, 1000} × {avg, max} ×
{全 1、全负、randn} 共 66 组用例，`torch.allclose(rtol=1e-4, atol=1e-6)` 全部通过；
残差仅为 float32 累加顺序噪声（最大 ~1.2e-7）。代表例：T=150 全 1 输入 avg 模式尾帧
= 1.0（与原版一致，错误补零实现会得到 0.5）。

**范围说明**：上述回归验证的是修订版在 CPU 上的归约语义等价；NPU 侧的兼容性与精度
需在实际昇腾环境中另行验证。原始贡献者的运行报告不能替代对本修订版在目标
模型和设备上的实测。

## 4. 调试技巧

NPU 算子异步下发，报错堆栈可能指向下一个同步点的无关模型（容易误判）。
定位真实出错算子：

```bash
export ASCEND_LAUNCH_BLOCKING=1   # 仅调试用，会显著降速，定位后取消
```

## 5. 预期行为

- 原始贡献者报告上述环境中的 fsmn-vad / paraformer / ct-transformer 可运行，
  cam++ 当时需池化规避；这不代表其他模型或软件栈已通过验证。
- 首次推理可能包含 NPU 图编译开销，但不能仅凭首次耗时认定是正常编译或排除故障。
  对同一输入分别记录模型加载、冷启动和预热后推理耗时，并在计时边界同步设备。
- 同时核对实际设备、调试开关及原始转录/时间戳等输出正确性；预热后仍慢、算子告警
  或输出异常需要单独定位，不能以“首次编译”解释所有问题。
