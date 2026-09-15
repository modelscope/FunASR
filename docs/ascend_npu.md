# 在昇腾 NPU（Ascend）上运行 FunASR

适用环境：Atlas 推理/训练卡 + CANN + torch_npu。
已在 Atlas 300I Pro（Ascend 310P3）、CANN 9.1.0-beta.1、torch 2.10.0+cpu、torch_npu 2.10.0、funasr 1.3.14 上完整验证。

## 1. 安装注意

昇腾环境要求 torch 为 CPU 版并与 torch_npu 严格配对（如 torch 2.10.0+cpu + torch_npu 2.10.0）。
建议使用 `--no-deps` 安装 funasr 以避免依赖解析替换 torch，随后手动补齐运行时依赖：

```bash
pip install funasr --no-deps
pip install torch_complex kaldiio omegaconf librosa kaldi-native-fbank \
    editdistance jieba zhconv tgt umap-learn praat-parselmouth \
    tensorboardX onnxruntime sentencepiece
```

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
`F.avg_pool1d(kernel_size=100, stride=100, ceil_mode=True)`。昇腾上该调用被
lower 为 `AvgPoolV2`，其融合算子仅支持 stride ∈ [1,63]，图编译失败并崩溃。

运行时等价替换（数值一致，仅用 NPU 支持的基础算子），需在首次前向之前执行：

```python
from funasr.models.campplus import components as cp
import torch.nn.functional as F

def seg_pooling(self, x, seg_len=100, stype="avg"):
    B, C, T = x.shape
    pad = (-T) % seg_len                      # 等价 ceil_mode
    xp = F.pad(x, (0, pad), value=0.0) if pad else x
    xg = xp.reshape(B, C, (T + pad) // seg_len, seg_len)
    seg = xg.mean(dim=-1) if stype == "avg" else xg.max(dim=-1).values
    shape = seg.shape
    seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
    return seg[..., :T]

cp.CAMLayer.seg_pooling = seg_pooling
```

## 4. 调试技巧

NPU 算子异步下发，报错堆栈可能指向下一个同步点的无关模型（容易误判）。
定位真实出错算子：

```bash
export ASCEND_LAUNCH_BLOCKING=1   # 仅调试用，会显著降速，定位后取消
```

## 5. 预期行为

- 首次推理包含 NPU 图编译开销（几十秒量级），属正常现象，非卡死；
- fsmn-vad / paraformer / ct-transformer 可直接运行；
- cam++ 需上述 seg_pooling 补丁后运行。
