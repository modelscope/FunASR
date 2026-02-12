# Paraformer LoRA 微调说明

本文档说明如何在 FunASR 中使用 LoRA 微调 Paraformer，并提供训练、推理与 CER 评测的完整示例。

## 1. 前置准备

1. 已准备好符合 FunASR 要求的 `train.jsonl` 与 `val.jsonl`。
2. 进入仓库根目录（示例路径）：

```bash
cd /workspace/FunASR
```

## 2. 训练配置

LoRA 配置文件：

```
examples/industrial_data_pretraining/paraformer/conf/paraformer_lora.yaml
```

关键字段说明：
- `model`: 基座模型名称或本地模型路径。
- `lora_only`: 是否只训练 LoRA 参数。
- `lora_bias`: LoRA 偏置训练策略（`none`/`all`/`lora_only`）。
- `encoder_conf.lora_*` / `decoder_conf.lora_*`: LoRA 参数（rank/alpha/dropout）。
- `train_data_set_list`/`valid_data_set_list`: 训练/验证集 jsonl。

如需覆盖配置，请通过命令行 `++key=value` 传参。

## 3. 训练脚本

脚本：

```
examples/industrial_data_pretraining/paraformer/lora_finetune.sh
```

你只需要确认脚本中的数据路径：

```bash
data_dir="${workspace}/data/list"
train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"
```

运行：

```bash
bash examples/industrial_data_pretraining/paraformer/lora_finetune.sh
```

训练日志与模型输出将保存在：

```
examples/industrial_data_pretraining/paraformer/outputs_lora
```

## 4. 推理脚本

推理脚本会读取 jsonl 输入并生成 `text.hyp` / `text.ref`：

- Python 脚本：`examples/industrial_data_pretraining/paraformer/lora_infer.py`
- Shell 封装：`examples/industrial_data_pretraining/paraformer/lora_infer.sh`

修改 `lora_infer.sh` 中路径后运行：

```bash
bash examples/industrial_data_pretraining/paraformer/lora_infer.sh
```

输出目录默认：

```
examples/industrial_data_pretraining/paraformer/outputs_lora/infer
```

## 5. CER 评测

评测脚本：

```
examples/industrial_data_pretraining/paraformer/lora_cer.sh
```

运行：

```bash
bash examples/industrial_data_pretraining/paraformer/lora_cer.sh
```

结果会输出 CER 统计到：

```
examples/industrial_data_pretraining/paraformer/outputs_lora/infer/text.cer
```

## 6. 常见问题

1. **训练不收敛或效果差**
   - 尝试调整 `lora_rank`、`lora_alpha`、`lora_dropout`。
   - 调整 `optim_conf.lr` 与 `train_conf.max_epoch`。

2. **推理报错找不到配置**
   - 确保训练输出目录中存在 `config.yaml`，并在推理脚本中设置正确的 `config_path` 和 `config_name`。

3. **多卡训练**
   - 设置 `CUDA_VISIBLE_DEVICES`，脚本会自动计算 `gpu_num`。

---

如需进一步定制，可直接在 `paraformer_lora.yaml` 中修改配置或在命令行传参覆盖。
