# ONNX Runtime 原生二进制 JSONL 输出

ONNX Runtime 原生二进制默认通过日志输出识别结果。当其他程序需要逐条读取结构化结果时，增加 `--output-format jsonl`，每个完成的输入会输出一行 JSON。

## 离线识别

在 `runtime/onnxruntime` 目录执行：

```shell
./build/bin/funasr-onnx-offline \
  --model-dir /path/to/timestamp-capable-model \
  --wav-path /path/to/audio.wav \
  --output-format jsonl \
  > results.jsonl 2> runtime.log
```

`funasr-onnx-offline-rtf` 同样支持该选项。

## 两遍识别

```shell
./build/bin/funasr-onnx-2pass \
  --model-dir /path/to/offline-model \
  --online-model-dir /path/to/online-model \
  --vad-dir /path/to/vad-model \
  --punc-dir /path/to/online-punctuation-model \
  --wav-path /path/to/audio.wav \
  --mode 2pass \
  --output-format jsonl \
  > results.jsonl 2> runtime.log
```

`funasr-onnx-2pass-rtf` 同样支持该选项。合法模式为 `offline`、`online` 和 `2pass`。原生两遍识别运行时需要同时提供 VAD 模型和在线标点模型才能完成推理。

## 输出结构

标准输出中的每一行都是一个独立 JSON 对象：

```json
{"key":"utt-001","mode":"offline","stamp_sents":[{"end":920,"start":0,"text_seg":"你好世界"}],"text":"你好世界","timestamp":[[0,420],[460,920]]}
```

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `key` | string | `wav.scp` 每行第一列；直接传入单个文件时为 `wav_default_id`。 |
| `mode` | string | `offline`、`online` 或 `2pass`。 |
| `text` | string | 完整转写文本；`2pass` 模式下为离线纠错后的最终文本。 |
| `timestamp` | array | 模型返回的原生词元或词级时间戳数组。 |
| `stamp_sents` | array | 二进制和模型支持时返回的原生句级时间戳对象；两遍识别二进制当前在此字段返回空数组。 |

需要使用支持时间戳的模型才能得到时间戳值。模型没有返回时间戳，或原生数据为空、格式错误、不是 JSON 数组时，相应字段固定输出 `[]`。

## 流式与批量行为

- JSON 记录只写入标准输出；运行诊断和现有进度日志仍写入标准错误。
- 输入 `wav.scp` 时，每个成功完成的 key 输出一行。
- RTF 二进制会保证单行 JSON 不被多个线程交错写入，但不同任务的完成顺序不固定。请通过 `key` 关联结果，不要依赖行顺序。
- 不传 `--output-format` 等同于 `--output-format log`，保留原有日志行为。
