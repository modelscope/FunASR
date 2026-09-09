(简体中文 | [English](./installation.md))

# 安装 Python SDK

> **只需 Fun-ASR-Nano 原生推理？** 使用 [Transformers 5.17.0 快速开始](../transformers_native_zh.md)。它加载独立的 `-hf` 权重，不要求 FunASR 工具库。本页保留 `funasr.AutoModel` 工具库路径，两者的依赖、参数和输出不可混用。

本页适用于 `from funasr import AutoModel`。需要打包好的 C++ 服务时，请先看 [Docker 与运行时镜像](./docker_zh.md)。安装完成后进入 [SDK 教程](../tutorial/README_zh.md)。

## 1. 创建独立环境

以下命令以已安装的 Python 3.11 为例，不表示所有模型或后端都支持所有 Python 版本。[setup.py](../../setup.py) 声明 `python_requires=">=3.7.0"`，但实际解析出的依赖和具体模型可能要求更高版本。[pyproject.toml](../../pyproject.toml) 定义构建后端，不是锁定的推理环境。

Linux/macOS：

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

也可以使用已有 Conda 环境，参见 [Conda 安装参考](https://docs.conda.io/en/latest/miniconda.html#windows-installers)。Apple Silicon 应使用架构一致的 arm64 解释器和 wheel，不要将 x86_64 Conda 环境与 arm64 包混用。

## 2. 先安装 PyTorch，再选择一种 FunASR 安装方式

根据解释器、操作系统和加速设备，在 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)或[版本兼容参考](https://pytorch.org/get-started/previous-versions/)中选择匹配的 `torch` 与 `torchaudio`。FunASR 核心包不会替你选择这些构建。不要仅凭本机 CUDA toolkit 版本判断 wheel 是否兼容。

### PyPI：使用已发布的软件包

```sh
python -m pip install --upgrade funasr
```

这会从已配置的索引安装可用软件包，不是当前 Git 工作区。当前源码文档中的模型或功能未必已包含在该发行包中。需要复现时，将不锁版本的安装命令改为已验证的精确版本。当前工作区版本记录在 [version.txt](../../funasr/version.txt)，不能据此证明索引上已有该版本。

### 源码：使用当前工作区及配套示例

在已有 FunASR 仓库根目录执行：

```sh
python -m pip install -e .
git rev-parse HEAD
```

尚无工作区时，可以先克隆[仓库](https://github.com/modelscope/FunASR)：

```sh
git clone https://github.com/modelscope/FunASR.git
cd FunASR
python -m pip install -e .
```

可编辑安装会直接导入该目录中的源码，请记录 commit 和本地修改；它不会另存一份代码。仓库中的 `examples/`、`runtime/` 和文档并不都会随 PyPI 包安装。

本工作区已将 `modelscope` 和 `huggingface_hub` 列为核心依赖，不需要再按“可选步骤”安装。模型专属依赖仍需单独处理。`knf` extra 为支持的无 torchaudio 路径提供 `kaldi-native-fbank`，`silero` extra 提供 Silero VAD；标准入门示例不需要它们。添加 extra 前请查看 [setup.py](../../setup.py) 和所选模型指南，避免在同一环境中混装互不兼容的模型依赖。

## 3. 验证解释器和导入

在之后用于推理的同一个已激活环境中执行：

```sh
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip --version
python -m pip check
python -c "import funasr, torch, torchaudio; from funasr import AutoModel; print('funasr:', funasr.__version__, funasr.__file__); print('torch:', torch.__version__, 'torchaudio:', torchaudio.__version__); print('CUDA available:', torch.cuda.is_available()); print('AutoModel import OK')"
```

这些命令仅检查依赖和导入，不是模型下载或推理测试。`funasr.__file__` 应指向预期安装位置；检查 PyPI 环境时，请避开其他源码工作区的导入遮蔽。教程显式使用 CPU 起步。导入成功或设备可用不代表某个模型已经在该设备上验证通过。

本源码工作区出现注册或导入失败时，可查看 `funasr.get_import_errors()`，或在启动 Python 前设置 `FUNASR_IMPORT_DEBUG=1`。其他可选模型的缺失依赖不一定影响当前模型。批量升级整个环境前，请先阅读[常见问题](../troubleshooting_zh.md)。

## 4. 模型、缓存与离线使用

`AutoModel` 接受模型 ID/别名或已有的本地模型目录。默认 `hub="ms"` 使用 ModelScope，`hub="hf"` 使用 Hugging Face。权重与 Python 包分开下载，由 hub 客户端管理缓存；加载后可通过 `model.model_path` 查看实际目录。请预留可写磁盘空间，并保留配置、分词器、前端资源与权重组成的完整目录。

需要复现或断网运行时：

1. 在允许联网的机器上准备精确模型快照和依赖，记录完整模型 ID、hub revision/commit、许可、软件包版本和文件校验和。VAD、标点、说话人模型也应分别准备。
2. 传输完整目录与依赖制品，把所有模型别名替换为本地目录，把输入 URL 替换为本地文件，并检查配置是否继续引用远程资源。
3. 设置 `disable_update=True` 可跳过 FunASR 启动版本检查，但它**不是** hub 客户端、模型代码或缺失资源的离线开关。应在禁用外连的环境中验证准备结果。

以下 Python 示例要求 `./models/paraformer-zh` 中已有完整、审查过的 Paraformer 兼容快照，并准备本地 `./audio.wav`：

```python
from pathlib import Path
from funasr import AutoModel

model_dir = Path("./models/paraformer-zh").resolve()
audio = Path("./audio.wav").resolve()
assert model_dir.is_dir(), model_dir
assert audio.is_file(), audio
model = AutoModel(
    model=str(model_dir), device="cpu", disable_update=True,
    trust_remote_code=False,
)
print(model.generate(input=str(audio)))
```

源码限制：ModelScope 辅助函数会将 `model_revision` 传给下载器，但本工作区的 Hugging Face 辅助函数调用 `snapshot_download(model)`，没有转发 revision。因此这里不能依赖 `AutoModel(..., hub="hf", model_revision=...)` 锁定版本；应通过 hub 客户端取得固定快照，再传入本地目录。参见[下载实现](../../funasr/download/download_model_from_hub.py)。

## 5. 信任与许可

- 默认保留 `trust_remote_code=False`，只有模型确实需要且代码已经审查时才开启。加载器在开启信任后可能安装模型的 `requirements.txt`，ModelScope 路径还可能导入配置的 `remote_code`。本地目录并不天然可信。
- 使用独立环境、审查过的制品和最小文件系统/网络权限。不要把 hub 凭据写进脚本、镜像、日志或问题反馈；使用 hub 客户端支持的认证方式。
- FunASR 软件采用 [MIT 许可](../../LICENSE)，模型权重采用各自条款，请查看精确模型卡和[模型仓库](../../model_zoo/readme_zh.md)。只有模型卡采用时，[模型许可协议](../../MODEL_LICENSE)才适用。第三方模型保留各自来源与许可。

下一步：[完成第一次转写](../tutorial/README_zh.md)。
