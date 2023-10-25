#  Paraformer online

该项目是一个基于iOS端的Paraformer流式识别的demo。该项目是基于onnx的推理c++版本。不需要依赖其他库和额外的配置，直接在xcode上编译运行即可。

## 获取模型
1. 通过命令 `git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx.git` 下载模型
2. 下载的是量化后的模型，该iOS项目需要用到四个文件，`model_quant.onnx`，`decoder_quant.onnx`，`am.mvn`和`config.yaml`
3. 将上面四个文件拖入xcode项目中的`model`文件夹下，建议在打开xcode项目后再拖入。

## 识别流程
1. mac设备上确保安装xcode；
2. 由于onnx是xcode的一个cocopod包，需要先依赖onnx环境，打开终端，去到包含Podfile文件的文件夹下，执行`pod install`，拉取onnx环境
3. mac连接iPhone设备，双击`paraformer_online.xcworkspace`，自动打开xcode，直接运行该xcode工程。 

## 未来工作
* coreml支持
* 加入流式标点

