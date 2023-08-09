#  Paraformer online

该项目是一个基于iOS端的Paraformer流式识别的demo。该项目是基于onnx的推理c++版本。不需要依赖其他库和额外的配置，直接在xcode上编译运行即可。

## 模型转换
拉取funasr最新版本代码，使用最新的转换脚本获取onnx模型，将生成的两个onnx模型，默认导出的model.onnx模型改为encoder.onnx，然后将decoder.onnx和encoder.onnx模型拖入xcode项目中的`model`文件夹下，建议在用xcode打开项目后，再拖入。

## 识别流程
1. mac设备上确保安装xcode；
2. 由于onnx是xcode的一个cocopod包，需要先依赖onnx环境，打开终端，去到包含Podfile文件的文件夹下，执行`pod install`，拉取onnx环境
3. mac连接iPhone设备，双击`paraformer_online.xcworkspace`，自动打开xcode，直接运行该xcode工程。 

## 未完成工作
* coreml支持
* 加入流式标点

