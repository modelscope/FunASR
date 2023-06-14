#  Paraformer online

该项目是一个基于iOS端的Paraformer流式识别的demo。该项目完全使用C++实现，可以方便的移植到不同平台。该项目不需要依赖其他库和额外的配置，直接在xcode上编译运行即可。

## 模型转换
这里是将torch模型转换成二进制格式的模型，方便通过内存映射的方式直接读取参数，加快模型读取速度。
```
tools/paraformer_convert.py model.pb
```
最终在当前目前下生成`model.bin`文件。然后将该文件拖入xcode项目中的`model`文件夹下，建议在用xcode打开项目后，再拖入。

## 识别流程
1. mac设备上确保安装xcode；
2. mac连接iPhone设备，双击`paraformer_online.xcodeproj`，自动打开xcode，直接运行该xcode工程。 

## 未完成工作
* 识别最后一个字丢失还未处理
* 动态chunk_size
* 推理速度还没有深入优化

