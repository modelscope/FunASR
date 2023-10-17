# AndroidClient

先说明，本项目是使用WebSocket连接服务器的语音识别服务，并不是将FunASR部署到Android里，服务启动方式请查看文档[SDK_advanced_guide_online_zh.md](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/docs/SDK_advanced_guide_online_zh.md)。

使用最新的 Android Studio 打开`AndroidClient`项目，运行即可，在运行之前还需要修改`ASR_HOST`参数，该参数是语音识别服务的WebSocket接口地址，需要修复为开发者自己的服务地址。

应用只有一个功能，按钮下开始识别，松开按钮结束识别。

应用效果图：

<div align="center">
  <img src="./images/demo.png" alt="应用效果图" width="300">
</div>
