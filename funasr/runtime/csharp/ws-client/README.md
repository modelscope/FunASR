# AliFsmnVadSharp
##### 简介：
本项目结合VAD模型、Websocket、NAudio等库开发FunASR-SDK客户端，实现了流式语音识别和音频文件转录功能。
##### 用途：
1.配合FunASR-SDK一键部署后台服务，实现了流式语音识别和音频文件转录功能;
2.实现了麦克风录制音频通过VAD检测，将有效音频保存在record文件夹下--代码已注释;
3.支持在此项目基础上修改，完成音频文件VAD检测，分割音频保存在本地--代码已注释;

##### 常用参数调整说明
VAD模型参数请参考cshape/AliFsmnVadSharp项目
serverUri:使用前请在代码内修改后台IP和端口
AudioFileQueue:转录文件路径保存在此队列中
##### 模型获取
speech_fsmn_vad_zh-cn-16k-common-pytorch
##### 调用方式：
编译完成后将config文件夹内文件复制到可执行程序路径下，如：AliFsmnVadSharp.Examples\bin\Debug\net6.0\
###### 1.添加项目引用
using AliFsmnVadSharp;

###### 2.初始化VAD模型和配置
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelFilePath = applicationBase + "./speech_fsmn_vad_zh-cn-16k-common-pytorch/model.onnx";
string configFilePath = applicationBase + "./speech_fsmn_vad_zh-cn-16k-common-pytorch/vad.yaml";
string mvnFilePath = applicationBase + "./speech_fsmn_vad_zh-cn-16k-common-pytorch/vad.mvn";
int batchSize = 2;//批量解码
AliFsmnVad aliFsmnVad = new AliFsmnVad(modelFilePath, configFilePath, mvnFilePath, batchSize);
```
###### 3.调用
方法一(适用于小文件)：
```csharp
SegmentEntity[] segments_duration = aliFsmnVad.GetSegments(samples);
```
方法二(适用于大文件)：
```csharp
SegmentEntity[] segments_duration = aliFsmnVad.GetSegmentsByStep(samples);
```
###### 4.输出结果：
```
Device Name: 麦克风阵列 (英特尔? 智音技术)
Device Channels:2
Device SampleRate:48000
Device BitsPerSample:32
load model and init config elapsed_milliseconds:796.3515625
[[0,10]]
[[0,10]]
文件名称:0.wav 文件转录内容: 试错的过程很简单啊,今特别是今天冒名插血卡的同学,你们可以听到后面的有专门的活动课,它会大大降低你的试错成本。其实你也可以过来听课,为什么你自己写嘛?我先今天写五个点,我就实试实验一下,发现这五个点不行,我再写五个点,这实再不行,那再写五个点嘛,你总会所谓的活动大神和所谓的高手就是只有一个把所有的。
[[0,10]]
[[0,10]]
文件名称:1.wav 文件转录内容: 正是因为存在绝对正义,所以我们接受现实的相对正义,但是不要因为现实的相对正义,我们就认为这个世界没有正义。因为如果当你认为这个世界没有正义。
[[0,10]]
文件名称:2.wav 文件转录内容: 但是现在不同啊,英国脱欧,欧盟内部完善的产业链的红利人。
[[0,10]]
文件名称:3.wav 文件转录内容: he must be home。now for the light is on,他一定在家,因为灯亮着就是有一种推理或者解释的一种感觉。
[[0,10]]
[[0,10]]
[[0,10]]
[[0,10]]
文件名称:4.wav 文件转录内容: after early nightfall, the yellow lamps would light up here in there, the squalid quarter of the broffles。
[[0,10]]
[[0,10]]
[[0,10]]
文件名称:5.wav 文件转录内容: 欢迎大家来体验达摩院推出的语音识别模型。
[[0,10]]
说明:本项目开启一个流式语音识别线程和一个音频文件转录线程，同时实现两种识别方式，文件转录需求高可在此基础上开启多线程同时进行文件转录。
```
输出的数据，例如：[0,10]，是以毫秒为单位的segement的起止时间，可以以此为依据对音频进行分片。其中静音噪音部分已被去除。无有效音频默认为[0,10]

其他说明：
测试用例：AliFsmnVadSharp.Examples。
测试环境：windows10。
测试用例中samples的计算,使用的是NAudio库。
未实现功能：此项目Websocket不支持SSL，启动FunASR_SDK请注意。
流式语音识别中使用的AliFsmnVadSharp项目中的kaldi-native-fbank-dll.dll已修改，否则会造成内存泄漏。
通过以下链接了解更多：
https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary
