# AliParaformerAsr
##### 简介：
项目中使用的Asr模型是阿里巴巴达摩院提供的Paraformer-large ASR模型。
**项目基于Net 6.0，使用C#编写，调用Microsoft.ML.OnnxRuntime对onnx模型进行解码，支持跨平台编译。项目以库的形式进行调用，部署非常方便。**
ASR整体流程的rtf在0.03左右。

##### 用途：
Paraformer是达摩院语音团队提出的一种高效的非自回归端到端语音识别框架。本项目为Paraformer中文通用语音识别模型，采用工业级数万小时的标注音频进行模型训练，保证了模型的通用识别效果。模型可以被应用于语音输入法、语音导航、智能会议纪要等场景。

##### Paraformer模型结构：
![](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

Paraformer模型结构如上图所示，由 Encoder、Predictor、Sampler、Decoder 与 Loss function 五部分组成。Encoder可以采用不同的网络结构，例如self-attention，conformer，SAN-M等。Predictor 为两层FFN，预测目标文字个数以及抽取目标文字对应的声学向量。Sampler 为无可学习参数模块，依据输入的声学向量和目标向量，生产含有语义的特征向量。Decoder 结构与自回归模型类似，为双向建模（自回归为单向建模）。Loss function 部分，除了交叉熵（CE）与 MWER 区分性优化目标，还包括了 Predictor 优化目标 MAE。

其核心点主要有：

Predictor 模块：基于 Continuous integrate-and-fire (CIF) 的 预测器 (Predictor) 来抽取目标文字对应的声学特征向量，可以更加准确的预测语音中目标文字个数。
Sampler：通过采样，将声学特征向量与目标文字向量变换成含有语义信息的特征向量，配合双向的 Decoder 来增强模型对于上下文的建模能力。
基于负样本采样的 MWER 训练准则。
更详细的细节见：

论文： [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317 "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition")

论文解读：[Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw "Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型")

##### ASR常用参数（参考：asr.yaml文件）：
用于解码的asr.yaml配置参数，取自官方模型配置config.yaml原文件。便于跟进和升级。

##### paraformer-large onnx模型下载
https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx

## 离线（非流式）模型调用方法：

###### 1.添加项目引用
using AliParaformerAsr;

###### 2.模型初始化和配置
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelFilePath = applicationBase + "./speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.onnx";
string configFilePath = applicationBase + "./speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/asr.yaml";
string mvnFilePath = applicationBase + "./speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn";
string tokensFilePath = applicationBase + "./speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.txt";
AliParaformerAsr.OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
```
###### 3.调用
```csharp
List<float[]> samples = new List<float[]>();
//这里省略wav文件转samples...
//具体参考示例（AliParaformerAsr.Examples）代码
List<string> results_batch = offlineRecognizer.GetResults(samples);
```
###### 4.输出结果：
```
欢迎大家来体验达摩院推出的语音识别模型

正是因为存在绝对正义所以我们接受现实的相对正义但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

非常的方便但是现在不同啊英国脱欧欧盟内部完善的产业链的红利人

he must be home now for the light is on他一定在家因为灯亮着就是有一种推理或者解释的那种感觉

after early nightfall the yellow lamps would light up here in there the squalid quarter of the broffles

elapsed_milliseconds:1502.8828125
total_duration:40525.6875
rtf:0.037084696280599808
end!
```
*
处理长音频，推荐结合AliFsmnVad一起使用：https://github.com/manyeyes/AliFsmnVad 
*

其他说明：
测试用例：AliParaformerAsr.Examples。
测试环境：windows11。
测试用例中samples的计算,使用的是NAudio库。

通过以下链接了解更多：
https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary



# AliCTTransformerPunc
##### 简介：
项目中使用的Punc模型是阿里巴巴达摩院提供的Controllable Time-delay Transformer模型。
**项目基于Net 6.0，使用C#编写，调用Microsoft.ML.OnnxRuntime对onnx模型进行解码，支持跨平台编译。项目以库的形式进行调用，部署非常方便。**

##### 用途：
可用于语音识别模型输出文本的标点预测。

##### CTTransformerPunc模型结构：
Controllable Time-delay Transformer是达摩院语音团队提出的高效后处理框架中的标点模块。本项目为中文通用标点模型，模型可以被应用于文本类输入的标点预测，也可应用于语音识别结果的后处理步骤，协助语音识别模块输出具有可读性的文本结果。

![](https://www.modelscope.cn/api/v1/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

Controllable Time-delay Transformer 模型结构如上图所示，由 Embedding、Encoder 和 Predictor 三部分组成。Embedding 是词向量叠加位置向量。Encoder可以采用不同的网络结构，例如self-attention，conformer，SAN-M等。Predictor 预测每个token后的标点类型。

在模型的选择上采用了性能优越的Transformer模型。Transformer模型在获得良好性能的同时，由于模型自身序列化输入等特性，会给系统带来较大时延。常规的Transformer可以看到未来的全部信息，导致标点会依赖很远的未来信息。这会给用户带来一种标点一直在变化刷新，长时间结果不固定的不良感受。基于这一问题，我们创新性的提出了可控时延的Transformer模型（Controllable Time-Delay Transformer, CT-Transformer），在模型性能无损失的情况下，有效控制标点的延时。

##### Punc常用参数（参考：punc.yaml文件）：
用于解码的punc.yaml配置参数，取自官方模型配置config.yaml原文件。

##### punc onnx模型下载
https://huggingface.co/manyeyes/punc_ct-transformer_zh-cn-common-vocab272727-onnx

## 模型调用方法：

###### 1.添加项目引用
using AliCTTransformerPunc;

###### 2.模型初始化和配置
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelFilePath = applicationBase + "./punc_ct-transformer_zh-cn-common-vocab272727-pytorch/model.onnx";
string configFilePath = applicationBase + "./punc_ct-transformer_zh-cn-common-vocab272727-pytorch/punc.yaml";
string tokensFilePath = applicationBase + "./punc_ct-transformer_zh-cn-common-vocab272727-pytorch/tokens.txt";
AliCTTransformerPunc.CTTransformer ctTransformer = new CTTransformer(modelFilePath, configFilePath, tokensFilePath);
```
###### 3.调用
```csharp
string text = "跨境河流是养育沿岸人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切愿意进一步完善双方联合工作机制凡是中方能做的我们都会去做而且会做得更好我请印度朋友们放心中国在上游的任何开发利用都会经过科学规划和论证兼顾上下游的利益";
string result = ctTransformer.GetResults(text);
```
###### 4.输出结果：
```
load_model_elapsed_milliseconds:979.125
跨境河流是养育沿岸人民的生命之源。长期以来，为帮助下游地区防灾减灾。中方技术人员在上游地区极为恶劣的自然条件下克服巨大困难，甚至冒着生命危险，向印方提供汛期水文资料处理紧急事件，中方重视印方在跨境河流问题上的关切，愿意进一步完善双方联合工作机制。凡是中方能做的，我们都会去做，而且会做得更好。我请印度朋友们放心中国在上游的任何开发利用，都会经过科学规划和论证，兼顾上下游的利益。
elapsed_milliseconds:381.6953125
end!
```

其他说明：
测试用例：AliCTTransformerPunc.Examples。
测试环境：windows11。

通过以下链接了解更多：
https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary



# AliFsmnVad
##### 简介：
项目中使用的VAD模型是阿里巴巴达摩院提供的FSMN-Monophone VAD模型。
**项目基于Net 6.0，使用C#编写，调用Microsoft.ML.OnnxRuntime对onnx模型进行解码，支持跨平台编译。项目以库的形式进行调用，部署非常方便。**
VAD整体流程的rtf在0.008左右。

##### 用途：
16k中文通用VAD模型：可用于检测长语音片段中有效语音的起止时间点.
FSMN-Monophone VAD是达摩院语音团队提出的高效语音端点检测模型，用于检测输入音频中有效语音的起止时间点信息，并将检测出来的有效音频片段输入识别引擎进行识别，减少无效语音带来的识别错误。

##### VAD常用参数调整说明（参考：vad.yaml文件）：
max_end_silence_time：尾部连续检测到多长时间静音进行尾点判停，参数范围500ms～6000ms，默认值800ms(该值过低容易出现语音提前截断的情况)。
speech_noise_thres：speech的得分减去noise的得分大于此值则判断为speech，参数范围：（-1,1）
取值越趋于-1，噪音被误判定为语音的概率越大，FA越高
取值越趋于+1，语音被误判定为噪音的概率越大，Pmiss越高
通常情况下，该值会根据当前模型在长语音测试集上的效果取balance


##### fsmnvad onnx模型下载
https://huggingface.co/manyeyes/speech_fsmn_vad_zh-cn-16k-common-onnx

##### 调用方式：
###### 1.添加项目引用
###### 2.初始化模型和配置
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
load model and init config elapsed_milliseconds:463.5390625
vad infer result:
[[70,2340][2620,6200][6480,23670][23950,26250][26780,28990][29950,31430][31750,37600][38210,46900][47310,49630][49910,56460][56740,59540][59820,70450]]
elapsed_milliseconds:662.796875
total_duration:70470.625
rtf:0.009405292985552491
```
输出的数据，例如：[70,2340]，是以毫秒为单位的segement的起止时间，可以以此为依据对音频进行分片。其中静音噪音部分已被去除。

其他说明：
测试用例：AliFsmnVad.Examples。
测试环境：windows11。
测试用例中samples的计算,使用的是NAudio库。

通过以下链接了解更多：
https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary
