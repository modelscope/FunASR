此目录是为了建设www.funasr.com网站，目前已经有了域名与公网服务器，欢迎社区用户参与建设，主要从3方面：

- http服务器搭建
- html网页设计
- funasr相关材料

其中，http服务器，不限制编程语言，推荐python-http（可维护性好？）？

html设计可以参考whisper（ https://openai.com/research/whisper ）

需要展示内容：

1、funasr介绍

    可以AIGC生成个语音相关的炫酷配图
    
    配上文字介绍

    FunASR希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过发布工业级语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的研究和生产，并推动语音识别生态的发展。让语音识别更有趣！

2、核心功能

    FunASR是一个基础语音识别工具包，提供多种功能，包括语音识别（ASR）、语音端点检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离和多人对话语音识别等。

3、Paraformer模型介绍

    Paraformer模型结构图（已有）
    
    文字描述:

        Paraformer是一种非自回归端到端语音识别模型。非自回归模型相比于目前主流的自回归模型，可以并行的对整条句子输出目标文字，特别适合利用GPU进行并行推理。Paraformer是目前已知的首个在工业大数据上可以获得和自回归端到端模型相同性能的非自回归模型。配合GPU推理，可以将推理效率提升10倍，从而将语音识别云服务的机器成本降低接近10倍。

4、离线文件转写服务

    原理图
    
    文字介绍:

        FunASR离线文件转写软件包，提供了一款功能强大的语音离线文件转写服务。拥有完整的语音识别链路，结合了语音端点检测、语音识别、标点等模型，可以将几十个小时的长音频与视频识别成带标点的文字，而且支持上百路请求同时进行转写。输出为带标点的文字，含有字级别时间戳，支持ITN与用户自定义热词等。服务端集成有ffmpeg，支持各种音视频格式输入。软件包提供有html、python、c++、java与c#等多种编程语言客户端，用户可以直接使用与进一步开发。

    在线体验：
        https://121.43.113.106:1335/static/index.html

    安装：
    
        https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline_zh.md
    
    使用：
    
        https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
    
    
    视频教程链接：
    
        https://www.bilibili.com/video/BV11a4y1U72y/?share_source=copy_web&vd_source=f6576748261a1b738a71ad618d380438

5、实时听写

    原理图
    
    文字介绍：

        FunASR实时语音听写软件包，集成了实时版本的语音端点检测模型、语音识别、语音识别、标点预测模型等。采用多模型协同，既可以实时的进行语音转文字，也可以在说话句尾用高精度转写文字修正输出，输出文字带有标点，支持多路请求。依据使用者场景不同，支持实时语音听写服务（online）、非实时一句话转写（offline）与实时与非实时一体化协同（2pass）3种服务模式。软件包提供有html、python、c++、java与c#等多种编程语言客户端，用户可以直接使用与进一步开发。
    
    在线体验：
        https://121.43.113.106:1336/static/index.html

    安装：
    
        https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online_zh.md
    
    使用：

        https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz

    视频教程链接：

        https://www.bilibili.com/video/BV1Yw411K7LY/?share_source=copy_web&vd_source=f6576748261a1b738a71ad618d380438
    
        
    

6、github
    
    https://github.com/alibaba-damo-academy/FunASR

7、社区交流

    
# 部署
git clone https://github.com/alibaba-damo-academy/FunASR.git
cd FunASR/web-pages
npm install
# 开发模式
npm run dev
# 产品模式
npm run example
