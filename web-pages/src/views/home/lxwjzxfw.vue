<template>
    <div class="lxwjzxfw-comp">
        <div class="lxwjzxfw-plan">
            <h3>
                离线文件转写服务
                <div></div>
            </h3>

            <div class="zxfl-lc">
                <div class="lc-top">
                    <div class="box box-1">
                        <div class="lc-plan">
                            <p>消息队列</p>
                            <p>(从客户端接收)</p>
                        </div>
                        <div class="icon-plan xxld">
                            <img src="./assets/images/lxwj-to-bottom.png" alt="">
                        </div>
                    </div>
                    <div class="box box-2">

                    </div>
                    <div class="box box-3 blue-box">
                        <div class="lc-plan">
                            <p>语言模型</p>
                            <p>Ngram/Token/Lex</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/lxwj-yymx.png" alt="">
                        </div>
                    </div>
                    <div class="box box-4 blue-box">
                        <div class="lc-plan">
                            <p>消息队列</p>
                            <p>(从客户端接收)</p>
                        </div>
                    </div>
                    <div class="box box-5">
                        <div class="lc-plan">
                            <p>消息队列</p>
                            <p>(从客户端接收)</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/lxwj-to-top.png" alt="">
                            <img class="text-img" src="./assets/images/lxwj-text.png" alt="">
                        </div>
                    </div>
                </div>

                <div class="lc-bottom">
                    <div class="box box-1">
                        <div class="lc-plan">
                            <p>语音端点检测</p>
                            <p>FSMN-VAD</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/lxwj-to-right.png" alt="">
                        </div>
                    </div>
                    <div class="box box-2">
                        <div class="lc-plan">
                            <p>声学模型</p>
                            <p>Paraformer</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/lxwj-to-right.png" alt="">
                        </div>
                    </div>
                    <div class="box box-3 blue-box">
                        <div class="lc-plan">
                            <p>解码器</p>
                            <p>Wfst decoder</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/lxwj-to-right.png" alt="">
                        </div>
                    </div>
                    <div class="box box-4">
                        <div class="lc-plan">
                            <p>标点预测</p>
                            <p>CT-Transformer</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/lxwj-to-right.png" alt="">
                        </div>
                    </div>
                    <div class="box box-5">
                        <div class="lc-plan">
                            <p>逆文本正则化</p>
                            <p>ITN</p>
                        </div>
                    </div>
                </div>

                <p class="lxwj-text">
                    FunASR离线文件转写软件包，提供了一款功能强大的语音离线文件转写服务。拥有完整的语音识别链路，结合了语音端点检测、语音识别、标点等模型，可以将几十个小时的长音频与视频识别成带标点的文字，而且支持上百路请求同时进行转写。输出为带标点的文字，含有字级别时间戳，支持ITN与用户自定义热词等。服务端集成有ffmpeg，支持各种音视频格式输入。软件包提供有html、python、c++、java与c#等多种编程语言客户端，用户可以直接使用与进一步开发。
                </p>
            </div>

            <div class="xq-box">
                <div class="video-box">
                    <iframe src="//player.bilibili.com/player.html?aid=663707374&bvid=BV11a4y1U72y&cid=1338069747&p=1&autoplay=0" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="false"> </iframe>
                    <!-- <LivePlayer
                        :autoplay="false"
                        :videoUrl="videoSrc"
                    ></LivePlayer> -->
                </div>

                <div class="btn-box">
                    <li v-for="(item, index) in btnList" :key="index" @click="toPage(item)">
                        <img :src="item.icon" alt="">
                        <p>{{item.title}}</p>
                    </li>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
// import LivePlayer from '@liveqing/liveplayer'
export default {
    name: 'lxwjzxfw-comp',
    // components: {
    //     LivePlayer
    // },
    data () {
        return {
            btnList: [
                {
                    icon: require('./assets/images/lxwj-zxty.png'),
                    title: '在线体验',
                    link: 'https://www.funasr.com/static/offline/index.html'
                },
                {
                    icon: require('./assets/images/lxwj-az.png'),
                    title: '安  装',
                    link: 'https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline_zh.md'
                },
                {
                    icon: require('./assets/images/lxwj-sy.png'),
                    title: '使  用',
                    link: 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz'
                }
            ],
            videoSrc: 'https://www.bilibili.com/video/BV11a4y1U72y/?share_source=copy_web&vd_source=f6576748261a1b738a71ad618d380438'
            // videoSrc: 'https://www.bilibili.com/video/BV11a4y1U72y/?share_source=copy_web&vd_source=f6576748261a1b738a71ad618d380438'
        }
    },
    methods: {
        toPage (item) {
            if (item.link) window.open(item.link)
        }
    }
}
</script>
<style src="./assets/css/lxwjzxfw.scss" lang="scss"></style>
