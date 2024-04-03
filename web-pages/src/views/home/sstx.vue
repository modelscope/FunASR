<template>
    <div class="sstx-comp">
        <div class="lxwjzxfw-plan">
            <div class="sstx-plan">
                <h3>
                    实时听写
                    <div></div>
                </h3>
            </div>

            <div class="sstx-lc">
                <div class="lc-top">
                    <div class="box box-1 box-min">
                        <div class="lc-plan">
                            <p>消息队列</p>
                            <p>(从客户端接收)</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/sstx-yp.png" alt="">
                            <p>音频</p>
                        </div>
                    </div>
                    <div class="box box-2">

                    </div>
                    <div class="box box-3 box-min">
                        <div class="lc-plan">
                            <p>消息队列</p>
                            <p>(发送到客户端)</p>
                        </div>
                        <div class="icon-plan">

                        </div>
                    </div>
                </div>
                <div class="lc-center">
                    <div class="box box-1">
                        <div class="lc-plan">
                            <p>FSMN-VAD-realtime</p>
                            <p>(端点检测实时)</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/sstx-fjyd.png" alt="">
                            <div>
                                <p>非静音段</p>
                                <p>间隔600ms</p>
                            </div>
                        </div>
                    </div>
                    <div class="box box-2">
                        <div class="lc-plan">
                            <p>Paraformmer-online</p>
                            <p>(语音识别实时)</p>
                        </div>
                        <div class="icon-plan1">
                            <img src="./assets/images/sstx-sbwz-max.png" alt="">
                            <div>
                                <p>识别文字</p>
                                <p>间隔600ms</p>
                            </div>
                        </div>
                        <div class="icon-plan2">
                            <img src="./assets/images/sstx-vad.png" alt="">
                            <p>VAD&nbsp;&nbsp;&nbsp;&nbsp;尾点</p>
                        </div>
                    </div>
                    <div class="box box-3">

                    </div>
                </div>
                <div class="lc-bottom">
                    <div class="box box-1 blue-box">
                        <div class="lc-plan">
                            <p>FSMN-VAD-realtime</p>
                            <p>(语音识别非实时)</p>
                        </div>
                        <div class="icon-plan1">
                            <img src="./assets/images/sstx-vad.png" alt="">
                            <p>VAD&nbsp;&nbsp;&nbsp;&nbsp;尾点</p>
                        </div>
                        <div class="icon-plan2">
                            <img src="./assets/images/sstx-sbwz.png" alt="">
                            <div>
                                <p>识别文字</p>
                            </div>
                        </div>
                    </div>
                    <div class="box box-2">
                        <div class="lc-plan">
                            <p>CT-Transformer</p>
                            <p>(标点预测)</p>
                        </div>
                        <div class="icon-plan">
                            <div class="icon-plan">
                                <img src="./assets/images/sstx-fjyd.png" alt="">
                            </div>
                        </div>
                    </div>
                    <div class="box box-3">
                        <div class="lc-plan">
                            <p>ITN</p>
                            <p>(逆文本正则化)</p>
                        </div>
                        <div class="icon-plan">
                            <img src="./assets/images/sstx-xz.png" alt="">
                            <div>
                                <p>修正&nbsp;&nbsp;&nbsp;识别文字</p>
                            </div>
                        </div>
                    </div>
                </div>

                <p class="lxwj-text">
                    FunASR实时语音听写软件包，集成了实时版本的语音端点检测模型、语音识别、语音识别、标点预测模型等。采用多模型协同，既可以实时的进行语音转文字，也可以在说话句尾用高精度转写文字修正输出，输出文字带有标点，支持多路请求。依据使用者场景不同，支持实时语音听写服务（online）、非实时一句话转写（offline）与实时与非实时一体化协同（2pass）3种服务模式。软件包提供有html、python、c++、java与c#等多种编程语言客户端，用户可以直接使用与进一步开发。
                </p>
            </div>

            <div class="xq-box">
                <div class="video-box">
                    <iframe src="//player.bilibili.com/player.html?aid=278639821&bvid=BV1Yw411K7LY&cid=1338073559&p=1&autoplay=0" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" > </iframe>
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
    name: 'sstx-comp',
    // components: {
    //     LivePlayer
    // },
    data () {
        return {
            btnList: [
                {
                    icon: require('./assets/images/lxwj-zxty.png'),
                    title: '在线体验',
                    link: 'https://www.funasr.com/static/online/index.html'
                },
                {
                    icon: require('./assets/images/lxwj-az.png'),
                    title: '安  装',
                    link: 'https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online_zh.md'
                },
                {
                    icon: require('./assets/images/lxwj-sy.png'),
                    title: '使  用',
                    link: 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz'
                }
            ],
            videoSrc: 'https://www.bilibili.com/video/BV1Yw411K7LY/?share_source=copy_web&vd_source=f6576748261a1b738a71ad618d380438'
        }
    },
    methods: {
        toPage (item) {
            if (item.link) window.open(item.link)
        }
    }
}
</script>
<style src="./assets/css/sstx.scss" lang="scss"></style>
