<template>
    <div class="banner-comp">
        <div class="swiper-container" ref="swiper-container" :class="{hiddenPagination: bannerList && bannerList.length<2}">
            <div class="swiper-wrapper">
                <div class="swiper-slide" v-for="(item,index) in bannerList" :key="index">
                    <div class="item">
                        <img class="banner-bg" :src="item.url" alt="">
                        <div v-if="item.flag === 1" class="content">
                            <h3 class="yjqd">
                                一键启动FUNASR
                            </h3>
                            <p class="text">
                                FUNASR希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过发布工业级<br />
                                语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的<br />
                                研究和生产，并推动语音识别生态的发展。让语音识别更有趣！
                            </p>

                            <div class="lxwm">
                                联系我们
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="swiper-pagination"></div>
        </div>
    </div>
</template>

<script>
import Swiper from 'swiper'
export default {
    name: 'banner-comp',
    data () {
        return {
            bannerList: [
                {
                    flag: 1,
                    url: require('./assets/images/banner.png')
                }
            ],
            swiperObj: null
        }
    },
    mounted () {
        this.$nextTick(() => {
            this.initSwiper()
        })
    },
    methods: {
        initSwiper () {
            if (this.swiperObj) {
                this.swiperObj.destroy()
            }
            // const that = this
            this.swiperObj = new Swiper(this.$refs['swiper-container'], {
                slidesPerView: 1,
                direction: 'vertical',
                pagination: {
                    el: '.swiper-pagination',
                    dynamicBullets: true
                },
                on: {
                    slideChangeTransitionEnd: function () {

                    },
                    init: function () {

                    }
                }
            })
        }
    }
}
</script>
<style src="./assets/css/banner.scss" lang="scss"></style>
