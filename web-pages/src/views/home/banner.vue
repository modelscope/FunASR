<template>
  <div class="banner-comp">
    <div
      class="swiper-container"
      ref="swiper-container"
      :class="{ hiddenPagination: bannerList && bannerList.length < 2 }"
    >
      <div class="swiper-wrapper">
        <div
          class="swiper-slide"
          v-for="(item, index) in bannerList"
          :key="index"
        >
          <div class="item">
            <img class="banner-bg" :src="item.url" alt="" />
            <div v-if="item.flag === 1" class="content">
              <h3 class="yjqd">一键启动FUNASR</h3>
              <p class="text">
                FUNASR希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过发布工业级<br />
                语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的<br />
                研究和生产，并推动语音识别生态的发展。让语音识别更有趣！
              </p>

              <div class="lxwm">联系我们</div>
              <div class="jzmd-wrap">
                <div class="jzmd-title">捐赠名单</div>
                <div class="jzmd-content">
                  <div
                    class="jzmd-item"
                    v-for="(item, index) in jzmdList"
                    :key="index"
                  >
                    <div class="name">{{ item.name }}</div>
                    <div class="num-text">
                      <span class="text">{{ item.num }}</span>
                      <span class="unit">元</span>
                    </div>
                  </div>
                </div>
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
import Swiper from "swiper";
export default {
  name: "banner-comp",
  data() {
    return {
      bannerList: [
        {
          flag: 1,
          url: require("./assets/images/banner.png"),
        },
      ],
      swiperObj: null,
      jzmdList: [
        {
          name: "程俊陶",
          num: 300,
        },
        {
          name: "黄明",
          num: 236,
        },
        {
          name: "高志付",
          num: 235,
        },
        {
          name: "陈翔",
          num: 200,
        },
        {
          name: "夏侯杰",
          num: 200,
        },
        {
          name: "黄药师",
          num: 198,
        },
        {
          name: "肖炜",
          num: 100,
        },
        {
          name: "刘柱",
          num: 100,
        },
        {
          name: "黄水杯",
          num: 85,
        },
        {
          name: "子龙",
          num: 85,
        },
        {
          name: "梁浩锋",
          num: 66,
        },
        {
          name: "Alan",
          num: 66,
        },
        {
          name: "橙汁",
          num: 50,
        },
        {
          name: "马勇",
          num: 50,
        },
        {
          name: "车先生",
          num: 30.66,
        },
        {
          name: "陈书东",
          num: 22.22,
        },
        {
          name: "葵",
          num: 22,
        },
        {
          name: "H2G2",
          num: 20.24,
        },
        {
          name: "白明智",
          num: 20,
        },
        {
          name: "向前一标",
          num: 20,
        },
        {
          name: "Hey Song",
          num: 20,
        },
        {
          name: "javaweh",
          num: 20,
        },
        {
          name: "刘志伟",
          num: 20,
        },
        {
          name: "打鸟肉的小剑侍",
          num: 10,
        },
        {
          name: "小邓",
          num: 10,
        },
      ],
    };
  },
  mounted() {
    this.$nextTick(() => {
      this.initSwiper();
    });
  },
  methods: {
    initSwiper() {
      if (this.swiperObj) {
        this.swiperObj.destroy();
      }
      // const that = this
      this.swiperObj = new Swiper(this.$refs["swiper-container"], {
        slidesPerView: 1,
        direction: "vertical",
        pagination: {
          el: ".swiper-pagination",
          dynamicBullets: true,
        },
        on: {
          slideChangeTransitionEnd: function () {},
          init: function () {},
        },
      });
    },
  },
};
</script>
<style src="./assets/css/banner.scss" lang="scss"></style>
