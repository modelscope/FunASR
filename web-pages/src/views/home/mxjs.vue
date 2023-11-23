<template>
    <div class="mxjs-comp">
        <div class="mxjs-plan">
            <h3>
                Paraformer模型介绍
                <div></div>
            </h3>
            <img src="./assets/images/mxjs.png" alt="">
            <p>
                Paraformer是一种非自回归端到端语音识别模型。非自回归模型相比于目前主流的自回归模型，可以并行的对整条句子输出目标文字，特别适合利用GPU进行并行推理。Paraformer是目前已知的首个在工业大数据上可以获得和自回归端到端模型相同性能的非自回归模型。配合GPU推理，可以将推理效率提升10倍，从而将语音识别云服务的机器成本降低接近10倍。
            </p>
        </div>
    </div>
</template>
<script>
export default {
    name: 'mxjs-comp',
    data () {
        return {
        }
    },
    methods: {}
}
</script>
<style src="./assets/css/mxjs.scss" lang="scss"></style>
