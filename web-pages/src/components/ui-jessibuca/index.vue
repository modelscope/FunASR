<template>
    <div class="ui-jessibuca"></div>
</template>

<script>
    export default {
        name: 'ui-jessibuca',
        props: {
            videoUrl: {
                type: String,
                default: ''
            },
            showBandwidth: {
                type: Boolean,
                default: true
            },
            fullscreenFlag: {
                type: Boolean,
                default: true
            },
            screenshotFlag: {
                type: Boolean,
                default: true
            },
            playFlag: {
                type: Boolean,
                default: true
            },
            audioFlag: {
                type: Boolean,
                default: true
            }
        },
        data () {
            return {
                jessibuca: null
            }
        },
        watch: {
            videoUrl: {
                immediate: true,
                handler (newVal, oldVal) {
                    if (newVal) {
                        this.init()
                        setTimeout(() => {
                            this.jessibuca.play(newVal)
                        }, 100)
                    } else if (this.jessibuca) {
                        this.jessibuca.destroy()
                        this.jessibuca = null
                    }
                }
            }
        },
        created () {},
        mounted () {
            this.init()
        },
        beforeDestroy () {
            this.jessibuca && this.jessibuca.destroy()
        },
        methods: {
            init () {
                const that = this
                if (!this.jessibuca && this.$el) {
                    this.jessibuca = new window.Jessibuca({
                        container: this.$el,
                        videoBuffer: 0.2, // 缓存时长
                        isResize: false,
                        text: '',
                        loadingText: '加载中',
                        debug: false,
                        useMSE: true, // 开启硬编码
                        showBandwidth: this.showBandwidth, // 显示网速
                        controlAutoHide: true,
                        operateBtns: {
                            fullscreen: this.fullscreenFlag,
                            screenshot: this.screenshotFlag,
                            play: this.playFlag,
                            audio: this.audioFlag
                        },
                        forceNoOffscreen: true,
                        isNotMute: false
                    })
                    this.jessibuca.on('error', function (error) {
                        console.log('error:', error)
                    })

                    this.jessibuca.on('fullscreen', function (flag) {
                        console.log('is fullscreen', flag)
                        that.$emit('fullscreen', flag)
                    })

                    this.jessibuca.on('webFullscreen', function (flag) {
                        console.log('is webFullscreen', flag)
                        that.$emit('webFullscreen', flag)
                    })
                }
            },
            setFullscreen (flag) {
                this.jessibuca.setFullscreen(flag)
            }
        }
    }
</script>
<style src="./assets/css/index.scss" lang="scss"></style>
