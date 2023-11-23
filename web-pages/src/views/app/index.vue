<template>
    <a-config-provider :locale="locale">
        <div
            class="app-container"
        >
            <div class="app-content" ref="app-content">
                <ui-scrollbar class="app-content-scrollbar" style="height: 100%;" :childOverWidth="false" :wrapStyle="[{ overflowX: 'hidden' }]">
                    <router-view :style="{ 'min-height': contentMinHeight + 'px' }" :contentMinHeight=" contentMinHeight" />
                </ui-scrollbar>
            </div>
        </div>
    </a-config-provider>
</template>

<script>
    import zhCN from 'ant-design-vue/lib/locale-provider/zh_CN'
    export default {
        name: 'app',
        components: {},
        props: { },
        data () {
            return {
                locale: zhCN,
                contentMinHeight: 0
            }
        },
        computed: {},
        beforeCreate () {
            const setHtmlFontSize = function () {
                const deviceWidth =
                    document.documentElement.clientWidth > 1280
                        ? document.documentElement.clientWidth <= 1920
                            ? document.documentElement.clientWidth
                            : 1920
                        : 1280
                document.getElementsByTagName('html')[0].style.cssText = 'font-size:' + deviceWidth / 100 + 'px !important'
            }
            let timerId
            if (window.addEventListener) {
                window.addEventListener(
                    'resize',
                    function () {
                        if (timerId) {
                            window.clearTimeout(timerId)
                        }
                        timerId = window.setTimeout(function () {
                            this.getContentMinHeight()
                        }, 300)
                    },
                    false
                )
            }
            setHtmlFontSize()
        },
        created () {},
        mounted () {
            this.getContentMinHeight()
        },
        methods: {
            getContentMinHeight () {
                this.contentMinHeight = this.$refs['app-content'].clientHeight
            }
        }
    }
</script>
<style src="./assets/css/index.scss" lang="scss"></style>
