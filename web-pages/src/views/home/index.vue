<template>
    <div class="page page-home">
        <header class="page-home-header">
            <img class="logo-img" src="./assets/images/logo.png" alt="">
            <ul>
                <li :ref="item.ref" v-for="(item, index) in headerList" :key="index" @click="anchorTo(item)">
                    <p>{{item.name}}</p>
                </li>
                <li @click="toPage({link: 'https://github.com/alibaba-damo-academy/FunASR'})">
                    Github
                </li>
                <li>
                    社区交流
                </li>
                <div :style="{width: lineWidth + 'px', left: lineLeft + 'px'}" class="line"></div>
            </ul>
            <div class="search-box">
                <img src="./assets/images/ss.png" alt="">
                <a-input placeholder="请输入关键字" />
            </div>
            <div class="language-box">
                中文/EN
            </div>
        </header>

        <banner-comp ref="funasrJs" />

        <hxgn-comp ref="hxgn" />

        <mxjs-comp ref="mxjs" />

        <lxwjzxfw-comp ref="lxwj" />

        <sstx-comp ref="sstx" />

        <footer class="page-home-footer">
            <div class="gywm">
                <img src="./assets/images/footer-logo.png" alt="">
                <ul v-for="(item, index) in footerList" :key="index">
                    <h3>{{item.title}}</h3>
                    <li @click="toPage(cItem)" v-for="(cItem, cIndex) in item.childer" :key="cIndex">
                        {{cItem.name}}
                    </li>
                </ul>
            </div>
        </footer>

    </div>
</template>

<script>
    import bannerComp from './banner.vue'
    import hxgnComp from './hxgn.vue'
    import mxjsComp from './mxjs.vue'
    import lxwjzxfwComp from './lxwjzxfw.vue'
    import sstxComp from './sstx.vue'
    export default {
        name: 'page-home',
        components: {
            bannerComp,
            hxgnComp,
            mxjsComp,
            lxwjzxfwComp,
            sstxComp
        },
        data () {
            return {
                headerList: [
                    {
                        name: 'funasr介绍',
                        ref: 'h-funasrJs',
                        isAnchor: true
                    },
                    {
                        name: '核心功能',
                        ref: 'h-hxgn',
                        isAnchor: true
                    },
                    {
                        name: 'Paraformer模型介绍',
                        ref: 'h-mxjs',
                        isAnchor: true
                    },
                    {
                        name: '离线文件转写服务',
                        ref: 'h-lxwj',
                        isAnchor: true
                    },
                    {
                        name: '实时听写',
                        ref: 'h-sstx',
                        isAnchor: true
                    }
                    // {
                    //     name: 'Github',
                    //     ref: 'h-github'
                    // },
                    // {
                    //     name: '社区交流',
                    //     ref: 'h-sqjl'
                    // }
                ],
                footerList: [
                    {
                        title: 'Funasr介绍',
                        childer: [
                            {
                                name: '关于我们',
                                link: ''
                            }
                        ]
                    },
                    {
                        title: '核心功能',
                        childer: [
                            {
                                name: '核心介绍',
                                link: ''
                            }
                        ]
                    },
                    {
                        title: 'Paraformer模型介绍',
                        childer: [
                            {
                                name: '模型结构固',
                                link: ''
                            },
                            {
                                name: '模型介绍',
                                link: ''
                            }
                        ]
                    },
                    {
                        title: '离线文件转写服务',
                        childer: [
                            {
                                name: '原理图',
                                link: ''
                            },
                            {
                                name: '文字介绍',
                                link: ''
                            },
                            {
                                name: '安装',
                                link: ''
                            },
                            {
                                name: '使用',
                                link: ''
                            },
                            {
                                name: '视频教程链接',
                                link: ''
                            }
                        ]
                    },
                    {
                        title: '实时听写',
                        childer: [
                            {
                                name: '原理图',
                                link: ''
                            },
                            {
                                name: '文字介绍',
                                link: ''
                            },
                            {
                                name: '安装',
                                link: ''
                            },
                            {
                                name: '使用',
                                link: ''
                            },
                            {
                                name: '视频教程链接',
                                link: ''
                            }
                        ]
                    }
                ],
                headerActive: 'h-funasrJs',
                lineWidth: 0,
                lineLeft: 0
            }
        },
        created () {},
        mounted () {
            this.$nextTick(() => {
                this.getLineWidth()
                this.handlerNodeScroll()
                this.getAnchorDetails()
            })
        },
        methods: {
            // 底部跳转
            toPage (item) {
                if (item.link) window.open(item.link)
            },
            // 获取元素的高度，距离顶部距离
            getAnchorDetails () {
                for (let i = 0; i < this.headerList.length; i++) {
                    const item = this.headerList[i]
                    if (item.isAnchor) {
                        const refName = item.ref.split('-')[1]
                        const dom = this.$refs[refName].$el
                        item.height = dom.offsetHeight
                        item.top = dom.offsetTop
                    }
                }
            },
            // 头部锚点按钮
            anchorTo (item) {
                this.headerActive = item.ref
                this.getLineWidth()
                if (item.isAnchor) {
                    const refName = item.ref.split('-')[1]
                    this.scrollToFn(refName)
                }
            },
            // 设置滚动距离
            scrollToFn (refName) {
                const dom = this.$refs[refName].$el
                const top = dom.offsetTop

                const appContentScrollbar = document.getElementsByClassName('app-content-scrollbar')[0]
                const uiModalWrap = appContentScrollbar.getElementsByClassName('ui-scrollbar__wrap')[0]

                uiModalWrap.scrollTo({
                    top: top - 70,
                    left: 0,
                    behavior: 'smooth'
                })
            },
            // 获取头部选中样式
            getLineWidth () {
                const dom = this.$refs[this.headerActive][0]
                if (!dom) return
                this.lineWidth = dom.offsetWidth
                this.lineLeft = dom.offsetLeft
            },
            // 监听元素滚动
            handlerNodeScroll () {
                const self = this
                // 函数防抖
                const fun = self.debounce(e => {
                    // 距顶部
                    const scrollTop = e.target.scrollTop
                    let toActive = null
                    for (let i = 0; i < self.headerList.length; i++) {
                        const item = self.headerList[i]
                        if (scrollTop > (item.top - 90)) {
                            toActive = item
                        }
                    }
                    if (toActive && toActive.top) {
                        self.headerActive = toActive.ref
                        self.getLineWidth()
                    }
                }, 100)

                // 监听滚动
                const appContentScrollbar = document.getElementsByClassName('app-content-scrollbar')[0]
                const uiModalWrap = appContentScrollbar.getElementsByClassName('ui-scrollbar__wrap')[0]
                uiModalWrap.addEventListener('scroll', function (e) {
                    fun(e)
                })
            },
            // 函数防抖
            debounce (handle, delay) {
                let timer = null
                return function () {
                    const _self = this
                    const _args = arguments
                    clearTimeout(timer)
                    timer = setTimeout(function () {
                        handle.apply(_self, _args)
                    }, delay)
                }
            }
        }
    }
</script>
<style src="./assets/css/index.scss" lang="scss"></style>
