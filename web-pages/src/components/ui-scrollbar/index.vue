
<script>
    import { addResizeListener, removeResizeListener, toObject } from './util'
    import scrollbarWidth from './scrollbar-width'
    import Bar from './bar.vue'
    require('./assets/css/index.scss')

    export default {
        name: 'ui-scrollbar',
        components: { Bar },
        props: {
            native: Boolean,
            childOverWidth: {
                type: Boolean,
                default: true
            },
            wrapStyle: {},
            wrapClass: {},
            viewClass: {},
            viewStyle: {},
            noresize: Boolean, // 如果 container 尺寸不会发生变化，最好设置它可以优化性能
            tag: {
                type: String,
                default: 'div'
            }
        },

        data () {
            return {
                sizeWidth: '0',
                sizeHeight: '0',
                moveX: 0,
                moveY: 0,
                scrollbarMiddle: 0
            }
        },

        computed: {
            wrap () {
                return this.$refs.wrap
            }
        },

        mounted () {
            if (this.native) return
            this.$nextTick(this.update)
            !this.noresize && addResizeListener(this.$refs.resize, this.update)
            !this.noresize && addResizeListener(this.$refs.wrap, this.update)
        },

        beforeDestroy () {
            if (this.native) return
            !this.noresize && removeResizeListener(this.$refs.resize, this.update)
            !this.noresize && removeResizeListener(this.$refs.wrap, this.update)
        },

        methods: {
            handleScroll () {
                const wrap = this.wrap

                this.moveY = (wrap.scrollTop * 100) / wrap.clientHeight
                this.moveX = (wrap.scrollLeft * 100) / wrap.clientWidth
                if (wrap.scrollTop + wrap.clientHeight === wrap.scrollHeight) {
                    this.$emit('scrollbottom')
                }
                if (wrap.scrollTop === 0) {
                    this.$emit('scrolltop')
                }
                this.$emit('scrollMove', {
                    moveX: wrap.scrollLeft,
                    moveY: wrap.scrollTop,
                    clientX: wrap.clientWidth,
                    clientY: wrap.clientHeight,
                    scrollHeight: wrap.scrollHeight,
                    scrollWidth: wrap.scrollWidth
                })
            },

            update () {
                let heightPercentage = null
                let widthPercentage = null
                const wrap = this.wrap
                if (!wrap) return

                heightPercentage = (wrap.clientHeight * 100) / wrap.scrollHeight
                widthPercentage = (wrap.clientWidth * 100) / wrap.scrollWidth

                this.sizeHeight = heightPercentage < 100 ? heightPercentage + '%' : ''
                this.sizeWidth = widthPercentage < 100 ? widthPercentage + '%' : ''
            },
            continueScroll (rowNum) {
                const wrap = this.wrap
                if (this.scrollbarMiddle === 0) {
                    this.scrollbarMiddle = wrap.scrollTop / 2 + wrap.clientHeight / 2 - wrap.clientHeight / rowNum
                }
                wrap.scrollTop = this.scrollbarMiddle
            }
        },

        render (h) {
            const gutter = scrollbarWidth()
            let style = this.wrapStyle

            if (gutter) {
                const gutterWith = `-${gutter}px`
                let gutterStyle = ''

                let overFlowXStr = ''
                let overflowYStr = ''
                if (style && style.length > 0) {
                    style.forEach((styleItem) => {
                        if (styleItem['overflow-x'] || styleItem.overflowX) {
                            overFlowXStr = styleItem['overflow-x'] || styleItem.overflowX
                        }
                        if (styleItem['overflow-y'] || styleItem.overflowY) {
                            overflowYStr = styleItem['overflow-y'] || styleItem.overflowY
                        }
                    })
                }

                if (overFlowXStr === 'hidden') {
                    gutterStyle = `margin-right: ${gutterWith};`
                } else {
                    gutterStyle = `margin-bottom: ${gutterWith}; margin-right: ${gutterWith};`
                }

                if (Array.isArray(this.wrapStyle)) {
                    style = toObject(this.wrapStyle)
                    if (overFlowXStr !== 'hidden') {
                        style.marginBottom = gutterWith
                    }
                    if (overflowYStr !== 'hidden') {
                        style.marginRight = gutterWith
                    }
                } else if (typeof this.wrapStyle === 'string') {
                    style += gutterStyle
                } else {
                    style = gutterStyle
                }
            }
            const view = h(
                this.tag,
                {
                    class: ['ui-scrollbar__view', this.viewClass],
                    style: this.viewStyle,
                    ref: 'resize'
                },
                this.$slots.default
            )
            const wrap = (
                <div
                    ref="wrap"
                    style={style}
                    onScroll={this.handleScroll}
                    class={[this.wrapClass, 'ui-scrollbar__wrap', gutter ? '' : 'ui-scrollbar__wrap--hidden-default']}
                >{[view]}</div>
            )
            let nodes

            if (!this.native) {
                nodes = [
                    wrap,
                    <Bar move={this.moveX} size={this.sizeWidth}></Bar>,
                    <Bar vertical move={this.moveY} size={this.sizeHeight}></Bar>
                ]
            } else {
                nodes = [
                <div ref="wrap" class={[this.wrapClass, 'ui-scrollbar__wrap']} style={style}>{[view]}</div>
                ]
            }
            return h('div', { class: ['ui-scrollbar', { 'child-over-width': !this.childOverWidth }] }, nodes)
        }
    }
</script>
