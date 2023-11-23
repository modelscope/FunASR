// 组件
import uiJessibuca from '@/components/ui-jessibuca/index.vue'
import uiScrollbar from '@/components/ui-scrollbar/index.vue'

const components = {
    // 组件
    uiJessibuca,
    uiScrollbar
}

const install = function (Vue, opts = {}) {
    Object.keys(components).forEach((name) => {
        Vue.component(name, components[name])
    })
}

export default {
    install
}
