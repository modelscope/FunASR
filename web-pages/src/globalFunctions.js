export default {
    install (Vue) {
        // px 转 rem
        Vue.prototype.pxTorem = (px, needUnit) => {
            if (needUnit) {
                return px / 19.2 + 'rem'
            } else {
                return px / 19.2
            }
        }
    }
}
