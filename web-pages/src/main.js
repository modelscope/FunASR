import Vue from 'vue'
// import queryString from 'query-string'
import app from '@/views/app/index.vue'
import { router } from '@/router/index.js'
import store from '@/store/index.js'
// 请求对象
import Axios from 'axios'
// 项目本地组件全局注册
import globalComponents from './globalComponents'
// 项目本地全局方法
import globalFunctions from './globalFunctions'

// 第三方组件
// ant-design-vue 组件 按需加载
import { ConfigProvider, Input } from 'ant-design-vue'

Vue.use(globalComponents)
Vue.use(globalFunctions)

Vue.use(ConfigProvider)
Vue.use(Input)

const axiosInstance = Axios.create({
    timeout: 60000
})

// 加载项目基础样式文件
require('@/assets/css/normalize.css')
// 加载项目组件覆盖样式，全局模块样式
require('@/assets/css/index.scss')
// 轮播插件swiper样式
require('swiper/css/swiper.min.css')

// 设置为 false 以阻止 vue 在启动时生成生产提示
Vue.config.productionTip = false
process.env.VUE_APP_env === 'devmock' && require('../mock')

Vue.prototype.$axios = axiosInstance

// axios 配置 请求和响应拦截
axiosInstance.interceptors.request.use(
    (config) => {
        // 禁用令牌
        if (
            typeof config.headers.disabletoken !== 'undefined' &&
            config.headers.disabletoken === true
        ) {
            delete config.headers.disabletoken
            return config
        }

        if (
            typeof config.headers.token === 'undefined' &&
            localStorage.getItem('token') !== null
        ) {
            config.headers.token = localStorage.getItem('token')
        }
        return config
    },
    (error) => {
        return Promise.reject(error)
    }
)
// 异常处理
axiosInstance.interceptors.response.use(
    (config) => {
        if (typeof config.headers.token !== 'undefined') {
            localStorage.setItem('token', config.headers.token)
        }
        if (
            config.data &&
            config.data.statusCode &&
            config.data.statusCode !== '00'
        ) {
            // 业务异常处理
            let msg = 'biz error, statusCode: ' + config.data.statusCode
            if (config.data.statusMsg) {
                msg = msg + ', statusMsg: ' + config.data.statusMsg
            }

            // 打印异常信息
            console.log(msg)
        }
        return config
    },
    (error) => {
        // 打印异常信息
        console.log(error)

        if (
            typeof error.response !== 'undefined' &&
            error.response.status === 401
        ) {
            console.log(error)
        }

        return Promise.reject(error)
    }
)

// 获取必要的数据
const checkNecessaryData = (callbackFun) => {
    const promiseList = []
    if (promiseList.length > 0) {
        Promise.all(promiseList)
            .then((res) => {
                localStorage.removeItem('getInitDataErrorCount')
                callbackFun && callbackFun()
            })
            .catch((res) => {
                const errorCount = localStorage.getItem(
                    'getInitDataErrorCount'
                )
                if (errorCount) {
                    let count = Number.parseInt(errorCount)
                    if (count <= 3) {
                        localStorage.setItem('getInitDataErrorCount', ++count)
                        // 菜单或者用户角色列表数据请求失败，就刷新页面
                        window.location.reload(window.location.href)
                    }
                } else {
                    localStorage.setItem('getInitDataErrorCount', 1)
                    // 菜单或者用户角色列表数据请求失败，就刷新页面
                    window.location.reload(window.location.href)
                }
            })
    } else {
        callbackFun && callbackFun()
    }
}

router.beforeEach((to, from, next) => {
    checkNecessaryData(() => {
        next()
    })
})

new Vue({
    render: (h) => h(app),
    router: router,
    store: store
}).$mount('#app')
