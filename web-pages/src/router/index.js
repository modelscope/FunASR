import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

const routerPush = Router.prototype.push
Router.prototype.push = function push (location) {
    return routerPush.call(this, location).catch(error => error)
}

const routerList = [
    {
        path: '/',
        component: resolve => require(['@/views/home/index.vue'], resolve),
        meta: {}
    }
]

const routerListMap = {}
routerList.forEach((item) => {
    routerListMap[item.path] = item
})

const createRouter = () => {
    return new Router({
        routes: routerList
    })
}

const router = createRouter()

export { router, routerListMap }
