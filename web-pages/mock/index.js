import Mock from 'mockjs'

import getUserInfo from './user/getUserInfo.js'
import getMenuList from './user/getMenuList.js'
Mock.setup({
    timeout: 500
})

console.log('启动mock请求数据')
getUserInfo(Mock)
getMenuList(Mock)
