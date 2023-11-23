import Vue from 'vue'
import Vuex from 'vuex'
import createPersistedState from 'vuex-persistedstate'

import common from './modules/common'
import getters from './getters'

Vue.use(Vuex)

const debug = process.env.NODE_ENV !== 'production'
const createPersisted = createPersistedState({
    storage: window.sessionStorage,
    paths: ['user']
})

const store = new Vuex.Store({
    plugins: debug ? [createPersisted] : [createPersisted],
    modules: {
        common
    },
    getters
})

export default store
