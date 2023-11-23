const common = {
    state: {
        isLoading: false,
        loadingText: '',
        loadingSpinner: '',
        loadingBackground: ''
    },

    mutations: {
        SET_IS_LOADING: (state, { isLoading, loadingText, loadingSpinner, loadingBackground }) => {
            state.isLoading = isLoading
            state.loadingText = loadingText || '加载中...'
            state.loadingSpinner = loadingSpinner || 'icon-loading-1'
            state.loadingBackground = loadingBackground || 'rgba(0, 0, 0, 0.8)'
        }
    }
}

export default common
