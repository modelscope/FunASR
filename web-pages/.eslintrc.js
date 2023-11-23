module.exports = {
    root: true,
    env: {
        node: true
    },
    extends: ['plugin:vue/essential', '@vue/standard'],
    parserOptions: {
        parser: '@babel/eslint-parser'
    },
    rules: {
        'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
        'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
        'quote-props': 'off',
        indent: 'off',
        // "vue/script-indent": [
        //     "error",
        //     4,
        //     {
        //         baseIndent: 1,
        //     },
        // ],
        'vue/order-in-components': 'error'
    }
}
