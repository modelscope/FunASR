const webpack = require('webpack')
// gzip压缩插件
const CompressionWebpackPlugin = require('compression-webpack-plugin')

const projectTitle = 'funasr'

// node命令行参数--env xxx=xxx保存至node的环境变量process.env.xxx=xxx
for (const arg of process.argv) {
    if (arg.indexOf('=') > 0) {
        process.env[arg.split('=')[0]] = arg.split('=')[1]
    }
}

module.exports = {
    pages: {
        index: {
            // 本地化的入口，格式/src/main-<配置>.js，当配置对应的入口文件不存在时，使用默认的入口文件./src/main.js
            entry: process.argv.indexOf('--mode') !== -1 && require('fs').existsSync('./src/main-' + process.argv[process.argv.indexOf('--mode') + 1] + '.js') ? './src/main-' + process.argv[process.argv.indexOf('--mode') + 1] + '.js' : './src/main.js',
            // 模板来源
            template: 'public/index.html',
            // 在 dist/index.html 的输出
            filename: 'index.html',
            // 当使用 title 选项时
            // template 中的 title 标签需要是 <title><%= htmlWebpackPlugin.options.title %></title>
            title: projectTitle
        }
    },
    // 根据环境配置项目名
    publicPath: process.env.NODE_ENV === 'production' ? './' : '/funasr-vue',
    // 保存时是否eslint检查
    lintOnSave: true,
    productionSourceMap: false,
    devServer: {
        // host: '127.0.0.1',
        port: 9018,
        // https: false, // https:{type:Boolean}
        // open: true,
        proxy: {
            '/api': {
                target: 'http://192.168.1.57:20611',
                // ws: true,
                changeOrigin: true,
                // logLevel: 'debug',
                pathRewrite: {
                    '/api': 'http://192.168.1.57:20611/sso-backend'
                }
            }
        },
        client: {
            overlay: false
        }
    },
    css: {
        loaderOptions: {
            // 给 sass-loader 传递选项
            sass: {
                // @/ 是 src/ 的别名
                // 所以这里假设你有 `src/variables.scss` 这个文件
                additionalData: '@import "~@/assets/css/common.scss";'
            }
        }
    },
    // 配置babel-loader 转译 node_modules 里面的文件
    transpileDependencies: [
        /[/\\]node_modules[/\\](.+?)?plugin-/ // 匹配编译plugin-开头的组件
    ],
    configureWebpack: (config) => {
        const plugins = []
        if (process.env.NODE_ENV !== 'production') {
            // 开发环境配置
            config.devtool = 'eval-source-map'
        } else {
            // 生产环境配置
            // 添加gzip压缩
            plugins.push(
                new CompressionWebpackPlugin({
                    filename: '[path].gz[query]',
                    algorithm: 'gzip',
                    test: new RegExp('\\.(' + ['js', 'css'].join('|') + ')$'),
                    threshold: 10240,
                    minRatio: 0.8
                })
            )
            config.optimization.minimizer[0].options.minimizer.options.compress.drop_console = true
            config.optimization.minimizer[0].options.minimizer.options.compress.drop_debugger = true
        }
        // 配置打包文件git信息
        plugins.push(
            new webpack.BannerPlugin({
                banner: (() => {
                    try {
                        const execSync = require('child_process').execSync
                        return JSON.stringify({
                            'remote-origin-url':
                                '' + execSync('git remote get-url origin'),
                            branch: '' + execSync('git rev-parse --abbrev-ref HEAD'),
                            commithash: '' + execSync('git rev-parse HEAD')
                        })
                    } catch (e) {
                        return 'not a git repository'
                    }
                })()
            })
        )

        config.plugins = [...config.plugins, ...plugins]

        const sassLoader = require.resolve('sass-loader')
        config.module.rules.filter(rule => {
            return rule.test.toString().indexOf('scss') !== -1
        }).forEach(rule => {
            rule.oneOf.forEach(oneOfRule => {
                const sassLoaderIndex = oneOfRule.use.findIndex(item => item.loader === sassLoader)
                oneOfRule.use.splice(sassLoaderIndex, 0, { loader: require.resolve('css-unicode-loader') })
            })
        })
    },
    chainWebpack: (config) => {
        // 为了补删除换行而加的配置
        // config.module
        //     .rule('vue')
        //     .use('vue-loader')
        //     .loader('vue-loader')
        //     .tap((options) => {
        //         // modify the options...
        //         options.compilerOptions.preserveWhitespace = false
        //         return options
        //     })
    }
}
