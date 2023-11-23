import ResizeObserver from 'resize-observer-polyfill'

export const BAR_MAP = {
    vertical: {
        offset: 'offsetHeight',
        scroll: 'scrollTop',
        scrollSize: 'scrollHeight',
        size: 'height',
        key: 'vertical',
        axis: 'Y',
        client: 'clientY',
        direction: 'top'
    },
    horizontal: {
        offset: 'offsetWidth',
        scroll: 'scrollLeft',
        scrollSize: 'scrollWidth',
        size: 'width',
        key: 'horizontal',
        axis: 'X',
        client: 'clientX',
        direction: 'left'
    }
}

export function renderThumbStyle ({ move, size, bar }) {
    const style = {}
    const translate = `translate${bar.axis}(${move}%)`

    style[bar.size] = size
    style.transform = translate
    style.msTransform = translate
    style.webkitTransform = translate

    return style
};

const isServer = typeof window === 'undefined'

/* istanbul ignore next */
const resizeHandler = function (entries) {
    for (const entry of entries) {
        const listeners = entry.target.__resizeListeners__ || []
        if (listeners.length) {
            listeners.forEach(fn => {
                fn()
            })
        }
    }
}

/* istanbul ignore next */
export const addResizeListener = function (element, fn) {
    if (isServer) return
    if (!element.__resizeListeners__) {
        element.__resizeListeners__ = []
        element.__ro__ = new ResizeObserver(resizeHandler)
        element.__ro__.observe(element)
    }
    element.__resizeListeners__.push(fn)
}

/* istanbul ignore next */
export const removeResizeListener = function (element, fn) {
    if (!element || !element.__resizeListeners__) return
    element.__resizeListeners__.splice(element.__resizeListeners__.indexOf(fn), 1)
    if (!element.__resizeListeners__.length) {
        element.__ro__.disconnect()
    }
}

function extend (to, _from) {
    for (const key in _from) {
        to[key] = _from[key]
    }
    return to
};

export function toObject (arr) {
    const res = {}
    for (let i = 0; i < arr.length; i++) {
        if (arr[i]) {
            extend(res, arr[i])
        }
    }
    return res
};

export const on = (function () {
    if (!isServer && document.addEventListener) {
        return function (element, event, handler) {
            if (element && event && handler) {
                element.addEventListener(event, handler, false)
            }
        }
    } else {
        return function (element, event, handler) {
            if (element && event && handler) {
                element.attachEvent('on' + event, handler)
            }
        }
    }
})()

/* istanbul ignore next */
export const off = (function () {
    if (!isServer && document.removeEventListener) {
        return function (element, event, handler) {
            if (element && event) {
                element.removeEventListener(event, handler, false)
            }
        }
    } else {
        return function (element, event, handler) {
            if (element && event) {
                element.detachEvent('on' + event, handler)
            }
        }
    }
})()
