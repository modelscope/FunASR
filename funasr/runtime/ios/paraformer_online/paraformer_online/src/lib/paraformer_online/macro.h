//
//  macro.h
//  paraformer_online
//
//  Created by 邱威 on 2023/6/9.
//

#ifndef macro_h
#define macro_h

#define DEFAULT_TAG "qnn"
// Log
#ifdef __ANDROID__
#include <android/log.h>
#define LOGDT(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_DEBUG, tag, ("%s " fmt), __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define LOGIT(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_INFO, tag, ("%s " fmt), __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_ERROR, tag, ("%s " fmt), __PRETTY_FUNCTION__, ##__VA_ARGS__);                                                                      \
    fprintf(stderr, ("E/%s: %s " fmt), tag, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#else //__ANDROID__

#define LOGDT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("D/%s: %s " fmt), tag, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define LOGIT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("I/%s: %s " fmt), tag, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...)                                                                                           \
    fprintf(stderr, ("E/%s: %s " fmt), tag, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#endif

#define LOGD(fmt, ...) LOGDT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGI(fmt, ...) LOGIT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGE(fmt, ...) LOGET(fmt, DEFAULT_TAG, ##__VA_ARGS__)

#endif /* macro_h */
