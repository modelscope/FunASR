
#ifndef COMDEFINE_H
#define COMDEFINE_H

#define S_BEGIN  0
#define S_MIDDLE 1
#define S_END    2
#define S_ALL    3
#define S_ERR    4

#ifndef MODEL_SAMPLE_RATE
#define MODEL_SAMPLE_RATE 16000
#endif

// model path
#define VAD_MODEL_PATH "vad-model"
#define VAD_CMVN_PATH "vad-cmvn"
#define VAD_CONFIG_PATH "vad-config"
#define AM_MODEL_PATH "am-model"
#define AM_CMVN_PATH "am-cmvn"
#define AM_CONFIG_PATH "am-config"
#define PUNC_MODEL_PATH "punc-model"
#define PUNC_CONFIG_PATH "punc-config"
#define WAV_PATH "wav-path"
#define WAV_SCP "wav-scp"
#define THREAD_NUM "thread-num"
#define PORT_ID "port-id"

// vad
#ifndef VAD_SILENCE_DURATION
#define VAD_SILENCE_DURATION 800
#endif

#ifndef VAD_MAX_LEN
#define VAD_MAX_LEN 15000
#endif

#ifndef VAD_SPEECH_NOISE_THRES
#define VAD_SPEECH_NOISE_THRES 0.9
#endif

#ifndef VAD_LFR_M
#define VAD_LFR_M 5
#endif

#ifndef VAD_LFR_N
#define VAD_LFR_N 1
#endif

// punc
#define UNK_CHAR "<unk>"
#define TOKEN_LEN     20

#define CANDIDATE_NUM   6
#define UNKNOW_INDEX 0
#define NOTPUNC_INDEX 1
#define COMMA_INDEX 2
#define PERIOD_INDEX 3
#define QUESTION_INDEX 4
#define DUN_INDEX 5
#define CACHE_POP_TRIGGER_LIMIT   200

#endif
