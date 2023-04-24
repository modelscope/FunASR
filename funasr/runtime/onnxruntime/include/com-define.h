
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

// vad
#ifndef VAD_SILENCE_DYRATION
#define VAD_SILENCE_DYRATION 15000
#endif

#ifndef VAD_MAX_LEN
#define VAD_MAX_LEN 800
#endif

#ifndef VAD_SPEECH_NOISE_THRES
#define VAD_SPEECH_NOISE_THRES 0.9
#endif

// punc
#define PUNC_MODEL_FILE  "punc_model.onnx"
#define PUNC_YAML_FILE "punc.yaml"

#define UNK_CHAR "<unk>"

#define  INPUT_NUM  2
#define  INPUT_NAME1 "input"
#define  INPUT_NAME2 "text_lengths"
#define  OUTPUT_NAME "logits"
#define  TOKEN_LEN     20

#define  CANDIDATE_NUM   6
#define UNKNOW_INDEX 0
#define NOTPUNC_INDEX 1
#define COMMA_INDEX 2
#define PERIOD_INDEX 3
#define QUESTION_INDEX 4
#define DUN_INDEX 5
#define  CACHE_POP_TRIGGER_LIMIT   200

#endif
