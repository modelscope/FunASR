#pragma once 

namespace funasr {
#define S_BEGIN  0
#define S_MIDDLE 1
#define S_END    2
#define S_ALL    3
#define S_ERR    4

#ifndef MODEL_SAMPLE_RATE
#define MODEL_SAMPLE_RATE 16000
#endif

// parser option
#define MODEL_DIR "model-dir"
#define OFFLINE_MODEL_DIR "model-dir"
#define ONLINE_MODEL_DIR "online-model-dir"
#define LM_DIR "lm-dir"
#define GLOB_BEAM "global-beam"
#define LAT_BEAM "lattice-beam"
#define AM_SCALE "am-scale"
// #define FST_HOTWORD "fst-hotword"
#define FST_INC_WTS "fst-inc-wts"
#define VAD_DIR "vad-dir"
#define PUNC_DIR "punc-dir"
#define QUANTIZE "quantize"
#define VAD_QUANT "vad-quant"
#define PUNC_QUANT "punc-quant"
#define ASR_MODE "mode"

#define WAV_PATH "wav-path"
#define WAV_SCP "wav-scp"
#define TXT_PATH "txt-path"
#define THREAD_NUM "thread-num"
#define PORT_ID "port-id"
#define HOTWORD_SEP " "
#define AUDIO_FS "audio-fs"

#define MODEL_PARA "Paraformer"
#define MODEL_SVS "SenseVoiceSmall"

// #define VAD_MODEL_PATH "vad-model"
// #define VAD_CMVN_PATH "vad-cmvn"
// #define VAD_CONFIG_PATH "vad-config"
// #define AM_MODEL_PATH "am-model"
// #define AM_CMVN_PATH "am-cmvn"
// #define AM_CONFIG_PATH "am-config"
// #define PUNC_MODEL_PATH "punc-model"
// #define PUNC_CONFIG_PATH "punc-config"

#define MODEL_NAME "model.onnx"
// hotword embedding compile model
#define MODEL_EB_NAME "model_eb.onnx"
#define TORCH_MODEL_EB_NAME "model_eb.torchscript"
#define QUANT_MODEL_NAME "model_quant.onnx"
#define VAD_CMVN_NAME "am.mvn"
#define VAD_CONFIG_NAME "config.yaml"

// gpu models
#define INFER_GPU "gpu"
#define BATCHSIZE "batch-size"
#define TORCH_MODEL_NAME "model.torchscript"
#define TORCH_QUANT_MODEL_NAME "model_quant.torchscript"
#define BLADE_MODEL_NAME "model_blade.torchscript"
#define BLADEDISC "bladedisc"

#define AM_CMVN_NAME "am.mvn"
#define AM_CONFIG_NAME "config.yaml"
#define LM_CONFIG_NAME "config.yaml"
#define PUNC_CONFIG_NAME "config.yaml"
#define MODEL_SEG_DICT "seg_dict"
#define TOKEN_PATH "tokens.json"
#define HOTWORD "hotword"
// #define NN_HOTWORD "nn-hotword"

#define ITN_DIR "itn-dir"
#define ITN_TAGGER_NAME "zh_itn_tagger.fst"
#define ITN_VERBALIZER_NAME "zh_itn_verbalizer.fst"

#define ENCODER_NAME "model.onnx"
#define QUANT_ENCODER_NAME "model_quant.onnx"
#define DECODER_NAME "decoder.onnx"
#define QUANT_DECODER_NAME "decoder_quant.onnx"

#define LM_FST_RES "TLG.fst"
#define LEX_PATH "lexicon.txt"

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

// asr
#ifndef PARA_LFR_M
#define PARA_LFR_M 7
#endif

#ifndef PARA_LFR_N
#define PARA_LFR_N 6
#endif

#ifndef ONLINE_STEP
#define ONLINE_STEP 9600
#endif

// punc
#define UNK_CHAR "<unk>"
#define TOKEN_LEN     20

#define CANDIDATE_NUM   6
#define UNKNOW_INDEX 0
#define NOTPUNC  "_"
#define NOTPUNC_INDEX 1
#define COMMA_INDEX 2
#define PERIOD_INDEX 3
#define QUESTION_INDEX 4
#define DUN_INDEX 5
#define CACHE_POP_TRIGGER_LIMIT   200

#define JIEBA_DICT "jieba.c.dict"
#define JIEBA_USERDICT "jieba_usr_dict"
#define JIEBA_HMM_MODEL "jieba.hmm"

} // namespace funasr
