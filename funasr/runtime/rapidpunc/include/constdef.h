#pragma once


#define UNK_CHAR "<unk>"


#define MODEL_FILE  "model.onnx"

#define YAML_FILE "punc.yaml"

/*
input
name: input
type: int64[batch_size,feats_length]
text_lengths
name: text_lengths
type: int32[batch_size]
*/

#define  INPUT_NAME1 "input"

#define  INPUT_NAME2 "text_lengths"


#define  INPUT_NUM  2

extern const char* INPUT_NAMES[];

/*

name: logits
type: float32[batch_size,logits_length,6]
*/
#define  OUTPUT_NAME "logits"




#define  TOKEN_LEN     20

#define ARRAY_SIZE(arr) (sizeof(arr)/sizeof((arr)[0]))


#define  CANDIDATE_NUM   6

#define UNKNOW_INDEX 0
#define NOTPUNC_INDEX 1
#define COMMA_INDEX 2
#define PERIOD_INDEX 3
#define QUESTION_INDEX 4
#define DUN_INDEX 5

#define  CACHE_POP_TRIGGER_LIMIT   200

//#define  LAST_ENG_ID  272726