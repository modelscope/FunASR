#pragma once
#include <map>
#include <vector>
#include <unordered_map>
#ifdef WIN32
#ifdef _FUNASR_API_EXPORT
#define  _FUNASRAPI __declspec(dllexport)
#else
#define  _FUNASRAPI __declspec(dllimport)
#endif
#else
#define _FUNASRAPI
#endif

#ifndef _WIN32
#define FUNASR_CALLBCK_PREFIX __attribute__((__stdcall__))
#else
#define FUNASR_CALLBCK_PREFIX __stdcall
#endif


typedef void* FUNASR_HANDLE;
typedef void* FUNASR_RESULT;
typedef void* FUNASR_DEC_HANDLE;
typedef unsigned char FUNASR_BOOL;

#define FUNASR_TRUE 1
#define FUNASR_FALSE 0
#define QM_DEFAULT_THREAD_NUM  4

typedef enum
{
 RASR_NONE=-1,
 RASRM_CTC_GREEDY_SEARCH=0,
 RASRM_CTC_RPEFIX_BEAM_SEARCH = 1,
 RASRM_ATTENSION_RESCORING = 2,
}FUNASR_MODE;

typedef enum {
	FUNASR_MODEL_PADDLE = 0,
	FUNASR_MODEL_PADDLE_2 = 1,
	FUNASR_MODEL_K2 = 2,
	FUNASR_MODEL_PARAFORMER = 3,
}FUNASR_MODEL_TYPE;

typedef enum {
	ASR_OFFLINE=0,
	ASR_ONLINE=1,
	ASR_TWO_PASS=2,
}ASR_TYPE;

typedef enum {
	PUNC_OFFLINE=0,
	PUNC_ONLINE=1,
}PUNC_TYPE;

typedef void (* QM_CALLBACK)(int cur_step, int n_total); // n_total: total steps; cur_step: Current Step.

// ASR
_FUNASRAPI FUNASR_HANDLE  	FunASRInit(std::map<std::string, std::string>& model_path, int thread_num, ASR_TYPE type=ASR_OFFLINE);
_FUNASRAPI FUNASR_HANDLE  	FunASROnlineInit(FUNASR_HANDLE asr_handle, std::vector<int> chunk_size={5,10,5});
_FUNASRAPI void         	FunASRReset(FUNASR_HANDLE handle, FUNASR_DEC_HANDLE dec_handle=nullptr);

// buffer
_FUNASRAPI FUNASR_RESULT	FunASRInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, FUNASR_MODE mode, QM_CALLBACK fn_callback, bool input_finished=true, int sampling_rate=16000, std::string wav_format="pcm");
// file, support wav & pcm
_FUNASRAPI FUNASR_RESULT	FunASRInfer(FUNASR_HANDLE handle, const char* sz_filename, FUNASR_MODE mode, QM_CALLBACK fn_callback, int sampling_rate=16000);

_FUNASRAPI const char*	FunASRGetResult(FUNASR_RESULT result,int n_index);
_FUNASRAPI const char*	FunASRGetStamp(FUNASR_RESULT result);
_FUNASRAPI const char*	FunASRGetStampSents(FUNASR_RESULT result);
_FUNASRAPI const char*	FunASRGetTpassResult(FUNASR_RESULT result,int n_index);
_FUNASRAPI const int	FunASRGetRetNumber(FUNASR_RESULT result);
_FUNASRAPI void			FunASRFreeResult(FUNASR_RESULT result);
_FUNASRAPI void			FunASRUninit(FUNASR_HANDLE handle);
_FUNASRAPI const float	FunASRGetRetSnippetTime(FUNASR_RESULT result);

// VAD
_FUNASRAPI FUNASR_HANDLE  	FsmnVadInit(std::map<std::string, std::string>& model_path, int thread_num);
_FUNASRAPI FUNASR_HANDLE  	FsmnVadOnlineInit(FUNASR_HANDLE fsmnvad_handle);
// buffer
_FUNASRAPI FUNASR_RESULT	FsmnVadInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, QM_CALLBACK fn_callback, bool input_finished=true, int sampling_rate=16000, std::string wav_format="pcm");
// file, support wav & pcm
_FUNASRAPI FUNASR_RESULT	FsmnVadInfer(FUNASR_HANDLE handle, const char* sz_filename, QM_CALLBACK fn_callback, int sampling_rate=16000);

_FUNASRAPI std::vector<std::vector<int>>*	FsmnVadGetResult(FUNASR_RESULT result,int n_index);
_FUNASRAPI void			 	FsmnVadFreeResult(FUNASR_RESULT result);
_FUNASRAPI void				FsmnVadUninit(FUNASR_HANDLE handle);
_FUNASRAPI const float		FsmnVadGetRetSnippetTime(FUNASR_RESULT result);

// PUNC
_FUNASRAPI FUNASR_HANDLE  		CTTransformerInit(std::map<std::string, std::string>& model_path, int thread_num, PUNC_TYPE type=PUNC_OFFLINE);
_FUNASRAPI FUNASR_RESULT     	CTTransformerInfer(FUNASR_HANDLE handle, const char* sz_sentence, FUNASR_MODE mode, QM_CALLBACK fn_callback, PUNC_TYPE type=PUNC_OFFLINE, FUNASR_RESULT pre_result=nullptr);
_FUNASRAPI const char* 			CTTransformerGetResult(FUNASR_RESULT result,int n_index);
_FUNASRAPI void					CTTransformerFreeResult(FUNASR_RESULT result);
_FUNASRAPI void					CTTransformerUninit(FUNASR_HANDLE handle);

//OfflineStream
_FUNASRAPI FUNASR_HANDLE  	FunOfflineInit(std::map<std::string, std::string>& model_path, int thread_num, bool use_gpu=false, int batch_size=1);
_FUNASRAPI void         	FunOfflineReset(FUNASR_HANDLE handle, FUNASR_DEC_HANDLE dec_handle=nullptr);
// buffer
_FUNASRAPI FUNASR_RESULT	FunOfflineInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, 
												  FUNASR_MODE mode, QM_CALLBACK fn_callback, const std::vector<std::vector<float>> &hw_emb, 
												  int sampling_rate=16000, std::string wav_format="pcm", bool itn=true, FUNASR_DEC_HANDLE dec_handle=nullptr,
												  std::string svs_lang="auto", bool svs_itn=true);
// file, support wav & pcm
_FUNASRAPI FUNASR_RESULT	FunOfflineInfer(FUNASR_HANDLE handle, const char* sz_filename, FUNASR_MODE mode, 
											QM_CALLBACK fn_callback, const std::vector<std::vector<float>> &hw_emb, 
											int sampling_rate=16000, bool itn=true, FUNASR_DEC_HANDLE dec_handle=nullptr);
//#if !defined(__APPLE__)
_FUNASRAPI const std::vector<std::vector<float>> CompileHotwordEmbedding(FUNASR_HANDLE handle, std::string &hotwords, ASR_TYPE mode=ASR_OFFLINE);
//#endif

_FUNASRAPI void				FunOfflineUninit(FUNASR_HANDLE handle);

//2passStream
_FUNASRAPI FUNASR_HANDLE  	FunTpassInit(std::map<std::string, std::string>& model_path, int thread_num);
_FUNASRAPI FUNASR_HANDLE    FunTpassOnlineInit(FUNASR_HANDLE tpass_handle, std::vector<int> chunk_size={5,10,5});
// buffer
_FUNASRAPI FUNASR_RESULT	FunTpassInferBuffer(FUNASR_HANDLE handle, FUNASR_HANDLE online_handle, const char* sz_buf, 
												int n_len, std::vector<std::vector<std::string>> &punc_cache, bool input_finished=true, 
												int sampling_rate=16000, std::string wav_format="pcm", ASR_TYPE mode=ASR_TWO_PASS, 
												const std::vector<std::vector<float>> &hw_emb={{0.0}}, bool itn=true, FUNASR_DEC_HANDLE dec_handle=nullptr,
												std::string svs_lang="auto", bool svs_itn=true);
_FUNASRAPI void				FunTpassUninit(FUNASR_HANDLE handle);
_FUNASRAPI void				FunTpassOnlineUninit(FUNASR_HANDLE handle);

// wfst decoder
_FUNASRAPI FUNASR_DEC_HANDLE	FunASRWfstDecoderInit(FUNASR_HANDLE handle, int asr_type, float glob_beam, float lat_beam, float am_scale);
_FUNASRAPI void			FunASRWfstDecoderUninit(FUNASR_DEC_HANDLE handle);
_FUNASRAPI void			FunWfstDecoderLoadHwsRes(FUNASR_DEC_HANDLE handle, int inc_bias, std::unordered_map<std::string, int> &hws_map);
_FUNASRAPI void			FunWfstDecoderUnloadHwsRes(FUNASR_DEC_HANDLE handle);

