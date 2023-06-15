#pragma once
#include <map>
#include <vector>

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

#ifdef __cplusplus 

extern "C" {
#endif

typedef void* FUNASR_HANDLE;
typedef void* FUNASR_RESULT;
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

typedef void (* QM_CALLBACK)(int cur_step, int n_total); // n_total: total steps; cur_step: Current Step.
	
// ASR
_FUNASRAPI FUNASR_HANDLE  	FunASRInit(std::map<std::string, std::string>& model_path, int thread_num);
// buffer
_FUNASRAPI FUNASR_RESULT	FunASRInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, FUNASR_MODE mode, QM_CALLBACK fn_callback, int sampling_rate=16000);
// file, support wav & pcm
_FUNASRAPI FUNASR_RESULT	FunASRInfer(FUNASR_HANDLE handle, const char* sz_filename, FUNASR_MODE mode, QM_CALLBACK fn_callback, int sampling_rate=16000);

_FUNASRAPI const char*	FunASRGetResult(FUNASR_RESULT result,int n_index);
_FUNASRAPI const int	FunASRGetRetNumber(FUNASR_RESULT result);
_FUNASRAPI void			FunASRFreeResult(FUNASR_RESULT result);
_FUNASRAPI void			FunASRUninit(FUNASR_HANDLE handle);
_FUNASRAPI const float	FunASRGetRetSnippetTime(FUNASR_RESULT result);

// VAD
_FUNASRAPI FUNASR_HANDLE  	FsmnVadInit(std::map<std::string, std::string>& model_path, int thread_num);
_FUNASRAPI FUNASR_HANDLE  	FsmnVadOnlineInit(FUNASR_HANDLE fsmnvad_handle);
// buffer
_FUNASRAPI FUNASR_RESULT	FsmnVadInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, QM_CALLBACK fn_callback, bool input_finished=true, int sampling_rate=16000);
// file, support wav & pcm
_FUNASRAPI FUNASR_RESULT	FsmnVadInfer(FUNASR_HANDLE handle, const char* sz_filename, QM_CALLBACK fn_callback, int sampling_rate=16000);

_FUNASRAPI std::vector<std::vector<int>>*	FsmnVadGetResult(FUNASR_RESULT result,int n_index);
_FUNASRAPI void			 	FsmnVadFreeResult(FUNASR_RESULT result);
_FUNASRAPI void				FsmnVadUninit(FUNASR_HANDLE handle);
_FUNASRAPI const float		FsmnVadGetRetSnippetTime(FUNASR_RESULT result);

// PUNC
_FUNASRAPI FUNASR_HANDLE  		CTTransformerInit(std::map<std::string, std::string>& model_path, int thread_num);
_FUNASRAPI const std::string	CTTransformerInfer(FUNASR_HANDLE handle, const char* sz_sentence, FUNASR_MODE mode, QM_CALLBACK fn_callback);
_FUNASRAPI void					CTTransformerUninit(FUNASR_HANDLE handle);

//OfflineStream
_FUNASRAPI FUNASR_HANDLE  	FunOfflineInit(std::map<std::string, std::string>& model_path, int thread_num);
// buffer
_FUNASRAPI FUNASR_RESULT	FunOfflineInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, FUNASR_MODE mode, QM_CALLBACK fn_callback, int sampling_rate=16000);
// file, support wav & pcm
_FUNASRAPI FUNASR_RESULT	FunOfflineInfer(FUNASR_HANDLE handle, const char* sz_filename, FUNASR_MODE mode, QM_CALLBACK fn_callback, int sampling_rate=16000);
_FUNASRAPI void				FunOfflineUninit(FUNASR_HANDLE handle);

#ifdef __cplusplus 

}
#endif
