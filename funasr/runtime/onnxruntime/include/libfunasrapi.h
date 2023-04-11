#pragma once

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

typedef void (* QM_CALLBACK)(int nCurStep, int nTotal); // nTotal: total steps; nCurStep: Current Step.
	
// APIs for qmasr
_FUNASRAPI FUNASR_HANDLE  FunASRInit(const char* szModelDir, int nThread, bool quantize);


// if not give a fnCallback ,it should be NULL 
_FUNASRAPI FUNASR_RESULT	FunASRRecogBuffer(FUNASR_HANDLE handle, const char* szBuf, int nLen, FUNASR_MODE Mode, QM_CALLBACK fnCallback);

_FUNASRAPI FUNASR_RESULT	FunASRRecogPCMBuffer(FUNASR_HANDLE handle, const char* szBuf, int nLen, FUNASR_MODE Mode, QM_CALLBACK fnCallback);

_FUNASRAPI FUNASR_RESULT	FunASRRecogPCMFile(FUNASR_HANDLE handle, const char* szFileName, FUNASR_MODE Mode, QM_CALLBACK fnCallback);

_FUNASRAPI FUNASR_RESULT	FunASRRecogFile(FUNASR_HANDLE handle, const char* szWavfile, FUNASR_MODE Mode, QM_CALLBACK fnCallback);

_FUNASRAPI const char*	FunASRGetResult(FUNASR_RESULT Result,int nIndex);

_FUNASRAPI const int		FunASRGetRetNumber(FUNASR_RESULT Result);

_FUNASRAPI void			FunASRFreeResult(FUNASR_RESULT Result);

_FUNASRAPI void			FunASRUninit(FUNASR_HANDLE Handle);

_FUNASRAPI const float	FunASRGetRetSnippetTime(FUNASR_RESULT Result);

#ifdef __cplusplus 

}
#endif
