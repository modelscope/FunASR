#pragma once


#ifdef WIN32


#ifdef _RPASR_API_EXPORT

#define  _RAPIDASRAPI __declspec(dllexport)
#else
#define  _RAPIDASRAPI __declspec(dllimport)
#endif
	

#else
#define _RAPIDASRAPI  
#endif





#ifndef _WIN32

#define RPASR_CALLBCK_PREFIX __attribute__((__stdcall__))

#else
#define RPASR_CALLBCK_PREFIX __stdcall
#endif
	

#ifdef __cplusplus 

extern "C" {
#endif

typedef void* RPASR_HANDLE;

typedef void* RPASR_RESULT;

typedef unsigned char RPASR_BOOL;

#define RPASR_TRUE 1
#define RPASR_FALSE 0
#define QM_DEFAULT_THREAD_NUM  4


typedef enum
{
 RASR_NONE=-1,
 RASRM_CTC_GREEDY_SEARCH=0,
 RASRM_CTC_RPEFIX_BEAM_SEARCH = 1,
 RASRM_ATTENSION_RESCORING = 2,
 
}RPASR_MODE;

typedef enum {

	RPASR_MODEL_PADDLE = 0,
	RPASR_MODEL_PADDLE_2 = 1,
	RPASR_MODEL_K2 = 2,
	RPASR_MODEL_PARAFORMER = 3,

}RPASR_MODEL_TYPE;


typedef void (* QM_CALLBACK)(int nCurStep, int nTotal); // nTotal: total steps; nCurStep: Current Step.
	
	// APIs for qmasr

_RAPIDASRAPI RPASR_HANDLE  RapidAsrInit(const char* szModelDir, int nThread);



// if not give a fnCallback ,it should be NULL 
_RAPIDASRAPI RPASR_RESULT	RapidAsrRecogBuffer(RPASR_HANDLE handle, const char* szBuf, int nLen, RPASR_MODE Mode, QM_CALLBACK fnCallback);
_RAPIDASRAPI RPASR_RESULT	RapidAsrRecogPCMBuffer(RPASR_HANDLE handle, const char* szBuf, int nLen, RPASR_MODE Mode, QM_CALLBACK fnCallback);

_RAPIDASRAPI RPASR_RESULT	RapidAsrRecogPCMFile(RPASR_HANDLE handle, const char* szFileName, RPASR_MODE Mode, QM_CALLBACK fnCallback);

_RAPIDASRAPI RPASR_RESULT	RapidAsrRecogFile(RPASR_HANDLE handle, const char* szWavfile, RPASR_MODE Mode, QM_CALLBACK fnCallback);

_RAPIDASRAPI const char*	RapidAsrGetResult(RPASR_RESULT Result,int nIndex);

_RAPIDASRAPI const int		RapidAsrGetRetNumber(RPASR_RESULT Result);
_RAPIDASRAPI void			RapidAsrFreeResult(RPASR_RESULT Result);


_RAPIDASRAPI void			RapidAsrUninit(RPASR_HANDLE Handle);


#ifdef __cplusplus 

}
#endif