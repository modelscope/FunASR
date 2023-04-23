#pragma once
#ifdef WIN32


#ifdef _QMPUC_API_EXPORT

#define  _RPPUNCRAPI __declspec(dllexport)
#else
#define  _RPPUNCAPI __declspec(dllimport)
#endif


#else
#define _RPPUNCAPI  
#endif




#ifdef __cplusplus
extern "C" {
#endif

#define RPPUNC_DEFAULT_THREADNUM  4
	typedef  void* RPPUNC_HANDLE;
	typedef  void* RPPUNC_RESULT;

	_RPPUNCAPI RPPUNC_HANDLE RapidPuncInit(const char* model_dir,int nThreadNum);
	_RPPUNCAPI void RapidPuncFinal(RPPUNC_HANDLE Handle);
	_RPPUNCAPI RPPUNC_RESULT RapidPuncAddPunc(RPPUNC_HANDLE Handle,const char * szText);

	_RPPUNCAPI int RapidPuncGetResultLength(RPPUNC_RESULT Result);
	_RPPUNCAPI const char * RapidPuncGetResultText(RPPUNC_RESULT Result);
	_RPPUNCAPI void RapidPuncFree(RPPUNC_RESULT Result);




#ifdef __cplusplus
}

#endif