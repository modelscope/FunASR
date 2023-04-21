#include "precomp.h"



#ifdef __cplusplus
extern "C" {
#endif





	_RPPUNCAPI RPPUNC_HANDLE RapidPuncInit(const char* model_dir,int nThreadNum)
	{
		auto PuncOB =new CRapidPuncOnnx(model_dir, nThreadNum);

		return (void*)PuncOB;

	}
	_RPPUNCAPI void RapidPuncFinal(RPPUNC_HANDLE Handle)
	{
		if (Handle)
		{
			delete (CRapidPuncOnnx*)Handle;
			Handle = nullptr;
		}

	}
	_RPPUNCAPI RPPUNC_RESULT RapidPuncAddPunc(RPPUNC_HANDLE Handle, const char* szText)
	{
		if (!Handle)
			return nullptr;
		CRapidPuncOnnx* pPuncObj = (CRapidPuncOnnx*)Handle;
		auto Result = new string();
		(*Result) = pPuncObj->AddPunc(szText);
		return Result;
	}

	_RPPUNCAPI int RapidPuncGetResultLength(RPPUNC_RESULT Result)
	{
		if (!Result)
			return 0;

		return ((string*)Result)->size();
	}
	_RPPUNCAPI const char* RapidPuncGetResultText(RPPUNC_RESULT Result)
	{
		if (!Result)
			return 0;

		return ((string*)Result)->c_str();
	}
	_RPPUNCAPI void RapidPuncFree(RPPUNC_RESULT Result)
	{

		if (!Result)
			return;

		delete (string*)Result;

	}







#ifdef __cplusplus
}
#endif

