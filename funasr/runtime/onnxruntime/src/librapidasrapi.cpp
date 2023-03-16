#include "precomp.h"
#ifdef __cplusplus 

extern "C" {
#endif


	// APIs for qmasr
	_RAPIDASRAPI RPASR_HANDLE  RapidAsrInit(const char* szModelDir, int nThreadNum)
	{


		Model* mm = create_model(szModelDir, nThreadNum); 

		return mm;
	}


	_RAPIDASRAPI RPASR_RESULT RapidAsrRecogBuffer(RPASR_HANDLE handle, const char* szBuf, int nLen, RPASR_MODE Mode, QM_CALLBACK fnCallback)
	{


		Model* pRecogObj = (Model*)handle;

		if (!pRecogObj)
			return nullptr;

		Audio audio(1);
		if (!audio.loadwav(szBuf, nLen))
			return nullptr;
		//audio.split();

		float* buff;
		int len;
		int flag=0;
		RPASR_RECOG_RESULT* pResult = new RPASR_RECOG_RESULT;
		pResult->snippet_time = audio.get_time_len();
		int nStep = 0;
		int nTotal = audio.get_queue_size();
		while (audio.fetch(buff, len, flag) > 0) {
			pRecogObj->reset();
			string msg = pRecogObj->forward(buff, len, flag);
			pResult->msg += msg;
			nStep++;
			if (fnCallback)
				fnCallback(nStep, nTotal);
		}


		return pResult;
	}

	_RAPIDASRAPI RPASR_RESULT RapidAsrRecogPCMBuffer(RPASR_HANDLE handle, const char* szBuf, int nLen, RPASR_MODE Mode, QM_CALLBACK fnCallback)
	{

		Model* pRecogObj = (Model*)handle;

		if (!pRecogObj)
			return nullptr;

		Audio audio(1);
		if (!audio.loadpcmwav(szBuf, nLen))
			return nullptr;
		//audio.split();

		float* buff;
		int len;
		int flag = 0;
		RPASR_RECOG_RESULT* pResult = new RPASR_RECOG_RESULT;
		pResult->snippet_time = audio.get_time_len();
		int nStep = 0;
		int nTotal = audio.get_queue_size();
		while (audio.fetch(buff, len, flag) > 0) {
			pRecogObj->reset();
			string msg = pRecogObj->forward(buff, len, flag);
			pResult->msg += msg;
			nStep++;
			if (fnCallback)
				fnCallback(nStep, nTotal);
		}


		return pResult;

	}

	_RAPIDASRAPI RPASR_RESULT RapidAsrRecogPCMFile(RPASR_HANDLE handle, const char* szFileName, RPASR_MODE Mode, QM_CALLBACK fnCallback)
	{

		Model* pRecogObj = (Model*)handle;

		if (!pRecogObj)
			return nullptr;

		Audio audio(1);
		if (!audio.loadpcmwav(szFileName))
			return nullptr;
		//audio.split();

		float* buff;
		int len;
		int flag = 0;
		RPASR_RECOG_RESULT* pResult = new RPASR_RECOG_RESULT;
		pResult->snippet_time = audio.get_time_len();
		int nStep = 0;
		int nTotal = audio.get_queue_size();
		while (audio.fetch(buff, len, flag) > 0) {
			pRecogObj->reset();
			string msg = pRecogObj->forward(buff, len, flag);
			pResult->msg += msg;
			nStep++;
			if (fnCallback)
				fnCallback(nStep, nTotal);
		}


		return pResult;

	}

	_RAPIDASRAPI RPASR_RESULT RapidAsrRecogFile(RPASR_HANDLE handle, const char* szWavfile, RPASR_MODE Mode, QM_CALLBACK fnCallback)
	{
		Model* pRecogObj = (Model*)handle;

		if (!pRecogObj)
			return nullptr;

		Audio audio(1);
		if(!audio.loadwav(szWavfile))
			return nullptr;
		//audio.split();

		float* buff;
		int len;
		int flag = 0;
		int nStep = 0;
		int nTotal = audio.get_queue_size();
		RPASR_RECOG_RESULT* pResult = new RPASR_RECOG_RESULT;
		pResult->snippet_time = audio.get_time_len();
		while (audio.fetch(buff, len, flag) > 0) {
			pRecogObj->reset();
			string msg = pRecogObj->forward(buff, len, flag);
			pResult->msg+= msg;
			nStep++;
			if (fnCallback)
				fnCallback(nStep, nTotal);
		}
	
	


		return pResult;
	}

	_RAPIDASRAPI const int RapidAsrGetRetNumber(RPASR_RESULT Result)
	{
		if (!Result)
			return 0;

		return 1;
		
	}


	_RAPIDASRAPI const float RapidAsrGetRetSnippetTime(RPASR_RESULT Result)
	{
		if (!Result)
			return 0.0f;

		return ((RPASR_RECOG_RESULT*)Result)->snippet_time;

	}

	_RAPIDASRAPI const char* RapidAsrGetResult(RPASR_RESULT Result,int nIndex)
	{
		RPASR_RECOG_RESULT * pResult = (RPASR_RECOG_RESULT*)Result;
		if(!pResult)
			return nullptr;

		return pResult->msg.c_str();
	
	}

	_RAPIDASRAPI void RapidAsrFreeResult(RPASR_RESULT Result)
	{

		if (Result)
		{
			delete (RPASR_RECOG_RESULT*)Result;

		}
	}

	_RAPIDASRAPI void RapidAsrUninit(RPASR_HANDLE handle)
	{

		Model* pRecogObj = (Model*)handle;


		if (!pRecogObj)
			return;

		delete pRecogObj;

	}



#ifdef __cplusplus 

}
#endif

