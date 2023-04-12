#include "precomp.h"
#ifdef __cplusplus 

extern "C" {
#endif

	// APIs for qmasr
	_FUNASRAPI FUNASR_HANDLE  FunASRInit(const char* szModelDir, int nThreadNum, bool quantize)
	{
		Model* mm = create_model(szModelDir, nThreadNum, quantize);
		return mm;
	}

	_FUNASRAPI FUNASR_RESULT FunASRRecogBuffer(FUNASR_HANDLE handle, const char* szBuf, int nLen, FUNASR_MODE Mode, QM_CALLBACK fnCallback)
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
		FUNASR_RECOG_RESULT* pResult = new FUNASR_RECOG_RESULT;
		pResult->snippet_time = audio.get_time_len();
		int nStep = 0;
		int nTotal = audio.get_queue_size();
		while (audio.fetch(buff, len, flag) > 0) {
			//pRecogObj->reset();
			string msg = pRecogObj->forward(buff, len, flag);
			pResult->msg += msg;
			nStep++;
			if (fnCallback)
				fnCallback(nStep, nTotal);
		}

		return pResult;
	}

	_FUNASRAPI FUNASR_RESULT FunASRRecogPCMBuffer(FUNASR_HANDLE handle, const char* szBuf, int nLen, FUNASR_MODE Mode, QM_CALLBACK fnCallback)
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
		FUNASR_RECOG_RESULT* pResult = new FUNASR_RECOG_RESULT;
		pResult->snippet_time = audio.get_time_len();
		int nStep = 0;
		int nTotal = audio.get_queue_size();
		while (audio.fetch(buff, len, flag) > 0) {
			//pRecogObj->reset();
			string msg = pRecogObj->forward(buff, len, flag);
			pResult->msg += msg;
			nStep++;
			if (fnCallback)
				fnCallback(nStep, nTotal);
		}

		return pResult;
	}

	_FUNASRAPI FUNASR_RESULT FunASRRecogPCMFile(FUNASR_HANDLE handle, const char* szFileName, FUNASR_MODE Mode, QM_CALLBACK fnCallback)
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
		FUNASR_RECOG_RESULT* pResult = new FUNASR_RECOG_RESULT;
		pResult->snippet_time = audio.get_time_len();
		int nStep = 0;
		int nTotal = audio.get_queue_size();
		while (audio.fetch(buff, len, flag) > 0) {
			//pRecogObj->reset();
			string msg = pRecogObj->forward(buff, len, flag);
			pResult->msg += msg;
			nStep++;
			if (fnCallback)
				fnCallback(nStep, nTotal);
		}

		return pResult;
	}

	_FUNASRAPI FUNASR_RESULT FunASRRecogFile(FUNASR_HANDLE handle, const char* szWavfile, FUNASR_MODE Mode, QM_CALLBACK fnCallback)
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
		FUNASR_RECOG_RESULT* pResult = new FUNASR_RECOG_RESULT;
		pResult->snippet_time = audio.get_time_len();
		while (audio.fetch(buff, len, flag) > 0) {
			//pRecogObj->reset();
			string msg = pRecogObj->forward(buff, len, flag);
			pResult->msg+= msg;
			nStep++;
			if (fnCallback)
				fnCallback(nStep, nTotal);
		}
	
		return pResult;
	}

	_FUNASRAPI const int FunASRGetRetNumber(FUNASR_RESULT Result)
	{
		if (!Result)
			return 0;

		return 1;
	}


	_FUNASRAPI const float FunASRGetRetSnippetTime(FUNASR_RESULT Result)
	{
		if (!Result)
			return 0.0f;

		return ((FUNASR_RECOG_RESULT*)Result)->snippet_time;
	}

	_FUNASRAPI const char* FunASRGetResult(FUNASR_RESULT Result,int nIndex)
	{
		FUNASR_RECOG_RESULT * pResult = (FUNASR_RECOG_RESULT*)Result;
		if(!pResult)
			return nullptr;

		return pResult->msg.c_str();
	}

	_FUNASRAPI void FunASRFreeResult(FUNASR_RESULT Result)
	{
		if (Result)
		{
			delete (FUNASR_RECOG_RESULT*)Result;
		}
	}

	_FUNASRAPI void FunASRUninit(FUNASR_HANDLE handle)
	{
		Model* pRecogObj = (Model*)handle;

		if (!pRecogObj)
			return;

		delete pRecogObj;
	}

#ifdef __cplusplus 

}
#endif

