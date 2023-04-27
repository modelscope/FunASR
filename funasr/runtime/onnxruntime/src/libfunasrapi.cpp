#include "precomp.h"
#ifdef __cplusplus 

extern "C" {
#endif

	// APIs for funasr
	_FUNASRAPI FUNASR_HANDLE  FunASRInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		Model* mm = CreateModel(model_path, thread_num);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  FunVadInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		Model* mm = CreateModel(model_path, thread_num);
		return mm;
	}

	_FUNASRAPI FUNASR_RESULT FunASRRecogBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		Model* recog_obj = (Model*)handle;
		if (!recog_obj)
			return nullptr;

		int32_t sampling_rate = -1;
		Audio audio(1);
		if (!audio.LoadWav(sz_buf, n_len, &sampling_rate))
			return nullptr;
		if(recog_obj->UseVad()){
			audio.Split(recog_obj);
		}

		float* buff;
		int len;
		int flag=0;
		FUNASR_RECOG_RESULT* p_result = new FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		int n_step = 0;
		int n_total = audio.GetQueueSize();
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = recog_obj->Forward(buff, len, flag);
			p_result->msg += msg;
			n_step++;
			if (fn_callback)
				fn_callback(n_step, n_total);
		}
		if(recog_obj->UsePunc()){
			string punc_res = recog_obj->AddPunc((p_result->msg).c_str());
			p_result->msg = punc_res;
		}

		return p_result;
	}

	_FUNASRAPI FUNASR_RESULT FunASRRecogPCMBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, int sampling_rate, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		Model* recog_obj = (Model*)handle;
		if (!recog_obj)
			return nullptr;

		Audio audio(1);
		if (!audio.LoadPcmwav(sz_buf, n_len, &sampling_rate))
			return nullptr;
		if(recog_obj->UseVad()){
			audio.Split(recog_obj);
		}

		float* buff;
		int len;
		int flag = 0;
		FUNASR_RECOG_RESULT* p_result = new FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		int n_step = 0;
		int n_total = audio.GetQueueSize();
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = recog_obj->Forward(buff, len, flag);
			p_result->msg += msg;
			n_step++;
			if (fn_callback)
				fn_callback(n_step, n_total);
		}
		if(recog_obj->UsePunc()){
			string punc_res = recog_obj->AddPunc((p_result->msg).c_str());
			p_result->msg = punc_res;
		}

		return p_result;
	}

	_FUNASRAPI FUNASR_RESULT FunASRRecogPCMFile(FUNASR_HANDLE handle, const char* sz_filename, int sampling_rate, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		Model* recog_obj = (Model*)handle;
		if (!recog_obj)
			return nullptr;

		Audio audio(1);
		if (!audio.LoadPcmwav(sz_filename, &sampling_rate))
			return nullptr;
		if(recog_obj->UseVad()){
			audio.Split(recog_obj);
		}

		float* buff;
		int len;
		int flag = 0;
		FUNASR_RECOG_RESULT* p_result = new FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		int n_step = 0;
		int n_total = audio.GetQueueSize();
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = recog_obj->Forward(buff, len, flag);
			p_result->msg += msg;
			n_step++;
			if (fn_callback)
				fn_callback(n_step, n_total);
		}
		if(recog_obj->UsePunc()){
			string punc_res = recog_obj->AddPunc((p_result->msg).c_str());
			p_result->msg = punc_res;
		}

		return p_result;
	}

	_FUNASRAPI FUNASR_RESULT FunASRRecogFile(FUNASR_HANDLE handle, const char* sz_wavfile, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		Model* recog_obj = (Model*)handle;
		if (!recog_obj)
			return nullptr;
		
		int32_t sampling_rate = -1;
		Audio audio(1);
		if(!audio.LoadWav(sz_wavfile, &sampling_rate))
			return nullptr;
		if(recog_obj->UseVad()){
			audio.Split(recog_obj);
		}

		float* buff;
		int len;
		int flag = 0;
		int n_step = 0;
		int n_total = audio.GetQueueSize();
		FUNASR_RECOG_RESULT* p_result = new FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = recog_obj->Forward(buff, len, flag);
			p_result->msg+= msg;
			n_step++;
			if (fn_callback)
				fn_callback(n_step, n_total);
		}
		if(recog_obj->UsePunc()){
			string punc_res = recog_obj->AddPunc((p_result->msg).c_str());
			p_result->msg = punc_res;
		}
	
		return p_result;
	}

	_FUNASRAPI const int FunASRGetRetNumber(FUNASR_RESULT result)
	{
		if (!result)
			return 0;

		return 1;
	}


	_FUNASRAPI const float FunASRGetRetSnippetTime(FUNASR_RESULT result)
	{
		if (!result)
			return 0.0f;

		return ((FUNASR_RECOG_RESULT*)result)->snippet_time;
	}

	_FUNASRAPI const char* FunASRGetResult(FUNASR_RESULT result,int n_index)
	{
		FUNASR_RECOG_RESULT * p_result = (FUNASR_RECOG_RESULT*)result;
		if(!p_result)
			return nullptr;

		return p_result->msg.c_str();
	}

	_FUNASRAPI void FunASRFreeResult(FUNASR_RESULT result)
	{
		if (result)
		{
			delete (FUNASR_RECOG_RESULT*)result;
		}
	}

	_FUNASRAPI void FunASRUninit(FUNASR_HANDLE handle)
	{
		Model* recog_obj = (Model*)handle;

		if (!recog_obj)
			return;

		delete recog_obj;
	}

#ifdef __cplusplus 

}
#endif

