#include "precomp.h"
#ifdef __cplusplus 

extern "C" {
#endif

	// APIs for Init
	_FUNASRAPI FUNASR_HANDLE  FunASRInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		Model* mm = CreateModel(model_path, thread_num);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  FunVadInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		VadModel* mm = CreateVadModel(model_path, thread_num);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  FunPuncInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		PuncModel* mm = CreatePuncModel(model_path, thread_num);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  FunOfflineInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		OfflineStream* mm = CreateOfflineStream(model_path, thread_num);
		return mm;
	}

	// APIs for ASR Infer
	_FUNASRAPI FUNASR_RESULT FunASRRecogBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		Model* recog_obj = (Model*)handle;
		if (!recog_obj)
			return nullptr;

		int32_t sampling_rate = -1;
		Audio audio(1);
		if (!audio.LoadWav(sz_buf, n_len, &sampling_rate))
			return nullptr;

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
	
		return p_result;
	}

	// APIs for VAD Infer
	_FUNASRAPI FUNASR_RESULT FunVadWavFile(FUNASR_HANDLE handle, const char* sz_wavfile, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		VadModel* vad_obj = (VadModel*)handle;
		if (!vad_obj)
			return nullptr;
		
		int32_t sampling_rate = -1;
		Audio audio(1);
		if(!audio.LoadWav(sz_wavfile, &sampling_rate))
			return nullptr;

		FUNASR_VAD_RESULT* p_result = new FUNASR_VAD_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		
		vector<std::vector<int>> vad_segments;
		audio.Split(vad_obj, vad_segments);
		p_result->segments = new vector<std::vector<int>>(vad_segments);

		return p_result;
	}

	// APIs for PUNC Infer
	_FUNASRAPI const std::string FunPuncInfer(FUNASR_HANDLE handle, const char* sz_sentence, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		PuncModel* punc_obj = (PuncModel*)handle;
		if (!punc_obj)
			return nullptr;

		string punc_res = punc_obj->AddPunc(sz_sentence);
		return punc_res;
	}

	// APIs for Offline-stream Infer
	_FUNASRAPI FUNASR_RESULT FunOfflineStream(FUNASR_HANDLE handle, const char* sz_wavfile, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		OfflineStream* offline_stream = (OfflineStream*)handle;
		if (!offline_stream)
			return nullptr;
		
		int32_t sampling_rate = -1;
		Audio audio(1);
		if(!audio.LoadWav(sz_wavfile, &sampling_rate))
			return nullptr;
		if(offline_stream->UseVad()){
			audio.Split(offline_stream);
		}

		float* buff;
		int len;
		int flag = 0;
		int n_step = 0;
		int n_total = audio.GetQueueSize();
		FUNASR_RECOG_RESULT* p_result = new FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = (offline_stream->asr_handle)->Forward(buff, len, flag);
			p_result->msg+= msg;
			n_step++;
			if (fn_callback)
				fn_callback(n_step, n_total);
		}
		if(offline_stream->UsePunc()){
			string punc_res = (offline_stream->punc_handle)->AddPunc((p_result->msg).c_str());
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

	// APIs for GetRetSnippetTime
	_FUNASRAPI const float FunASRGetRetSnippetTime(FUNASR_RESULT result)
	{
		if (!result)
			return 0.0f;

		return ((FUNASR_RECOG_RESULT*)result)->snippet_time;
	}

	_FUNASRAPI const float FunVadGetRetSnippetTime(FUNASR_RESULT result)
	{
		if (!result)
			return 0.0f;

		return ((FUNASR_VAD_RESULT*)result)->snippet_time;
	}

	// APIs for GetResult
	_FUNASRAPI const char* FunASRGetResult(FUNASR_RESULT result,int n_index)
	{
		FUNASR_RECOG_RESULT * p_result = (FUNASR_RECOG_RESULT*)result;
		if(!p_result)
			return nullptr;

		return p_result->msg.c_str();
	}

	_FUNASRAPI vector<std::vector<int>>* FunVadGetResult(FUNASR_RESULT result,int n_index)
	{
		FUNASR_VAD_RESULT * p_result = (FUNASR_VAD_RESULT*)result;
		if(!p_result)
			return nullptr;

		return p_result->segments;
	}

	// APIs for FreeResult
	_FUNASRAPI void FunASRFreeResult(FUNASR_RESULT result)
	{
		if (result)
		{
			delete (FUNASR_RECOG_RESULT*)result;
		}
	}

	_FUNASRAPI void FunVadFreeResult(FUNASR_RESULT result)
	{
		FUNASR_VAD_RESULT * p_result = (FUNASR_VAD_RESULT*)result;
		if (p_result)
		{
			if(p_result->segments){
				delete p_result->segments;
			}
			delete p_result;
		}
	}

	// APIs for Uninit
	_FUNASRAPI void FunASRUninit(FUNASR_HANDLE handle)
	{
		Model* recog_obj = (Model*)handle;

		if (!recog_obj)
			return;

		delete recog_obj;
	}

	_FUNASRAPI void FunVadUninit(FUNASR_HANDLE handle)
	{
		VadModel* recog_obj = (VadModel*)handle;

		if (!recog_obj)
			return;

		delete recog_obj;
	}

	_FUNASRAPI void FunPuncUninit(FUNASR_HANDLE handle)
	{
		PuncModel* punc_obj = (PuncModel*)handle;

		if (!punc_obj)
			return;

		delete punc_obj;
	}

	_FUNASRAPI void FunOfflineUninit(FUNASR_HANDLE handle)
	{
		OfflineStream* offline_stream = (OfflineStream*)handle;

		if (!offline_stream)
			return;

		delete offline_stream;
	}

#ifdef __cplusplus 

}
#endif

