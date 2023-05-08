#include "precomp.h"
#ifdef __cplusplus 

extern "C" {
#endif

	// APIs for Init
	_FUNASRAPI FUNASR_HANDLE  FunASRInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		funasr::Model* mm = funasr::CreateModel(model_path, thread_num);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  FsmnVadInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		funasr::VadModel* mm = funasr::CreateVadModel(model_path, thread_num);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  FunPuncInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		funasr::PuncModel* mm = funasr::CreatePuncModel(model_path, thread_num);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  FunOfflineInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		funasr::OfflineStream* mm = funasr::CreateOfflineStream(model_path, thread_num);
		return mm;
	}

	// APIs for ASR Infer
	_FUNASRAPI FUNASR_RESULT FunASRRecogBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		funasr::Model* recog_obj = (funasr::Model*)handle;
		if (!recog_obj)
			return nullptr;

		int32_t sampling_rate = -1;
		funasr::Audio audio(1);
		if (!audio.LoadWav(sz_buf, n_len, &sampling_rate))
			return nullptr;

		float* buff;
		int len;
		int flag=0;
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
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
		funasr::Model* recog_obj = (funasr::Model*)handle;
		if (!recog_obj)
			return nullptr;

		funasr::Audio audio(1);
		if (!audio.LoadPcmwav(sz_buf, n_len, &sampling_rate))
			return nullptr;

		float* buff;
		int len;
		int flag = 0;
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
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
		funasr::Model* recog_obj = (funasr::Model*)handle;
		if (!recog_obj)
			return nullptr;

		funasr::Audio audio(1);
		if (!audio.LoadPcmwav(sz_filename, &sampling_rate))
			return nullptr;

		float* buff;
		int len;
		int flag = 0;
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
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
		funasr::Model* recog_obj = (funasr::Model*)handle;
		if (!recog_obj)
			return nullptr;
		
		int32_t sampling_rate = -1;
		funasr::Audio audio(1);
		if(!audio.LoadWav(sz_wavfile, &sampling_rate))
			return nullptr;

		float* buff;
		int len;
		int flag = 0;
		int n_step = 0;
		int n_total = audio.GetQueueSize();
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
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
	_FUNASRAPI FUNASR_RESULT FsmnVadWavFile(FUNASR_HANDLE handle, const char* sz_wavfile, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		funasr::VadModel* vad_obj = (funasr::VadModel*)handle;
		if (!vad_obj)
			return nullptr;
		
		int32_t sampling_rate = -1;
		funasr::Audio audio(1);
		if(!audio.LoadWav(sz_wavfile, &sampling_rate))
			return nullptr;

		funasr::FUNASR_VAD_RESULT* p_result = new funasr::FUNASR_VAD_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		
		vector<std::vector<int>> vad_segments;
		audio.Split(vad_obj, vad_segments);
		p_result->segments = new vector<std::vector<int>>(vad_segments);

		return p_result;
	}

	// APIs for PUNC Infer
	_FUNASRAPI const std::string FunPuncInfer(FUNASR_HANDLE handle, const char* sz_sentence, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		funasr::PuncModel* punc_obj = (funasr::PuncModel*)handle;
		if (!punc_obj)
			return nullptr;

		string punc_res = punc_obj->AddPunc(sz_sentence);
		return punc_res;
	}

	// APIs for Offline-stream Infer
	_FUNASRAPI FUNASR_RESULT FunOfflineRecogFile(FUNASR_HANDLE handle, const char* sz_wavfile, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		funasr::OfflineStream* offline_stream = (funasr::OfflineStream*)handle;
		if (!offline_stream)
			return nullptr;
		
		int32_t sampling_rate = -1;
		funasr::Audio audio(1);
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
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
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

	_FUNASRAPI FUNASR_RESULT FunOfflineRecogPCMBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, int sampling_rate, FUNASR_MODE mode, QM_CALLBACK fn_callback)
	{
		funasr::OfflineStream* offline_stream = (funasr::OfflineStream*)handle;
		if (!offline_stream)
			return nullptr;

		funasr::Audio audio(1);
		if (!audio.LoadPcmwav(sz_buf, n_len, &sampling_rate))
			return nullptr;
		if(offline_stream->UseVad()){
			audio.Split(offline_stream);
		}

		float* buff;
		int len;
		int flag = 0;
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		int n_step = 0;
		int n_total = audio.GetQueueSize();
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = (offline_stream->asr_handle)->Forward(buff, len, flag);
			p_result->msg += msg;
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

		return ((funasr::FUNASR_RECOG_RESULT*)result)->snippet_time;
	}

	_FUNASRAPI const float FsmnVadGetRetSnippetTime(FUNASR_RESULT result)
	{
		if (!result)
			return 0.0f;

		return ((funasr::FUNASR_VAD_RESULT*)result)->snippet_time;
	}

	// APIs for GetResult
	_FUNASRAPI const char* FunASRGetResult(FUNASR_RESULT result,int n_index)
	{
		funasr::FUNASR_RECOG_RESULT * p_result = (funasr::FUNASR_RECOG_RESULT*)result;
		if(!p_result)
			return nullptr;

		return p_result->msg.c_str();
	}

	_FUNASRAPI vector<std::vector<int>>* FsmnVadGetResult(FUNASR_RESULT result,int n_index)
	{
		funasr::FUNASR_VAD_RESULT * p_result = (funasr::FUNASR_VAD_RESULT*)result;
		if(!p_result)
			return nullptr;

		return p_result->segments;
	}

	// APIs for FreeResult
	_FUNASRAPI void FunASRFreeResult(FUNASR_RESULT result)
	{
		if (result)
		{
			delete (funasr::FUNASR_RECOG_RESULT*)result;
		}
	}

	_FUNASRAPI void FsmnVadFreeResult(FUNASR_RESULT result)
	{
		funasr::FUNASR_VAD_RESULT * p_result = (funasr::FUNASR_VAD_RESULT*)result;
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
		funasr::Model* recog_obj = (funasr::Model*)handle;

		if (!recog_obj)
			return;

		delete recog_obj;
	}

	_FUNASRAPI void FsmnVadUninit(FUNASR_HANDLE handle)
	{
		funasr::VadModel* recog_obj = (funasr::VadModel*)handle;

		if (!recog_obj)
			return;

		delete recog_obj;
	}

	_FUNASRAPI void FunPuncUninit(FUNASR_HANDLE handle)
	{
		funasr::PuncModel* punc_obj = (funasr::PuncModel*)handle;

		if (!punc_obj)
			return;

		delete punc_obj;
	}

	_FUNASRAPI void FunOfflineUninit(FUNASR_HANDLE handle)
	{
		funasr::OfflineStream* offline_stream = (funasr::OfflineStream*)handle;

		if (!offline_stream)
			return;

		delete offline_stream;
	}

#ifdef __cplusplus 

}
#endif

