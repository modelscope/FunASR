#include "precomp.h"
#include <vector>
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

	_FUNASRAPI FUNASR_HANDLE  FsmnVadOnlineInit(FUNASR_HANDLE fsmnvad_handle)
	{
		funasr::VadModel* mm = funasr::CreateVadModel(fsmnvad_handle);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  CTTransformerInit(std::map<std::string, std::string>& model_path, int thread_num, PUNC_TYPE type)
	{
		funasr::PuncModel* mm = funasr::CreatePuncModel(model_path, thread_num, type);
		return mm;
	}

	_FUNASRAPI FUNASR_HANDLE  FunOfflineInit(std::map<std::string, std::string>& model_path, int thread_num)
	{
		funasr::OfflineStream* mm = funasr::CreateOfflineStream(model_path, thread_num);
		return mm;
	}

	// APIs for ASR Infer
	_FUNASRAPI FUNASR_RESULT FunASRInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, FUNASR_MODE mode, QM_CALLBACK fn_callback, int sampling_rate, std::string wav_format)
	{
		funasr::Model* recog_obj = (funasr::Model*)handle;
		if (!recog_obj)
			return nullptr;

		funasr::Audio audio(1);
		if(wav_format == "pcm" || wav_format == "PCM"){
			if (!audio.LoadPcmwav(sz_buf, n_len, &sampling_rate))
				return nullptr;
		}else{
			if (!audio.FfmpegLoad(sz_buf, n_len))
				return nullptr;
		}

		float* buff;
		int len;
		int flag = 0;
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		if(p_result->snippet_time == 0){
            return p_result;
        }
		int n_step = 0;
		int n_total = audio.GetQueueSize();
    // TODO: hotword not implement, use a empty embedding
    std::vector<std::vector<float>> emb;
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = recog_obj->Forward(buff, len, flag, emb);
			p_result->msg += msg;
			n_step++;
			if (fn_callback)
				fn_callback(n_step, n_total);
		}

		return p_result;
	}

	_FUNASRAPI FUNASR_RESULT FunASRInfer(FUNASR_HANDLE handle, const char* sz_filename, FUNASR_MODE mode, QM_CALLBACK fn_callback, int sampling_rate)
	{
		funasr::Model* recog_obj = (funasr::Model*)handle;
		if (!recog_obj)
			return nullptr;

		funasr::Audio audio(1);
		if(funasr::is_target_file(sz_filename, "wav")){
			int32_t sampling_rate_ = -1;
			if(!audio.LoadWav(sz_filename, &sampling_rate_))
				return nullptr;
		}else if(funasr::is_target_file(sz_filename, "pcm")){
			if (!audio.LoadPcmwav(sz_filename, &sampling_rate))
				return nullptr;
		}else{
			if (!audio.FfmpegLoad(sz_filename))
				return nullptr;
		}

		float* buff;
		int len;
		int flag = 0;
		int n_step = 0;
		int n_total = audio.GetQueueSize();
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		if(p_result->snippet_time == 0){
            return p_result;
        }
    // TODO: hotword not implement, use a empty embedding
    std::vector<std::vector<float>> emb;
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = recog_obj->Forward(buff, len, flag, emb);
			p_result->msg += msg;
			n_step++;
			if (fn_callback)
				fn_callback(n_step, n_total);
		}

		return p_result;
	}

	// APIs for VAD Infer
	_FUNASRAPI FUNASR_RESULT FsmnVadInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, QM_CALLBACK fn_callback, bool input_finished, int sampling_rate, std::string wav_format)
	{
		funasr::VadModel* vad_obj = (funasr::VadModel*)handle;
		if (!vad_obj)
			return nullptr;

		funasr::Audio audio(1);
		if(wav_format == "pcm" || wav_format == "PCM"){
			if (!audio.LoadPcmwav(sz_buf, n_len, &sampling_rate))
				return nullptr;
		}else{
			if (!audio.FfmpegLoad(sz_buf, n_len))
				return nullptr;
		}

		funasr::FUNASR_VAD_RESULT* p_result = new funasr::FUNASR_VAD_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		if(p_result->snippet_time == 0){
            return p_result;
        }
		
		vector<std::vector<int>> vad_segments;
		audio.Split(vad_obj, vad_segments, input_finished);
		p_result->segments = new vector<std::vector<int>>(vad_segments);

		return p_result;
	}

	_FUNASRAPI FUNASR_RESULT FsmnVadInfer(FUNASR_HANDLE handle, const char* sz_filename, QM_CALLBACK fn_callback, int sampling_rate)
	{
		funasr::VadModel* vad_obj = (funasr::VadModel*)handle;
		if (!vad_obj)
			return nullptr;

		funasr::Audio audio(1);
		if(funasr::is_target_file(sz_filename, "wav")){
			int32_t sampling_rate_ = -1;
			if(!audio.LoadWav(sz_filename, &sampling_rate_))
				return nullptr;
		}else if(funasr::is_target_file(sz_filename, "pcm")){
			if (!audio.LoadPcmwav(sz_filename, &sampling_rate))
				return nullptr;
		}else{
			if (!audio.FfmpegLoad(sz_filename))
				return nullptr;
		}

		funasr::FUNASR_VAD_RESULT* p_result = new funasr::FUNASR_VAD_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		if(p_result->snippet_time == 0){
            return p_result;
        }
		
		vector<std::vector<int>> vad_segments;
		audio.Split(vad_obj, vad_segments, true);
		p_result->segments = new vector<std::vector<int>>(vad_segments);

		return p_result;
	}

	// APIs for PUNC Infer
	_FUNASRAPI FUNASR_RESULT CTTransformerInfer(FUNASR_HANDLE handle, const char* sz_sentence, FUNASR_MODE mode, QM_CALLBACK fn_callback, PUNC_TYPE type, FUNASR_RESULT pre_result)
	{
		funasr::PuncModel* punc_obj = (funasr::PuncModel*)handle;
		if (!punc_obj)
			return nullptr;
		
		FUNASR_RESULT p_result = nullptr;
		if (type==PUNC_OFFLINE){
			p_result = (FUNASR_RESULT)new funasr::FUNASR_PUNC_RESULT;
			((funasr::FUNASR_PUNC_RESULT*)p_result)->msg = punc_obj->AddPunc(sz_sentence);
		}else if(type==PUNC_ONLINE){
			if (!pre_result)
				p_result = (FUNASR_RESULT)new funasr::FUNASR_PUNC_RESULT;
			else
				p_result = pre_result;
			((funasr::FUNASR_PUNC_RESULT*)p_result)->msg = punc_obj->AddPunc(sz_sentence, ((funasr::FUNASR_PUNC_RESULT*)p_result)->arr_cache);
		}else{
			LOG(ERROR) << "Wrong PUNC_TYPE";
			exit(-1);
		}

		return p_result;
	}

	// APIs for Offline-stream Infer
	_FUNASRAPI FUNASR_RESULT FunOfflineInferBuffer(FUNASR_HANDLE handle, const char* sz_buf, int n_len, FUNASR_MODE mode, QM_CALLBACK fn_callback, const std::vector<std::vector<float>> &hw_emb, int sampling_rate, std::string wav_format)
	{
		funasr::OfflineStream* offline_stream = (funasr::OfflineStream*)handle;
		if (!offline_stream)
			return nullptr;

		funasr::Audio audio(1);
		if(wav_format == "pcm" || wav_format == "PCM"){
			if (!audio.LoadPcmwav(sz_buf, n_len, &sampling_rate))
				return nullptr;
		}else{
			if (!audio.FfmpegLoad(sz_buf, n_len))
				return nullptr;
		}

		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		if(p_result->snippet_time == 0){
            return p_result;
        }
		if(offline_stream->UseVad()){
			audio.Split(offline_stream);
		}

		float* buff;
		int len;
		int flag = 0;

		int n_step = 0;
		int n_total = audio.GetQueueSize();
		while (audio.Fetch(buff, len, flag) > 0) {
      //LOG(INFO) << "InferBuffer len " << len;
			string msg = (offline_stream->asr_handle)->Forward(buff, len, flag, hw_emb);
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

	_FUNASRAPI FUNASR_RESULT FunOfflineInfer(FUNASR_HANDLE handle, const char* sz_filename, FUNASR_MODE mode, QM_CALLBACK fn_callback, const std::vector<std::vector<float>> &hw_emb, int sampling_rate)
	{
		funasr::OfflineStream* offline_stream = (funasr::OfflineStream*)handle;
		if (!offline_stream)
			return nullptr;
		
		funasr::Audio audio(1);
		if(funasr::is_target_file(sz_filename, "wav")){
			int32_t sampling_rate_ = -1;
			if(!audio.LoadWav(sz_filename, &sampling_rate_))
				return nullptr;
		}else if(funasr::is_target_file(sz_filename, "pcm")){
			if (!audio.LoadPcmwav(sz_filename, &sampling_rate))
				return nullptr;
		}else{
			if (!audio.FfmpegLoad(sz_filename))
				return nullptr;
		}
		funasr::FUNASR_RECOG_RESULT* p_result = new funasr::FUNASR_RECOG_RESULT;
		p_result->snippet_time = audio.GetTimeLen();
		if(p_result->snippet_time == 0){
            return p_result;
        }
		if(offline_stream->UseVad()){
			audio.Split(offline_stream);
		}

		float* buff;
		int len;
		int flag = 0;
		int n_step = 0;
		int n_total = audio.GetQueueSize();
    //std::vector<std::vector<float>> emb;
		while (audio.Fetch(buff, len, flag) > 0) {
			string msg = (offline_stream->asr_handle)->Forward(buff, len, flag, hw_emb);
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

	_FUNASRAPI const std::vector<std::vector<float>> CompileHotwordEmbedding(FUNASR_HANDLE handle, std::string &hotwords) {
		funasr::OfflineStream* offline_stream = (funasr::OfflineStream*)handle;
    std::vector<std::vector<float>> emb;
		if (!offline_stream)
			return emb;
		return (offline_stream->asr_handle)->CompileHotwordEmbedding(hotwords);

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

	_FUNASRAPI const char* CTTransformerGetResult(FUNASR_RESULT result,int n_index)
	{
		funasr::FUNASR_PUNC_RESULT * p_result = (funasr::FUNASR_PUNC_RESULT*)result;
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

	_FUNASRAPI void CTTransformerFreeResult(FUNASR_RESULT result)
	{
		if (result)
		{
			delete (funasr::FUNASR_PUNC_RESULT*)result;
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

	_FUNASRAPI void CTTransformerUninit(FUNASR_HANDLE handle)
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

