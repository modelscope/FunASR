/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <thread>
#include <glog/logging.h>
#include "funasrruntime.h"
#include "tclap/CmdLine.h"
#include "com-define.h"
#include "audio.h"

using namespace std;

std::atomic<int> wav_index(0);
std::mutex mtx;

bool is_target_file(const std::string& filename, const std::string target) {
    std::size_t pos = filename.find_last_of(".");
    if (pos == std::string::npos) {
        return false;
    }
    std::string extension = filename.substr(pos + 1);
    return (extension == target);
}

void GetValue(TCLAP::ValueArg<std::string>& value_arg, string key, std::map<std::string, std::string>& model_path)
{
    model_path.insert({key, value_arg.getValue()});
    LOG(INFO)<< key << " : " << value_arg.getValue();
}


void runReg(FUNASR_HANDLE tpass_handle, std::vector<int> chunk_size, vector<string> wav_list, vector<string> wav_ids,
            float* total_length, long* total_time, int core_id, ASR_TYPE asr_mode_) {
    
    struct timeval start, end;
    long seconds = 0;
    float n_total_length = 0.0f;
    long n_total_time = 0;
    
    // init online features
    FUNASR_HANDLE tpass_online_handle=FunTpassOnlineInit(tpass_handle, chunk_size);

    // warm up
    for (size_t i = 0; i < 2; i++)
    {
        int32_t sampling_rate_ = 16000;
        funasr::Audio audio(1);
		if(is_target_file(wav_list[0].c_str(), "wav")){
			if(!audio.LoadWav2Char(wav_list[0].c_str(), &sampling_rate_)){
				LOG(ERROR)<<"Failed to load "<< wav_list[0];
                exit(-1);
            }
		}else if(is_target_file(wav_list[0].c_str(), "pcm")){
			if (!audio.LoadPcmwav2Char(wav_list[0].c_str(), &sampling_rate_)){
				LOG(ERROR)<<"Failed to load "<< wav_list[0];
                exit(-1);
            }
		}else{
			if (!audio.FfmpegLoad(wav_list[0].c_str(), true)){
				LOG(ERROR)<<"Failed to load "<< wav_list[0];
                exit(-1);
            }
		}
        char* speech_buff = audio.GetSpeechChar();
        int buff_len = audio.GetSpeechLen()*2;

        int step = 1600*2;
        bool is_final = false;

        std::vector<std::vector<string>> punc_cache(2);
        for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
            if (sample_offset + step >= buff_len - 1) {
                    step = buff_len - sample_offset;
                    is_final = true;
                } else {
                    is_final = false;
            }
            FUNASR_RESULT result = FunTpassInferBuffer(tpass_handle, tpass_online_handle, speech_buff+sample_offset, step, punc_cache, is_final, sampling_rate_, "pcm", (ASR_TYPE)asr_mode_);
            if (result)
            {
                FunASRFreeResult(result);
            }
        }
    }

    while (true) {
        // 使用原子变量获取索引并递增
        int i = wav_index.fetch_add(1);
        if (i >= wav_list.size()) {
            break;
        }
        int32_t sampling_rate_ = 16000;
        funasr::Audio audio(1);
		if(is_target_file(wav_list[i].c_str(), "wav")){
			if(!audio.LoadWav2Char(wav_list[i].c_str(), &sampling_rate_)){
				LOG(ERROR)<<"Failed to load "<< wav_list[i];
                exit(-1);
            }
		}else if(is_target_file(wav_list[i].c_str(), "pcm")){
			if (!audio.LoadPcmwav2Char(wav_list[i].c_str(), &sampling_rate_)){
				LOG(ERROR)<<"Failed to load "<< wav_list[i];
                exit(-1);
            }
		}else{
			if (!audio.FfmpegLoad(wav_list[i].c_str(), true)){
				LOG(ERROR)<<"Failed to load "<< wav_list[i];
                exit(-1);
            }
		}
        char* speech_buff = audio.GetSpeechChar();
        int buff_len = audio.GetSpeechLen()*2;

        int step = 1600*2;
        bool is_final = false;

        string online_res="";
        string tpass_res="";
        std::vector<std::vector<string>> punc_cache(2);
        for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
            if (sample_offset + step >= buff_len - 1) {
                    step = buff_len - sample_offset;
                    is_final = true;
                } else {
                    is_final = false;
            }
            gettimeofday(&start, NULL);
            FUNASR_RESULT result = FunTpassInferBuffer(tpass_handle, tpass_online_handle, speech_buff+sample_offset, step, punc_cache, is_final, sampling_rate_, "pcm", (ASR_TYPE)asr_mode_);
            gettimeofday(&end, NULL);
            seconds = (end.tv_sec - start.tv_sec);
            long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
            n_total_time += taking_micros;

            if (result)
            {
                string online_msg = FunASRGetResult(result, 0);
                online_res += online_msg;
                if(online_msg != ""){
                    LOG(INFO)<< wav_ids[i] <<" : "<<online_msg;
                }
                string tpass_msg = FunASRGetTpassResult(result, 0);
                tpass_res += tpass_msg;
                if(tpass_msg != ""){
                    LOG(INFO)<< wav_ids[i] <<" offline results : "<<tpass_msg;
                }
                float snippet_time = FunASRGetRetSnippetTime(result);
                n_total_length += snippet_time;
                FunASRFreeResult(result);
            }
            else
            {
                LOG(ERROR) << ("No return data!\n");
            }
        }
        if(asr_mode_ == 2){
            LOG(INFO) <<"Thread: " << this_thread::get_id() <<" " << wav_ids[i] << " Final online  results "<<" : "<<online_res;
        }
        if(asr_mode_==1){
            LOG(INFO) <<"Thread: " << this_thread::get_id() <<" " << wav_ids[i] << " Final online  results "<<" : "<<tpass_res;
        }
        if(asr_mode_ == 0 || asr_mode_==2){
            LOG(INFO) <<"Thread: " << this_thread::get_id() <<" " << wav_ids[i] << " Final offline results " <<" : "<<tpass_res;
        }

    }
    {
        lock_guard<mutex> guard(mtx);
        *total_length += n_total_length;
        if(*total_time < n_total_time){
            *total_time = n_total_time;
        }
    }
    FunTpassOnlineUninit(tpass_online_handle);
}


int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-2pass", ' ', "1.0");
    TCLAP::ValueArg<std::string>    offline_model_dir("", OFFLINE_MODEL_DIR, "the asr offline model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
    TCLAP::ValueArg<std::string>    online_model_dir("", ONLINE_MODEL_DIR, "the asr online model path, which contains model.onnx, decoder.onnx, config.yaml, am.mvn", true, "", "string");
    TCLAP::ValueArg<std::string>    quantize("", QUANTIZE, "true (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    vad_dir("", VAD_DIR, "the vad online model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
    TCLAP::ValueArg<std::string>    vad_quant("", VAD_QUANT, "true (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    punc_dir("", PUNC_DIR, "the punc online model path, which contains model.onnx, punc.yaml", false, "", "string");
    TCLAP::ValueArg<std::string>    punc_quant("", PUNC_QUANT, "true (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    asr_mode("", ASR_MODE, "offline, online, 2pass", false, "2pass", "string");
    TCLAP::ValueArg<std::int32_t>   onnx_thread("", "onnx-inter-thread", "onnxruntime SetIntraOpNumThreads", false, 1, "int32_t");
    TCLAP::ValueArg<std::int32_t>   thread_num_("", THREAD_NUM, "multi-thread num for rtf", false, 1, "int32_t");

    TCLAP::ValueArg<std::string> wav_path("", WAV_PATH, "the input could be: wav_path, e.g.: asr_example.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)", true, "", "string");

    cmd.add(offline_model_dir);
    cmd.add(online_model_dir);
    cmd.add(quantize);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_quant);
    cmd.add(wav_path);
    cmd.add(asr_mode);
    cmd.add(onnx_thread);
    cmd.add(thread_num_);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(offline_model_dir, OFFLINE_MODEL_DIR, model_path);
    GetValue(online_model_dir, ONLINE_MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);
    GetValue(wav_path, WAV_PATH, model_path);
    GetValue(asr_mode, ASR_MODE, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    int thread_num = onnx_thread.getValue();
    int asr_mode_ = -1;
    if(model_path[ASR_MODE] == "offline"){
        asr_mode_ = 0;
    }else if(model_path[ASR_MODE] == "online"){
        asr_mode_ = 1;
    }else if(model_path[ASR_MODE] == "2pass"){
        asr_mode_ = 2;
    }else{
        LOG(ERROR) << "Wrong asr-mode : " << model_path[ASR_MODE];
        exit(-1);
    }
    FUNASR_HANDLE tpass_hanlde=FunTpassInit(model_path, thread_num);

    if (!tpass_hanlde)
    {
        LOG(ERROR) << "FunTpassInit init failed";
        exit(-1);
    }

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "Model initialization takes " << (double)modle_init_micros / 1000000 << " s";

    // read wav_path
    vector<string> wav_list;
    vector<string> wav_ids;
    string default_id = "wav_default_id";
    string wav_path_ = model_path.at(WAV_PATH);

    if(is_target_file(wav_path_, "scp")){
        ifstream in(wav_path_);
        if (!in.is_open()) {
            LOG(ERROR) << "Failed to open file: " << model_path.at(WAV_SCP) ;
            return 0;
        }
        string line;
        while(getline(in, line))
        {
            istringstream iss(line);
            string column1, column2;
            iss >> column1 >> column2;
            wav_list.emplace_back(column2);
            wav_ids.emplace_back(column1);
        }
        in.close();
    }else{
        wav_list.emplace_back(wav_path_);
        wav_ids.emplace_back(default_id);
    }

    std::vector<int> chunk_size = {5,10,5};
    // 多线程测试
    float total_length = 0.0f;
    long total_time = 0;
    std::vector<std::thread> threads;

    int rtf_threds = thread_num_.getValue();
    for (int i = 0; i < rtf_threds; i++)
    {
        threads.emplace_back(thread(runReg, tpass_hanlde, chunk_size, wav_list, wav_ids, &total_length, &total_time, i, (ASR_TYPE)asr_mode_));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    LOG(INFO) << "total_time_wav " << (long)(total_length * 1000) << " ms";
    LOG(INFO) << "total_time_comput " << total_time / 1000 << " ms";
    LOG(INFO) << "total_rtf " << (double)total_time/ (total_length*1000000);
    LOG(INFO) << "speedup " << 1.0/((double)total_time/ (total_length*1000000));

    FunTpassUninit(tpass_hanlde);
    return 0;
}

