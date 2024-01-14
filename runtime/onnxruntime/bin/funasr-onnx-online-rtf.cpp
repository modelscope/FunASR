/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include <glog/logging.h>
#include "funasrruntime.h"
#include "tclap/CmdLine.h"
#include "com-define.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <map>
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

void runReg(FUNASR_HANDLE asr_handle, vector<string> wav_list, vector<string> wav_ids, int audio_fs,
            float* total_length, long* total_time, int core_id) {
    
    struct timeval start, end;
    long seconds = 0;
    float n_total_length = 0.0f;
    long n_total_time = 0;
    
    // init online features
    FUNASR_HANDLE online_handle=FunASROnlineInit(asr_handle);

    // warm up
    for (size_t i = 0; i < 10; i++)
    {
        int32_t sampling_rate_ = audio_fs;
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

        int step = 9600*2;
        bool is_final = false;

        string final_res="";
        for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
            if (sample_offset + step >= buff_len - 1) {
                    step = buff_len - sample_offset;
                    is_final = true;
                } else {
                    is_final = false;
            }
            FUNASR_RESULT result = FunASRInferBuffer(online_handle, speech_buff+sample_offset, step, RASR_NONE, NULL, is_final, sampling_rate_);
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
        int32_t sampling_rate_ = audio_fs;
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

        int step = 9600*2;
        bool is_final = false;

        string final_res="";
        for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
            if (sample_offset + step >= buff_len - 1) {
                    step = buff_len - sample_offset;
                    is_final = true;
                } else {
                    is_final = false;
            }
            gettimeofday(&start, NULL);
            FUNASR_RESULT result = FunASRInferBuffer(online_handle, speech_buff+sample_offset, step, RASR_NONE, NULL, is_final, sampling_rate_);
            gettimeofday(&end, NULL);
            seconds = (end.tv_sec - start.tv_sec);
            long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
            n_total_time += taking_micros;

            if (result)
            {
                string msg = FunASRGetResult(result, 0);
                final_res += msg;
                LOG(INFO) << "Thread: " << this_thread::get_id() << "," << wav_ids[i] << " : " << msg;
                float snippet_time = FunASRGetRetSnippetTime(result);
                n_total_length += snippet_time;
                FunASRFreeResult(result);
            }
            else
            {
                LOG(ERROR) << ("No return data!\n");
            }
        }
        LOG(INFO) << "Thread: " << this_thread::get_id() << ", Final results " << wav_ids[i] << " : " << final_res;

    }
    {
        lock_guard<mutex> guard(mtx);
        *total_length += n_total_length;
        if(*total_time < n_total_time){
            *total_time = n_total_time;
        }
    }
    FunASRUninit(online_handle);
}

void GetValue(TCLAP::ValueArg<std::string>& value_arg, string key, std::map<std::string, std::string>& model_path)
{
    if (value_arg.isSet()){
        model_path.insert({key, value_arg.getValue()});
        LOG(INFO)<< key << " : " << value_arg.getValue();
    }
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-online-rtf", ' ', "1.0");
    TCLAP::ValueArg<std::string>    model_dir("", MODEL_DIR, "the model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
    TCLAP::ValueArg<std::string>    quantize("", QUANTIZE, "true (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    vad_dir("", VAD_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
    TCLAP::ValueArg<std::string>    vad_quant("", VAD_QUANT, "true (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    punc_dir("", PUNC_DIR, "the punc model path, which contains model.onnx, punc.yaml", false, "", "string");
    TCLAP::ValueArg<std::string>    punc_quant("", PUNC_QUANT, "true (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "true", "string");

    TCLAP::ValueArg<std::string> wav_path("", WAV_PATH, "the input could be: wav_path, e.g.: asr_example.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)", true, "", "string");
    TCLAP::ValueArg<std::int32_t>   audio_fs("", AUDIO_FS, "the sample rate of audio", false, 16000, "int32_t");
    TCLAP::ValueArg<std::int32_t> thread_num("", THREAD_NUM, "multi-thread num for rtf", true, 0, "int32_t");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_quant);
    cmd.add(wav_path);
    cmd.add(audio_fs);
    cmd.add(thread_num);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);
    GetValue(wav_path, WAV_PATH, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    FUNASR_HANDLE asr_handle=FunASRInit(model_path, 1, ASR_ONLINE);

    if (!asr_handle)
    {
        LOG(ERROR) << "FunASR init failed";
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
    if(is_target_file(wav_path_, "wav") || is_target_file(wav_path_, "pcm")){
        wav_list.emplace_back(wav_path_);
        wav_ids.emplace_back(default_id);
    }
    else if(is_target_file(wav_path_, "scp")){
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
        LOG(ERROR)<<"Please check the wav extension!";
        exit(-1);
    }

    // 多线程测试
    float total_length = 0.0f;
    long total_time = 0;
    std::vector<std::thread> threads;

    int rtf_threds = thread_num.getValue();
    for (int i = 0; i < rtf_threds; i++)
    {
        threads.emplace_back(thread(runReg, asr_handle, wav_list, wav_ids, audio_fs.getValue(), &total_length, &total_time, i));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    LOG(INFO) << "total_time_wav " << (long)(total_length * 1000) << " ms";
    LOG(INFO) << "total_time_comput " << total_time / 1000 << " ms";
    LOG(INFO) << "total_rtf " << (double)total_time/ (total_length*1000000);
    LOG(INFO) << "speedup " << 1.0/((double)total_time/ (total_length*1000000));

    FunASRUninit(asr_handle);
    return 0;
}
