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
#include <vector>
#include <glog/logging.h>
#include "funasrruntime.h"
#include "tclap/CmdLine.h"
#include "com-define.h"
#include "audio.h"

using namespace std;

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
    if (value_arg.isSet()){
        model_path.insert({key, value_arg.getValue()});
        LOG(INFO)<< key << " : " << value_arg.getValue();
    }
}

void print_segs(vector<vector<int>>* vec, string &wav_id) {
    if((*vec).size() == 0){
        return;
    }    
    string seg_out=wav_id + ": [";
    for (int i = 0; i < vec->size(); i++) {
        vector<int> inner_vec = (*vec)[i];
        if(inner_vec.size() == 0){
            continue;
        }
        seg_out += "[";
        for (int j = 0; j < inner_vec.size(); j++) {
            seg_out += to_string(inner_vec[j]);
            if (j != inner_vec.size() - 1) {
                seg_out += ",";
            }
        }
        seg_out += "]";
        if (i != vec->size() - 1) {
            seg_out += ",";
        }
    }
    seg_out += "]";
    LOG(INFO)<<seg_out;
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-offline-vad", ' ', "1.0");
    TCLAP::ValueArg<std::string>    model_dir("", MODEL_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", true, "", "string");
    TCLAP::ValueArg<std::string>    quantize("", QUANTIZE, "false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "false", "string");

    TCLAP::ValueArg<std::string>    wav_path("", WAV_PATH, "the input could be: wav_path, e.g.: asr_example.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)", true, "", "string");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(wav_path);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(wav_path, WAV_PATH, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    int thread_num = 1;
    FUNASR_HANDLE asr_hanlde=FunASRInit(model_path, thread_num, 1);

    if (!asr_hanlde)
    {
        LOG(ERROR) << "FunVad init failed";
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
    // init online features
    float snippet_time = 0.0f;
    long taking_micros = 0;
    for (int i = 0; i < wav_list.size(); i++) {
        auto& wav_file = wav_list[i];
        auto& wav_id = wav_ids[i];

        int32_t sampling_rate_ = -1;
        funasr::Audio audio(1);
		if(is_target_file(wav_file.c_str(), "wav")){
			int32_t sampling_rate_ = -1;
			if(!audio.LoadWav2Char(wav_file.c_str(), &sampling_rate_)){
				LOG(ERROR)<<"Failed to load "<< wav_file;
                exit(-1);
            }
		}else if(is_target_file(wav_file.c_str(), "pcm")){
			if (!audio.LoadPcmwav2Char(wav_file.c_str(), &sampling_rate_)){
				LOG(ERROR)<<"Failed to load "<< wav_file;
                exit(-1);
            }
		}else{
			LOG(ERROR)<<"Wrong wav extension";
			exit(-1);
		}
        char* speech_buff = audio.GetSpeechChar();
        int buff_len = audio.GetSpeechLen()*2;

        int step = 9600*2;
        bool is_final = false;

        for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
            if (sample_offset + step >= buff_len - 1) {
                    step = buff_len - sample_offset;
                    is_final = true;
                } else {
                    is_final = false;
            }
            gettimeofday(&start, NULL);
            FUNASR_RESULT result = FunASRInferBuffer(asr_hanlde, speech_buff+sample_offset, step, RASR_NONE, NULL, is_final, 16000);
            gettimeofday(&end, NULL);
            seconds = (end.tv_sec - start.tv_sec);
            taking_micros += ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

            if (result)
            {
                string msg = FunASRGetResult(result, 0);
                LOG(INFO)<< wav_id <<" : "<<msg;
                snippet_time += FunASRGetRetSnippetTime(result);
                FunASRFreeResult(result);
            }
            else
            {
                LOG(ERROR) << ("No return data!\n");
            }
        }
    }

    LOG(INFO) << "Audio length: " << (double)snippet_time << " s";
    LOG(INFO) << "Model inference takes: " << (double)taking_micros / 1000000 <<" s";
    LOG(INFO) << "Model inference RTF: " << (double)taking_micros/ (snippet_time*1000000);
    FsmnVadUninit(asr_hanlde);
    return 0;
}

