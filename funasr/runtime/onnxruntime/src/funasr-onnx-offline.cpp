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
#include <glog/logging.h>
#include "libfunasrapi.h"
#include "tclap/CmdLine.h"
#include "com-define.h"

using namespace std;

void GetValue(TCLAP::ValueArg<std::string>& value_arg, string key, std::map<std::string, std::string>& model_path)
{
    if (value_arg.isSet()){
        model_path.insert({key, value_arg.getValue()});
        LOG(INFO)<< key << " : " << value_arg.getValue();
    }
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-offline", ' ', "1.0");
    TCLAP::ValueArg<std::string>    model_dir("", MODEL_DIR, "the asr model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
    TCLAP::ValueArg<std::string>    quantize("", QUANTIZE, "false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "false", "string");
    TCLAP::ValueArg<std::string>    vad_dir("", VAD_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
    TCLAP::ValueArg<std::string>    vad_quant("", VAD_QUANT, "false (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "false", "string");
    TCLAP::ValueArg<std::string>    punc_dir("", PUNC_DIR, "the punc model path, which contains model.onnx, punc.yaml", false, "", "string");
    TCLAP::ValueArg<std::string>    punc_quant("", PUNC_QUANT, "false (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "false", "string");

    TCLAP::ValueArg<std::string> wav_path("", WAV_PATH, "wave file path", false, "", "string");
    TCLAP::ValueArg<std::string> wav_scp("", WAV_SCP, "wave scp path", false, "", "string");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_quant);
    cmd.add(wav_path);
    cmd.add(wav_scp);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);
    GetValue(wav_path, WAV_PATH, model_path);
    GetValue(wav_scp, WAV_SCP, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    int thread_num = 1;
    FUNASR_HANDLE asr_hanlde=FunOfflineInit(model_path, thread_num);

    if (!asr_hanlde)
    {
        LOG(ERROR) << "FunASR init failed";
        exit(-1);
    }

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "Model initialization takes " << (double)modle_init_micros / 1000000 << " s";

    // read wav_path and wav_scp
    vector<string> wav_list;

    if(model_path.find(WAV_PATH)!=model_path.end()){
        wav_list.emplace_back(model_path.at(WAV_PATH));
    }
    if(model_path.find(WAV_SCP)!=model_path.end()){
        ifstream in(model_path.at(WAV_SCP));
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
        }
        in.close();
    }
    
    float snippet_time = 0.0f;
    long taking_micros = 0;
    for(auto& wav_file : wav_list){
        gettimeofday(&start, NULL);
        FUNASR_RESULT result=FunOfflineStream(asr_hanlde, wav_file.c_str(), RASR_NONE, NULL);
        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        taking_micros += ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        if (result)
        {
            string msg = FunASRGetResult(result, 0);
            LOG(INFO)<<"Result: "<<msg;
            snippet_time += FunASRGetRetSnippetTime(result);
            FunASRFreeResult(result);
        }
        else
        {
            LOG(ERROR) << ("No return data!\n");
        }
    }
 
    LOG(INFO) << "Audio length: " << (double)snippet_time << " s";
    LOG(INFO) << "Model inference takes: " << (double)taking_micros / 1000000 <<" s";
    LOG(INFO) << "Model inference RTF: " << (double)taking_micros/ (snippet_time*1000000);
    FunOfflineUninit(asr_hanlde);
    return 0;
}

