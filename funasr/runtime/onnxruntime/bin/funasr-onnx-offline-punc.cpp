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
#include "funasrruntime.h"
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

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-offline-punc", ' ', "1.0");
    TCLAP::ValueArg<std::string>    model_dir("", MODEL_DIR, "the punc model path, which contains model.onnx, punc.yaml", true, "", "string");
    TCLAP::ValueArg<std::string>    quantize("", QUANTIZE, "false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "false", "string");
    TCLAP::ValueArg<std::string> txt_path("", TXT_PATH, "txt file path, one sentence per line", true, "", "string");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(txt_path);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(txt_path, TXT_PATH, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    int thread_num = 1;
    FUNASR_HANDLE punc_hanlde=CTTransformerInit(model_path, thread_num);

    if (!punc_hanlde)
    {
        LOG(ERROR) << "FunASR init failed";
        exit(-1);
    }

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "Model initialization takes " << (double)modle_init_micros / 1000000 << " s";

    // read txt_path
    vector<string> txt_list;

    if(model_path.find(TXT_PATH)!=model_path.end()){
        ifstream in(model_path.at(TXT_PATH));
        if (!in.is_open()) {
            LOG(ERROR) << "Failed to open file: " << model_path.at(TXT_PATH) ;
            return 0;
        }
        string line;
        while(getline(in, line))
        {
            txt_list.emplace_back(line); 
        }
        in.close();
    }
    
    long taking_micros = 0;
    for(auto& txt_str : txt_list){
        gettimeofday(&start, NULL);
        FUNASR_RESULT result=CTTransformerInfer(punc_hanlde, txt_str.c_str(), RASR_NONE, NULL);
        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        taking_micros += ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        string msg = FunASRGetResult(result, 0);
        LOG(INFO)<<"Results: "<<msg;
        CTTransformerFreeResult(result);
    }

    LOG(INFO) << "Model inference takes: " << (double)taking_micros / 1000000 <<" s";
    CTTransformerUninit(punc_hanlde);
    return 0;
}

