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

void print_segs(vector<vector<int>>* vec) {
    string seg_out="[";
    for (int i = 0; i < vec->size(); i++) {
        vector<int> inner_vec = (*vec)[i];
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

    TCLAP::ValueArg<std::string> wav_path("", WAV_PATH, "wave file path", false, "", "string");
    TCLAP::ValueArg<std::string> wav_scp("", WAV_SCP, "wave scp path", false, "", "string");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(wav_path);
    cmd.add(wav_scp);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(wav_path, WAV_PATH, model_path);
    GetValue(wav_scp, WAV_SCP, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    int thread_num = 1;
    FUNASR_HANDLE vad_hanlde=FunVadInit(model_path, thread_num);

    if (!vad_hanlde)
    {
        LOG(ERROR) << "FunVad init failed";
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
        FUNASR_RESULT result=FunVadWavFile(vad_hanlde, wav_file.c_str(), RASR_NONE, NULL);
        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        taking_micros += ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        if (result)
        {
            vector<std::vector<int>>* vad_segments = FunVadGetResult(result, 0);
            print_segs(vad_segments);
            snippet_time += FunVadGetRetSnippetTime(result);
            FunVadFreeResult(result);
        }
        else
        {
            LOG(ERROR) << ("No return data!\n");
        }
    }
 
    LOG(INFO) << "Audio length: " << (double)snippet_time << " s";
    LOG(INFO) << "Model inference takes: " << (double)taking_micros / 1000000 <<" s";
    LOG(INFO) << "Model inference RTF: " << (double)taking_micros/ (snippet_time*1000000);
    FunVadUninit(vad_hanlde);
    return 0;
}

