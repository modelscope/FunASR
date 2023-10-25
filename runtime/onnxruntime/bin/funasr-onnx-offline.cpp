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
#include <unordered_map>
#include "util/text-utils.h"
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

void ExtractHws(string hws_file, unordered_map<string, int> &hws_map)
{
    std::string line;
    std::ifstream ifs_hws(hws_file.c_str());
    while (getline(ifs_hws, line)) {
      kaldi::Trim(&line);
      if (line.empty()) {
        continue;
      }
      float score = 1.0f;
      std::vector<std::string> text;
      kaldi::SplitStringToVector(line, "\t", true, &text);
      if (text.size() > 1) {
        score = std::stof(text[1]);
      } else if (text.empty()) {
        continue;
      }
      hws_map.emplace(text[0], score);
    }
    ifs_hws.close();
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-offline", ' ', "1.0");
    TCLAP::ValueArg<std::string>    model_dir("", MODEL_DIR, "the asr model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
    TCLAP::ValueArg<std::string>    quantize("", QUANTIZE, "true (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    vad_dir("", VAD_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
    TCLAP::ValueArg<std::string>    vad_quant("", VAD_QUANT, "true (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    punc_dir("", PUNC_DIR, "the punc model path, which contains model.onnx, punc.yaml", false, "", "string");
    TCLAP::ValueArg<std::string>    punc_quant("", PUNC_QUANT, "true (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    fst_dir("", FST_DIR, "the fst resource path", false, "", "string");
    TCLAP::ValueArg<std::string>    hws_dir("", HWS_DIR, "the fst hotwords list", false, "", "string");
    TCLAP::ValueArg<std::int32_t>   hws_inc_bias("", HWS_INC_BIAS, "the fst hotwords incremental bias", false, 20, "int32_t");
    TCLAP::ValueArg<std::string>    itn_dir("", ITN_DIR, "the itn model(fst) path, which contains zh_itn_tagger.fst and zh_itn_verbalizer.fst", false, "", "string");

    TCLAP::ValueArg<std::string> wav_path("", WAV_PATH, "the input could be: wav_path, e.g.: asr_example.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)", true, "", "string");
    TCLAP::ValueArg<std::string> hotword("", HOTWORD, "*.txt(one hotword perline) or hotwords seperate by space (could be: 阿里巴巴 达摩院)", false, "", "string");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_quant);
    cmd.add(itn_dir);
    cmd.add(fst_dir);
    cmd.add(hws_dir);
    cmd.add(hws_inc_bias);
    cmd.add(wav_path);
    cmd.add(hotword);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);
    GetValue(itn_dir, ITN_DIR, model_path);
    GetValue(fst_dir, FST_DIR, model_path);
    GetValue(hws_dir, HWS_DIR, model_path);
    GetValue(wav_path, WAV_PATH, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    int thread_num = 1;
    FUNASR_HANDLE asr_hanlde=FunOfflineInit(model_path, thread_num);

    if (!asr_hanlde)
    {
        LOG(ERROR) << "FunASR init failed";
        exit(-1);
    }

    // init wfst decoder
    FUNASR_DEC_HANDLE decoder_handle = FunASRWfstDecoderInit(asr_hanlde, ASR_OFFLINE);

    // process fst hotwords list
    unordered_map<string, int> hws_map;
    if (hws_dir.isSet()) {
      ExtractHws(model_path.at(HWS_DIR), hws_map);
    }

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "Model initialization takes " << (double)modle_init_micros / 1000000 << " s";

    // read hotwords
    std::string hotword_ = hotword.getValue();
    std::string hotwords_;

    if(is_target_file(hotword_, "txt")){
        ifstream in(hotword_);
        if (!in.is_open()) {
            LOG(ERROR) << "Failed to open file: " << model_path.at(HOTWORD) ;
            return 0;
        }
        string line;
        while(getline(in, line))
        {
            hotwords_ +=line+HOTWORD_SEP;
        }
        in.close();
    }else{
        hotwords_ = hotword_;
    }

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
    
    float snippet_time = 0.0f;
    long taking_micros = 0;

    // load hotwords list and build graph
    FunWfstDecoderLoadHwsRes(decoder_handle, hws_inc_bias.getValue(), hws_map);
	
    std::vector<std::vector<float>> hotwords_embedding = CompileHotwordEmbedding(asr_hanlde, hotwords_);
    for (int i = 0; i < wav_list.size(); i++) {
        auto& wav_file = wav_list[i];
        auto& wav_id = wav_ids[i];
        gettimeofday(&start, NULL);
        FunOfflineReset(asr_hanlde, decoder_handle);
        FUNASR_RESULT result=FunOfflineInfer(asr_hanlde, wav_file.c_str(), RASR_NONE, NULL, hotwords_embedding, 16000, false, decoder_handle);
        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        taking_micros += ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        if (result)
        {
            string msg = FunASRGetResult(result, 0);
            LOG(INFO)<< wav_id <<" : "<<msg;
            string stamp = FunASRGetStamp(result);
            if(stamp !=""){
                LOG(INFO)<< wav_id <<" : "<<stamp;
            }
            snippet_time += FunASRGetRetSnippetTime(result);
            FunASRFreeResult(result);
        }
        else
        {
            LOG(ERROR) << ("No return data!\n");
        }
    }
    FunWfstDecoderUnloadHwsRes(decoder_handle);
    LOG(INFO) << "Audio length: " << (double)snippet_time << " s";
    LOG(INFO) << "Model inference takes: " << (double)taking_micros / 1000000 <<" s";
    LOG(INFO) << "Model inference RTF: " << (double)taking_micros/ (snippet_time*1000000);
    
    FunASRWfstDecoderUninit(decoder_handle);
    FunOfflineUninit(asr_hanlde);
    return 0;
}

