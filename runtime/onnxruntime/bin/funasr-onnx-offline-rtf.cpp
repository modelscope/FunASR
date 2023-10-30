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
#include <unordered_map>
#include "util/text-utils.h"
using namespace std;

std::atomic<int> wav_index(0);
std::mutex mtx;
void ExtractHws(string hws_file, unordered_map<string, int> &hws_map)
{
    std::string line;
    std::ifstream ifs_hws(hws_file.c_str());
    if(!ifs_hws.is_open()){
        LOG(ERROR) << "Failed to open file: " << hws_file ;
        return;
    }
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

void runReg(FUNASR_HANDLE asr_handle, vector<string> wav_list, vector<string> wav_ids,
            float* total_length, long* total_time, int core_id, string nn_hotwords_, int fst_inc_wts = 20, string fst_hotword_path = "") {
    
    struct timeval start, end;
    long seconds = 0;
    float n_total_length = 0.0f;
    long n_total_time = 0;
	
    // init wfst decoder
    FUNASR_DEC_HANDLE decoder_handle = FunASRWfstDecoderInit(asr_handle, ASR_OFFLINE);

    // process fst hotwords list
    unordered_map<string, int> hws_map;
    ExtractHws(fst_hotword_path, hws_map);

    // load hotwords list and build graph
    FunWfstDecoderLoadHwsRes(decoder_handle, fst_inc_wts, hws_map);

    std::vector<std::vector<float>> hotwords_embedding = CompileHotwordEmbedding(asr_handle, nn_hotwords_);
    
    // warm up
    for (size_t i = 0; i < 1; i++)
    {
        FunOfflineReset(asr_handle, decoder_handle);
        FUNASR_RESULT result=FunOfflineInfer(asr_handle, wav_list[0].c_str(), RASR_NONE, NULL, hotwords_embedding, 16000, false, decoder_handle);
        if(result){
            FunASRFreeResult(result);
        }
    }

    while (true) {
        // 使用原子变量获取索引并递增
        int i = wav_index.fetch_add(1);
        if (i >= wav_list.size()) {
            break;
        }

        gettimeofday(&start, NULL);
        FUNASR_RESULT result=FunOfflineInfer(asr_handle, wav_list[i].c_str(), RASR_NONE, NULL, hotwords_embedding, 16000, false, decoder_handle);

        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        n_total_time += taking_micros;

        if(result){
            string msg = FunASRGetResult(result, 0);
            LOG(INFO) << "Thread: " << this_thread::get_id() << "," << wav_ids[i] << " : " << msg;
            string stamp = FunASRGetStamp(result);
            if(stamp !=""){
                LOG(INFO) << "Thread: " << this_thread::get_id() << "," << wav_ids[i] << " : " << stamp;
            }
            float snippet_time = FunASRGetRetSnippetTime(result);
            n_total_length += snippet_time;
            FunASRFreeResult(result);
        }else{
            LOG(ERROR) << wav_ids[i] << (": No return data!\n");
        }
    }
    {
        lock_guard<mutex> guard(mtx);
        *total_length += n_total_length;
        if(*total_time < n_total_time){
            *total_time = n_total_time;
        }
    }
    FunWfstDecoderUnloadHwsRes(decoder_handle);
    FunASRWfstDecoderUninit(decoder_handle);
}

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

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-offline-rtf", ' ', "1.0");
    TCLAP::ValueArg<std::string>    model_dir("", MODEL_DIR, "the model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
    TCLAP::ValueArg<std::string>    quantize("", QUANTIZE, "true (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    vad_dir("", VAD_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
    TCLAP::ValueArg<std::string>    vad_quant("", VAD_QUANT, "true (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    punc_dir("", PUNC_DIR, "the punc model path, which contains model.onnx, punc.yaml", false, "", "string");
    TCLAP::ValueArg<std::string>    punc_quant("", PUNC_QUANT, "true (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "true", "string");
    TCLAP::ValueArg<std::string>    lm_dir("", LM_DIR, "the lm model path, which contains compiled models: TLG.fst, config.yaml ", false, "", "string");
    TCLAP::ValueArg<std::string>    fst_hotword("", FST_HOTWORD, "the fst hotwords file, one hotword perline, Format: Hotword [tab] Weight (could be: 阿里巴巴 \t 20)", false, "", "string");
    TCLAP::ValueArg<std::int32_t>   fst_inc_wts("", FST_INC_WTS, "the fst hotwords incremental bias", false, 20, "int32_t");
    TCLAP::ValueArg<std::string>    itn_dir("", ITN_DIR, "the itn model(fst) path, which contains zh_itn_tagger.fst and zh_itn_verbalizer.fst", false, "", "string");

    TCLAP::ValueArg<std::string> wav_path("", WAV_PATH, "the input could be: wav_path, e.g.: asr_example.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)", true, "", "string");
    TCLAP::ValueArg<std::int32_t> thread_num("", THREAD_NUM, "multi-thread num for rtf", true, 0, "int32_t");
    TCLAP::ValueArg<std::string> nn_hotword("", NN_HOTWORD,
        "the nn hotwords file, one hotword perline, Format: Hotword (could be: 阿里巴巴)", false, "", "string");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_quant);
    cmd.add(itn_dir);
    cmd.add(lm_dir);
    cmd.add(fst_hotword);
    cmd.add(fst_inc_wts);
    cmd.add(wav_path);
    cmd.add(thread_num);
    cmd.add(nn_hotword);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);
    GetValue(itn_dir, ITN_DIR, model_path);
    GetValue(lm_dir, LM_DIR, model_path);
    GetValue(fst_hotword, FST_HOTWORD, model_path);
    GetValue(wav_path, WAV_PATH, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    FUNASR_HANDLE asr_handle=FunOfflineInit(model_path, 1);

    if (!asr_handle)
    {
        LOG(ERROR) << "FunASR init failed";
        exit(-1);
    }

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "Model initialization takes " << (double)modle_init_micros / 1000000 << " s";

    // nn hotword file
    std::string nn_hotwords_;
    std::string file_nn_hotword = nn_hotword.getValue();
    std::string line;
    std::ifstream file(file_nn_hotword);
    LOG(INFO) << "nn hotword path: " << file_nn_hotword;

    if (file.is_open()) {
        while (getline(file, line)) {
            nn_hotwords_ += line+HOTWORD_SEP;
        }
        LOG(INFO) << "nn hotwords: " << nn_hotwords_;
        file.close();
    } else {
        LOG(ERROR) << "Unable to open nn hotwords file: " << file_nn_hotword 
            << ". If you have not set nn hotwords, please ignore this message.";
    }

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
    string fst_hotword_path;
    int value_bias = 20;
    if (fst_hotword.isSet()) {
      fst_hotword_path = model_path.at(FST_HOTWORD);
      value_bias = fst_inc_wts.getValue();
    }
    for (int i = 0; i < rtf_threds; i++)
    {
        threads.emplace_back(thread(runReg, asr_handle, wav_list, wav_ids, &total_length, &total_time, i, nn_hotwords_, value_bias, fst_hotword_path));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    LOG(INFO) << "total_time_wav " << (long)(total_length * 1000) << " ms";
    LOG(INFO) << "total_time_comput " << total_time / 1000 << " ms";
    LOG(INFO) << "total_rtf " << (double)total_time/ (total_length*1000000);
    LOG(INFO) << "speedup " << 1.0/((double)total_time/ (total_length*1000000));

    FunOfflineUninit(asr_handle);
    return 0;
}
