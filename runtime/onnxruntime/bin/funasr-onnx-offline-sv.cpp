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
#include "cam-sv-model.h"
#include "audio.h"
#include <vector>
#include <cmath>

using namespace std;

void GetValue(TCLAP::ValueArg<std::string> &value_arg, string key, std::map<std::string, std::string> &model_path)
{

    model_path.insert({key, value_arg.getValue()});
    LOG(INFO)<< key << " : " << value_arg.getValue();
}
float CosineSimilarity(const std::vector<float> &emb1,
                       const std::vector<float> &emb2)
{
    CHECK_EQ(emb1.size(), emb2.size());
    float dot = 0.f;
    float emb1_sum = 0.f;
    float emb2_sum = 0.f;
    for (size_t i = 0; i < emb1.size(); i++)
    {
        dot += emb1[i] * emb2[i];
        emb1_sum += emb1[i] * emb1[i];
        emb2_sum += emb2[i] * emb2[i];
    }
    dot /= std::max(std::sqrt(emb1_sum) * std::sqrt(emb2_sum),
                    std::numeric_limits<float>::epsilon());
    return dot;
}

bool is_target_file(const std::string &filename, const std::string target)
{
    std::size_t pos = filename.find_last_of(".");
    if (pos == std::string::npos)
    {
        return false;
    }
    std::string extension = filename.substr(pos + 1);
    return (extension == target);
}

std::vector<float> readwavfile(const std::string &wav_file)
{
    funasr::Audio audio(1);
    int32_t sampling_rate = 16000;
    std::string wav_format = "pcm";

    if (is_target_file(wav_file.c_str(), "wav"))
    {
        int32_t sampling_rate = -1;
        if (!audio.LoadWav(wav_file.c_str(), &sampling_rate))
        {
            printf("audio.LoadWav failed!\n");
            std::vector<float> data;
            return data;
        }
    }
    int len;
    float *buff;
    int flag;
    if (audio.Fetch(buff, len, flag) != 1)
    {
        printf("audio.Fetch\n");
    }

    std::vector<float> data(buff, buff + len);
    return data;
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    TCLAP::CmdLine cmd("funasr-onnx-offline-sv", ' ', "1.0");
    // TCLAP::ValueArg<std::string> model_dir("", SV_DIR, "the cam model path, which contains model.onnx, cam.yaml", true, "", "string");
    // TCLAP::ValueArg<std::string> sv_quant("", SV_QUANT, "false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "true", "string");
    // TCLAP::ValueArg<std::string> wav_file1("", "wav_file1", "the input could be: wav_path, e.g.: asr_example1.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, ", true, "", "string");
    // TCLAP::ValueArg<std::string> wav_file2("", "wav_file2", "the input could be: wav_path, e.g.: asr_example2.wav; pcm_path, e.g.: asr_example.pcm; wav.scp,", true, "", "string");
    // TCLAP::ValueArg<std::int32_t> onnx_thread("", "model-thread-num", "onnxruntime SetIntraOpNumThreads", false, 1, "int32_t");

    TCLAP::ValueArg<std::string> model_dir("", SV_DIR, "the cam model path, which contains model.onnx, cam.yaml", false, "/workspace/models/weights2/camplus_sv_zh-cn-16k-common-onnx", "string");
    TCLAP::ValueArg<std::string> sv_quant("", SV_QUANT, "false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "true", "string");
    TCLAP::ValueArg<std::string> wav_file1("", "wav_file1", "the input could be: wav_path, e.g.: asr_example1.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, ", false, "/home/wzp/project/FunASR/speaker1_a_cn_16k.wav", "string");
    TCLAP::ValueArg<std::string> wav_file2("", "wav_file2", "the input could be: wav_path, e.g.: asr_example2.wav; pcm_path, e.g.: asr_example.pcm; wav.scp,", false, "/home/wzp/project/FunASR/speaker1_a_cn_16k.wav", "string");
    TCLAP::ValueArg<std::int32_t> onnx_thread("", "model-thread-num", "onnxruntime SetIntraOpNumThreads", false, 1, "int32_t");

    cmd.add(model_dir);
    cmd.add(sv_quant);
    cmd.add(wav_file1);
    cmd.add(wav_file2);
    cmd.add(onnx_thread);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, SV_DIR, model_path);
    GetValue(sv_quant, SV_QUANT, model_path);

    std::string audio_file1 = wav_file1.getValue();
    std::string audio_file2 = wav_file2.getValue();
    
    std::vector<float> data1 = readwavfile(audio_file1);
    std::vector<float> data2 = readwavfile(audio_file2);
    int thread_num = onnx_thread.getValue();

    FUNASR_HANDLE sv_hanlde = CamPPlusSvInit(model_path, thread_num);
    std::vector<std::vector<float>> result1 = CamPPlusSvInfer(sv_hanlde, data1);
    std::vector<std::vector<float>> result2 = CamPPlusSvInfer(sv_hanlde, data2);
    float sim = CosineSimilarity(result1[0], result2[0]);
    for(int i=0;i<10;i++)
    {
        printf("%f\n",result1[0][i]);
    }
    printf("声纹相似度=%f\n", sim);
    CamPPlusSvUninit(sv_hanlde);
    return 0;
}
