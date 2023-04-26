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

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-offline", ' ', "1.0");
    TCLAP::ValueArg<std::string> vad_model("", VAD_MODEL_PATH, "vad model path", false, "", "string");
    TCLAP::ValueArg<std::string> vad_cmvn("", VAD_CMVN_PATH, "vad cmvn path", false, "", "string");
    TCLAP::ValueArg<std::string> vad_config("", VAD_CONFIG_PATH, "vad config path", false, "", "string");

    TCLAP::ValueArg<std::string> am_model("", AM_MODEL_PATH, "am model path", true, "", "string");
    TCLAP::ValueArg<std::string> am_cmvn("", AM_CMVN_PATH, "am cmvn path", true, "", "string");
    TCLAP::ValueArg<std::string> am_config("", AM_CONFIG_PATH, "am config path", true, "", "string");

    TCLAP::ValueArg<std::string> punc_model("", PUNC_MODEL_PATH, "punc model path", false, "", "string");
    TCLAP::ValueArg<std::string> punc_config("", PUNC_CONFIG_PATH, "punc config path", false, "", "string");

    TCLAP::ValueArg<std::string> wav_path("", WAV_PATH, "wave file path", false, "", "string");
    TCLAP::ValueArg<std::string> wav_scp("", WAV_SCP, "wave scp path", false, "", "string");

    cmd.add(vad_model);
    cmd.add(vad_cmvn);
    cmd.add(vad_config);
    cmd.add(am_model);
    cmd.add(am_cmvn);
    cmd.add(am_config);
    cmd.add(punc_model);
    cmd.add(punc_config);
    cmd.add(wav_path);
    cmd.add(wav_scp);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(vad_model, VAD_MODEL_PATH, model_path);
    GetValue(vad_cmvn, VAD_CMVN_PATH, model_path);
    GetValue(vad_config, VAD_CONFIG_PATH, model_path);
    GetValue(am_model, AM_MODEL_PATH, model_path);
    GetValue(am_cmvn, AM_CMVN_PATH, model_path);
    GetValue(am_config, AM_CONFIG_PATH, model_path);
    GetValue(punc_model, PUNC_MODEL_PATH, model_path);
    GetValue(punc_config, PUNC_CONFIG_PATH, model_path);
    GetValue(wav_path, WAV_PATH, model_path);
    GetValue(wav_scp, WAV_SCP, model_path);


    struct timeval start, end;
    gettimeofday(&start, NULL);
    int thread_num = 1;
    FUNASR_HANDLE asr_hanlde=FunASRInit(model_path, thread_num);

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
        FUNASR_RESULT result=FunASRRecogFile(asr_hanlde, wav_file.c_str(), RASR_NONE, NULL);
        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        taking_micros += ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        if (result)
        {
            string msg = FunASRGetResult(result, 0);
            setbuf(stdout, NULL);
            printf("Result: %s \n", msg.c_str());
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
    FunASRUninit(asr_hanlde);
    return 0;
}

