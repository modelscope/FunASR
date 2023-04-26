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
#include "libfunasrapi.h"
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

using namespace std;

std::atomic<int> wav_index(0);
std::mutex mtx;

void runReg(FUNASR_HANDLE asr_handle, vector<string> wav_list, 
            float* total_length, long* total_time, int core_id) {
    
    struct timeval start, end;
    long seconds = 0;
    float n_total_length = 0.0f;
    long n_total_time = 0;
    
    // warm up
    for (size_t i = 0; i < 1; i++)
    {
        FUNASR_RESULT result=FunASRRecogFile(asr_handle, wav_list[0].c_str(), RASR_NONE, NULL);
    }

    while (true) {
        // 使用原子变量获取索引并递增
        int i = wav_index.fetch_add(1);
        if (i >= wav_list.size()) {
            break;
        }

        gettimeofday(&start, NULL);
        FUNASR_RESULT result=FunASRRecogFile(asr_handle, wav_list[i].c_str(), RASR_NONE, NULL);

        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        n_total_time += taking_micros;

        if(result){
            string msg = FunASRGetResult(result, 0);
            LOG(INFO) << "Thread: " << this_thread::get_id() <<" Result: " << msg.c_str();

            float snippet_time = FunASRGetRetSnippetTime(result);
            n_total_length += snippet_time;
            FunASRFreeResult(result);
        }else{
            LOG(ERROR) << ("No return data!\n");
        }
    }
    {
        lock_guard<mutex> guard(mtx);
        *total_length += n_total_length;
        if(*total_time < n_total_time){
            *total_time = n_total_time;
        }
    }
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
    TCLAP::ValueArg<std::string> vad_model("", VAD_MODEL_PATH, "vad model path", false, "", "string");
    TCLAP::ValueArg<std::string> vad_cmvn("", VAD_CMVN_PATH, "vad cmvn path", false, "", "string");
    TCLAP::ValueArg<std::string> vad_config("", VAD_CONFIG_PATH, "vad config path", false, "", "string");

    TCLAP::ValueArg<std::string> am_model("", AM_MODEL_PATH, "am model path", false, "", "string");
    TCLAP::ValueArg<std::string> am_cmvn("", AM_CMVN_PATH, "am cmvn path", false, "", "string");
    TCLAP::ValueArg<std::string> am_config("", AM_CONFIG_PATH, "am config path", false, "", "string");

    TCLAP::ValueArg<std::string> punc_model("", PUNC_MODEL_PATH, "punc model path", false, "", "string");
    TCLAP::ValueArg<std::string> punc_config("", PUNC_CONFIG_PATH, "punc config path", false, "", "string");

    TCLAP::ValueArg<std::string> wav_scp("", WAV_SCP, "wave scp path", true, "", "string");
    TCLAP::ValueArg<std::int32_t> thread_num("", THREAD_NUM, "multi-thread num for rtf", true, 0, "int32_t");

    cmd.add(vad_model);
    cmd.add(vad_cmvn);
    cmd.add(vad_config);
    cmd.add(am_model);
    cmd.add(am_cmvn);
    cmd.add(am_config);
    cmd.add(punc_model);
    cmd.add(punc_config);
    cmd.add(wav_scp);
    cmd.add(thread_num);
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
    GetValue(wav_scp, WAV_SCP, model_path);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    FUNASR_HANDLE asr_handle=FunASRInit(model_path, 1);

    if (!asr_handle)
    {
        LOG(ERROR) << "FunASR init failed";
        exit(-1);
    }

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "Model initialization takes " << (double)modle_init_micros / 1000000 << " s";

    // read wav_scp
    vector<string> wav_list;
    if(model_path.find(WAV_SCP)!=model_path.end()){
        ifstream in(model_path.at(WAV_SCP));
        if (!in.is_open()) {
            LOG(ERROR) << "Failed to open file: " << model_path.at(WAV_SCP);
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

    // 多线程测试
    float total_length = 0.0f;
    long total_time = 0;
    std::vector<std::thread> threads;

    int rtf_threds = thread_num.getValue();
    for (int i = 0; i < rtf_threds; i++)
    {
        threads.emplace_back(thread(runReg, asr_handle, wav_list, &total_length, &total_time, i));
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
