
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include "libfunasrapi.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
using namespace std;

std::atomic<int> index(0);
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
        int i = index.fetch_add(1);
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
            printf("Thread: %d Result: %s \n", this_thread::get_id(), msg.c_str());

            float snippet_time = FunASRGetRetSnippetTime(result);
            n_total_length += snippet_time;
            FunASRFreeResult(result);
        }else{
            cout <<"No return data!";
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

int main(int argc, char *argv[])
{

    if (argc < 5)
    {
        printf("Usage: %s /path/to/model_dir /path/to/wav.scp quantize(true or false) thread_num \n", argv[0]);
        exit(-1);
    }

    // read wav.scp
    vector<string> wav_list;
    ifstream in(argv[2]);
    if (!in.is_open()) {
        printf("Failed to open file: %s", argv[2]);
        return 0;
    }
    string line;
    while(getline(in, line))
    {
        istringstream iss(line);
        string column1, column2;
        iss >> column1 >> column2;
        wav_list.push_back(column2); 
    }
    in.close();

    // model init
    struct timeval start, end;
    gettimeofday(&start, NULL);
    // is quantize
    bool quantize = false;
    istringstream(argv[3]) >> boolalpha >> quantize;
    // thread num
    int thread_num = 1;
    thread_num = atoi(argv[4]);

    FUNASR_HANDLE asr_handle=FunASRInit(argv[1], 1, quantize);
    if (!asr_handle)
    {
        printf("Cannot load ASR Model from: %s, there must be files model.onnx and vocab.txt", argv[1]);
        exit(-1);
    }
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs.\n", (double)modle_init_micros / 1000000);

    // 多线程测试
    float total_length = 0.0f;
    long total_time = 0;
    std::vector<std::thread> threads;

    for (int i = 0; i < thread_num; i++)
    {
        threads.emplace_back(thread(runReg, asr_handle, wav_list, &total_length, &total_time, i));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    printf("total_time_wav %ld ms.\n", (long)(total_length * 1000));
    printf("total_time_comput %ld ms.\n", total_time / 1000);
    printf("total_rtf %05lf .\n", (double)total_time/ (total_length*1000000));
    printf("speedup %05lf .\n", 1.0/((double)total_time/ (total_length*1000000)));

    FunASRUninit(asr_handle);
    return 0;
}
