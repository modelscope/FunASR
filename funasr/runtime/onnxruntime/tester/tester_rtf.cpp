
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include "librapidasrapi.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
using namespace std;

void runReg(vector<string> wav_list, RPASR_HANDLE AsrHanlde)
{
    for (size_t i = 0; i < wav_list.size(); i++)
    {
        RPASR_RESULT Result=RapidAsrRecogFile(AsrHanlde, wav_list[i].c_str(), RASR_NONE, NULL);

        if(Result){
            string msg = RapidAsrGetResult(Result, 0);
            printf("Result: %s \n", msg.c_str());
            RapidAsrFreeResult(Result);
        }else{
            cout <<"No return data!";
        }
    }
}

int main(int argc, char *argv[])
{

    if (argc < 4)
    {
        printf("Usage: %s /path/to/model_dir /path/to/wav.scp quantize(true or false) \n", argv[0]);
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
    int nThreadNum = 1;
    // is quantize
    bool quantize = false;
    istringstream(argv[3]) >> boolalpha >> quantize;

    RPASR_HANDLE AsrHanlde=RapidAsrInit(argv[1], nThreadNum, quantize);
    if (!AsrHanlde)
    {
        printf("Cannot load ASR Model from: %s, there must be files model.onnx and vocab.txt", argv[1]);
        exit(-1);
    }
    
    std::thread t1(runReg, wav_list, AsrHanlde);
    std::thread t2(runReg, wav_list, AsrHanlde);

    t1.join();
    t2.join();

    //runReg(wav_list, AsrHanlde);

    RapidAsrUninit(AsrHanlde);
    return 0;
}
