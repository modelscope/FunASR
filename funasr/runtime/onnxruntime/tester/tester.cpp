
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include "librapidasrapi.h"

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("Usage: %s /path/to/model_dir /path/to/wav/file quantize(true or false)", argv[0]);
        exit(-1);
    }
    struct timeval start, end;
    gettimeofday(&start, NULL);
    int nThreadNum = 4;
    // is quantize
    bool quantize = false;
    istringstream(argv[3]) >> boolalpha >> quantize;
    RPASR_HANDLE AsrHanlde=RapidAsrInit(argv[1], nThreadNum, quantize);

    if (!AsrHanlde)
    {
        printf("Cannot load ASR Model from: %s, there must be files model.onnx and vocab.txt", argv[1]);
        exit(-1);
    }

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs.\n", (double)modle_init_micros / 1000000);

    gettimeofday(&start, NULL);
    float snippet_time = 0.0f;

    RPASR_RESULT Result=RapidAsrRecogFile(AsrHanlde, argv[2], RASR_NONE, NULL);

    gettimeofday(&end, NULL);
   
    if (Result)
    {
        string msg = RapidAsrGetResult(Result, 0);
        setbuf(stdout, NULL);
        cout << "Result: \"";
        cout << msg << endl;
        cout << "\"." << endl;
        snippet_time = RapidAsrGetRetSnippetTime(Result);
        RapidAsrFreeResult(Result);
    }
    else
    {
        cout <<"no return data!";
    }
 
    //char* buff = nullptr;
    //int len = 0;
    //ifstream ifs(argv[2], std::ios::binary | std::ios::in);
    //if (ifs.is_open())
    //{
    //    ifs.seekg(0, std::ios::end);
    //    len = ifs.tellg();
    //    ifs.seekg(0, std::ios::beg);

    //    buff = new char[len];

    //    ifs.read(buff, len);


    //    //RPASR_RESULT Result = RapidAsrRecogPCMFile(AsrHanlde, argv[2], RASR_NONE, NULL);

    //    RPASR_RESULT Result=RapidAsrRecogPCMBuffer(AsrHanlde, buff,len, RASR_NONE, NULL);
    //    //RPASR_RESULT Result = RapidAsrRecogPCMFile(AsrHanlde, argv[2], RASR_NONE, NULL);
    //    gettimeofday(&end, NULL);
    //   
    //    if (Result)
    //    {
    //        string msg = RapidAsrGetResult(Result, 0);
    //        setbuf(stdout, NULL);
    //        cout << "Result: \"";
    //        cout << msg << endl;
    //        cout << "\"." << endl;
    //        snippet_time = RapidAsrGetRetSnippetTime(Result);
    //        RapidAsrFreeResult(Result);
    //    }
    //    else
    //    {
    //        cout <<"no return data!";
    //    }
  
    //   
    //delete[]buff;
    //}
 
    printf("Audio length %lfs.\n", (double)snippet_time);
    seconds = (end.tv_sec - start.tv_sec);
    long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model inference takes %lfs.\n", (double)taking_micros / 1000000);
    printf("Model inference RTF: %04lf.\n", (double)taking_micros/ (snippet_time*1000000));

    RapidAsrUninit(AsrHanlde);

    return 0;
}

    