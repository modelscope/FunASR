
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include "libfunasrapi.h"
#include <sstream>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        printf("Usage: %s /path/to/model_dir /path/to/wav/file quantize(true or false) use_vad(true or false) \n", argv[0]);
        exit(-1);
    }
    struct timeval start, end;
    gettimeofday(&start, NULL);
    int nThreadNum = 1;
    // is quantize
    bool quantize = false;
    bool use_vad = false;
    istringstream(argv[3]) >> boolalpha >> quantize;
    istringstream(argv[4]) >> boolalpha >> use_vad;
    FUNASR_HANDLE AsrHanlde=FunASRInit(argv[1], nThreadNum, quantize, use_vad);

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
    FUNASR_RESULT Result=FunASRRecogFile(AsrHanlde, argv[2], RASR_NONE, NULL, use_vad);
    gettimeofday(&end, NULL);

    float snippet_time = 0.0f;
    if (Result)
    {
        string msg = FunASRGetResult(Result, 0);
        setbuf(stdout, NULL);
        printf("Result: %s \n", msg.c_str());
        snippet_time = FunASRGetRetSnippetTime(Result);
        FunASRFreeResult(Result);
    }
    else
    {
        printf("no return data!\n");
    }
 
    printf("Audio length %lfs.\n", (double)snippet_time);
    seconds = (end.tv_sec - start.tv_sec);
    long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model inference takes %lfs.\n", (double)taking_micros / 1000000);
    printf("Model inference RTF: %04lf.\n", (double)taking_micros/ (snippet_time*1000000));

    FunASRUninit(AsrHanlde);

    return 0;
}

    
