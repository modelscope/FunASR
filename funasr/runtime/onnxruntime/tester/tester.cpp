
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include "librapidasrapi.h"

#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf("Usage: %s /path/to/model_dir /path/to/wav/file", argv[0]);
        exit(-1);
    }
    struct timeval start, end;
    gettimeofday(&start, NULL);
    int nThreadNum = 4;
    RPASR_HANDLE AsrHanlde=RapidAsrInit(argv[1], nThreadNum);

    if (!AsrHanlde)
    {
        printf("Cannot load ASR Model from: %s, there must be files model.onnx and vocab.txt", argv[1]);
        exit(-1);
    }
    
 

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs.\n", (double)micros / 1000000);

    setbuf(stdout, NULL);
    cout << "Result: \"";
    gettimeofday(&start, NULL);

    RPASR_RESULT Result=RapidAsrRecogPCMFile(AsrHanlde, argv[2], RASR_NONE, NULL);
    gettimeofday(&end, NULL);

    if (Result)
    {
        string msg = RapidAsrGetResult(Result, 0);
        cout << msg << endl;
        cout << "\"." << endl;
        RapidAsrFreeResult(Result);
    }
    else
    {
        cout <<("no return data!");
    }
  
    seconds = (end.tv_sec - start.tv_sec);
    long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model inference takes %lfs.\n", (double)micros / 1000000);

    printf("Model inference RTF: %04lf.\n", (double)taking_micros/micros );

    RapidAsrUninit(AsrHanlde);

    return 0;
}

    