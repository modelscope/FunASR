#include <iostream>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include <Audio.h>
#include <Model.h>

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
    Model* mm = create_model(argv[1], nThreadNum);
    if (!mm)
    {
        printf("Cannot load ASR Model from: %s, there must be files model.onnx and vocab.txt", argv[1]);
        exit(-1);
    }
    
  
    Audio audio(0);
    if (!audio.loadwav(argv[2]))
    {
        printf("cannot load %s\n", argv[2]);
        return -1;
    }
    audio.disp();
  

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs.\n", (double)micros / 1000000);
    audio.split();

    setbuf(stdout, NULL);
    cout << "Result: \"";
    gettimeofday(&start, NULL);
    float *buff;
    int len;
    int flag;
    while (audio.fetch(buff, len, flag) > 0) {
        mm->reset();
        string msg = mm->forward(buff, len, flag);
        cout << msg;
    }

    gettimeofday(&end, NULL);

    cout << "\"." << endl;

    seconds = (end.tv_sec - start.tv_sec);
    long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model inference takes %lfs.\n", (double)micros / 1000000);

    printf("Model inference RTF: %04lf.\n", (double)taking_micros/micros );

    delete mm;

    return 0;
}
