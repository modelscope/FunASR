
#ifndef AUDIO_H
#define AUDIO_H

#include <queue>
#include <stdint.h>
#include "model.h"

#ifndef WAV_HEADER_SIZE
#define WAV_HEADER_SIZE 44
#endif

using namespace std;

class AudioFrame {
  private:
    int start;
    int end;
    int len;

  public:
    AudioFrame();
    AudioFrame(int len);

    ~AudioFrame();
    int SetStart(int val);
    int SetEnd(int val);
    int GetStart();
    int GetLen();
    int Disp();
};

class Audio {
  private:
    float *speech_data;
    int16_t *speech_buff;
    int speech_len;
    int speech_align_len;
    int offset;
    float align_size;
    int data_type;
    queue<AudioFrame *> frame_queue;

  public:
    Audio(int data_type);
    Audio(int data_type, int size);
    ~Audio();
    void Disp();
    bool LoadWav(const char* filename, int32_t* sampling_rate);
    void WavResample(int32_t sampling_rate, const float *waveform, int32_t n);
    bool LoadWav(const char* buf, int n_len, int32_t* sampling_rate);
    bool LoadPcmwav(const char* buf, int n_file_len, int32_t* sampling_rate);
    bool LoadPcmwav(const char* filename, int32_t* sampling_rate);
    int FetchChunck(float *&dout, int len);
    int Fetch(float *&dout, int &len, int &flag);
    void Padding();
    void Split(Model* recog_obj);
    float GetTimeLen();
    int GetQueueSize() { return (int)frame_queue.size(); }
};

#endif
