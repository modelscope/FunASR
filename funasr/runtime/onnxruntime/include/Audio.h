
#ifndef AUDIO_H
#define AUDIO_H

#include <ComDefine.h>
#include <queue>
#include <stdint.h>

#ifndef model_sample_rate
#define model_sample_rate 16000
#endif
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
    int set_start(int val);
    int set_end(int val, int max_len);
    int get_start();
    int get_len();
    int disp();
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
    void disp();
    bool loadwav(const char* filename, int32_t* sampling_rate);
    void wavResample(int32_t sampling_rate, const float *waveform, int32_t n);
    bool loadwav(const char* buf, int nLen, int32_t* sampling_rate);
    bool loadpcmwav(const char* buf, int nFileLen, int32_t* sampling_rate);
    bool loadpcmwav(const char* filename, int32_t* sampling_rate);
    int fetch_chunck(float *&dout, int len);
    int fetch(float *&dout, int &len, int &flag);
    void padding();
    void split();
    float get_time_len();

    int get_queue_size() { return (int)frame_queue.size(); }
};

#endif
