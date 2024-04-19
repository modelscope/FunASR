#ifndef AUDIO_H
#define AUDIO_H

#include <queue>
#include <stdint.h>
#include "vad-model.h"
#include "offline-stream.h"
#include "com-define.h"

#ifndef WAV_HEADER_SIZE
#define WAV_HEADER_SIZE 44
#endif

using namespace std;
namespace funasr {

class AudioFrame {
  private:
    int start;
    int end;


  public:
    AudioFrame();
    AudioFrame(int len);
    AudioFrame(const AudioFrame &other);
    AudioFrame(int start, int end, bool is_final);

    ~AudioFrame();
    int SetStart(int val);
    int SetEnd(int val);
    int GetStart();
    int GetLen();
    int Disp();
    // 2pass
    bool is_final = false;
    float* data = nullptr;
    int len;
    int global_start = 0; // the start of a frame in the global time axis. in ms
    int global_end = 0;   // the end of a frame in the global time axis. in ms
};

#ifdef _WIN32
#ifdef _FUNASR_API_EXPORT
#define DLLAPI __declspec(dllexport)
#else
#define DLLAPI __declspec(dllimport)
#endif
#else
#define DLLAPI 
#endif
class DLLAPI Audio {
  private:
    float *speech_data=nullptr;
    int16_t *speech_buff=nullptr;
    char* speech_char=nullptr;
    int speech_len;
    int speech_align_len;
    float align_size;
    int data_type;
    queue<AudioFrame *> frame_queue;
    queue<AudioFrame *> asr_online_queue;
    queue<AudioFrame *> asr_offline_queue;
    int dest_sample_rate;
  public:
    Audio(int data_type);
    Audio(int model_sample_rate,int data_type);
    Audio(int model_sample_rate,int data_type, int size);
    ~Audio();
    void ClearQueue(std::queue<AudioFrame*>& q);
    void Disp();
    void WavResample(int32_t sampling_rate, const float *waveform, int32_t n);
    bool LoadWav(const char* buf, int n_len, int32_t* sampling_rate);
    bool LoadWav(const char* filename, int32_t* sampling_rate, bool resample=true);
    bool LoadWav2Char(const char* filename, int32_t* sampling_rate);
    bool LoadPcmwav(const char* buf, int n_file_len, int32_t* sampling_rate);
    bool LoadPcmwav(const char* filename, int32_t* sampling_rate, bool resample=true);
    bool LoadPcmwav2Char(const char* filename, int32_t* sampling_rate);
    bool LoadOthers2Char(const char* filename);
    bool FfmpegLoad(const char *filename, bool copy2char=false);
    bool FfmpegLoad(const char* buf, int n_file_len);
    int FetchChunck(AudioFrame *&frame);
    int FetchTpass(AudioFrame *&frame);
    int Fetch(float *&dout, int &len, int &flag);
    int Fetch(float *&dout, int &len, int &flag, float &start_time);
    int Fetch(float **&dout, int *&len, int *&flag, float*& start_time, int batch_size, int &batch_in);
    int FetchDynamic(float **&dout, int *&len, int *&flag, float*& start_time, int batch_size, int &batch_in);
    void Padding();
    void Split(OfflineStream* offline_streamj);
    void CutSplit(OfflineStream* offline_streamj, std::vector<int> &index_vector);
    void Split(VadModel* vad_obj, vector<std::vector<int>>& vad_segments, bool input_finished=true);
    void Split(VadModel* vad_obj, int chunk_len, bool input_finished=true, ASR_TYPE asr_mode=ASR_TWO_PASS);
    float GetTimeLen();
    int GetQueueSize() { return (int)frame_queue.size(); }
    char* GetSpeechChar(){return speech_char;}
    int GetSpeechLen(){return speech_len;}

    // 2pass
    vector<float> all_samples;
    int offset = 0;
    int speech_start=-1, speech_end=0;
    int speech_offline_start=-1;

    int seg_sample = MODEL_SAMPLE_RATE/1000;
    bool LoadPcmwavOnline(const char* buf, int n_file_len, int32_t* sampling_rate);
    void ResetIndex(){
      speech_start=-1;
      speech_end=0;
      speech_offline_start=-1;
      offset = 0;
      all_samples.clear();
    }
};

} // namespace funasr
#endif
