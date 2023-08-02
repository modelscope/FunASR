#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <assert.h>

#include "audio.h"
#include "precomp.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

using namespace std;

namespace funasr {
// see http://soundfile.sapp.org/doc/WaveFormat/
// Note: We assume little endian here
struct WaveHeader {
  bool Validate() const {
    //                 F F I R
    if (chunk_id != 0x46464952) {
      printf("Expected chunk_id RIFF. Given: 0x%08x\n", chunk_id);
      return false;
    }
    //               E V A W
    if (format != 0x45564157) {
      printf("Expected format WAVE. Given: 0x%08x\n", format);
      return false;
    }

    if (subchunk1_id != 0x20746d66) {
      printf("Expected subchunk1_id 0x20746d66. Given: 0x%08x\n",
                       subchunk1_id);
      return false;
    }

    if (subchunk1_size != 16) {  // 16 for PCM
      printf("Expected subchunk1_size 16. Given: %d\n",
                       subchunk1_size);
      return false;
    }

    if (audio_format != 1) {  // 1 for PCM
      printf("Expected audio_format 1. Given: %d\n", audio_format);
      return false;
    }

    if (num_channels != 1) {  // we support only single channel for now
      printf("Expected single channel. Given: %d\n", num_channels);
      return false;
    }
    if (byte_rate != (sample_rate * num_channels * bits_per_sample / 8)) {
      return false;
    }

    if (block_align != (num_channels * bits_per_sample / 8)) {
      return false;
    }

    if (bits_per_sample != 16) {  // we support only 16 bits per sample
      printf("Expected bits_per_sample 16. Given: %d\n",
                       bits_per_sample);
      return false;
    }
    return true;
  }

  // See https://en.wikipedia.org/wiki/WAV#Metadata and
  // https://www.robotplanet.dk/audio/wav_meta_data/riff_mci.pdf
  void SeekToDataChunk(std::istream &is) {
    //                              a t a d
    while (is && subchunk2_id != 0x61746164) {
      // const char *p = reinterpret_cast<const char *>(&subchunk2_id);
      // printf("Skip chunk (%x): %c%c%c%c of size: %d\n", subchunk2_id, p[0],
      //        p[1], p[2], p[3], subchunk2_size);
      is.seekg(subchunk2_size, std::istream::cur);
      is.read(reinterpret_cast<char *>(&subchunk2_id), sizeof(int32_t));
      is.read(reinterpret_cast<char *>(&subchunk2_size), sizeof(int32_t));
    }
  }

  int32_t chunk_id;
  int32_t chunk_size;
  int32_t format;
  int32_t subchunk1_id;
  int32_t subchunk1_size;
  int16_t audio_format;
  int16_t num_channels;
  int32_t sample_rate;
  int32_t byte_rate;
  int16_t block_align;
  int16_t bits_per_sample;
  int32_t subchunk2_id;    // a tag of this chunk
  int32_t subchunk2_size;  // size of subchunk2
};
static_assert(sizeof(WaveHeader) == WAV_HEADER_SIZE, "");

class AudioWindow {
  private:
    int *window;
    int in_idx;
    int out_idx;
    int sum;
    int window_size = 0;

  public:
    AudioWindow(int window_size) : window_size(window_size)
    {
        window = (int *)calloc(sizeof(int), window_size + 1);
        in_idx = 0;
        out_idx = 1;
        sum = 0;
    };
    ~AudioWindow(){
        free(window);
    };
    int put(int val)
    {
        sum = sum + val - window[out_idx];
        window[in_idx] = val;
        in_idx = in_idx == window_size ? 0 : in_idx + 1;
        out_idx = out_idx == window_size ? 0 : out_idx + 1;
        return sum;
    };
};

AudioFrame::AudioFrame(){}
AudioFrame::AudioFrame(int len) : len(len)
{
    start = 0;
}
AudioFrame::AudioFrame(const AudioFrame &other)
{
    start = other.start;
    end = other.end;
    len = other.len;
    is_final = other.is_final;
}
AudioFrame::AudioFrame(int start, int end, bool is_final):start(start),end(end),is_final(is_final){
    len = end - start;
}
AudioFrame::~AudioFrame(){
    if(data != NULL){
        free(data);
    }
}
int AudioFrame::SetStart(int val)
{
    start = val < 0 ? 0 : val;
    return start;
}

int AudioFrame::SetEnd(int val)
{
    end = val;
    len = end - start;
    return end;
}

int AudioFrame::GetStart()
{
    return start;
}

int AudioFrame::GetLen()
{
    return len;
}

int AudioFrame::Disp()
{
    LOG(ERROR) << "Not imp!!!!";
    return 0;
}

Audio::Audio(int data_type) : data_type(data_type)
{
    speech_buff = NULL;
    speech_data = NULL;
    align_size = 1360;
}

Audio::Audio(int data_type, int size) : data_type(data_type)
{
    speech_buff = NULL;
    speech_data = NULL;
    align_size = (float)size;
}

Audio::~Audio()
{
    if (speech_buff != NULL) {
        free(speech_buff);
    }
    if (speech_data != NULL) {
        free(speech_data);
    }
    if (speech_char != NULL) {
        free(speech_char);
    }
}

void Audio::Disp()
{
    LOG(INFO) << "Audio time is " << (float)speech_len / MODEL_SAMPLE_RATE << " s. len is " << speech_len;
}

float Audio::GetTimeLen()
{
    return (float)speech_len / MODEL_SAMPLE_RATE;
}

void Audio::WavResample(int32_t sampling_rate, const float *waveform,
                          int32_t n)
{
    LOG(INFO) << "Creating a resampler:\n"
              << "   in_sample_rate: "<< sampling_rate << "\n"
              << "   output_sample_rate: " << static_cast<int32_t>(MODEL_SAMPLE_RATE);
    float min_freq =
        std::min<int32_t>(sampling_rate, MODEL_SAMPLE_RATE);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;

    auto resampler = std::make_unique<LinearResample>(
          sampling_rate, MODEL_SAMPLE_RATE, lowpass_cutoff, lowpass_filter_width);
    std::vector<float> samples;
    resampler->Resample(waveform, n, true, &samples);
    //reset speech_data
    speech_len = samples.size();
    if (speech_data != NULL) {
        free(speech_data);
    }
    speech_data = (float*)malloc(sizeof(float) * speech_len);
    memset(speech_data, 0, sizeof(float) * speech_len);
    copy(samples.begin(), samples.end(), speech_data);
}

bool Audio::FfmpegLoad(const char *filename, bool copy2char){
    // from file
    AVFormatContext* formatContext = avformat_alloc_context();
    if (avformat_open_input(&formatContext, filename, NULL, NULL) != 0) {
        printf("Error: Could not open input file.");
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return false;
    }

    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        printf("Error: Could not find stream information.");
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return false;
    }
    const AVCodec* codec = NULL;
    AVCodecParameters* codecParameters = NULL;
    int audioStreamIndex = av_find_best_stream(formatContext, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (audioStreamIndex >= 0) {
        codecParameters = formatContext->streams[audioStreamIndex]->codecpar;
    }
    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        fprintf(stderr, "Failed to allocate codec context\n");
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return false;
    }
    if (avcodec_parameters_to_context(codecContext, codecParameters) != 0) {
        printf("Error: Could not copy codec parameters to codec context.");
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        avcodec_free_context(&codecContext);
        return false;
    }
    if (avcodec_open2(codecContext, codec, NULL) < 0) {
        printf("Error: Could not open audio decoder.");
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        avcodec_free_context(&codecContext);
        return false;
    }
    SwrContext *swr_ctx = swr_alloc_set_opts(
        nullptr, // allocate a new context
        AV_CH_LAYOUT_MONO, // output channel layout (stereo)
        AV_SAMPLE_FMT_S16, // output sample format (signed 16-bit)
        16000, // output sample rate (same as input)
        av_get_default_channel_layout(codecContext->channels), // input channel layout
        codecContext->sample_fmt, // input sample format
        codecContext->sample_rate, // input sample rate
        0, // logging level
        nullptr // parent context
    );
    if (swr_ctx == nullptr) {
        std::cerr << "Could not initialize resampler" << std::endl;
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        avcodec_free_context(&codecContext);
        return false;
    }
    if (swr_init(swr_ctx) != 0) {
        std::cerr << "Could not initialize resampler" << std::endl;
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        avcodec_free_context(&codecContext);
        swr_free(&swr_ctx);
        return false;
    }

    // to pcm
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<uint8_t> resampled_buffers;
    while (av_read_frame(formatContext, packet) >= 0) {
        if (packet->stream_index == audioStreamIndex) {
            if (avcodec_send_packet(codecContext, packet) >= 0) {
                while (avcodec_receive_frame(codecContext, frame) >= 0) {
                    // Resample audio if necessary
                    std::vector<uint8_t> resampled_buffer;
                    int in_samples = frame->nb_samples;
                    uint8_t **in_data = frame->extended_data;
                    int out_samples = av_rescale_rnd(in_samples,
                                                    16000,
                                                    codecContext->sample_rate,
                                                    AV_ROUND_DOWN);
                    
                    int resampled_size = out_samples * av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);
                    if (resampled_buffer.size() < resampled_size) {
                        resampled_buffer.resize(resampled_size);
                    }                    
                    uint8_t *resampled_data = resampled_buffer.data();
                    int ret = swr_convert(
                        swr_ctx,
                        &resampled_data, // output buffer
                        resampled_size, // output buffer size
                        (const uint8_t **)(frame->data), //(const uint8_t **)(frame->extended_data)
                        in_samples // input buffer size
                    );
                    if (ret < 0) {
                        std::cerr << "Error resampling audio" << std::endl;
                        break;
                    }
                    std::copy(resampled_buffer.begin(), resampled_buffer.end(), std::back_inserter(resampled_buffers));
                }
            }
        }
        av_packet_unref(packet);
    }

    avformat_close_input(&formatContext);
    avformat_free_context(formatContext);
    avcodec_free_context(&codecContext);
    swr_free(&swr_ctx);
    av_packet_free(&packet);
    av_frame_free(&frame);

    if (speech_data != NULL) {
        free(speech_data);
    }
    if (speech_buff != NULL) {
        free(speech_buff);
    }
    if (speech_char != NULL) {
        free(speech_char);
    }
    offset = 0;
    
    if(copy2char){
        speech_char = (char *)malloc(resampled_buffers.size());
        memset(speech_char, 0, resampled_buffers.size());
        memcpy((void*)speech_char, (const void*)resampled_buffers.data(), resampled_buffers.size());
    }

    speech_len = (resampled_buffers.size()) / 2;
    speech_buff = (int16_t*)malloc(sizeof(int16_t) * speech_len);
    if (speech_buff)
    {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        memcpy((void*)speech_buff, (const void*)resampled_buffers.data(), speech_len * sizeof(int16_t));

        speech_data = (float*)malloc(sizeof(float) * speech_len);
        memset(speech_data, 0, sizeof(float) * speech_len);

        float scale = 1;
        if (data_type == 1) {
            scale = 32768;
        }
        for (int32_t i = 0; i != speech_len; ++i) {
            speech_data[i] = (float)speech_buff[i] / scale;
        }

        AudioFrame* frame = new AudioFrame(speech_len);
        frame_queue.push(frame);
    
        return true;
    }
    else
        return false;
    
}

bool Audio::FfmpegLoad(const char* buf, int n_file_len){
    // from buf
    char* buf_copy = (char *)malloc(n_file_len);
    memcpy(buf_copy, buf, n_file_len);

    AVIOContext* avio_ctx = avio_alloc_context(
        (unsigned char*)buf_copy, // buffer
        n_file_len, // buffer size
        0, // write flag (0 for read-only)
        nullptr, // opaque pointer (not used here)
        nullptr, // read callback (not used here)
        nullptr, // write callback (not used here)
        nullptr // seek callback (not used here)
    );
    AVFormatContext* formatContext = avformat_alloc_context();
    formatContext->pb = avio_ctx;
    if (avformat_open_input(&formatContext, "", NULL, NULL) != 0) {
        printf("Error: Could not open input file.");
        avio_context_free(&avio_ctx);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return false;
    }

    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        printf("Error: Could not find stream information.");
        avio_context_free(&avio_ctx);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return false;
    }
    const AVCodec* codec = NULL;
    AVCodecParameters* codecParameters = NULL;
    int audioStreamIndex = av_find_best_stream(formatContext, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (audioStreamIndex >= 0) {
        codecParameters = formatContext->streams[audioStreamIndex]->codecpar;
    }
    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        fprintf(stderr, "Failed to allocate codec context\n");
        avio_context_free(&avio_ctx);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return false;
    }
    if (avcodec_parameters_to_context(codecContext, codecParameters) != 0) {
        printf("Error: Could not copy codec parameters to codec context.");
        avio_context_free(&avio_ctx);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        avcodec_free_context(&codecContext);
        return false;
    }
    if (avcodec_open2(codecContext, codec, NULL) < 0) {
        printf("Error: Could not open audio decoder.");
        avio_context_free(&avio_ctx);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        avcodec_free_context(&codecContext);
        return false;
    }
    SwrContext *swr_ctx = swr_alloc_set_opts(
        nullptr, // allocate a new context
        AV_CH_LAYOUT_MONO, // output channel layout (stereo)
        AV_SAMPLE_FMT_S16, // output sample format (signed 16-bit)
        16000, // output sample rate (same as input)
        av_get_default_channel_layout(codecContext->channels), // input channel layout
        codecContext->sample_fmt, // input sample format
        codecContext->sample_rate, // input sample rate
        0, // logging level
        nullptr // parent context
    );
    if (swr_ctx == nullptr) {
        std::cerr << "Could not initialize resampler" << std::endl;
        avio_context_free(&avio_ctx);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        avcodec_free_context(&codecContext);
        return false;
    }
    if (swr_init(swr_ctx) != 0) {
        std::cerr << "Could not initialize resampler" << std::endl;
        avio_context_free(&avio_ctx);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        avcodec_free_context(&codecContext);
        swr_free(&swr_ctx);
        return false;
    }

    // to pcm
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<uint8_t> resampled_buffers;
    while (av_read_frame(formatContext, packet) >= 0) {
        if (packet->stream_index == audioStreamIndex) {
            if (avcodec_send_packet(codecContext, packet) >= 0) {
                while (avcodec_receive_frame(codecContext, frame) >= 0) {
                    // Resample audio if necessary
                    std::vector<uint8_t> resampled_buffer;
                    int in_samples = frame->nb_samples;
                    uint8_t **in_data = frame->extended_data;
                    int out_samples = av_rescale_rnd(in_samples,
                                                    16000,
                                                    codecContext->sample_rate,
                                                    AV_ROUND_DOWN);
                    
                    int resampled_size = out_samples * av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);
                    if (resampled_buffer.size() < resampled_size) {
                        resampled_buffer.resize(resampled_size);
                    }                    
                    uint8_t *resampled_data = resampled_buffer.data();
                    int ret = swr_convert(
                        swr_ctx,
                        &resampled_data, // output buffer
                        resampled_size, // output buffer size
                        (const uint8_t **)(frame->data), //(const uint8_t **)(frame->extended_data)
                        in_samples // input buffer size
                    );
                    if (ret < 0) {
                        std::cerr << "Error resampling audio" << std::endl;
                        break;
                    }
                    std::copy(resampled_buffer.begin(), resampled_buffer.end(), std::back_inserter(resampled_buffers));
                }
            }
        }
        av_packet_unref(packet);
    }

    avio_context_free(&avio_ctx);
    avformat_close_input(&formatContext);
    avformat_free_context(formatContext);
    avcodec_free_context(&codecContext);
    swr_free(&swr_ctx);
    av_packet_free(&packet);
    av_frame_free(&frame);

    if (speech_data != NULL) {
        free(speech_data);
    }
    if (speech_buff != NULL) {
        free(speech_buff);
    }
    offset = 0;

    speech_len = (resampled_buffers.size()) / 2;
    speech_buff = (int16_t*)malloc(sizeof(int16_t) * speech_len);
    if (speech_buff)
    {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        memcpy((void*)speech_buff, (const void*)resampled_buffers.data(), speech_len * sizeof(int16_t));

        speech_data = (float*)malloc(sizeof(float) * speech_len);
        memset(speech_data, 0, sizeof(float) * speech_len);

        float scale = 1;
        if (data_type == 1) {
            scale = 32768;
        }
        for (int32_t i = 0; i != speech_len; ++i) {
            speech_data[i] = (float)speech_buff[i] / scale;
        }

        AudioFrame* frame = new AudioFrame(speech_len);
        frame_queue.push(frame);
    
        return true;
    }
    else
        return false;
    
}


bool Audio::LoadWav(const char *filename, int32_t* sampling_rate)
{
    WaveHeader header;
    if (speech_data != NULL) {
        free(speech_data);
    }
    if (speech_buff != NULL) {
        free(speech_buff);
    }
    
    offset = 0;
    std::ifstream is(filename, std::ifstream::binary);
    is.read(reinterpret_cast<char *>(&header), sizeof(header));
    if(!is){
        LOG(ERROR) << "Failed to read " << filename;
        return false;
    }

    if (!header.Validate()) {
        return false;
    }

    header.SeekToDataChunk(is);
    if (!is) {
        return false;
    }
    
    if (!header.Validate()) {
        return false;
    }

    header.SeekToDataChunk(is);
    if (!is) {
        return false;
    }
    
    *sampling_rate = header.sample_rate;
    // header.subchunk2_size contains the number of bytes in the data.
    // As we assume each sample contains two bytes, so it is divided by 2 here
    speech_len = header.subchunk2_size / 2;
    speech_buff = (int16_t *)malloc(sizeof(int16_t) * speech_len);

    if (speech_buff)
    {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        is.read(reinterpret_cast<char *>(speech_buff), header.subchunk2_size);
        if (!is) {
            LOG(ERROR) << "Failed to read " << filename;
            return false;
        }
        speech_data = (float*)malloc(sizeof(float) * speech_len);
        memset(speech_data, 0, sizeof(float) * speech_len);

        float scale = 1;
        if (data_type == 1) {
            scale = 32768;
        }
        for (int32_t i = 0; i != speech_len; ++i) {
            speech_data[i] = (float)speech_buff[i] / scale;
        }

        //resample
        if(*sampling_rate != MODEL_SAMPLE_RATE){
            WavResample(*sampling_rate, speech_data, speech_len);
        }

        AudioFrame* frame = new AudioFrame(speech_len);
        frame_queue.push(frame);

        return true;
    }
    else
        return false;
}

bool Audio::LoadWav2Char(const char *filename, int32_t* sampling_rate)
{
    WaveHeader header;
    if (speech_char != NULL) {
        free(speech_char);
    }
    offset = 0;
    std::ifstream is(filename, std::ifstream::binary);
    is.read(reinterpret_cast<char *>(&header), sizeof(header));
    if(!is){
        LOG(ERROR) << "Failed to read " << filename;
        return false;
    }
    if (!header.Validate()) {
        return false;
    }
    header.SeekToDataChunk(is);
        if (!is) {
            return false;
    }
    if (!header.Validate()) {
        return false;
    }
    header.SeekToDataChunk(is);
    if (!is) {
        return false;
    }
    
    *sampling_rate = header.sample_rate;
    // header.subchunk2_size contains the number of bytes in the data.
    // As we assume each sample contains two bytes, so it is divided by 2 here
    speech_len = header.subchunk2_size / 2;
    speech_char = (char *)malloc(header.subchunk2_size);
    memset(speech_char, 0, header.subchunk2_size);
    is.read(speech_char, header.subchunk2_size);

    return true;
}

bool Audio::LoadWav(const char* buf, int n_file_len, int32_t* sampling_rate)
{ 
    WaveHeader header;
    if (speech_data != NULL) {
        free(speech_data);
    }
    if (speech_buff != NULL) {
        free(speech_buff);
    }
    offset = 0;

    std::memcpy(&header, buf, sizeof(header));

    *sampling_rate = header.sample_rate;
    speech_len = header.subchunk2_size / 2;
    speech_buff = (int16_t *)malloc(sizeof(int16_t) * speech_len);
    if (speech_buff)
    {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        memcpy((void*)speech_buff, (const void*)(buf + WAV_HEADER_SIZE), speech_len * sizeof(int16_t));

        speech_data = (float*)malloc(sizeof(float) * speech_len);
        memset(speech_data, 0, sizeof(float) * speech_len);

        float scale = 1;
        if (data_type == 1) {
            scale = 32768;
        }

        for (int32_t i = 0; i != speech_len; ++i) {
            speech_data[i] = (float)speech_buff[i] / scale;
        }
        
        //resample
        if(*sampling_rate != MODEL_SAMPLE_RATE){
            WavResample(*sampling_rate, speech_data, speech_len);
        }

        AudioFrame* frame = new AudioFrame(speech_len);
        frame_queue.push(frame);

        return true;
    }
    else
        return false;
}

bool Audio::LoadPcmwav(const char* buf, int n_buf_len, int32_t* sampling_rate)
{
    if (speech_data != NULL) {
        free(speech_data);
    }
    if (speech_buff != NULL) {
        free(speech_buff);
    }
    offset = 0;

    speech_len = n_buf_len / 2;
    speech_buff = (int16_t*)malloc(sizeof(int16_t) * speech_len);
    if (speech_buff)
    {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        memcpy((void*)speech_buff, (const void*)buf, speech_len * sizeof(int16_t));

        speech_data = (float*)malloc(sizeof(float) * speech_len);
        memset(speech_data, 0, sizeof(float) * speech_len);

        float scale = 1;
        if (data_type == 1) {
            scale = 32768;
        }

        for (int32_t i = 0; i != speech_len; ++i) {
            speech_data[i] = (float)speech_buff[i] / scale;
        }
        
        //resample
        if(*sampling_rate != MODEL_SAMPLE_RATE){
            WavResample(*sampling_rate, speech_data, speech_len);
        }

        AudioFrame* frame = new AudioFrame(speech_len);
        frame_queue.push(frame);
        return true;

    }
    else
        return false;
}

bool Audio::LoadPcmwavOnline(const char* buf, int n_buf_len, int32_t* sampling_rate)
{
    if (speech_data != NULL) {
        free(speech_data);
    }
    if (speech_buff != NULL) {
        free(speech_buff);
    }
    if (speech_char != NULL) {
        free(speech_char);
    }

    speech_len = n_buf_len / 2;
    speech_buff = (int16_t*)malloc(sizeof(int16_t) * speech_len);
    if (speech_buff)
    {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        memcpy((void*)speech_buff, (const void*)buf, speech_len * sizeof(int16_t));

        speech_data = (float*)malloc(sizeof(float) * speech_len);
        memset(speech_data, 0, sizeof(float) * speech_len);

        float scale = 1;
        if (data_type == 1) {
            scale = 32768;
        }

        for (int32_t i = 0; i != speech_len; ++i) {
            speech_data[i] = (float)speech_buff[i] / scale;
        }
        
        //resample
        if(*sampling_rate != MODEL_SAMPLE_RATE){
            WavResample(*sampling_rate, speech_data, speech_len);
        }

        for (int32_t i = 0; i != speech_len; ++i) {
            all_samples.emplace_back(speech_data[i]);
        }

        AudioFrame* frame = new AudioFrame(speech_len);
        frame_queue.push(frame);
        return true;

    }
    else
        return false;
}

bool Audio::LoadPcmwav(const char* filename, int32_t* sampling_rate)
{
    if (speech_data != NULL) {
        free(speech_data);
    }
    if (speech_buff != NULL) {
        free(speech_buff);
    }
    offset = 0;

    FILE* fp;
    fp = fopen(filename, "rb");
    if (fp == nullptr)
	{
        LOG(ERROR) << "Failed to read " << filename;
        return false;
	}
    fseek(fp, 0, SEEK_END);
    uint32_t n_file_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    speech_len = (n_file_len) / 2;
    speech_buff = (int16_t*)malloc(sizeof(int16_t) * speech_len);
    if (speech_buff)
    {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        int ret = fread(speech_buff, sizeof(int16_t), speech_len, fp);
        fclose(fp);

        speech_data = (float*)malloc(sizeof(float) * speech_len);
        memset(speech_data, 0, sizeof(float) * speech_len);

        float scale = 1;
        if (data_type == 1) {
            scale = 32768;
        }
        for (int32_t i = 0; i != speech_len; ++i) {
            speech_data[i] = (float)speech_buff[i] / scale;
        }

        //resample
        if(*sampling_rate != MODEL_SAMPLE_RATE){
            WavResample(*sampling_rate, speech_data, speech_len);
        }

        AudioFrame* frame = new AudioFrame(speech_len);
        frame_queue.push(frame);
    
        return true;
    }
    else
        return false;

}

bool Audio::LoadPcmwav2Char(const char* filename, int32_t* sampling_rate)
{
    if (speech_char != NULL) {
        free(speech_char);
    }
    offset = 0;

    FILE* fp;
    fp = fopen(filename, "rb");
    if (fp == nullptr)
	{
        LOG(ERROR) << "Failed to read " << filename;
        return false;
	}
    fseek(fp, 0, SEEK_END);
    uint32_t n_file_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    speech_len = (n_file_len) / 2;
    speech_char = (char *)malloc(n_file_len);
    memset(speech_char, 0, n_file_len);
    fread(speech_char, sizeof(int16_t), n_file_len/2, fp);
    fclose(fp);
    
    return true;
}

bool Audio::LoadOthers2Char(const char* filename)
{
    if (speech_char != NULL) {
        free(speech_char);
    }

    FILE* fp;
    fp = fopen(filename, "rb");
    if (fp == nullptr)
	{
        LOG(ERROR) << "Failed to read " << filename;
        return false;
	}
    fseek(fp, 0, SEEK_END);
    uint32_t n_file_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    speech_len = n_file_len;
    speech_char = (char *)malloc(n_file_len);
    memset(speech_char, 0, n_file_len);
    fread(speech_char, 1, n_file_len, fp);
    fclose(fp);
    
    return true;
}

int Audio::FetchTpass(AudioFrame *&frame)
{
    if (asr_offline_queue.size() > 0) {
        frame = asr_offline_queue.front();
        asr_offline_queue.pop();
        return 1;
    } else {
        return 0;
    }
}

int Audio::FetchChunck(AudioFrame *&frame)
{
    if (asr_online_queue.size() > 0) {
        frame = asr_online_queue.front();
        asr_online_queue.pop();
        return 1;
    } else {
        return 0;
    }
}

int Audio::Fetch(float *&dout, int &len, int &flag)
{
    if (frame_queue.size() > 0) {
        AudioFrame *frame = frame_queue.front();
        frame_queue.pop();

        dout = speech_data + frame->GetStart();
        len = frame->GetLen();
        delete frame;
        flag = S_END;
        return 1;
    } else {
        return 0;
    }
}

void Audio::Padding()
{
    float num_samples = speech_len;
    float frame_length = 400;
    float frame_shift = 160;
    float num_frames = floor((num_samples + (frame_shift / 2)) / frame_shift);
    float num_new_samples = (num_frames - 1) * frame_shift + frame_length;
    float num_padding = num_new_samples - num_samples;
    float num_left_padding = (frame_length - frame_shift) / 2;
    float num_right_padding = num_padding - num_left_padding;

    float *new_data = (float *)malloc(num_new_samples * sizeof(float));
    int i;
    int tmp_off = 0;
    for (i = 0; i < num_left_padding; i++) {
        int ii = num_left_padding - i - 1;
        new_data[i] = speech_data[ii];
    }
    tmp_off = num_left_padding;
    memcpy(new_data + tmp_off, speech_data, speech_len * sizeof(float));
    tmp_off += speech_len;

    for (i = 0; i < num_right_padding; i++) {
        int ii = speech_len - i - 1;
        new_data[tmp_off + i] = speech_data[ii];
    }
    free(speech_data);
    speech_data = new_data;
    speech_len = num_new_samples;

    AudioFrame *frame = new AudioFrame(num_new_samples);
    frame_queue.push(frame);
    frame = frame_queue.front();
    frame_queue.pop();
    delete frame;
}

void Audio::Split(OfflineStream* offline_stream)
{
    AudioFrame *frame;

    frame = frame_queue.front();
    frame_queue.pop();
    int sp_len = frame->GetLen();
    delete frame;
    frame = NULL;

    std::vector<float> pcm_data(speech_data, speech_data+sp_len);
    vector<std::vector<int>> vad_segments = (offline_stream->vad_handle)->Infer(pcm_data);
    for(vector<int> segment:vad_segments)
    {
        frame = new AudioFrame();
        int start = segment[0]*seg_sample;
        int end = segment[1]*seg_sample;
        frame->SetStart(start);
        frame->SetEnd(end);
        frame_queue.push(frame);
        frame = NULL;
    }
}

void Audio::Split(VadModel* vad_obj, vector<std::vector<int>>& vad_segments, bool input_finished)
{
    AudioFrame *frame;

    frame = frame_queue.front();
    frame_queue.pop();
    int sp_len = frame->GetLen();
    delete frame;
    frame = NULL;

    std::vector<float> pcm_data(speech_data, speech_data+sp_len);
    vad_segments = vad_obj->Infer(pcm_data, input_finished);
}

// 2pass
void Audio::Split(VadModel* vad_obj, bool input_finished, ASR_TYPE asr_mode)
{
    AudioFrame *frame;

    frame = frame_queue.front();
    frame_queue.pop();
    int sp_len = frame->GetLen();
    delete frame;
    frame = NULL;

    std::vector<float> pcm_data(speech_data, speech_data+sp_len);
    vector<std::vector<int>> vad_segments = vad_obj->Infer(pcm_data, input_finished);

    speech_end += sp_len/seg_sample;
    if(vad_segments.size() == 0){
        if(speech_start != -1){
            int start = speech_start*seg_sample;
            int end = speech_end*seg_sample;
            int buff_len = end-start;
            int step = ONLINE_STEP;

            if(asr_mode != ASR_OFFLINE){
                if(buff_len >= step){
                    frame = new AudioFrame(step);
                    frame->data = (float*)malloc(sizeof(float) * step);
                    memcpy(frame->data, all_samples.data()+start-offset, step*sizeof(float));
                    asr_online_queue.push(frame);
                    frame = NULL;
                    speech_start += step/seg_sample;
                }
            }
        }
    }else{
        for(auto vad_segment: vad_segments){
            int speech_start_i=-1, speech_end_i=-1;
            if(vad_segment[0] != -1){
                speech_start_i = vad_segment[0];
            }
            if(vad_segment[1] != -1){
                speech_end_i = vad_segment[1];
            }

            // [1, 100]
            if(speech_start_i != -1 && speech_end_i != -1){
                int start = speech_start_i*seg_sample;
                int end = speech_end_i*seg_sample;

                if(asr_mode != ASR_OFFLINE){
                    frame = new AudioFrame(end-start);
                    frame->is_final = true;
                    frame->data = (float*)malloc(sizeof(float) * (end-start));
                    memcpy(frame->data, all_samples.data()+start-offset, (end-start)*sizeof(float));
                    asr_online_queue.push(frame);
                    frame = NULL;
                }

                if(asr_mode != ASR_ONLINE){
                    frame = new AudioFrame(end-start);
                    frame->is_final = true;
                    frame->data = (float*)malloc(sizeof(float) * (end-start));
                    memcpy(frame->data, all_samples.data()+start-offset, (end-start)*sizeof(float));
                    asr_offline_queue.push(frame);
                    frame = NULL;
                }

                speech_start = -1;
                speech_offline_start = -1;
            // [70, -1]
            }else if(speech_start_i != -1){
                speech_start = speech_start_i;
                speech_offline_start = speech_start_i;
                
                int start = speech_start*seg_sample;
                int end = speech_end*seg_sample;
                int buff_len = end-start;
                int step = ONLINE_STEP;

                if(asr_mode != ASR_OFFLINE){
                    if(buff_len >= step){
                        frame = new AudioFrame(step);
                        frame->data = (float*)malloc(sizeof(float) * step);
                        memcpy(frame->data, all_samples.data()+start-offset, step*sizeof(float));
                        asr_online_queue.push(frame);
                        frame = NULL;
                        speech_start += step/seg_sample;
                    }
                }

            }else if(speech_end_i != -1){ // [-1,100]
                if(speech_start == -1 or speech_offline_start == -1){
                    LOG(ERROR) <<"Vad start is null while vad end is available." ;
                    exit(-1);
                }

                int start = speech_start*seg_sample;
                int offline_start = speech_offline_start*seg_sample;
                int end = speech_end_i*seg_sample;
                int buff_len = end-start;
                int step = ONLINE_STEP;

                if(asr_mode != ASR_ONLINE){
                    frame = new AudioFrame(end-offline_start);
                    frame->is_final = true;
                    frame->data = (float*)malloc(sizeof(float) * (end-offline_start));
                    memcpy(frame->data, all_samples.data()+offline_start-offset, (end-offline_start)*sizeof(float));
                    asr_offline_queue.push(frame);
                    frame = NULL;
                }

                if(asr_mode != ASR_OFFLINE){
                    if(buff_len > 0){
                        for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
                            bool is_final = false;
                            if (sample_offset + step >= buff_len - 1) {
                                step = buff_len - sample_offset;
                                is_final = true;
                            }
                            frame = new AudioFrame(step);
                            frame->is_final = is_final;
                            frame->data = (float*)malloc(sizeof(float) * step);
                            memcpy(frame->data, all_samples.data()+start-offset+sample_offset, step*sizeof(float));
                            asr_online_queue.push(frame);
                            frame = NULL;
                        }
                    }else{
                        frame = new AudioFrame(0);
                        frame->is_final = true;
                        asr_online_queue.push(frame);
                        frame = NULL;
                    }
                }
                speech_start = -1;
                speech_offline_start = -1;
            }
        }
    }

    // erase all_samples
    int vector_cache = MODEL_SAMPLE_RATE*2;
    if(speech_offline_start == -1){
        if(all_samples.size() > vector_cache){
            int erase_num = all_samples.size() - vector_cache;
            all_samples.erase(all_samples.begin(), all_samples.begin()+erase_num);
            offset += erase_num;
        }
    }else{
        int offline_start = speech_offline_start*seg_sample;
         if(offline_start-offset > vector_cache){
            int erase_num = offline_start-offset - vector_cache;
            all_samples.erase(all_samples.begin(), all_samples.begin()+erase_num);
            offset += erase_num;
        }       
    }
    
}

} // namespace funasr