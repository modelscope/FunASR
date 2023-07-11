#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>

extern "C" {
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

int main(int argc, char* argv[]) {
    // from buff
    FILE* fp;
    fp = fopen(argv[1], "rb");
    if (fp == nullptr)
	{
        return -1;
	}
    fseek(fp, 0, SEEK_END);
    uint32_t n_file_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char* buf = (char *)malloc(n_file_len);
    memset(buf, 0, n_file_len);
    fread(buf, 1, n_file_len, fp);
    fclose(fp);

    AVIOContext* avio_ctx = avio_alloc_context(
        (unsigned char*)buf, // buffer
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
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return -1;
    }

    // from file
    // AVFormatContext* formatContext = avformat_alloc_context();
    // if (avformat_open_input(&formatContext, argv[1], NULL, NULL) != 0) {
    //     printf("Error: Could not open input file.");
    //     avformat_close_input(&formatContext);
    //     avformat_free_context(formatContext);
    //     return -1;
    // }


    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        printf("Error: Could not find stream information.");
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return -1;
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
        return -1;
    }
    if (avcodec_parameters_to_context(codecContext, codecParameters) != 0) {
        printf("Error: Could not copy codec parameters to codec context.");
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return -1;
    }
    if (avcodec_open2(codecContext, codec, NULL) < 0) {
        printf("Error: Could not open audio decoder.");
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return -1;
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
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return -1;
    }
    if (swr_init(swr_ctx) != 0) {
        std::cerr << "Could not initialize resampler" << std::endl;
        swr_free(&swr_ctx);
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(formatContext);
        return -1;
    }

    // to pcm
    FILE *out_file = fopen("output.pcm", "wb");
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<uint8_t> resampled_buffer;
    while (av_read_frame(formatContext, packet) >= 0) {
        if (packet->stream_index == audioStreamIndex) {
            if (avcodec_send_packet(codecContext, packet) >= 0) {
                while (avcodec_receive_frame(codecContext, frame) >= 0) {
                    // Resample audio if necessary
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
                    fwrite(resampled_buffer.data(), sizeof(int8_t), resampled_size, out_file);
                }
            }
        }
        av_packet_unref(packet);
    }
    fclose(out_file);

    avio_context_free(&avio_ctx);
    avformat_close_input(&formatContext);
    avformat_free_context(formatContext);
    avcodec_free_context(&codecContext);
    swr_free(&swr_ctx);
    av_packet_free(&packet);
    av_frame_free(&frame);
}
