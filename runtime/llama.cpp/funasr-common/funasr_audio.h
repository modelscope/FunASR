// funasr_audio.h — load any audio file (WAV/MP3/FLAC/OGG...) as 16 kHz mono f32,
// via miniaudio's decoder (handles arbitrary sample rate, channels, bit depth).
// One TU must define FUNASR_AUDIO_IMPLEMENTATION before including.
#ifndef FUNASR_AUDIO_H
#define FUNASR_AUDIO_H
#include <vector>
// returns true on success; out = mono 16 kHz float samples in [-1,1]
bool funasr_load_audio_16k_mono(const char * path, std::vector<float> & out);
#endif

#ifdef FUNASR_AUDIO_IMPLEMENTATION
// keep only the decoder + data conversion (resampler/channel mixer); drop playback/capture
#define MA_NO_DEVICE_IO
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_NO_THREADING
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <cstdio>
bool funasr_load_audio_16k_mono(const char * path, std::vector<float> & out) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, 16000); // f32, mono, 16k
    ma_decoder dec;
    if (ma_decoder_init_file(path, &cfg, &dec) != MA_SUCCESS) {
        fprintf(stderr, "audio: failed to open/decode %s (supported: wav/mp3/flac)\n", path);
        return false;
    }
    ma_uint64 nframes = 0;
    if (ma_decoder_get_length_in_pcm_frames(&dec, &nframes) != MA_SUCCESS || nframes == 0) {
        // length unknown (e.g. some mp3): read in chunks
        out.clear();
        std::vector<float> buf(16000);
        ma_uint64 got;
        do { got = 0; ma_decoder_read_pcm_frames(&dec, buf.data(), buf.size(), &got);
             out.insert(out.end(), buf.begin(), buf.begin() + got); } while (got == buf.size());
    } else {
        out.resize(nframes);
        ma_uint64 got = 0;
        ma_decoder_read_pcm_frames(&dec, out.data(), nframes, &got);
        out.resize(got);
    }
    ma_decoder_uninit(&dec);
    return !out.empty();
}
#endif
