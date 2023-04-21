
#ifndef COMDEFINE_H
#define COMDEFINE_H

#define S_BEGIN  0
#define S_MIDDLE 1
#define S_END    2
#define S_ALL    3
#define S_ERR    4

#ifndef MODEL_SAMPLE_RATE
#define MODEL_SAMPLE_RATE 16000
#endif

#ifndef VAD_SILENCE_DYRATION
#define VAD_SILENCE_DYRATION 15000
#endif

#ifndef VAD_MAX_LEN
#define VAD_MAX_LEN 800
#endif

#ifndef VAD_SPEECH_NOISE_THRES
#define VAD_SPEECH_NOISE_THRES 0.9
#endif


#endif
