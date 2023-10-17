/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  AudioCapture.m
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#import "AudioCapture.h"
#import <AVFoundation/AVFoundation.h>
#include <thread>

#import "AudioRecorder.h"

#include "precomp.h"

#define Recorder_Sample_Rate 16000 
#define Samples_Per_Frame (Recorder_Sample_Rate/100)

#define k_input_frames 10
#define k_left_padding_frames 5
#define k_right_padding_frames 5
#define k_input_samples 960 // (60ms)


static AudioCapture *selfClass = nil;

@interface AudioCapture ()<AVCaptureAudioDataOutputSampleBufferDelegate>

@property (nonatomic, strong) AVCaptureSession *capture;

@property (nonatomic, copy) NSMutableData *sampleData;

@property (nonatomic, strong) AudioRecorder *audioRecorder;

@property (nonatomic, assign) BOOL isRecording;

@property (nonatomic, strong) NSLock *lock;

@end

using namespace funasr;

@implementation AudioCapture
{
    const char* output_path;
    void* denoiser;
    void* resampler;
    
//    Paraformer_stream *stream_;
    
    FUNASR_HANDLE asr_handle_;
    FUNASR_HANDLE online_handle_;
    
    bool is_onnx;
    
    char *speech_buff;
}
int packetIndex = 0;

- (id)initWithOnnxModel:(BOOL)onnxModel
{
    self = [super init];
    if (self) {
        is_onnx = onnxModel ? true : false;
        
        if (is_onnx) {
            [self initASROnnx];
        } else {
//            [self initASR];
        }
        self.isRecording = NO;
        
        NSLog(@"model init done!");
    }
    return self;
}

void S16ToFloatS16_1(const int16_t* src, size_t size, float* dest) {
    for (size_t i = 0; i < size; ++i)
        dest[i] = (float)src[i];
}

void CharToFloat(const char* src, size_t size, float* dst) {
    const int16_t* sample_data = reinterpret_cast<const int16_t*>(src);
    S16ToFloatS16_1(sample_data, size/2, dst);
}

void CharToFloat_1(const char* src, size_t size, float* dst) {
    const int16_t* sample_data = reinterpret_cast<const int16_t*>(src);
//    length = length/2;
    float data_f[Samples_Per_Frame] = {0.0};
    S16ToFloatS16_1(sample_data, size/2, data_f);
    
//    float data_f_norm[480];
    for (int i = 0; i < Samples_Per_Frame; i++) {
        dst[i] = data_f[i] / 32767.0;
    }
}

void CharToS16(const char* src, size_t src_size, short* dst) {
    const int16_t* sample_data = reinterpret_cast<const int16_t*>(src);
    for (int i = 0; i < src_size/2; i++) {
        dst[i] = sample_data[i];
    }
}

- (void)initASROnnx {
    NSString *model_file_path = [[NSBundle mainBundle] pathForResource:@"config" ofType:@".yaml" inDirectory:@"model"];
    model_file_path = [model_file_path stringByReplacingOccurrencesOfString:@"config.yaml" withString:@""];
    const char* model_dir= [model_file_path UTF8String];
    
    std::map<std::string, std::string> model_path;
    model_path.insert({MODEL_DIR, model_dir});
    model_path.insert({QUANTIZE, "true"});
    
    struct timeval start, end;
//    gettimeofday(&start, NULL);
    int thread_num = 1;
    asr_handle_ = FunASRInit(model_path, thread_num, ASR_ONLINE);

    if (!asr_handle_)
    {
        std::cout << "FunVad init failed" << std::endl;
    }

//    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    std::cout << "Model initialization takes " << (double)modle_init_micros / 1000000 << " s" << std::endl;

    string default_id = "wav_default_id";

    // init online features
    online_handle_ = FunASROnlineInit(asr_handle_);
    
}

//static FILE *pf_file_out = NULL;
//int audio_index = 0;
- (void)startRecorder {
    selfClass = self;
    
    __weak __typeof(self) weakSelf = self;
    [self.sampleData setLength:0];
    // audio record call-back
//    float speech[10 * 960];
    speech_buff = (char *)calloc(10*960*2, sizeof(char));
    __block int speech_idx = 0;
    self.audioRecorder.inputBlock = ^(NSData *speechData) {
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong __typeof(weakSelf) strongSelf = weakSelf;
            if (weakSelf.isRecording) {
//                [weakSelf appendPCMData:speechData]; // DUBUG USE: append audio data, Memory increment
                const char* buffer = (const char*)speechData.bytes;
                int length = (int)speechData.length;
                
                if (strongSelf->is_onnx) {
//                    char speech_buff[9600*2];
                    int step = 9600*2;
                    
                    memcpy(speech_buff+length*speech_idx, buffer, length*sizeof(char));
                    
                    // FIX: You can change it to a ring buffer
                    if (speech_idx == k_input_frames*6-1) {
                        FUNASR_RESULT result = FunASRInferBuffer(strongSelf->online_handle_, speech_buff, step, RASR_NONE, NULL, false, 16000);
                        
                        memset(speech_buff, 0, step * sizeof(char));
                        speech_idx = 0;
                        
                        if (result)
                        {
                            string msg = FunASRGetResult(result, 0);
//                            std::cout<<msg << std::endl;
                            FunASRFreeResult(result);
                            
                            if (strongSelf.resultBlock) {
                                strongSelf.resultBlock([NSString stringWithUTF8String:msg.c_str()]);
                            }
                        }
                        
                        
                    } else {
                        speech_idx++;
                    }
                    
                    
                } else {
                    
                    
                }
            }
        });
    };
    [self.audioRecorder start];
    self.isRecording = YES;
}

- (void)pushData {
}

- (void)stopRecorder {
    self.isRecording = NO;
    if (is_onnx) {
        FunASRUninit(asr_handle_);
        FunASRUninit(online_handle_);
    }
    
    free(speech_buff);
    
    [self.audioRecorder stop];
    [self writeData];
    
    /////
//    const char* buffer = (const char*)self.sampleData.bytes;
//    int length = (int)self.sampleData.length;
//
//    float *input_data = (float *)malloc(sizeof(float) * (length/2));
//    CharToFloat(buffer, length, input_data);
//
//    std::string msg = stream_->Process(input_data, length/2);
//    NSString *result = [NSString stringWithUTF8String:msg.c_str()];
//
//    if (self.resultBlock) {
//        self.resultBlock(result);
//    }
//
//    free(input_data);
    
}

- (BOOL)writeData {
    return [self.sampleData writeToFile:[self getPCMPath] atomically:NO];
}

- (void)appendPCMData:(NSData *)pcmData {
    [self audioProcessing:pcmData];
}

- (void)audioProcessing:(NSData *)data {
    [self.sampleData appendData:data];
}

- (NSMutableData *)sampleData {
    if (!_sampleData) {
        _sampleData = [NSMutableData data];
    }

    return _sampleData;
}

- (AudioRecorder *)audioRecorder {
    if (!_audioRecorder) {
        _audioRecorder = [[AudioRecorder alloc] init];
    }
    
    return _audioRecorder;
}

- (NSLock *)lock {
    if (!_lock) {
        _lock = [NSLock new];
    }

    return _lock;
}

- (NSString *)getPCMPath {
    NSString *directoryS = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
    NSString *directory = [directoryS stringByAppendingPathComponent:@"mic_ori.pcm"];
    return directory;
}


@end
