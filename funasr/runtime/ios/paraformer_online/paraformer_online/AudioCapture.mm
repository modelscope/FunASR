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
#include "model_infer.hpp"

#define Recorder_Sample_Rate 16000 
#define Samples_Per_Frame (Recorder_Sample_Rate/100)

static AudioCapture *selfClass = nil;

@interface AudioCapture ()<AVCaptureAudioDataOutputSampleBufferDelegate>

@property (nonatomic, strong) AVCaptureSession *capture;

@property (nonatomic, copy) NSMutableData *sampleData;

@property (nonatomic, strong) AudioRecorder *audioRecorder;

@property (nonatomic, assign) BOOL isRecording;

@property (nonatomic, strong) NSLock *lock;

@end

@implementation AudioCapture
{
    const char* output_path;
    void* denoiser;
    void* resampler;
    
    paraformer_online::ModelInfer *infer_;
//    Paraformer_stream *stream_;
}
int packetIndex = 0;

- (instancetype)init
{
    self = [super init];
    if (self) {

        NSString * model_dir = @"model";
        NSString *model_path = [[NSBundle mainBundle] pathForResource:@"model" ofType:@".bin" inDirectory:model_dir];
        model_path = [model_path stringByReplacingOccurrencesOfString:@"model.bin" withString:@""];
        const char* model_file = [model_path UTF8String];

//        stream_ = new Paraformer_stream(model_file, vad_model_file);
        infer_ = new paraformer_online::ModelInfer(model_file);
        
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

//static FILE *pf_file_out = NULL;
//int audio_index = 0;
- (void)startRecorder {
    selfClass = self;
    
    __weak __typeof(self) weakSelf = self;
    [self.sampleData setLength:0];
    // audio record call-back
//    float speech[10 * 960];
    float *speech = (float *)calloc(10*960, sizeof(float));
    __block int speech_idx = 0;
    self.audioRecorder.inputBlock = ^(NSData *speechData) {
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong __typeof(weakSelf) strongSelf = weakSelf;
            if (weakSelf.isRecording) {
//                [weakSelf appendPCMData:speechData]; // DUBUG USE: append audio data, Memory increment
                const char* buffer = (const char*)speechData.bytes;
                int length = (int)speechData.length;
                
                float input_data[Samples_Per_Frame];
                CharToFloat(buffer, length, input_data);
                
                memcpy(speech+Samples_Per_Frame*speech_idx, input_data, Samples_Per_Frame*sizeof(float));
                
                // FIX: You can change it to a ring buffer
                if (speech_idx == 59) {
                    string result = strongSelf->infer_->forward(speech, Samples_Per_Frame*speech_idx);
                    memset(speech, 0, Samples_Per_Frame*speech_idx * sizeof(float));
                    speech_idx = 0;
                    if (weakSelf.resultBlock) {
                        weakSelf.resultBlock([NSString stringWithUTF8String:result.c_str()]);
                    }
                } else {
                    speech_idx++;
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
