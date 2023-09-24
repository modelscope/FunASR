/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  AudioRecorder.h
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#import <Foundation/Foundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <AVFoundation/AVFoundation.h>


NS_ASSUME_NONNULL_BEGIN

#define RECORDER_NOTIFICATION_CALLBACK_NAME @"recorderNotificationCallBackName"
#define kNumberAudioQueueBuffers 3 //缓冲区设定3个
#define kDefaultSampleRate 16000    //采样率
#define kBufferByteSize (kDefaultSampleRate/100*2)//960
#define kDataSize 2048

typedef void (^AudioRecorder_inputBlock)(NSData *speechData);

@interface AudioRecorder : NSObject
{
    AudioQueueRef _audioQueue;
    AudioStreamBasicDescription audioFormat;
    AudioQueueBufferRef _audioBuffers[kNumberAudioQueueBuffers];
    char data[kDataSize];
    int offset;

}

@property (nonatomic, assign) BOOL isRecording;
@property (atomic, assign) NSUInteger sampleRate;

@property (nonatomic,copy) AudioRecorder_inputBlock inputBlock;

-(void)start;

-(void)stop;

@end

NS_ASSUME_NONNULL_END
