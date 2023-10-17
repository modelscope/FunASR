/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  AudioRecorder.m
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#import "AudioRecorder.h"



@implementation AudioRecorder

- (id)init
{
    self = [super init];
    if (self) {

        self.sampleRate = kDefaultSampleRate;

        //设置录音 初始化录音参数
        [self setupAudioFormat:kAudioFormatLinearPCM SampleRate:(int)self.sampleRate];

    }
    return self;
}
//设置录音 初始化录音参数
- (void)setupAudioFormat:(UInt32)inFormatID SampleRate:(int)sampeleRate
{

    memset(&audioFormat, 0, sizeof(audioFormat));

    audioFormat.mSampleRate = sampeleRate;//采样率
    audioFormat.mChannelsPerFrame = 1;//单声道

    audioFormat.mFormatID = inFormatID;//采集pcm 格式
    audioFormat.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    audioFormat.mBitsPerChannel = 16;//每个通道采集2个byte
    audioFormat.mBytesPerPacket = audioFormat.mBytesPerFrame = (audioFormat.mBitsPerChannel / 8) * audioFormat.mChannelsPerFrame;
    audioFormat.mFramesPerPacket = 1;

}

//回调函数 不断采集声音。
void inputHandler(void *inUserData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer, const AudioTimeStamp *inStartTime,UInt32 inNumPackets, const AudioStreamPacketDescription *inPacketDesc)
{

    AudioRecorder *recorder = (__bridge AudioRecorder*)inUserData;

    int k = 0;
    if (inBuffer->mAudioDataByteSize > 0) {
//        NSLog(@"size: %d", inBuffer->mAudioDataByteSize);
        memcpy(recorder->data+recorder->offset,inBuffer->mAudioData,inBuffer->mAudioDataByteSize);//通过recorder->offset 偏移把语音数据存入recorder->data

        recorder->offset+=inBuffer->mAudioDataByteSize;//记录语音数据的大小

        k = recorder->offset/kBufferByteSize;//计算语音数据有多个960个字节语音

        for(int i = 0; i <k; i++)
        {

            NSData *SpeechData = [[NSData alloc]initWithBytes:recorder->data+i*kBufferByteSize length:kBufferByteSize];//把recorder->data 数据以960个字节分切放入 传出的数组中。
            [[NSNotificationCenter defaultCenter] postNotificationName:RECORDER_NOTIFICATION_CALLBACK_NAME object:SpeechData];
            if (recorder.inputBlock) {
                recorder.inputBlock(SpeechData);
            }
        }

//        NSLog(@"sampleRate: %lu", (unsigned long)recorder->_sampleRate);
        memcpy(recorder->data,recorder->data+k*kBufferByteSize,recorder->offset-(k*kBufferByteSize));//把剩下的语音数据放入原来的数组中

        recorder->offset-=(k*kBufferByteSize);//计算剩下的语音数据大小


    }

    if (recorder.isRecording) {

        AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, NULL);

    }

}
//开始录音
-(void)start
{
    NSError *error = nil;

    //录音的设置和初始化
    BOOL ret = [[AVAudioSession sharedInstance] setCategory:AVAudioSessionCategoryPlayAndRecord error:&error];
    if (!ret) {
        return;
    }

    //启用audio session
    ret = [[AVAudioSession sharedInstance] setActive:YES error:&error];
    if (!ret)
    {
        return;
    }


    //初始化缓冲语音数据数组
    memset(data,0,kDataSize); // 清空
    offset = 0;
    audioFormat.mSampleRate = self.sampleRate;
    //初始化音频输入队列
    AudioQueueNewInput(&audioFormat, inputHandler, (__bridge void *)(self), NULL, NULL, 0, &_audioQueue);

    int bufferByteSize = kBufferByteSize; //设定采集一帧960个字节

    //创建缓冲器
    for (int i = 0; i < kNumberAudioQueueBuffers; ++i){
        AudioQueueAllocateBuffer(_audioQueue, bufferByteSize, &_audioBuffers[i]);
        AudioQueueEnqueueBuffer(_audioQueue, _audioBuffers[i], 0, NULL);
    }

    //开始录音
    AudioQueueStart(_audioQueue, NULL);
    self.isRecording = YES;
}

//结束录音
-(void)stop
{
    if (self.isRecording) {
        self.isRecording = NO;
        AudioQueueStop(_audioQueue, true);
        AudioQueueDispose(_audioQueue, true);
        [[AVAudioSession sharedInstance] setActive:NO error:nil];
        [[AVAudioSession sharedInstance] setCategory:AVAudioSessionCategorySoloAmbient error:nil];
    }
}

@end
