/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  AudioCapture.h
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^SpeechRecognitionBlock)(NSString *result);

@interface AudioCapture : NSObject

@property (nonatomic,copy) SpeechRecognitionBlock resultBlock;

- (void)startRecorder;
- (void)stopRecorder;

@end

NS_ASSUME_NONNULL_END
