/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  ViewController.m
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#import "ViewController.h"

#include "AudioCapture.h"

@interface ViewController ()<UITextViewDelegate>

@property (nonatomic, strong) AudioCapture *audioCapture;
@property (nonatomic, copy) NSString *pre_partial_text;

@property (weak, nonatomic) IBOutlet UITextView *resultView;
@property (weak, nonatomic) IBOutlet UIButton *startButton;


@end

@implementation ViewController
{
    bool is_mic_;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    self.startButton.layer.cornerRadius = 60;
    self.startButton.layer.masksToBounds = YES;
    self.startButton.selected = NO;
    self.resultView.editable = NO;
    
    is_mic_ = true;
    
    if (is_mic_) {
        self.pre_partial_text = @"";
        self.audioCapture = [[AudioCapture alloc] init];
        __weak __typeof(self) weakSelf = self;

        self.audioCapture.resultBlock = ^(NSString *result) {
            if ([result length] == 0) {
                return;
            }
            NSString *text = nil;
            text = [NSString stringWithFormat:@"%@%@", weakSelf.pre_partial_text, result];
            weakSelf.pre_partial_text = text;
            dispatch_async(dispatch_get_main_queue(), ^{
                weakSelf.resultView.text = text;
                
            });
        };
    }
}

- (IBAction)startButtonClicked:(id)sender {
    self.startButton.enabled = NO;
    if (!self.startButton.selected) {
        if (is_mic_) [self.audioCapture startRecorder];
    } else {
        if (is_mic_) [self.audioCapture stopRecorder];
    }

    self.startButton.selected = !self.startButton.selected;
    self.startButton.enabled = YES;
}

- (void)textViewDidChange:(UITextView *)textView {
    [textView scrollRangeToVisible:NSMakeRange([textView.text length] - 1, 1)];
}


@end
