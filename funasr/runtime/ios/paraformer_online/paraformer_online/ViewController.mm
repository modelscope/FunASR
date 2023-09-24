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

@property (strong, nonatomic) UIView *coverView;
@property (strong, nonatomic) UIButton *launchOnnxButton;
@property (strong, nonatomic) UIButton *launchButton;
@property (strong, nonatomic) UIActivityIndicatorView *activityIndicator;


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
    
    [self.view addSubview:self.coverView];
    [self initEngine:YES];
}

- (UIView *)coverView {
    if (!_coverView) {
        NSInteger screenWidth = self.view.bounds.size.width;
        NSInteger screenHeight = self.view.bounds.size.height;
        _coverView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, screenWidth, screenHeight)];
        _coverView.backgroundColor = [UIColor whiteColor];
        
        NSInteger x = 30;
        NSInteger heigh = 60;
        NSInteger y = screenHeight/2-120;
        NSInteger width = (screenWidth-30*2);
        
        UILabel *tip = [[UILabel alloc] initWithFrame:CGRectMake(x, y, width, heigh)];
        tip.backgroundColor = [UIColor clearColor];
        tip.textColor = [UIColor grayColor];
        tip.font = [UIFont systemFontOfSize:20];
        tip.textAlignment = NSTextAlignmentCenter;
        tip.text = @"正在初始化模型...";
        [_coverView addSubview:tip];
        _coverView.alpha = 0.8;
    }
    
    return _coverView;
}

- (void)initEngine:(BOOL)onnxModel {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSInteger screenWidth = self.view.bounds.size.width;
        NSInteger screenHeight = self.view.bounds.size.height;
        NSInteger activityIndicatorWith = screenWidth/2;
        NSInteger activityIndicatorHeight = activityIndicatorWith;
    //    UIActivityIndicatorView *activityIndicator = [[UIActivityIndicatorView alloc] initWithFrame:CGRectMake((screenWidth-activityIndicatorWith)/2, (screenHeight-activityIndicatorHeight)/2, activityIndicatorWith, activityIndicatorHeight)];
        self.activityIndicator = [[UIActivityIndicatorView alloc] initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleLarge];
        self.activityIndicator.frame = CGRectMake((screenWidth-activityIndicatorWith)/2, (screenHeight-activityIndicatorHeight)/2, activityIndicatorWith, activityIndicatorHeight);
        [self.view addSubview:self.activityIndicator];
        [self.activityIndicator startAnimating];
        
        
    });
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self initASR:onnxModel];
    });
}

- (void)initASR:(BOOL)onnxModel {
    if (is_mic_) {
        self.pre_partial_text = @"";
        self.audioCapture = [[AudioCapture alloc] initWithOnnxModel:onnxModel];
        __weak __typeof(self) weakSelf = self;

        self.audioCapture.resultBlock = ^(NSString *result) {
            if ([result length] == 0) {
                return;
            }
            NSLog(@"%@", result);
            NSString *text = nil;
            text = [NSString stringWithFormat:@"%@%@", weakSelf.pre_partial_text, result];
            weakSelf.pre_partial_text = text;
            dispatch_async(dispatch_get_main_queue(), ^{
                weakSelf.resultView.text = text;
                
            });
        };
    }
    
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.activityIndicator stopAnimating];
        [self.activityIndicator removeFromSuperview];
        [self.coverView removeFromSuperview];
    });
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
