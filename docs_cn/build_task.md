# 搭建自定义任务
FunASR类似ESPNet，以`Task`为通用接口，从而实现模型的训练和推理。每一个`Task`是一个类，其需要继承`AbsTask`，其对应的具体代码见`funasr/tasks/abs_task.py`。下面给出其包含的主要函数及功能介绍：
```python
class AbsTask(ABC):
    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        pass
    
    @classmethod
    def build_preprocess_fn(cls, args, train):
        (...)
    
    @classmethod
    def build_collate_fn(cls, args: argparse.Namespace):
        (...)

    @classmethod
    def build_model(cls, args):
        (...)
    
    @classmethod
    def main(cls, args):
        (...)
```
- add_task_arguments：添加特定`Task`需要的参数
- build_preprocess_fn：定义如何处理对样本进行预处理
- build_collate_fn：定义如何将多个样本组成一个`batch`
- build_model：定义模型
- main：训练入口，通过`Task.main()`来启动训练

下面我们将以语音识别任务为例，介绍如何定义一个新的`Task`，具体代码见`funasr/tasks/asr.py`中的`ASRTask`。 定义新的`Task`的过程，其实就是根据任务需求，重定义上述函数的过程。
- add_task_arguments
```python
@classmethod
def add_task_arguments(cls, parser: argparse.ArgumentParser):
    group = parser.add_argument_group(description="Task related")
    group.add_argument(
        "--token_list",
        type=str_or_none,
        default=None,
        help="A text mapping int-id to token",
    )
    (...)
```
对于语音识别任务，需要的特定参数包括`token_list`等。根据不同任务的特定需求，用户可以在此函数中定义相应的参数。

- build_preprocess_fn
```python
@classmethod
def build_preprocess_fn(cls, args, train):
    if args.use_preprocessor:
        retval = CommonPreprocessor(
                    train=train,
                    token_type=args.token_type,
                    token_list=args.token_list,
                    bpemodel=args.bpemodel,
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_cleaner=args.cleaner,
                    ...
                )
    else:
        retval = None
    return retval
```
该函数定义了如何对样本进行预处理。具体地，语音识别任务的输入包括音频和抄本。对于音频，在此实现了(可选)对音频加噪声，加混响等功能；对于抄本，在此实现了(可选)根据bpe处理抄本，将抄本映射成`tokenid`等功能。用户可以自己选择需要对样本进行的预处理操作，实现方法可以参考`CommonPreprocessor`。

- build_collate_fn
```python
@classmethod
def build_collate_fn(cls, args, train):
    return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
```
该函数定义了如何将多个样本组成一个`batch`。对于语音识别任务，在此实现的是将不同的音频和抄本，通过`padding`的方式来得到等长的数据。具体地，自此我们默认用`0.0`来作为音频的填充值，用`-1`作为抄本的默认填充值。用户可以在此定义不同的组`batch`操作，实现方法可以参考`CommonCollateFn`。

- build_model
```python
@classmethod
def build_model(cls, args, train):
    with open(args.token_list, encoding="utf-8") as f:
        token_list = [line.rstrip() for line in f]
        vocab_size = len(token_list)
        frontend = frontend_class(**args.frontend_conf)
        specaug = specaug_class(**args.specaug_conf)
        normalize = normalize_class(**args.normalize_conf)
        preencoder = preencoder_class(**args.preencoder_conf)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)
        postencoder = postencoder_class(input_size=encoder_output_size, **args.postencoder_conf)
        decoder = decoder_class(vocab_size=vocab_size, encoder_output_size=encoder_output_size,  **args.decoder_conf)
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf)
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            token_list=token_list,
            **args.model_conf,
        )
    return model
```
该函数定义了具体的模型。对于不同的语音识别模型，往往可以共用同一个语音识别`Task`，然后在此函数中定义特定的模型。例如，这里给出的是一个标准的encoder-decoder结构的语音识别模型。具体地，先定义该模型的各个模块，包括encoder，decoder等，然后在将这些模块组合在一起得到一个完整的模型。在FunASR中，模型需要继承`AbsESPnetModel`，其具体代码见`funasr/train/abs_espnet_model.py`，主要需要实现的是`forward`函数。
