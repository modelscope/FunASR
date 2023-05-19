# Build custom tasks
FunASR is similar to ESPNet, which applies `Task`  as the general interface ti achieve the training and inference of models. Each `Task` is a class inherited from `AbsTask` and its corresponding code can be seen in `funasr/tasks/abs_task.py`. The main functions of `AbsTask` are shown as follows:
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
- add_task_arguments：Add parameters required by a specified `Task`
- build_preprocess_fn：定义如何处理对样本进行预处理 define how to preprocess samples
- build_collate_fn：define how to combine multiple samples into a `batch`
- build_model：define the model
- main：training interface, starting training through `Task.main()`

Next, we take the speech recognition as an example to introduce how to define a new `Task`. For the corresponding code, please see `ASRTask` in `funasr/tasks/asr.py`. The procedure of defining a new `Task` is actually the procedure of redefining the above functions according to the requirements of the specified `Task`.

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
For speech recognition tasks, specific parameters required include `token_list`, etc. According to the specific requirements of different tasks, users can define corresponding parameters in this function.

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
This function defines how to preprocess samples. Specifically, the input of speech recognition tasks includes speech and text. For speech, functions such as (optional) adding noise and reverberation to the speech are supported. For text, functions such as (optional) processing text according to bpe and mapping text to `tokenid` are supported. Users can choose the preprocessing operation that needs to be performed on the sample. For the detail implementation, please refer to `CommonPreprocessor`.

- build_collate_fn
```python
@classmethod
def build_collate_fn(cls, args, train):
    return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
```
This function defines how to combine multiple samples into a `batch`. For speech recognition tasks, `padding` is employed to obtain equal-length data from different speech and text. Specifically, we set `0.0` as the default padding value for speech and `-1` as the default padding value for text. Users can define different `batch` operations here. For the detail implementation, please refer to `CommonCollateFn`.

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
This function defines the detail of the model. For different speech recognition models, the same speech recognition `Task` can usually be shared and the remaining thing needed to be done is to define a specific model in this function. For example, a speech recognition model with a standard encoder-decoder structure has been shown above. Specifically, it first defines each module of the model, including encoder, decoder, etc. and then combine these modules together to generate a complete model. In FunASR, the model needs to inherit `FunASRModel` and the corresponding code can be seen in `funasr/train/abs_espnet_model.py`. The main function needed to be implemented is the `forward` function.

Next, we take `SANMEncoder` as an example to introduce how to use a custom encoder as a part of the model when defining the specified model and the corresponding code can be seen in `funasr/models/encoder/sanm_encoder.py`. For a custom encoder, in addition to inheriting the common encoder class `AbsEncoder`, it is also necessary to define the `forward` function to achieve the forward computation of the `encoder`. After defining the `encoder`, it should also be registered in the `Task`. The corresponding code example can be seen as below:
```python
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
        mfcca_enc=MFCCAEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
```
In this code, `sanm=SANMEncoder` takes the newly defined `SANMEncoder` as an optional choice of the `encoder`. Once the user specifies the `encoder` as `sanm` in the configuration file, the `SANMEncoder` will be correspondingly employed as the `encoder` module of the model.