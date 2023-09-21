# Build custom models
Although FunASR has already provided various models, users can also customize their own models based on FunASR. Before introducing how to customize models, we assume that users are already familiar with the training and testing in FunASR. For users who are not familiar with this, please read [here](https://alibaba-damo-academy.github.io/FunASR/en/academic_recipe/asr_recipe.html) first.

The customization of the model is implemented through `funasr/build_utils/build_model.py`, which is as follows:
```python
def build_model(args):
    if args.task_name == "asr":
        model = build_asr_model(args)
    elif args.task_name == "pretrain":
        model = build_pretrain_model(args)
    elif args.task_name == "lm":
        model = build_lm_model(args)
    elif args.task_name == "punc":
        model = build_punc_model(args)
    elif args.task_name == "vad":
        model = build_vad_model(args)
    elif args.task_name == "diar":
        model = build_diar_model(args)
    elif args.task_name == "sv":
        model = build_sv_model(args)
    elif args.task_name == "ss":
        model = build_ss_model(args)
    else:
        raise NotImplementedError("Not supported task: {}".format(args.task_name))

    return model

```
We have implemented different model customization codes for corresponding tasks. If the model for the user's task has been already included, users can directly reuse the corresponding model customization codes (with minor modifications if necessary). Otherwise, users can implement the model customization code for a new task referring to the existing codes.

Next, we will take `paraformer` as an example to introduce how to customize a model in detail. Assume we have already implement `conformer` by the function `build_asr_model` and we want to implement `paraformer`. The main code of `build_asr_model` function is like follows (the detail code can be seen in `funasr/build_utils/build_asr_model.py`):
```python
...
# encoder
encoder_class = encoder_choices.get_class("conformer")
encoder = encoder_class(input_size=input_size, **encoder_conf)
# decoder
decoder_class = decoder_choices.get_class("transformer")
decoder = decoder_class(vocab_size=vocab_size, encoder_output_size=encoder.output_size(), **decoder_conf)
...
model_class = model_choices.get_class("asr")
model = model_class(
    ...,
    encoder=encoder,
    decoder=decoder,
    **model_conf,
)
```
As you can see, we first define the components of `conformer`, including the encoder, decoder and so on. Then, these components are passed as parameters to define `conformer`. In FunASR, we use `ClassChoices` (the detail code can be seen in funasr/train/class_choices.py) to implement each component. We take "model_choices" as an example and the following is its definition:
```python
model_choices = ClassChoices(
    "model",
    classes=dict(
        asr=ASRModel,
        ...,
    ),
    default="asr",
)
```
Users can define different models by adding key-value pairs in `classes` like `asr=ASRModel`. Here `asr` is a string parameter to specify the model type and it denotes the standard encoder-decoder ASR model in FunASR. Then the usage is as follows:
```python
model_class = model_choices.get_class("asr")
model = model_class(
    ...,
    encoder=encoder,
    decoder=decoder,
    **model_conf,
)
```
We use `get_class` function and `asr` parameter to get `ASRModel`, which is then used to instantiate the model. Compared to `conformer`, `paraformer` has an additional component, namely the predictor, in addition to the encoder and decoder. As described above, we should define all the components of `paraformer` first. For the encoder and decoder, we should specify the component type as `encoder_class = encoder_choices.get_class("conformer")` and `decoder_class=decoder_choices.get_class("paraformer_decoder_san")`. For the additional `predictor`, we should first implement predictor_choices` like follows:
```python
predictor_choices = ClassChoices(
    name="predictor",
    classes=dict(
        cif_predictor=CifPredictor,
        ...,
    ),
    default="cif_predictor",
    optional=True,
)
```
Then we can specify the predictor like the encoder and decoder as `predictor_class = predictor_choices.get_class("cif_predictor")`. Once define all the componts of `paraformer`, we can finally define `paraformer` as follows:
```python
model_class = model_choices.get_class("paraformer")
model = model_class(
    ...,
    encoder=encoder,
    decoder=decoder,
    predictor=predictor,
    **model_conf,
)
```