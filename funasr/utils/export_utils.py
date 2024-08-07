import os
import torch
import functools


def export(
    model, data_in=None, quantize: bool = False, opset_version: int = 14, type="onnx", **kwargs
):
    model_scripts = model.export(**kwargs)
    export_dir = kwargs.get("output_dir", os.path.dirname(kwargs.get("init_param")))
    os.makedirs(export_dir, exist_ok=True)

    if not isinstance(model_scripts, (list, tuple)):
        model_scripts = (model_scripts,)
    for m in model_scripts:
        m.eval()
        if type == "onnx":
            _onnx(
                m,
                data_in=data_in,
                quantize=quantize,
                opset_version=opset_version,
                export_dir=export_dir,
                **kwargs,
            )
        elif type == "torchscript":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Exporting torchscripts on device {}".format(device))
            _torchscripts(m, path=export_dir, device=device)
        elif type == "bladedisc":
            assert (
                torch.cuda.is_available()
            ), "Currently bladedisc optimization for FunASR only supports GPU"
            # bladedisc only optimizes encoder/decoder modules
            if hasattr(m, "encoder") and hasattr(m, "decoder"):
                _bladedisc_opt_for_encdec(m, path=export_dir, enable_fp16=True)
            else:
                _torchscripts(m, path=export_dir, device="cuda")
        print("output dir: {}".format(export_dir))

    return export_dir


def _onnx(
    model,
    data_in=None,
    quantize: bool = False,
    opset_version: int = 14,
    export_dir: str = None,
    **kwargs,
):

    dummy_input = model.export_dummy_inputs()

    verbose = kwargs.get("verbose", False)

    if isinstance(model.export_name, str):
        export_name = model.export_name + ".onnx"
    else:
        export_name = model.export_name()
    model_path = os.path.join(export_dir, export_name)
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=verbose,
        opset_version=opset_version,
        input_names=model.export_input_names(),
        output_names=model.export_output_names(),
        dynamic_axes=model.export_dynamic_axes(),
    )

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        import onnx

        quant_model_path = model_path.replace(".onnx", "_quant.onnx")
        onnx_model = onnx.load(model_path)
        nodes = [n.name for n in onnx_model.graph.node]
        nodes_to_exclude = [
            m for m in nodes if "output" in m or "bias_encoder" in m or "bias_decoder" in m
        ]
        print("Quantizing model from {} to {}".format(model_path, quant_model_path))
        quantize_dynamic(
            model_input=model_path,
            model_output=quant_model_path,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            nodes_to_exclude=nodes_to_exclude,
        )


def _torchscripts(model, path, device="cuda"):
    dummy_input = model.export_dummy_inputs()
    
    if device == "cuda":
        model = model.cuda()
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = dummy_input.cuda()
        else:
            dummy_input = tuple([i.cuda() for i in dummy_input])
    
    model_script = torch.jit.trace(model, dummy_input)
    if isinstance(model.export_name, str):
        model_script.save(os.path.join(path, f"{model.export_name}".replace("onnx", "torchscript")))
    else:
        model_script.save(os.path.join(path, f"{model.export_name()}".replace("onnx", "torchscript")))


def _bladedisc_opt(model, model_inputs, enable_fp16=True):
    model = model.eval()
    try:
        import torch_blade
    except Exception as e:
        print(
            f"Warning, if you are exporting bladedisc, please install it and try it again: pip install -U torch_blade\n"
        )
    torch_config = torch_blade.config.Config()
    torch_config.enable_fp16 = enable_fp16
    with torch.no_grad(), torch_config:
        opt_model = torch_blade.optimize(
            model,
            allow_tracing=True,
            model_inputs=model_inputs,
        )
    return opt_model


def _rescale_input_hook(m, x, scale):
    if len(x) > 1:
        return (x[0] / scale, *x[1:])
    else:
        return (x[0] / scale,)


def _rescale_output_hook(m, x, y, scale):
    if isinstance(y, tuple):
        return (y[0] / scale, *y[1:])
    else:
        return y / scale


def _rescale_encoder_model(model, input_data):
    # Calculate absmax
    absmax = torch.tensor(0).cuda()

    def stat_input_hook(m, x, y):
        val = x[0] if isinstance(x, tuple) else x
        absmax.copy_(torch.max(absmax, val.detach().abs().max()))

    encoders = model.encoder.model.encoders
    hooks = [m.register_forward_hook(stat_input_hook) for m in encoders]
    model = model.cuda()
    model(*input_data)
    for h in hooks:
        h.remove()

    # Rescale encoder modules
    fp16_scale = int(2 * absmax // 65536)
    print(f"rescale encoder modules with factor={fp16_scale}")
    model.encoder.model.encoders0.register_forward_pre_hook(
        functools.partial(_rescale_input_hook, scale=fp16_scale),
    )
    for name, m in model.encoder.model.named_modules():
        if name.endswith("self_attn"):
            m.register_forward_hook(functools.partial(_rescale_output_hook, scale=fp16_scale))
        if name.endswith("feed_forward.w_2"):
            state_dict = {k: v / fp16_scale for k, v in m.state_dict().items()}
            m.load_state_dict(state_dict)


def _bladedisc_opt_for_encdec(model, path, enable_fp16):
    # Get input data
    # TODO: better to use real data
    input_data = model.export_dummy_inputs()
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.cuda()
    else:
        input_data = tuple([i.cuda() for i in input_data])

    # Get input data for decoder module
    decoder_inputs = list()

    def get_input_hook(m, x):
        decoder_inputs.extend(list(x))

    hook = model.decoder.register_forward_pre_hook(get_input_hook)
    model = model.cuda()
    model(*input_data)
    hook.remove()

    # Prevent FP16 overflow
    if enable_fp16:
        _rescale_encoder_model(model, input_data)

    # Export and optimize encoder/decoder modules
    model.encoder = _bladedisc_opt(model.encoder, input_data[:2])
    model.decoder = _bladedisc_opt(model.decoder, tuple(decoder_inputs))
    model_script = torch.jit.trace(model, input_data)
    model_script.save(os.path.join(path, f"{model.export_name}_blade.torchscript"))
