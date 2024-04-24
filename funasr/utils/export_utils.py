import os
import torch


def export_onnx(model, data_in=None, quantize: bool = False, opset_version: int = 14, **kwargs):
    model_scripts = model.export(**kwargs)
    export_dir = kwargs.get("output_dir", os.path.dirname(kwargs.get("init_param")))
    os.makedirs(export_dir, exist_ok=True)

    if not isinstance(model_scripts, (list, tuple)):
        model_scripts = (model_scripts,)
    for m in model_scripts:
        m.eval()
        _onnx(
            m,
            data_in=data_in,
            quantize=quantize,
            opset_version=opset_version,
            export_dir=export_dir,
            **kwargs
        )
        print("output dir: {}".format(export_dir))

    return export_dir


def _onnx(
    model,
    data_in=None,
    quantize: bool = False,
    opset_version: int = 14,
    export_dir: str = None,
    **kwargs
):

    dummy_input = model.export_dummy_inputs()

    verbose = kwargs.get("verbose", False)

    export_name = model.export_name() if hasattr(model, "export_name") else "model.onnx"
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
        if not os.path.exists(quant_model_path):
            onnx_model = onnx.load(model_path)
            nodes = [n.name for n in onnx_model.graph.node]
            nodes_to_exclude = [
                m for m in nodes if "output" in m or "bias_encoder" in m or "bias_decoder" in m
            ]
            quantize_dynamic(
                model_input=model_path,
                model_output=quant_model_path,
                op_types_to_quantize=["MatMul"],
                per_channel=True,
                reduce_range=False,
                weight_type=QuantType.QUInt8,
                nodes_to_exclude=nodes_to_exclude,
            )
