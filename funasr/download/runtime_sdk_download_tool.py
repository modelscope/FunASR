import os
import argparse
from pathlib import Path

from funasr.utils.type_utils import str2bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--export-dir", type=str, required=True)
    parser.add_argument("--export", type=str2bool, default=True, help="whether to export model")
    parser.add_argument("--type", type=str, default="onnx", help='["onnx", "torchscript", "bladedisc"]')
    parser.add_argument("--device", type=str, default="cpu", help='["cpu", "cuda"]')
    parser.add_argument("--quantize", type=str2bool, default=False, help="export quantized model")
    parser.add_argument("--fallback-num", type=int, default=0, help="amp fallback number")
    parser.add_argument("--audio_in", type=str, default=None, help='["wav", "wav.scp"]')
    parser.add_argument("--model_revision", type=str, default=None, help="model_revision")
    parser.add_argument("--calib_num", type=int, default=200, help="calib max num")
    args = parser.parse_args()

    model_dir = args.model_name
    output_dir = args.model_name
    if not Path(args.model_name).exists():
        from modelscope.hub.snapshot_download import snapshot_download

        try:
            model_dir = snapshot_download(
                args.model_name, cache_dir=args.export_dir, revision=args.model_revision
            )
            output_dir = os.path.join(args.export_dir, args.model_name)
        except:
            raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
                model_dir
            )
    if args.export:
        model_file = os.path.join(model_dir, "model.onnx")
        if args.quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        if args.type == "torchscript":
            model_file = os.path.join(model_dir, "model.torchscript")
            args.device = "cuda"
        elif args.type == "bladedisc":
            model_file = os.path.join(model_dir, "model_blade.torchscript")
            args.device = "cuda"
        if not os.path.exists(model_file):
            print("model is not exist, begin to export " + model_file)
            from funasr import AutoModel

            export_model = AutoModel(model=args.model_name, output_dir=output_dir, device=args.device)
            export_model.export(
                    quantize=args.quantize,
                    type=args.type,
                    )


if __name__ == "__main__":
    main()
