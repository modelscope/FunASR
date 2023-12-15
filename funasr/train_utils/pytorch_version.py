import torch


def pytorch_cudnn_version() -> str:
    message = (
        f"pytorch.version={torch.__version__}, "
        f"cuda.available={torch.cuda.is_available()}, "
    )

    if torch.backends.cudnn.enabled:
        message += (
            f"cudnn.version={torch.backends.cudnn.version()}, "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark}, "
            f"cudnn.deterministic={torch.backends.cudnn.deterministic}"
        )
    return message
