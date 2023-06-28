import torch


class SGD(torch.optim.SGD):
    """Thin inheritance of torch.optim.SGD to bind the required arguments, 'lr'

    Note that
    the arguments of the optimizer invoked by AbsTask.main()
    must have default value except for 'param'.

    I can't understand why only SGD.lr doesn't have the default value.
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
