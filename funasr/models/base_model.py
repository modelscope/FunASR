import torch


class FunASRModel(torch.nn.Module):
    """The common model class

    """

    def __init__(self):
        super().__init__()
        self.num_updates = 0

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def get_num_updates(self):
        return self.num_updates
