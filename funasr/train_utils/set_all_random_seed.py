import random

import numpy as np
import torch


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
