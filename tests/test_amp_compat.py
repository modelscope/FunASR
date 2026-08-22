import pytest
import torch

from funasr.utils.amp import autocast


@pytest.mark.parametrize("args", [(False,), (False, torch.float16, False)])
def test_autocast_preserves_legacy_positional_arguments(args):
    with autocast(*args):
        pass
