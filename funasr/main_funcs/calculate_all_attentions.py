from collections import defaultdict
from typing import Dict
from typing import List

import torch

from funasr.modules.rnn.attentions import AttAdd
from funasr.modules.rnn.attentions import AttCov
from funasr.modules.rnn.attentions import AttCovLoc
from funasr.modules.rnn.attentions import AttDot
from funasr.modules.rnn.attentions import AttForward
from funasr.modules.rnn.attentions import AttForwardTA
from funasr.modules.rnn.attentions import AttLoc
from funasr.modules.rnn.attentions import AttLoc2D
from funasr.modules.rnn.attentions import AttLocRec
from funasr.modules.rnn.attentions import AttMultiHeadAdd
from funasr.modules.rnn.attentions import AttMultiHeadDot
from funasr.modules.rnn.attentions import AttMultiHeadLoc
from funasr.modules.rnn.attentions import AttMultiHeadMultiResLoc
from funasr.modules.rnn.attentions import NoAtt
from funasr.modules.attention import MultiHeadedAttention


from funasr.models.base_model import FunASRModel


@torch.no_grad()
def calculate_all_attentions(
    model: FunASRModel, batch: Dict[str, torch.Tensor]
) -> Dict[str, List[torch.Tensor]]:
    """Derive the outputs from the all attention layers

    Args:
        model:
        batch: same as forward
    Returns:
        return_dict: A dict of a list of tensor.
        key_names x batch x (D1, D2, ...)

    """
    bs = len(next(iter(batch.values())))
    assert all(len(v) == bs for v in batch.values()), {
        k: v.shape for k, v in batch.items()
    }

    # 1. Register forward_hook fn to save the output from specific layers
    outputs = {}
    handles = {}
    for name, modu in model.named_modules():

        def hook(module, input, output, name=name):
            if isinstance(module, MultiHeadedAttention):
                # NOTE(kamo): MultiHeadedAttention doesn't return attention weight
                # attn: (B, Head, Tout, Tin)
                outputs[name] = module.attn.detach().cpu()
            elif isinstance(module, AttLoc2D):
                c, w = output
                # w: previous concate attentions
                # w: (B, nprev, Tin)
                att_w = w[:, -1].detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, (AttCov, AttCovLoc)):
                c, w = output
                assert isinstance(w, list), type(w)
                # w: list of previous attentions
                # w: nprev x (B, Tin)
                att_w = w[-1].detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, AttLocRec):
                # w: (B, Tin)
                c, (w, (att_h, att_c)) = output
                att_w = w.detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(
                module,
                (
                    AttMultiHeadDot,
                    AttMultiHeadAdd,
                    AttMultiHeadLoc,
                    AttMultiHeadMultiResLoc,
                ),
            ):
                c, w = output
                # w: nhead x (B, Tin)
                assert isinstance(w, list), type(w)
                att_w = [_w.detach().cpu() for _w in w]
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(
                module,
                (
                    AttAdd,
                    AttDot,
                    AttForward,
                    AttForwardTA,
                    AttLoc,
                    NoAtt,
                ),
            ):
                c, w = output
                att_w = w.detach().cpu()
                outputs.setdefault(name, []).append(att_w)

        handle = modu.register_forward_hook(hook)
        handles[name] = handle

    # 2. Just forward one by one sample.
    # Batch-mode can't be used to keep requirements small for each models.
    keys = []
    for k in batch:
        if not k.endswith("_lengths"):
            keys.append(k)

    return_dict = defaultdict(list)
    for ibatch in range(bs):
        # *: (B, L, ...) -> (1, L2, ...)
        _sample = {
            k: batch[k][ibatch, None, : batch[k + "_lengths"][ibatch]]
            if k + "_lengths" in batch
            else batch[k][ibatch, None]
            for k in keys
        }

        # *_lengths: (B,) -> (1,)
        _sample.update(
            {
                k + "_lengths": batch[k + "_lengths"][ibatch, None]
                for k in keys
                if k + "_lengths" in batch
            }
        )
        model(**_sample)

        # Derive the attention results
        for name, output in outputs.items():
            if isinstance(output, list):
                if isinstance(output[0], list):
                    # output: nhead x (Tout, Tin)
                    output = torch.stack(
                        [
                            # Tout x (1, Tin) -> (Tout, Tin)
                            torch.cat([o[idx] for o in output], dim=0)
                            for idx in range(len(output[0]))
                        ],
                        dim=0,
                    )
                else:
                    # Tout x (1, Tin) -> (Tout, Tin)
                    output = torch.cat(output, dim=0)
            else:
                # output: (1, NHead, Tout, Tin) -> (NHead, Tout, Tin)
                output = output.squeeze(0)
            # output: (Tout, Tin) or (NHead, Tout, Tin)
            return_dict[name].append(output)
        outputs.clear()

    # 3. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return dict(return_dict)
