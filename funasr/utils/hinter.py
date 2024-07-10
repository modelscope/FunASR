import sys
import torch.distributed
import logging

HINTED = set()


def hint_once(content, uid, rank=None):
    if (rank is None) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == rank:
        if uid not in HINTED:
            logging.info(content)
            HINTED.add(uid)

