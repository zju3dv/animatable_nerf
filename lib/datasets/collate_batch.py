from torch.utils.data.dataloader import default_collate
import torch
import numpy as np


def meta_anisdf_collator(batch):
    batch = [[default_collate([b]) for b in batch_] for batch_ in batch]
    return batch


_collators = {'meta_anisdf': meta_anisdf_collator}


def make_collator(cfg, is_train):
    collator = cfg.train.collator if is_train else cfg.test.collator
    if collator in _collators:
        return _collators[collator]
    else:
        return default_collate
