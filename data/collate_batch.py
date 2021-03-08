# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):

    if len(list(zip(*batch))) == 3:
        imgs, pids, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        print(torch.stack(imgs, dim=0).shape)
        return torch.stack(imgs, dim=0), pids
    
    elif len(list(zip(*batch))) == 4:
        imgs, pids, _, real_forg  = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, real_forg


def val_collate_fn(batch):
    imgs, pids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids
