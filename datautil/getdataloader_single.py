# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import numpy as np
from torch.utils.data import DataLoader

import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset

import datautil.actdata.cross_people as cross_people
import torch
from torch.utils.data import DataLoader, Subset

task_act = {'cross_people': cross_people}

def get_curriculum_loader(dataset, domain_labels, batch_size, domain_order=None, shuffle=True, num_workers=0):
    """
    Returns a list of DataLoaders, each with an increasing subset of domains.
    """
    if domain_order is None:
        domain_order = sorted(list(set(domain_labels)))
    loaders = []
    for k in range(1, len(domain_order) + 1):
        domains_included = domain_order[:k]
        indices = [i for i, d in enumerate(domain_labels) if d in domains_included]
        subset = Subset(dataset, indices)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers))
    return loaders

def get_dataloader(args, tr, val, tar):
    train_loader = DataLoader(dataset=tr, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    valid_loader = DataLoader(dataset=val, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    target_loader = DataLoader(dataset=tar, batch_size=args.batch_size,
                               num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader

def get_act_dataloader(args, return_dataset=False):
    source_datasetlist = []
    target_datalist = []
    pcross_act = task_act[args.task]

    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)
    for i, item in enumerate(tmpp):
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=actutil.act_train())
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)
            if len(tdata)/args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata)/args.batch_size
    rate = 0.2
    args.steps_per_epoch = int(args.steps_per_epoch*(1-rate))
    tdata = combindataset(args, source_datasetlist)
    l = len(tdata.labels)
    indexall = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l*rate)
    indextr, indexval = indexall[ted:], indexall[:ted]
    tr = subdataset(args, tdata, indextr)
    val = subdataset(args, tdata, indexval)
    targetdata = combindataset(args, target_datalist)
    train_loader, train_loader_noshuffle, valid_loader, target_loader = get_dataloader(
        args, tr, val, targetdata)
    
    # Patch: handle return_dataset for curriculum learning
    if return_dataset:
        return train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata
    else:
        return train_loader, train_loader_noshuffle, valid_loader, target_loader, None, None, None
