#!/usr/bin/env python
# -*- coding: utf-8 -*-

# usage: python -m src.test.test_communicater

import os
import torch
from torch.multiprocessing import Process

from src.distributed import new_process_group, send, recv
from src.distributed.group import Group


def run(group: Group):
    """ Distributed function to be implemented later. """
    tensor = torch.zeros(1)
    if group.rank == 0:
        tensor += 1
        # Send the tensor to process 1
        send(tensor, group, 1)
    else:
        # Receive tensor from process 0
        recv(tensor, group, src=0)
    print(os.getpid(), '-> Rank ',group.rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    group = new_process_group("0", backend=backend, rank=rank, world_size=size)

    # mydist.init_process_group("1", backend, rank=rank, world_size=size)

    fn(group)

# usage: python -m src.test.test_communicater
if __name__ == "__main__":
    size = 2
    # size = 1
    processes = []
    # init_process(0, size, run)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
