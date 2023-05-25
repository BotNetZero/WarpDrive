#!/usr/bin/env python
# -*- coding: utf-8 -*-

# usage: python -m src.test.test_communicater

import os
import torch
from torch.multiprocessing import Process

from src.distributed import new_process_group, send, recv, new_sub_group
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
    print(f"pid ({os.getpid()}) name ({ group.name }) -> Rank ({ group.rank }) has data ({ tensor })")


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    group = new_process_group("0", backend=backend, rank=rank, world_size=size)

    # subgroup1 = new_sub_group("1", group)
    subgroup1 = new_sub_group("1", group, ranks=[0, 1])

    fn(subgroup1)


def one_group_run():
    size = 1
    init_process(0, size, run)


def two_group_run():
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# usage: python -m src.test.test_communicater
if __name__ == "__main__":
    # one_group_run()
    two_group_run()
