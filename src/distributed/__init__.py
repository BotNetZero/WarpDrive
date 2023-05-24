#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Union
from datetime import timedelta
from typing import Union, Optional, Any

from torch.autograd.profiler import record_function
from torch.distributed import Backend, rendezvous, default_pg_timeout
from torch._C._distributed_c10d import Store, PrefixStore, ProcessGroup

from .group import Group
from .manager import _group_manager
from .helper import new_process_group_helper, store_based_barrier
from .communicater import send, recv, _check_group, _exist_group

logger = logging.getLogger(__name__)

__all__ = [
    "new_process_group",
    "close_process_group",
]


def close_process_group(group: Union[str, Group]):
    if group == None:
        return
    _group_manager.del_group(group)


def new_process_group(
    group_name: str,
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: timedelta = default_pg_timeout,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    pg_options: Optional[ProcessGroup.Options] = None,
) -> Group:
    if not isinstance(timeout, timedelta):
        raise RuntimeError(
            "Expected timeout argument to be of type" "datetime.timedelta"
        )

    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")
    
    # 目前仅支持 NCCL and GLOO backend
    # TODO 支持其他 backend: MPI ...
    if backend not in [Backend.NCCL, Backend.GLOO]:
        raise Exception("backend not supported")

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(
            init_method, rank, world_size, timeout=timeout
        )
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore("default_pg", store)

    group: Group = new_process_group_helper(
        world_size,
        rank,
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    # exec barrier
    # TODO 支持其他 backend: MPI ...
    store_based_barrier(group, timeout)

    return group


def new_sub_group(
    sub_group_name: str,
    parent_group: Group,
    ranks: List[int]=None,
    timeout: timedelta=default_pg_timeout,
    backend: Union[str, Backend]=None,
    pg_options: Optional[ProcessGroup.Options]=None
) -> Group:
    """
    Creates a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    .. warning::
        Using multiple process groups with the ``NCCL`` backend concurrently
        is not safe and the user should perform explicit synchronization in
        their application to ensure only one process group is used at a time.
        This means collectives from one process group should have completed
        execution on the device (not just enqueued since CUDA execution is
        async) before collectives from another process group are enqueued.
        See `Using multiple NCCL communicators concurrently <https://docs.nvid
        ia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using
        -multiple-nccl-communicators-concurrently>`_ for more details.
    """
    _check_group(parent_group)
    _exist_group(sub_group_name)

    default_pg = parent_group.pg
    default_backend, default_store = parent_group.backend, parent_group.store
    default_rank = default_pg.rank()
    default_world_size = default_pg.size()

    # Default to the same backend as the global process group
    # if the backend is not specified.
    if not backend:
        backend = default_backend

    # checks the input ranks
    if ranks is not None:
        ranks = sorted(ranks)
        sub_group_world_size = len(ranks)
        if sub_group_world_size > default_world_size:
            raise RuntimeError(
                "the new group's world size should be less or "
                "equal to the world size set by "
                "new_process_group"
            )
        # check ranks' sanity
        for rank in ranks:
            if rank < 0 or rank >= default_world_size:
                raise RuntimeError(
                    "The new group's rank should be within the "
                    "the world_size set by new_process_group"
                )
        if default_rank in ranks:
            sub_group_rank = ranks.index(default_rank)
        else:
            sub_group_rank = None
    else:
        ranks = list(range(default_world_size))
        sub_group_world_size = default_world_size
        sub_group_rank = default_rank

    backend = Backend(backend)

    with record_function(f"## process_group:init with ranks: {ranks}"):
        group: Group = new_process_group_helper(
            sub_group_world_size,
            sub_group_rank,
            # ranks,
            backend,
            default_store,
            group_name=sub_group_name,
            group_level=1,
            parent_name=parent_group.name,
            pg_options=pg_options,
            timeout=timeout,
        )

    # exec barrier
    # TODO 支持其他 backend: MPI ...
    store_based_barrier(group, timeout)

    return group
