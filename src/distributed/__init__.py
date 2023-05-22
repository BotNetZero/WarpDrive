#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import timedelta
from typing import Union, Optional, Any

from torch.distributed import Backend, rendezvous, default_pg_timeout
from torch._C._distributed_c10d import Store, PrefixStore

from .group import Group
from .manager import _group_manager
from .helper import new_process_group_helper, store_based_barrier

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
    pg_options: Optional[Any] = None,
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
        world_size, rank, backend, store,
        pg_options=pg_options, group_name=group_name, timeout=timeout,
    )

    # exec barrier
    # TODO 支持其他 backend: MPI ...
    store_based_barrier(group, timeout)

    return group
