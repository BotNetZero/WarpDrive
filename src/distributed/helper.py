#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import logging
from typing import Union
from datetime import timedelta

import torch
from torch.distributed import Backend, default_pg_timeout, BackendConfig
from torch._C._distributed_c10d import (
    Store, PrefixStore, ProcessGroup, DebugLevel, get_debug_level
)

NCCL_AVAILABLE = True
GLOO_AVAILABLE = True

try:
    from torch._C._distributed_c10d import ProcessGroupNCCL
    ProcessGroupNCCL.__module__ = "torch.distributed.distributed_c10d"
except ImportError:
    NCCL_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupGloo
    from torch._C._distributed_c10d import _ProcessGroupWrapper
    ProcessGroupGloo.__module__ = "torch.distributed.distributed_c10d"
except ImportError:
    GLOO_AVAILABLE = False

from .group import Group
from .manager import _group_manager


logger = logging.getLogger(__name__)


PG_WRAPPER_STORE_PREFIX = "pg_wrapper"
STORE_BASED_BARRIER_PREFIX = "store_based_barrier_key"


def new_process_group_helper(
    group_size: int,
    group_rank: int,
    backend: Union[str, Backend],
    store: PrefixStore,
    group_name: str=None,
    group_level: int=0,
    parent_name: str=None,
    pg_options: ProcessGroup.Options=None,
    timeout: timedelta=default_pg_timeout,
) -> Group:
    global _group_manager

    if not group_name:
        group_name = str(_group_manager.size)

    if _group_manager.exist_group(group_name):
        raise RuntimeError(
            "The specified group name has already been "
            "created, please use a different group name"
        )

    if not isinstance(timeout, timedelta):
        raise RuntimeError(
            "Expected timeout argument to be of type" "datetime.timedelta"
        )

    prefix_store = PrefixStore(f"{group_name}/", store)
    base_pg_options = ProcessGroup.Options(backend=str(backend))
    base_pg_options._timeout = timeout
    pg: ProcessGroup = ProcessGroup(prefix_store, group_rank, group_size, base_pg_options)
    backend_config = BackendConfig(backend)
    for device, backend_str in backend_config.get_device_backend_map().items():
        # Use the group name as prefix in the default store, such that
        # a single store can be reused by multiple groups.
        backend_prefix_store = PrefixStore(f"{device}/", prefix_store)

        if backend_str not in [Backend.GLOO, Backend.NCCL]:
            raise RuntimeError(f"Unsupported backend: {backend_str}")

        if backend_str == Backend.GLOO:
            # TODO: remove this check after lazy initialization is supported
            # if pg_options is not None:
            #     raise RuntimeError("GLOO options not supported")
            backend_class = ProcessGroupGloo(backend_prefix_store, group_rank, group_size, timeout=timeout)
            backend_type = ProcessGroup.BackendType.GLOO
        else:
            if not NCCL_AVAILABLE:
                raise RuntimeError("Distributed package doesn't have NCCL " "built in")
            if pg_options is not None:
                assert isinstance(
                    pg_options, ProcessGroupNCCL.Options
                ), "Expected pg_options argument to be of type ProcessGroupNCCL.Options"
            else:
                # default pg_options for NCCL
                pg_options = ProcessGroupNCCL.Options()
                pg_options.is_high_priority_stream = False
                pg_options._timeout = timeout

            backend_class = ProcessGroupNCCL(backend_prefix_store, group_rank, group_size, pg_options)
            backend_type = ProcessGroup.BackendType.NCCL

        # Set sequence numbers for gloo and nccl backends.
        if backend_str in [Backend.GLOO, Backend.NCCL]:
            backend_class._set_sequence_number_for_group()
        # If the type is a sublcass of ProcessGroup then return this process group immediately
        # TODO: This defaults to the old behavior for PythonProcessGroups which overwrites the
        # ProcessGroup instance
        if issubclass(type(backend_class), ProcessGroup):
            pg = backend_class
            break

        # Process group wrapper initialization for supported PGs when TORCH_DISTRIBUTED_DEBUG is set
        if backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]:
            # In debug mode and if GLOO is available, wrap in a wrapper PG that
            # enables enhanced collective checking for debuggability.
            if get_debug_level() == DebugLevel.DETAIL:
                if not GLOO_AVAILABLE:
                    logger.info(
                        """TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but
                                GLOO is not available. Build with Gloo to
                                create a wrapper process group in debug mode
                                to aid collective desynchronization debugging."""
                    )
                else:
                    backend_class = _create_process_group_wrapper(
                        wrapped_pg=backend_class,
                        store_prefix=group_name,
                        store=backend_prefix_store,
                        rank=group_rank,
                        world_size=group_size,
                        timeout=timeout,
                    )

        # only create single backend pg when backend is set to gloo, nccl, mpi, and ucc
        if backend in [Backend.GLOO, Backend.NCCL, Backend.UCC, Backend.MPI]:
            for device in backend_config.get_device_backend_map().keys():
                pg._register_backend(torch.device(device), backend_type, backend_class)

            # break out of outer loop to not create any more backends
            break
        else:
            pg._register_backend(torch.device(device), backend_type, backend_class)

    # new group add it to the group manager
    group = Group(group_name, group_size, group_rank, pg, backend, store, str(backend_config), level=group_level, parent_name=parent_name)
    _group_manager.add_group(group)

    return group


def _create_process_group_wrapper(
    wrapped_pg: ProcessGroup,
    store_prefix: str,
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta = default_pg_timeout,
):
    # Create a separate prefix store for the helper process group.
    prefix = f"{PG_WRAPPER_STORE_PREFIX}:{store_prefix}"
    store = PrefixStore(prefix, store)
    helper_pg = ProcessGroupGloo(store, rank, world_size, timeout=timeout)
    # Wrap the underlying pg with ProcessGroupWrapper.
    wrapped_pg = _ProcessGroupWrapper(wrapped_pg, helper_pg)
    return wrapped_pg


def store_based_barrier(group: Group, timeout: timedelta):
    """
    Barrier based on store which is used for synchronizing processes after
    ``init_process_group`` or ``new_group``. Intended to be used only with
    those two methods and is not a generic alternative to ``barrier()``.
    """
    store_key = "{}:{}".format(STORE_BASED_BARRIER_PREFIX, group.name)
    group.store.add(store_key, 1)
    logger.info("Added key: {} to store for rank: {}".format(store_key, group.pg.rank()))

    # Use 'add' instead of 'get' since for some store implementations 'add'
    # doesn't work well with 'get'. Ideally the store implementations should
    # be fixed, but for backward compatiblity reasons it is risky to change
    # the store implementations. Once, we completely migrate away from these
    # legacy stores, we can use 'get' here instead.
    worker_count = group.store.add(store_key, 0)
    start = time.time()
    log_time = time.time()
    while worker_count != group.pg.size():
        time.sleep(0.01)
        worker_count = group.store.add(store_key, 0)

        # Print status periodically to keep track.
        if timedelta(seconds=(time.time() - log_time)) > timedelta(seconds=10):
            logger.info(
                "Waiting in store based barrier to initialize process group for "
                "rank: {}, key: {} (world_size={}, worker_count={}, timeout={})".format(
                    group.pg.rank(), store_key, group.pg.size(), worker_count, timeout
                )
            )
            log_time = time.time()

        if timedelta(seconds=(time.time() - start)) > timeout:
            raise RuntimeError(
                "Timed out initializing process group in store based barrier on "
                "rank: {}, for key: {} (world_size={}, worker_count={}, timeout={})".format(
                    group.pg.rank(), store_key, group.pg.size(), worker_count, timeout
                )
            )
