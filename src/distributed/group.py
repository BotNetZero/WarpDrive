#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from torch.distributed import Backend, default_pg_timeout
from torch._C._distributed_c10d import (
    Store, PrefixStore, ProcessGroup, DebugLevel, get_debug_level
)

from .constants import (
    _NCCL_AVAILABLE, _GLOO_AVAILABLE,
    ProcessGroupNCCL, ProcessGroupGloo, _ProcessGroupWrapper
)

class Group:
	def __init__(
			self,
			name: str,
			world_size: int,
			rank: int,
			pg: ProcessGroup,
			backend: Backend,
			store: PrefixStore,
			config: str,
			level: int = 0,
		):
		"""
		_world.pg_map[pg] = (backend, prefix_store)
    	_world.pg_names[pg] = group_name
    	_world.pg_backend_config[pg] = str(backend_config)
		_world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
		"""
		self.name = name

		self.world_size = world_size
		self.rank = rank

		self.pg = pg

		self.backend = backend
		self.store = store

		self.config = config
		self.rank_map = {i: i for i in range(pg.size())}

		self.level = level
