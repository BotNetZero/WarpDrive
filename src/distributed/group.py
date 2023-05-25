#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.distributed import Backend
from torch._C._distributed_c10d import PrefixStore, ProcessGroup


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
			parent_name: str = None,
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
		self.parent_name = parent_name
		if self.level == 0 and self.parent_name is not None:
			raise RuntimeError(
				"Top level group should not have parent group"
			)
