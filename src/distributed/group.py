#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from torch.distributed import Backend
from torch._C._distributed_c10d import ProcessGroup, PrefixStore

class Group:
	def __init__(
			self,
			name: str,
			pg: ProcessGroup,
			backend: Backend,
			store: PrefixStore,
			config: str,
			level: int = 0,
		):
		"""
		"""
		self.name = name
		self.pg = pg
		self.backend = backend
		self.store = store
		self.config = config
		self.level = level
