#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from group import Group

# 管理所有 group
# TODO 增删改查的时候需要加锁

class GroupManager:
    def __init__(self) -> None:
        self.size = 0
        self.table = {}
        self.pg_map = {}
        self.pg_names = {}
        self.pg_backend_config = {}
        self.pg_group_ranks = {}
    
    def add_group(self, group: Group):
        """
		_world.pg_map[pg] = (backend, prefix_store)
    	_world.pg_names[pg] = group_name
    	_world.pg_backend_config[pg] = str(backend_config)
		_world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
        """
        if group == None:
            return
        if self.exist_group(group.name):
            raise Exception("group name already exists")
        # 全局数据
        self.size += 1
        self.table[group.name] = group
        # 统计数据
        self.pg_map[group.pg] = (group.backend, group.store)
        self.pg_names[group.pg] = group.name
        self.pg_backend_config[group.pg] = group.config
        self.pg_group_ranks[group.pg] = {i: i for i in range(group.pg.size())}
    
    def exist_group(self, group: Union[str, Group]):
        if isinstance(group, str):
            return group in self.table
        if isinstance(group, Group):
            return group.name in self.table
        raise Exception("group instance type error")
    
    def del_group(self, group: Union[str, Group]):
        if group is None:
            return
        if isinstance(group, str):
            if self.exist_group(group):
                group = self.table[group]
                self._del_group(group)
        if isinstance(group, Group):
            if self.exist_group(group):
                self._del_group(group)
        raise Exception("group instance type error")
    
    def _del_group(self, group: Group):
        if group == None:
            return
        if not self.exist_group(group):
            return
        # 清理统计数据
        del self.pg_map[group.pg]
        del self.pg_names[group.pg]
        del self.pg_backend_config[group.pg]
        del self.pg_group_ranks[group.pg]
        # 清理全局数据
        del self.table[group.name]
        self.size -= 1
