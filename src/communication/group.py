# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/10
Description   : group structure
"""


class Group:
	def __init__(self, grp_name, grp_type="main") -> None:
		"""
		"""
		self.grp_name = grp_name
		self.grp_type = grp_type
		self.pg = None

