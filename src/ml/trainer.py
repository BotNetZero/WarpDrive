# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/09
Description   : trainer
"""


class Trainer:
	"""
	"""
	def __init__(self, model, optimizer, ) -> None:
		self.model = model
		self.model.train()
		self.optimizer = optimizer


class Evaluator:
	def __init__(self) -> None:
		pass