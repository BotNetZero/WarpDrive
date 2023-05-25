# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/09
Description   : trainer
"""
import torch
import torch.nn as nn
import torch.cuda as cuda
from src.common.logger import logger
from src.distributed.comm_utils import get_main_group_comm, get_pp_group_rank, get_pp_world_size


class Trainer:
	"""
	按照rank所处stage实现training的功能:

	"""
	def __init__(self, args, configs, device, model, optimizer, scheduler) -> None:
		"""
		:param args:
		:param device: current device
		:param model:
		:param optimizer:
		:param scheduler: LR scheduler for optimizer
		"""
		self.args = args
		self.configs = configs
		self.device = device
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.loss_fn = nn.CrossEntropyLoss()
		self.comm = get_main_group_comm()		# dist. communicator

		#
		self.compute_stream = cuda.default_stream(device)	#
		self.send_stream = cuda.Stream(device)
		self.recv_stream = cuda.Stream(device)

		#
		self.pp_rank = get_pp_group_rank()
		if self.pp_rank == 0:
			pass


	def _step(self):
		pass

	def _forward_step(self, input_ids, batch_Y=None):
		"""
		training step
		"""
		self.optimizer.zero_grand()

	def _backward_step(self):
		pass


	def __call__(self, input_ids, targets, print_step=100):
		if self.pp_rank == 0:
			assert input_ids is not None
			self.input_micro_batches = input_ids.chunk(self.args.micro_batch_num)
			



class Evaluator:
	def __init__(self) -> None:
		pass