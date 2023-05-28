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
from src.distributed.comm_utils import (
	get_main_group_comm,
	get_pp_group_rank,
	get_pp_world_size,
	get_pp_prev_global_rank,
	get_pp_next_global_rank,
)
from src.utils.conversion import normalize_precision
from src.parallel.schedule import SequenceSchedule

class Trainer:
	"""
	按照rank所处stage实现training的功能:
	1/ init: forward, backward tensor
	2/ forward step
	3/ backward step
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
		self.dtype = normalize_precision(configs.torch_dtype)
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.loss_fn = nn.CrossEntropyLoss()
		self.comm = get_main_group_comm()		# dist. communicator
		#
		self.compute_stream = cuda.default_stream(device)	#
		self.send_stream = cuda.Stream(device)
		self.recv_stream = cuda.Stream(device)
		self.collect_stream = cuda.Stream(device)			#
		#
		self.pp_rank = get_pp_group_rank()
		self.pp_prev_rank = get_pp_prev_global_rank()
		self.pp_next_rank = get_pp_next_global_rank()
		self.pp_world_size = get_pp_world_size()
		# first stage
		if self.pp_rank == 0:
			self.input_micro_batches = None			# micro batched input ids
			self.output_micro_batches = [
				torch.zeros(
					(self.args.micro_batch_size, self.args.seq_length, self.configs.embedding_size),
					requires_grad=False, dtype=self.dtype, device=self.device
				) for _ in range(self.args.micro_batch_num)
			]
			self.output_micro_batches_grad = [		# recv tensor from next rank
				torch.zeros(
					(self.args.micro_batch_size, self.args.seq_length, self.configs.embedding_size),
					requires_grad=False, dtype=self.dtype, device=self.device
				) for _ in range(self.args.micro_batch_num)
			]
		# last stage
		elif self.pp_rank == self.pp_world_size-1:
			self.input_micro_batches = [			# recv tensor from previous rank
				torch.zeros(
					(self.args.micro_batch_size, self.args.seq_length, self.configs.embedding_size),
					requires_grad=True, dtype=self.dtype, device=self.device
				) for _ in range(self.args.micro_batch_num)
			]
			self.output_micro_batches_grad = None		# labels
		# middle stage
		else:
			self.input_micro_batches = [			# recv tensor from previous rank
				torch.zeros(
					(self.args.micro_batch_size, self.args.seq_length, self.configs.embedding_size),
					requires_grad=True, dtype=self.dtype, device=self.device
				) for _ in range(self.args.micro_batch_num)
			]
			self.output_micro_batches_grad = [		# recv tensor from next rank
				torch.zeros(
					(self.args.micro_batch_size, self.args.seq_length, self.configs.embedding_size),
					requires_grad=False, dtype=self.dtype, device=self.device
				) for _ in range(self.args.micro_batch_num)
			]

		# 全局数据
		self.global_step = 0
		self.scheduler = SequenceSchedule(self.args.micro_batch_num)

	def zero_input_grad(self):
		"""
		input_micro_batches的参数没有在optimizer中 手动清零
		"""
		if self.input_micro_batches:
			for input_micro_batch in self.input_micro_batches:
				if input_micro_batch.grad is not None:
					input_micro_batch.grad.zero_()

	def _step(self, input_ids, targets):
		"""
		one training step
		"""
		# input_ids [batch_size, seq_len] ==> [micro_batch_size, seq_len] * micro_batch_num
		if self.pp_rank == 0:						# first stage处理input tokens
			assert input_ids is not None
			self.input_micro_batches = input_ids.chunk(self.args.micro_batch_num, dim=0)
		if self.pp_rank == self.pp_world_size-1:	# last stage处理labels
			assert targets is not None
			self.target_micro_batches = targets.chunk(self.args.micro_batch_num, dim=0)
		# schedule
		for action, micro_batch_idx in self.scheduler:
			if action == "wait":
				continue
			elif action == "fw":
				micro_batch = self._forward_step(micro_batch_idx)
			elif action == "bw":
				self._backward_step()
			else:
				raise ValueError(f"action [{action}] not support YET!!")


	def _forward_step(self, micro_batch_idx):
		"""
		1/ recev input
		2/ forward pass
		3/ send output
		:param input_ids:
		"""
		if self.pp_rank == 0:		# first stage
			# step1: None
			# step2: fw
			with cuda.stream(self.compute_stream):
				micro_output = self.model(self.input_micro_batches[micro_batch_idx])
			# step3: send
			self.comm.send(self.send_stream, micro_output, self.pp_next_rank, self.compute_stream, None)

		elif self.pp_rank == self.pp_world_size-1:	# last stage
			# step1: recv input
			self.comm.recv(self.recv_stream, self.input_micro_batches[micro_batch_idx], self.pp_prev_rank, None, None)
			# step2: fw
			with cuda.stream(self.compute_stream):
				self.compute_stream.wait_stream(self.recv_stream)	# synchronize recv
				micro_output = self.model(self.input_micro_batches[micro_batch_idx])
			# step3: None

		else:	# middle stage
			# step1: recv input
			self.comm.recv(self.recv_stream, self.input_micro_batches[micro_batch_idx], self.pp_prev_rank, None, None)
			# step2: fw
			with cuda.stream(self.compute_stream):
				self.compute_stream.wait_stream(self.recv_stream)	# synchronize recv
				micro_output = self.model(self.input_micro_batches[micro_batch_idx])
			# step3: send
			self.comm.send(self.send_stream, micro_output, self.pp_next_rank, self.compute_stream, None)

		return micro_output


	def _backward_step(self):
		pass


	def __call__(self, input_ids, targets):
		"""
		steps:
		1/ init: micro batches,
		2/ pipeline schedule
		3/ training step
		"""
		# update loss scale
		if not self.args.loss_scale:	# dynamic scaler
			scales_buffer = [torch.ones_like(self.optimizer.grad_scaler._scale) for _ in range(self.args.pipeline_group_size)]
			self.comm.all_gather(self.optimizer.grad_scaler._scale, scales_buffer)
			self.optimizer.grad_scaler._scale.data[:] = min([s.item() for s in scales_buffer])

		# zero grad
		self.zero_input_grad()
		self.optimizer.zero_grad(set_to_none=False)

		#
		self._step(input_ids, targets)

		self.global_step += 1


class Evaluator:
	def __init__(self) -> None:
		pass