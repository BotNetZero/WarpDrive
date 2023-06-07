# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/09
Description   : trainer
"""
import torch
import torch.nn as nn
import torch.functional as F
import torch.cuda as cuda
from torch.amp.autocast_mode import autocast
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


def loss_func(logits, targets):
	"""
	prev token ==> next token

	:param logits: lm_head outputs
	:param targets:
	"""
	shift_logits = logits[..., :-1, :].contiguous()		# prev tokens' logits: [batch_size, seq_len, vocab_size]
	shift_labels = targets[..., 1:].contiguous()		# next tokens label id: [batch_size, seq_len]

	unfold_logits = shift_logits.view(-1, shift_logits.size(-1)) 	# [batch_size*seq_len, vocab_size]
	unfold_labels = shift_labels.view(-1)							# target: [batch_size*seq_len]
	#
	loss = F.cross_entropy(unfold_logits, unfold_labels)

	return loss


class Trainer:
	"""
	按照rank所处stage实现training的功能:
	1/ init: forward, backward tensor
	2/ forward step
	3/ backward step
	"""
	def __init__(self, args, configs, device, model, optimizer, lr_scheduler, grad_scaler) -> None:
		"""
		:param args:
		:param device: current device
		:param model:
		:param optimizer:
		:param lr_scheduler: LR scheduler for optimizer
		:param grad_scaler:
		"""
		self.args = args
		self.configs = configs
		self.device = device
		self.dtype = normalize_precision(configs.torch_dtype)
		self.model = model
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler		#
		self.grad_scaler = grad_scaler			#
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
			self.output_micro_batches_grad = [		# gradients from next stage
				torch.zeros(
					(self.args.micro_batch_size, self.args.seq_length, self.configs.embedding_size),
					requires_grad=False, dtype=self.dtype, device=self.device
				) for _ in range(self.args.micro_batch_num)
			]
		# last stage
		elif self.pp_rank == self.pp_world_size-1:
			self.input_micro_batches = [			# input tensor with grad from previous rank
				torch.zeros(
					(self.args.micro_batch_size, self.args.seq_length, self.configs.embedding_size),
					requires_grad=True, dtype=self.dtype, device=self.device
				) for _ in range(self.args.micro_batch_num)
			]
			self.output_micro_batches_grad = None	#
		# middle stage
		else:
			self.input_micro_batches = [			# input tensor with grad from previous rank
				torch.zeros(
					(self.args.micro_batch_size, self.args.seq_length, self.configs.embedding_size),
					requires_grad=True, dtype=self.dtype, device=self.device
				) for _ in range(self.args.micro_batch_num)
			]
			self.output_micro_batches_grad = [		# gradients from next stage
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
		stage 1:N model的input tensor做grad清零
		因为各staged model的input tensor不在optimzier中, 需要手动清零
		"""
		if self.input_micro_batches:
			for input_micro_batch in self.input_micro_batches:
				if input_micro_batch.grad is not None:
					input_micro_batch.grad.zero_()

	def _step(self, input_ids, targets):
		"""
		one training step
		staged model compute gragh
		FW:
			input_micro_batch --> model_0 --> pred_micro_batch
			stage_0 send(pred_micro_batch) --> stage_1 recv(input_micro_batch)
			input_micro_batch --> model_1 --> pred_micro_batch
			...
		BW:
			loss --> model_N --> input_micro_batch.grad
			stage_N send(input_micro_batch.grad) --> stage_N-1 recv(pred_micro_batch.grad)
			pred_micro_batch --> model_N-1 --> input_micro_batch.grad
			...
		"""
		# input_ids [batch_size, seq_len] ==> [micro_batch_size, seq_len] * micro_batch_num
		if self.pp_rank == 0:						# first stage处理input tokens
			assert input_ids is not None
			self.input_micro_batches = input_ids.chunk(self.args.micro_batch_num, dim=0)

		# targets [batch_size, seq_len] ==> [micro_batch_size, seq_len] * micro_batch_num
		if self.pp_rank == self.pp_world_size-1:	# last stage处理labels
			assert targets is not None
			target_micro_batches = targets.chunk(self.args.micro_batch_num, dim=0)
		else:
			target_micro_batches = [None for _ in range(self.args.micro_batch_num)]

		pred_micro_batches = [None for _ in range(self.args.micro_batch_num)]	# staged model preds

		# schedule
		for action, micro_batch_idx in self.scheduler:
			print("in rank [{self.pp_rank}], current action: {action}_{micro_batch_idx}")
			if action == "wait":
				continue
			elif action == "fw":
				pred_micro_batch = self._forward_step(micro_batch_idx)
				pred_micro_batches[micro_batch_idx] = pred_micro_batch
			elif action == "bw":
				pred = pred_micro_batches[micro_batch_idx]
				target = target_micro_batches[micro_batch_idx]
				self._backward_step(micro_batch_idx, pred, target)
			else:
				raise ValueError(f"action [{action}] not support YET!!")
			# TODO: synchronize

		# all reduce
		self.grad_scaler.step(self.optimizer)
		self.lr_scheduler.step()
		self.grad_scaler.update()

	def _forward_step(self, micro_batch_idx):
		"""
		1/ recev input
		2/ forward pass
		3/ send output
		"""
		if self.pp_rank == 0:		# first stage
			# step1: None
			# step2: fw
			with cuda.stream(self.compute_stream):
				with autocast(device_type="cuda", dtype=self.dtype):
					micro_pred = self.model(self.input_micro_batches[micro_batch_idx])
			# step3: send
			self.comm.send(self.send_stream, micro_pred, self.pp_next_rank, self.compute_stream, None)

		elif self.pp_rank == self.pp_world_size-1:	# last stage
			# step1: recv input
			self.comm.recv(self.recv_stream, self.input_micro_batches[micro_batch_idx], self.pp_prev_rank, None, None)
			# step2: fw
			with cuda.stream(self.compute_stream):
				self.compute_stream.wait_stream(self.recv_stream)	# synchronize recv
				with autocast(device_type="cuda", dtype=self.dtype):
					micro_pred = self.model(self.input_micro_batches[micro_batch_idx])
			# step3: None

		else:	# middle stage
			# step1: recv input
			self.comm.recv(self.recv_stream, self.input_micro_batches[micro_batch_idx], self.pp_prev_rank, None, None)
			# step2: fw
			with cuda.stream(self.compute_stream):
				self.compute_stream.wait_stream(self.recv_stream)	# synchronize recv
				with autocast(device_type="cuda", dtype=self.dtype):
					micro_pred = self.model(self.input_micro_batches[micro_batch_idx])
			# step3: send
			self.comm.send(self.send_stream, micro_pred, self.pp_next_rank, self.compute_stream, None)

		return micro_pred

	def _backward_step(self, micro_batch_idx, pred, target):
		"""
		1/ calculate loss and scale, or recv grad
		2/ recompute, bw
		3/ send grad
		"""
		if self.pp_rank == self.pp_world_size-1:	# last stage
			with cuda.stream(self.compute_stream):
				# step1: loss, fp32
				with autocast(device_type="cuda", dtype=self.dtype):
					loss = loss_func(pred, target)
				# step2: bw
				self.grad_scaler.scale(loss).backward()
			# step 3: send grad
			self.comm.send(self.send_stream, self.input_micro_batches[micro_batch_idx].grad, self.pp_prev_rank, self.compute_stream, None)

		elif self.pp_rank == 0: 	# first stage
			# step1: recv grad
			self.comm.recv(self.recv_stream, self.output_micro_batches_grad[micro_batch_idx], self.pp_next_rank, None, None)
			# step2: backward
			with cuda.stream(self.compute_stream):
				self.compute_stream.wait_stream(self.recv_stream)	# synchronize
				pred.backward(gradient=self.output_micro_batches_grad[micro_batch_idx])
			# step3: None

		else:	# middle stage
			# step1: recv grad
			self.comm.recv(self.recv_stream, self.output_micro_batches_grad[micro_batch_idx], self.pp_next_rank, None, None)
			# step2: backward
			with cuda.stream(self.compute_stream):
				self.compute_stream.wait_stream(self.recv_stream)	# synchronize
				pred.backward(gradient=self.output_micro_batches_grad[micro_batch_idx])
			# step3: send grad
			self.comm.send(self.send_stream, self.input_micro_batches[micro_batch_idx].grad, self.pp_prev_rank, self.compute_stream, None)


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