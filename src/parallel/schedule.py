# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/26
Description   : pipeline schedule
"""
from src.distributed.comm_utils import (
	get_pp_group_rank,
	get_pp_world_size,
	get_pp_prev_global_rank,
	get_pp_next_global_rank,
)


class PipeSchedule:
	"""
	pipeline schedule for micro batches
	因为是static schedule 所以可以根据micro batch number预先设计好各rank在每个clock的action
	"""
	def __init__(self, micro_batch_num, main_group=None) -> None:
		"""
		:param micro_batch_num:
		:param main_group:
		"""
		self.micro_batch_num = micro_batch_num
		self.main_group = main_group
		self.pp_world_size = get_pp_world_size()		# total stages
		self.pp_rank = get_pp_group_rank()				# current stage
		self.prev_rank = get_pp_prev_global_rank()		# prev stage
		self.next_rank = get_pp_next_global_rank()		# next stage
		self._clock = 0					# current clock
		self.total_steps = None			# total clocks

	def steps(self):
		"""
		generate actions: fw, bw, wait
		"""
		raise NotImplementedError()

	def get_action(self, clock):
		"""
		当前clock下rank所执行的action
		"""
		raise NotImplementedError()

	def current_time(self):
		return self._clock

	def __iter__(self):
		self.it = None
		return self

	def __next__(self):
		if self.it is None:
			self.it = self.steps()
		return next(self.it)


class SequenceSchedule(PipeSchedule):
	"""
	sequence schedule:

	clock	0		1		2		3		4		5		6		7		8		9		10
	rank0	f1		f2		f3		wait	wait	wait	wait	b3		b2		b1		all_reduce
	rank1	wait	f1		f2		f3		wait	wait	b3		b2		b1		wait	all_reduce
	rank2	wait	wait	f1		f2		f3		b3		b2		b1		wait	wait	all_reduce
	"""
	def __init__(self, micro_batch_num, main_group=None) -> None:
		super().__init__(micro_batch_num, main_group)
		self.total_steps = 2*(self.micro_batch_num+self.pp_world_size-1)	# 2*micro_batch_num=fw+bw, 2*(world_size-1)=wait

	def steps(self):
		"""
		decide the micro_batch id and action for each rank at every clock
		"""
		for clock in range(self.total_steps):	# clock: 0, 1, 2, ...; pp_rank: 0, 1, 2, ...
			self._clock = clock
			action, micro_batch_idx = self._get_action(clock)
			cmd = (action, micro_batch_idx)
			yield cmd

	def _get_action(self, clock):
		"""
		forward, backward, micro_batch_id
		"""
		if clock < self.pp_rank:
			return "wait", 0
		#
		tmp_clock = clock - self.pp_rank
		if tmp_clock < self.micro_batch_num:
			return "fw", tmp_clock

		diff = self.pp_world_size - 1 - self.pp_rank	# diff btw current stage and last stage
		tmp_clock = clock - self.pp_rank - self.micro_batch_num
		if tmp_clock < 2*diff:
			return "wait", 0

		tmp_clock = clock - self.pp_rank - self.micro_batch_num - 2*diff
		if tmp_clock < self.micro_batch_num:
			return "bw", tmp_clock

		return "wait", 0


class F1B1Schedule(PipeSchedule):
	"""
	1f1b schedule:

	clock	1
	rank0
	rank1
	rank2
	"""
	pass


class InterleaveSchedule(PipeSchedule):
	"""
	1f1b interleave schedule
	"""
	pass


class InferenceSchedule(PipeSchedule):
	"""
	inference schedule
	"""
	pass
