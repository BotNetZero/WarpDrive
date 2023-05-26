# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/26
Description   : pipeline schedule
"""
from src.distributed.comm_utils import get_pp_group_rank, get_pp_world_size, get_pp_prev_global_rank, get_pp_next_global_rank

class PipeSchedule:
	"""
	pipeline schedule.
	因为是static schedule 所以可以根据micro batch number预先设计好各rank在各clock的action
	"""

	def __init__(self, micro_batch_num, main_group=None) -> None:
		"""
		:param micro_batch_num:
		:param main_group:
		"""
		self.micro_batch_num = micro_batch_num
		self.main_group = main_group
		self.pp_world_size = get_pp_world_size()
		self.pp_rank = get_pp_group_rank()
		self.prev_rank = get_pp_prev_global_rank()
		self.next_rank = get_pp_next_global_rank()
		self._clock = 0
		self.total_steps = None

	def steps(self):
		"""
		generate actions
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

	clock	1		2		3		4		5		6		7		8		9		10		11
	rank0	f1		f2		f3		wait	wait	wait	wait	b3		b2		b1		all_reduce
	rank1	wait	f1		f2		f3		wait	wait	b3		b2		b1		wait	all_reduce
	rank2	wait	wait	f1		f2		f3		b3		b2		b1		wait	wait	all_reduce
	"""
	def __init__(self, micro_batch_num, main_group=None) -> None:
		super().__init__(micro_batch_num, main_group)
		self.total_steps = 2*(self.micro_batch_num+self.pp_world_size-1)	# fw, bw, wait

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
	pass
