import os, sys
sys.path.append(os.getcwd())


class TestSchedule:
	def __init__(self, pp_rank, world_size, micro_batch_num) -> None:
		self.pp_rank = pp_rank
		self.pp_world_size = world_size
		self.micro_batch_num = micro_batch_num
		self.total_steps = 2*(self.micro_batch_num+self.pp_world_size-1)

	def steps(self):
		for clock in range(self.total_steps):	# clock: 0, 1, 2, ...; pp_rank: 0, 1, 2, ...
			self._clock = clock
			#
			action, micro_batch_idx = self._get_action(clock)
			if action == "wait":
				cmd = "wait"
			else:
				cmd = f"{action}_{micro_batch_idx}"
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

	def __iter__(self):
		self.it = None
		return self

	def __next__(self):
		if self.it is None:
			self.it = self.steps()
		return next(self.it)



if __name__ == "__main__":
	scheduler = TestSchedule(pp_rank=0, world_size=3, micro_batch_num=3)
	for cmd in scheduler:
		print(cmd)

