# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/05
Description   : 
"""
import torch
import torch.cuda as cuda
import torch.distributed as dist


class NCCLCommunicator:
	def __init__(self, device_id, rank, grp_size, grp_name) -> None:
		self.device_id = device_id
		self.rank = rank
		self.grp_size = grp_size
		self.pg = None


	def send(self, send_stream, tensor, dst_rank):
		"""
		execute send task under send_stream ctrl
		"""
		with cuda.stream(send_stream):
			dist.send(tensor, dst_rank, )

	def recv(self, tensor, src):
		pass

