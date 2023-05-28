# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/05
Description   :
"""
import torch
import torch.cuda as cuda
import src.distributed.c10d as dist

# TODO: redesign...
class Communicator:
	"""
	封装torch.distributed的collective comm和p2p comm, 在task stream的控制下通信
	"""
	def __init__(self, device_id, rank, pg=None) -> None:
		"""
		:param device_id: cuda id
		:param rank: global rank
		:param pg: main process group
		"""
		self.device_id = device_id
		self.rank = rank
		self.pg = pg

	def _check_sub_pg(self, sub_pg):
		"""
		check whethere sub process group belongs to main process group
		"""
		raise NotImplementedError()

	def send(self, send_stream, tensor, dst_rank, wait_stream, sub_pg):
		"""
		execute send task under send_stream ctrl
		:param send_stream: cuda stream for p2p send
		:param tensor:
		:param dst_rank: global rank of destination
		:param wait_stream: synchronize with wait_stream
		:param sub_pg:
		"""
		with cuda.stream(send_stream):
			send_stream.wait_stream(wait_stream)
			dist.send(tensor, dst_rank, sub_pg)

	def recv(self, recv_stream, tensor, src_rank, wait_stream, sub_pg):
		with cuda.stream(recv_stream):
			recv_stream.wait_stream(wait_stream)
			dist.recv(tensor, src_rank, sub_pg)

	def broadcast(self, collective_stream, tensor, src_rank, wait_stream, sub_pg):
		"""
		execute broadcast task under collective_stream ctrl
		"""
		with cuda.stream(collective_stream):
			if wait_stream is not None:
				collective_stream.wait_stream(wait_stream)
			dist.broadcast(tensor, src_rank, sub_pg)

	def all_gather(self, collective_stream, tensor, output_tensor_list, wait_stream, sub_pg):
		"""
		gather tensor from each rank into output_tensor_list
		"""
		with cuda.stream(collective_stream):
			if wait_stream is not None:
				collective_stream.wait_stream(wait_stream)
			dist.all_gather(output_tensor_list, tensor, group=sub_pg)
