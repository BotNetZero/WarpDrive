# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/10
Description   : 分布式环境测试

"""
import os, sys
sys.path.append(os.getcwd())

import traceback
import torch
import torch.cuda as cuda

import src.distributed.c10d as dist
from src.utils.arguments import parse_args
from src.distributed.comm_utils import init_distributed_env, destroy_distributed_env, get_main_group_comm, get_pp_group
from src.distributed.comm_utils import get_pp_prev_global_rank, get_pp_next_global_rank

def main():
	args, _ = parse_args()
	init_distributed_env(args)
	comm = get_main_group_comm()

	device = torch.device(args.cuda_id)
	cuda.set_device(device)

	compute_stream = cuda.default_stream(device)		# stream for computing
	send_stream = cuda.Stream(device)					# stream for p2p send
	recv_stream = cuda.Stream(device)					# stream for p2p recv
	try:
		for i in range(10):
			print(f"<<<<<< iter: {i} >>>>>>")
			with cuda.stream(compute_stream):
				send_tensor = torch.ones(2,2, device=device) + args.global_rank + i
				recv_tensor = torch.zeros(2,2, device=device)
			#
			if args.global_rank == 0:
			# 	with cuda.stream(send_stream):
			# 		send_stream.wait_stream(compute_stream)
			# 		print("send stream synchronize with compute stream")
			# 		dist.send(send_tensor, 1)
			# 	#
			# 	with cuda.stream(recv_stream):
			# 		recv_stream.wait_stream(send_stream)
			# 		print("recv stream synchronize with send stream")
			# 		dist.recv(recv_tensor, 1)
				comm.send(send_stream, send_tensor, 1, compute_stream, None)
				comm.recv(recv_stream, recv_tensor, 1, send_stream, None)
			else:
			# 	with cuda.stream(recv_stream):
			# 		recv_stream.wait_stream(compute_stream)
			# 		print("recv stream synchronize with compute stream")
			# 		dist.recv(recv_tensor, 0)
			# 	#
			# 	with cuda.stream(send_stream):
			# 		send_stream.wait_stream(recv_stream)
			# 		print("send stream synchronize with compute stream")
			# 		dist.send(send_tensor, 0)
				comm.recv(recv_stream, recv_tensor, 0, compute_stream, None)
				comm.send(send_stream, send_tensor, 0, recv_stream, None)
			#
			print("send tensor:", send_tensor)
			print("recv tensor:", recv_tensor)
			print()
	except Exception as exc:
		print(traceback.format_exception(exc))

	destroy_distributed_env()
	print("destroy distributed environment...")


if __name__ == "__main__":
	main()
