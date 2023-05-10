# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/10
Description   : 分布式环境测试

"""
import os, sys
sys.path.append(os.getcwd())

import torch
import torch.cuda as cuda
from src.utils.arguments import parse_args
from src.communication.comm_utils import init_distributed_env, destroy_distributed_env, get_main_group_comm, get_pp_group
from src.communication.comm_utils import get_pp_prev_global_rank, get_pp_next_global_rank

def main():
	args = parse_args()
	init_distributed_env(args)
	comm = get_main_group_comm()
	device = torch.device(args.cuda_id)
	#
	compute_stream = cuda.default_stream(device)		# stream for computing
	send_stream = cuda.Stream(device)					# stream for p2p send
	recv_stream = cuda.Stream(device)					# stream for p2p recv

	pp_grp = get_pp_group()
	for i in range(10):
		print(f"<<<<< iteration {i} in rank {args.global_rank} >>>>")
		with cuda.stream(compute_stream):
			send_tensor = torch.empty(2,2, device=device) + args.global_rank + i
			recv_tensor = torch.empty(2,2, device=device)
		send_stream.wait_stream(compute_stream)
		print("synchronize with compute stream")
		if args.global_rank == 0:
			next_rank = get_pp_next_global_rank()
			print("communicate with rank:", next_rank)
			comm.send(send_stream, send_tensor, next_rank, pp_grp)
			comm.recv(recv_stream, recv_tensor, next_rank, pp_grp)
		elif args.global_rank == 1:
			next_rank = get_pp_prev_global_rank()
			print("communicate with rank:", next_rank)
			comm.send(send_stream, send_tensor, next_rank, pp_grp)
			comm.recv(recv_stream, recv_tensor, next_rank, pp_grp)
		print("send tensor:", send_tensor)
		print("recv tensor:", recv_tensor)
		print()

	destroy_distributed_env()
	print("destroy distributed environment...")


if __name__ == "__main__":
	main()
