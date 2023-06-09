# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/10
Description   : 训练过程中的通信

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
	brod_stream = cuda.Stream(device)					# stream for broadcast
	#
	stop_flag = torch.zeros(1, dtype=torch.int32).to(device)
	data = torch.zeros(3, 5, dtype=torch.int, device=device)
	cuda.synchronize()
	try:
		if args.global_rank == 0:
			stop_flag.data[:] = 1
			for i in range(10):
				print("epoch:", i)
				comm.broadcast(brod_stream, stop_flag, 0, None, None)		# broadcast stop_flag for all ranks
				brod_stream.synchronize()
				print("brod_stream state:", brod_stream.query())
				#
				if stop_flag.item() == 2:
					print("finished training, then stop....")
					break
				data = torch.full((3, 5), i+1, dtype=torch.int, device=device)
				print("prepare send data:", data)
				compute_stream.synchronize()
				print("compute_stream state:", compute_stream.query())
				#
				comm.send(send_stream, data, 1, None, None)
				send_stream.synchronize()
				print("send_stream state:", send_stream.query())
				#
				if i == 5:
					print("setting stop flag....")
					stop_flag.data[:] = 2
				compute_stream.synchronize()
				print("compute_stream state:", compute_stream.query())
				print()
		else:
			while True:
				comm.broadcast(brod_stream, stop_flag, 0, None, None)		# broadcast stop_flag for all ranks
				brod_stream.synchronize()
				print("brod_stream state:", brod_stream.query())
				#
				if stop_flag.item() == 0:
					print("not receive correct stop_flag....")
					continue
				if stop_flag.item() == 2:
					print("finished training, then stop....")
					break
				compute_stream.synchronize()
				print("compute_stream state:", compute_stream.query())
				#
				comm.recv(recv_stream, data, 0, None, None)
				recv_stream.synchronize()
				print("recv stream state:", recv_stream.query())
				print("recv data:", data)
				#
				compute_stream.synchronize()
				print("compute_stream state:", compute_stream.query())
				print()

	except Exception as exc:
		traceback.print_exc()

	destroy_distributed_env()
	print("destroy distributed environment...")


if __name__ == "__main__":
	main()
