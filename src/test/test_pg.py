# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/04/22
Description   : process group init, 测试机器通信
"""
import os, sys
sys.path.append(os.getcwd())

import logging
import argparse
import datetime
import traceback
import torch
import torch.cuda as cuda
import src.distributed.c10d as dist

logger = logging.getLogger(__name__)

def main():
	parser = argparse.ArgumentParser(description='simple distributed training job')
	parser.add_argument('--rank', type=int)
	parser.add_argument('--size', type=int)
	parser.add_argument('--ip', type=str)
	parser.add_argument('--port', type=str)
	args = parser.parse_args()

	tcp_init = f"tcp://{args.ip}:{args.port}"

	print("starting with master:", tcp_init)

	#172.17.8.193
	try:
		dist.init_process_group(
			backend="nccl",
			init_method=tcp_init,
			timeout=datetime.timedelta(seconds=60),
			rank=args.rank,
			world_size=args.size,
		)
		end_time = datetime.datetime.now()
		print("success... finished init PG at: ", end_time)
	except Exception as exc:
		end_time = datetime.datetime.now()
		print("fail!!! Init PG at: ", end_time)
		print(traceback.format_exception(exc))

	dist.destroy_process_group()
	print("destroy PG.....")


if __name__ == "__main__":
	main()
