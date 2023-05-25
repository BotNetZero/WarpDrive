# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/17
Description   : pipeline模型测试
"""
import os, sys
sys.path.append(os.getcwd())
import time
import traceback
import torch
import torch.cuda as cuda
import src.distributed.c10d as dist
from src.utils.arguments import parse_args
from src.distributed.comm_utils import init_distributed_env, destroy_distributed_env, get_main_group_comm, get_pp_group
from src.ml.gptneox import GPTStageFirst, GPTStageMiddle, GPTStageLast
from src.ml.tokenizer import Tokenizer
from src.common.constants import MODEL_PATH

def main():
	args, configs = parse_args()
	model_path = os.path.join(MODEL_PATH, args.model_name.lower())
	#
	try:
		init_distributed_env(args)
		device = torch.device(args.cuda_id)
		cuda.set_device(device)

		assert args.world_size == 2

		if args.global_rank == 0:
			tokenizer = Tokenizer(model_path)
			model = GPTStageFirst(args, configs, device)


		if args.global_rank == 1:
			model = GPTStageLast(args, configs, device)

	except Exception as exc:
		print(traceback.format_exc())

	time.sleep(5*60)
	destroy_distributed_env()
	print("destroy distributed environment...")

if __name__ == "__main__":
	main()
