# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/09
Description   : pretrain
"""
import os, sys
sys.path.append(os.getcwd())
import traceback
import torch
import torch.cuda as cuda
import torch.distributed as dist
from src.utils.arguments import parse_args
from src.distributed.comm_utils import init_distributed_env, destroy_distributed_env, get_main_group_comm, get_pp_group
from src.distributed.comm_utils import get_pp_prev_global_rank, get_pp_next_global_rank


def get_model():
	pass


def pretrain():
	"""

	"""
	args = parse_args()
	init_distributed_env(args)
	comm = get_main_group_comm()
	device = torch.device(args.cuda_id)

	# model, optimizer, dataloader
	get_model()

	# traing & eval


if __name__ == "__main__":
	try:
		pretrain()
	except Exception as exc:
		print(traceback.print_exc(exc))
	#
	destroy_distributed_env()
