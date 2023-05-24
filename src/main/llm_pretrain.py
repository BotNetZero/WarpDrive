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
from src.distributed.comm_utils import get_pp_group_rank, get_pp_prev_global_rank, get_pp_next_global_rank
from src.data.data_utils import get_train_data_loader
from src.common.constants import MODEL_PATH
from src.ml.tokenizer import Tokenizer
from src.ml.gptneox import GPTStageFirst, GPTStageLast, GPTStageMiddle

def get_model(args, configs, device):
	pp_rank = get_pp_group_rank()
	if pp_rank == 0:
		model = GPTStageFirst(args, configs, device)
	


def pretrain():
	"""

	"""
	args, configs = parse_args()
	init_distributed_env(args)
	comm = get_main_group_comm()
	device = torch.device(args.cuda_id)

	# model, optimizer, dataloader
	model_path = os.path.join(MODEL_PATH, "pythia_7b")
	tokenizer = Tokenizer(model_path)

	model = get_model(args, configs)
	train_dataloader = get_train_data_loader(args, tokenizer)

	# traing & eval


if __name__ == "__main__":
	try:
		pretrain()
	except Exception as exc:
		print(traceback.print_exc(exc))
	#
	destroy_distributed_env()
