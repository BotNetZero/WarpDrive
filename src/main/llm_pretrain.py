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
from transformers import get_linear_schedule_with_warmup
from src.utils.arguments import parse_args
from src.distributed.comm_utils import init_distributed_env, destroy_distributed_env, get_main_group_comm, get_pp_group
from src.distributed.comm_utils import get_pp_group_rank, get_pp_world_size, get_pp_prev_global_rank, get_pp_next_global_rank
from src.data.data_utils import get_train_data_loader
from src.common.constants import MODEL_PATH
from src.ml.tokenizer import Tokenizer
from src.ml.gptneox import GPTStageFirst, GPTStageLast, GPTStageMiddle, GPTStageFull
from src.optimization.optimizer import create_optimizer, get_fp16_optimizer


def get_model(args, configs, device):
	"""
	基于rank所处stage创建model
	"""
	pp_rank = get_pp_group_rank()
	pp_world_size = get_pp_world_size()
	if pp_world_size == 1:
		model = GPTStageFull(args, configs, device)
	elif pp_rank == 0:
		model = GPTStageFirst(args, configs, device)
	elif pp_rank == pp_world_size-1:
		model = GPTStageLast(args, configs, device)
	else:
		model = GPTStageMiddle(args, configs, device)
	return model


def get_optimizer(args, configs, model, device):
	"""
	optimizer under fp16
	"""
	if args.training_mode == "pretrain":
		learning_rate = configs.lr
	else:
		learning_rate = args.lr
	#
	tmp_optimizer = create_optimizer(
		model, optimizer_type=getattr(args, 'optimizer', 'adamw'), learning_rate=learning_rate)
	optimizer = get_fp16_optimizer(args, tmp_optimizer, device)
	scheduler = get_linear_schedule_with_warmup(tmp_optimizer, args.warmup_steps, args.total_steps)

	return optimizer, scheduler


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

	model = get_model(args, configs, device)
	train_dataloader = get_train_data_loader(args, tokenizer)
	optimizer, scheduler = get_optimizer(args, configs, model, device)

	# traing & eval


if __name__ == "__main__":
	try:
		pretrain()
	except Exception as exc:
		print(traceback.print_exc(exc))
	#
	destroy_distributed_env()
