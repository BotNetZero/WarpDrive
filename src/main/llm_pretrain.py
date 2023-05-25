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
from src.ml.trainer import Trainer, Evaluator
from src.ml.gptneox import GPTStageFirst, GPTStageLast, GPTStageMiddle, GPTStageFull
from src.optimization.optimizer import create_optimizer, get_fp16_optimizer

pp_rank = get_pp_group_rank()
pp_world_size = get_pp_world_size()

def get_model(args, configs, device):
	"""
	基于rank所处stage创建model
	"""
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
	模型预训练
	"""
	args, configs = parse_args()
	init_distributed_env(args)
	comm = get_main_group_comm()

	device = torch.device(args.cuda_id)
	cuda.set_device(device)

	# model, optimizer, dataloader
	model_path = os.path.join(MODEL_PATH, "pythia_7b")
	tokenizer = Tokenizer(model_path)

	model = get_model(args, configs, device)
	train_dataloader = get_train_data_loader(args, tokenizer)
	optimizer, scheduler = get_optimizer(args, configs, model, device)
	optimizer.reload_model_params()

	# traing & eval
	trainer = Trainer(args, configs, device, model, optimizer, scheduler)
	# evaluater = Evaluator()

	#
	assert args.data_group_size == 1		# TODO: 暂时忽略data parallel

	stop_flag = torch.zeros(1, dtype=torch.int32).to(device)					# barrier flag
	input_ids = torch.zeros(													# recv from master rank
		args.batch_size, args.seq_length, dtype=torch.int32, device=device		#
	)
	stop_stream = cuda.Stream(device)	# stream for stop flag broadcast
	data_stream = cuda.Stream(device)	# stream for data broadcast

	# master rank: pp_rank 0, dp_rank 0
	if pp_rank == 0:						# TODO: dp_rank也是判断条件
		steps = 0
		for i, global_batch_X in enumerate(train_dataloader, 1):
			steps += 1

			comm.broadcast(stop_stream, stop_flag, 0, None, None)		# broadcast stop_flag for all ranks
			if stop_flag.item() == 1:
				break

			global_input_ids = global_batch_X["input_ids"]				#

			# input_ids_list = global_input_ids.chunk()						# # TODO: 分发数据做data parallel
			input_ids = global_input_ids.to(device)							# 当前rank的数据
			comm.broadcast(data_stream, input_ids, 0, stop_stream, None)	# input_ids在last stage用作label数据

			trainer()

			if steps > args.total_steps:
				stop_flag.data[:] = 1
	# last stage
	elif pp_rank == pp_world_size-1:
		while True:
			comm.broadcast(stop_stream, stop_flag, 0, None, None)			# broadcast stop_flag for all ranks
			if stop_flag.item() == 1:
				break
			# 接收input_ids
			comm.broadcast(data_stream, input_ids, 0, stop_stream, None)
			labels = input_ids.clone()

			trainer()

	# middle stage
	else:
		while True:
			comm.broadcast(stop_stream, stop_flag, 0, None, None)			# broadcast stop_flag for all ranks
			if stop_flag.item() == 1:
				break
			# 接收input_ids, middle stage不需要
			comm.broadcast(data_stream, input_ids, 0, stop_stream, None)	# TODO: 改进communicator, 在subgroup中做broadcast


if __name__ == "__main__":
	try:
		pretrain()
	except Exception as exc:
		print(traceback.print_exc(exc))
	#
	destroy_distributed_env()
