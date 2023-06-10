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
from torch.cuda.amp.grad_scaler import GradScaler
from transformers import get_linear_schedule_with_warmup
from src.utils.arguments import parse_args
from src.distributed.comm_utils import init_distributed_env, destroy_distributed_env, get_main_group_comm, get_pp_group
from src.distributed.comm_utils import get_pp_group_rank, get_pp_world_size, get_pp_prev_global_rank, get_pp_next_global_rank
from src.data.data_utils import get_train_data_loader
from src.common.constants import MODEL_PATH
from src.ml.tokenizer import Tokenizer
from src.ml.trainer import Trainer, Evaluator
from src.ml.gptneox import GPTStageFirst, GPTStageLast, GPTStageMiddle, GPTStageFull
from src.ml.optimizer import create_optimizer


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
	optimizer = create_optimizer(
		model, optimizer_type=getattr(args, 'optimizer', 'adamw'), learning_rate=learning_rate)

	lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps)

	grad_scaler = GradScaler(
		init_scale=args.initial_scale,
		growth_interval=args.growth_interval,
	)

	return optimizer, lr_scheduler, grad_scaler


def pretrain():
	"""
	模型预训练
	"""
	args, configs = parse_args()
	init_distributed_env(args)
	comm = get_main_group_comm()

	pp_rank = get_pp_group_rank()
	pp_world_size = get_pp_world_size()

	device = torch.device(args.cuda_id)
	cuda.set_device(device)

	# model, optimizer, dataloader
	model_path = os.path.join(MODEL_PATH, "pythia_7b")
	tokenizer = Tokenizer(model_path)

	model = get_model(args, configs, device)
	train_dataloader = get_train_data_loader(args, tokenizer)
	optimizer, lr_scheduler, grad_scaler = get_optimizer(args, configs, model, device)

	# traing & eval
	trainer = Trainer(args, configs, device, model, optimizer, lr_scheduler, grad_scaler)
	# evaluater = Evaluator()

	#
	assert args.data_group_size == 1		# TODO: 暂时忽略data parallel

	stop_flag = torch.zeros(1, dtype=torch.int32).to(device)					# barrier flag
	input_ids = torch.zeros(													# recv from master rank
		args.batch_size, args.seq_length, dtype=torch.long, device=device		#
	)
	stop_stream = cuda.Stream(device)	# stream for stop flag broadcast
	data_stream = cuda.Stream(device)	# stream for data broadcast

	# master rank: pp_rank 0, dp_rank 0
	if pp_rank == 0:						# TODO: dp_rank也是判断条件
		for _, global_batch_X in enumerate(train_dataloader, 1):		# master rank加载所有训练数据
			#
			stop_stream.wait_stream(cuda.current_stream())				# make sure prev iteration is over and tensors for comm is ready
			comm.broadcast(stop_stream, stop_flag, 0, None, None)		# broadcast stop_flag for all ranks
			stop_stream.synchronize()									# make sure comm is finished, and all ranks get the same stop_flag

			# under defult stream
			if stop_flag.item() == 1:
				print("finished training, then stop....")
				break
			#
			global_input_ids = global_batch_X["input_ids"]				#

			# input_ids_list = global_input_ids.chunk(dp_size)			# # TODO: 分发数据做data parallel
			input_ids = global_input_ids.to(device)						# master rank的数据
			print("broadcast input ids:", input_ids)

			# broadcast input data under data_stream
			data_stream.wait_stream(cuda.current_stream())				# make sure tensors for comm is updated
			comm.broadcast(data_stream, input_ids, 0, None, None)		# input_ids在last stage用作label数据

			# start training while broadcasting...
			trainer(input_ids, None)									# one training step
			
			# # TODO: evaluator
			# if trainer.global_step % args.evaluation_steps == 0:
			# 	evaluator()

			# under default stream
			if trainer.global_step > args.total_steps:
				stop_flag.data[:] = 1

	# last stage
	elif pp_rank == pp_world_size-1:
		while True:
			stop_stream.wait_stream(cuda.current_stream())				# make sure prev iteration is over
			comm.broadcast(stop_stream, stop_flag, 0, None, None)		# broadcast stop_flag for all ranks
			stop_stream.synchronize()									# make sure data is received

			# default stream
			if stop_flag.item() == 1:
				print("finished training, then stop....")
				break

			# recv input_ids under data_stream
			data_stream.wait_stream(cuda.current_stream())				# make sure stop_flag is checked
			comm.broadcast(data_stream, input_ids, 0, None, None)		# recv input_ids
			data_stream.synchronize()									# make sure comm is finished

			# default stream
			print("broadcast input ids:", input_ids)
			labels = input_ids.clone()

			# one training step
			trainer(None, labels)										# one training step

			# # TODO: evaluator
			# if trainer.global_step % args.evaluation_steps == 0:
			# 	evaluator()

	# middle stage
	else:
		while True:
			stop_stream.wait_stream(cuda.current_stream())				# make sure prev iteration is over
			comm.broadcast(stop_stream, stop_flag, 0, None, None)		# recv stop_flag
			stop_stream.synchronize()									# make sure data is received

			# under default stream
			if stop_flag.item() == 1:
				print("finished training, then stop....")
				break

			# recv input_ids
			data_stream.wait_stream(cuda.current_stream())				# make sure stop_flag is checked
			comm.broadcast(data_stream, input_ids, 0, None, None)		# TODO: 改进communicator, 在subgroup中做broadcast
			# data_stream.synchronize()									# middle stage don't use input_ids, therefore no synchronize...

			# starting training while broadcasting
			trainer(None, None)											# one training step

			# # TODO: evaluator
			# if trainer.global_step % args.evaluation_steps == 0:
			# 	evaluator()

	# makesure training is over...
	cuda.synchronize()

if __name__ == "__main__":
	try:
		pretrain()
	except Exception as exc:
		traceback.print_exc()
	#
	destroy_distributed_env()
