# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/05
Description   : 
"""
from datetime import datetime
import torch
import torch.cuda as cuda
import torch.distributed as dist

from src.common.logger import logger


_PIPELINE_PARALLEL_COMM = None
_PIPELINE_PARALLEL_RANK = None
_PIPELINE_PARALLEL_WORLD_SIZE = None

_DATA_PARALLEL_COMM = None
_DATA_PARALLEL_RANK = None
_DATA_PARALLEL_WORLD_SIZE = None

_TENSOR_PARALLEL_COMM = None
_TENSOR_PARALLEL_RANK = None
_TENSOR_PARALLEL_WORLD_SIZE = None


def get_pipeline_parallel_comm():
	assert _PIPELINE_PARALLEL_COMM is not None
	return _PIPELINE_PARALLEL_COMM


def get_pipeline_parallel_rank():
	assert _PIPELINE_PARALLEL_RANK is not None
	return _PIPELINE_PARALLEL_RANK


def get_pipeline_parallel_world_size():
	assert _PIPELINE_PARALLEL_WORLD_SIZE is not None
	return _PIPELINE_PARALLEL_WORLD_SIZE



def init_distributed_env(*args):
	"""
	initialize torch.distributed DEFAULT PG
	"""
	device_count = cuda.device_count()
	
	# init args check
	if device_count == 0:
		if args.pp_backend.lower() == "nccl":
			raise ValueError(f"Not support NCCL backend of pp mode, because Node has no GPU...")
		if args.data_group_size > 1 and args.dp_backend.lower() == "nccl":
			raise ValueError(f"Not support NCCL backend of dp mode, because Node has no GPU...")

	if args.dp_backend.lower() == "nccl" and args.data_group_size > device_count:
		logger.warning(f"argument data_group_size [{args.data_group_size}] is larger than available gpus [{device_count}] in this node")
		args.data_group_size = device_count
		logger.warning(f"reset argument data_group_size = {device_count}")

	if args.local_rank >= args.data_group_size:
		raise ValueError(f"local rank [{args.local_rank}] is larger than data_group_size [{args.data_group_size}]")
	
	# set pipeline, data parallel vars
	global _PIPELINE_PARALLEL_COMM
	global _PIPELINE_PARALLEL_RANK
	global _PIPELINE_PARALLEL_WORLD_SIZE
	global _DATA_PARALLEL_COMM
	global _DATA_PARALLEL_RANK
	global _DATA_PARALLEL_WORLD_SIZE

	_PIPELINE_PARALLEL_RANK = args.pipeline_stage
	_PIPELINE_PARALLEL_WORLD_SIZE = args.pipeline_group_size

	if args.data_group_size > 1:
		_DATA_PARALLEL_RANK = args.local_rank
		_DATA_PARALLEL_WORLD_SIZE = args.data_group_size

	# init default PG
	init_main_pg(args.pp_backend, args.global_rank, args.world_size, args.dist_url, args.group_name)
	cuda.set_device()


def init_main_pg(backend, rank, world_size, dist_url, grp_name):
	"""
	main pg建群, pp grp和dp grp负责实际工作
	"""
	if dist.is_initialized():	# default pg exist
		logger.info("MAIN PG with same name exists. Now destroy and rebuild it")
		dist.destroy_process_group()
	#
	grp_name = f"{grp_name}_mainPG_{world_size}:{rank}"
	logger.info(f"Global rank[{rank}] start building DEFAULT PG: [{grp_name}]...")
	dist.init_process_group(
		backend=backend, 							# 
		timeout=datetime.timedelta(seconds=2*60), 	# 2 min
		init_method=dist_url, 						# tcp init
		world_size=world_size,
		rank=rank,
		group_name=grp_name
	)
