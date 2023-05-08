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
from src.communication.nccl_backend import NCCLCommunicator

_PIPELINE_PARALLEL_GROUP = None
_PIPELINE_PARALLEL_RANKS = None 
_PIPELINE_PARALLEL_WORLD_SIZE = None

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_RANKS = None
_DATA_PARALLEL_WORLD_SIZE = None

_TENSOR_PARALLEL_GROUP = None
_TENSOR_PARALLEL_RANKS = None
_TENSOR_PARALLEL_WORLD_SIZE = None


def get_pp_group():
	assert _PIPELINE_PARALLEL_GROUP is not None
	return _PIPELINE_PARALLEL_GROUP

def get_pp_ranks():
	assert _PIPELINE_PARALLEL_RANKS is not None
	return _PIPELINE_PARALLEL_RANKS

def get_pp_world_size():
	assert _PIPELINE_PARALLEL_WORLD_SIZE is not None
	return _PIPELINE_PARALLEL_WORLD_SIZE


def init_distributed_env(args):
	"""
	initialize torch.distributed DEFAULT PG
	"""
	device_count = cuda.device_count()

	# ignore data parallel, tensor parallel now....
	if args.data_group_size > 1 or args.tensor_group_size > 1:
		raise ValueError(f"Not support data parallel and tensor parallel YET....")
	
	# init args check
	if args.mode.lower() not in ("cluster", "cs"):
		raise ValueError(f"Current mode [{args.mode}] no support. Available modes: cluster, CS")
	if device_count == 0:
		raise ValueError(f"Node has no GPU. WarpDrive requires CUDA.")
	if args.cuda_id >= device_count:
		raise ValueError(f"CUDA idx [{args.cuda_id}] exceeds GPU number of node")

	if args.dp_backend.lower() == "nccl" and args.data_group_size*args.tensor_group_size > device_count:
		raise ValueError(f"At stage [{args.stage}], GPUs [{device_count}] < data_group_size [{args.data_group_size}] x tensor_group_size [{args.tensor_group_size}]")
	
	if args.local_rank >= args.data_group_size:
		raise ValueError(f"local rank [{args.local_rank}] is larger than data_group_size [{args.data_group_size}]")

	# init main PG
	init_main_pg(args.pp_backend, args.global_rank, args.world_size, args.dist_url, args.group_name)
	cuda.set_device(args.cuda_id)	# cuda_id --> global_rank

	# set pipeline, data parallel vars
	global _PIPELINE_PARALLEL_WORLD_SIZE
	global _DATA_PARALLEL_WORLD_SIZE
	global _TENSOR_PARALLEL_WORLD_SIZE

	# ranks of each group
	ppg_ranks, dpg_ranks, tpg_ranks = rank_topology(args)

	_PIPELINE_PARALLEL_WORLD_SIZE = args.pipeline_group_size
	init_pipeline_parallel_group(args.pp_backend, ppg_ranks, args.global_rank)
	
	if args.data_group_size > 1:
		_DATA_PARALLEL_WORLD_SIZE = args.data_group_size
		init_data_parallel_group(args.dp_backend, dpg_ranks, args.global_rank)

	if args.tensor_group_size > 1:
		_TENSOR_PARALLEL_WORLD_SIZE = args.tensor_group_size
		init_tensor_parallel_group(args.tp_backend, tpg_ranks, args.global_rank)

def rank_topology(args):
	"""
	分配每个GPU的global rank和group
	:return: 
		pp_ranks: ranks of each pipeline parallel group
		dp_ranks: ranks of each data parallel group
		tp_ranks: ranks of each ensor parallel group
	"""
	args.world_size = sum(args.gpus)
	args.pipeline_group_size = len(args.gpus)	# pipeline group size
	if args.pipeline_group_size < 2:
		raise ValueError(f"pipeline_group_size [{args.pipeline_group_size}] should be larger than 2")

	global_rank = 0
	for stage, gpus in enumerate(args.gpus):
		if stage < args.stage:
			global_rank += gpus
	args.global_rank = global_rank + args.local_rank
	#
	ppg_ranks = []		# ranks of each pp group
	dpg_ranks = []		# ranks of each dp group
	tpg_ranks = []		# ranks of each tp group
	min_gpus = min(args.gpus)
	max_gpus = max(args.gpus)
	if args.mode.lower() == "cluster":
		if min_gpus != max_gpus:
			raise ValueError(f"GPUs at each stage must be equal")
		if args.world_size % (args.tensor_group_size * args.pipeline_group_size) != 0:
			raise RuntimeError(
				f"world_size [{args.world_size}] is not divisible by tensor_group_size "
				f"[{args.tensor_group_size}] x pipeline_group_size [{args.pipeline_group_size}])"
      		)
		if args.world_size != (args.tensor_group_size * args.pipeline_group_size*args.data_group_size ):
			raise RuntimeError(
				f"world_size [{args.world_size}] != "
			)
		num_pp_grps = args.world_size // args.pipeline_group_size		# number of pipeline parallel groups
		num_tp_grps = args.world_size // args.tensor_group_size 		# number of tensor parallel groups
		for i in range(num_pp_grps):
			# pp ranks
			ranks = range(i, args.world_size, num_pp_grps)		# e.g.: [(0, 2, 4), (1, 3, 5)]
			ppg_ranks.append(list(ranks))
			# dp ranks
			start_rank = i * num_pp_grps
			end_rank = (i + 1) * num_pp_grps
			for j in range(args.tensor_group_size):
				ranks = range(start_rank+j, end_rank, args.tensor_group_size)
				dpg_ranks.append(list(ranks))
		#
		for i in range(num_tp_grps):
			ranks = range(i*args.tensor_group_size, (i+1)*args.tensor_group_size)
			tpg_ranks.append(list(ranks))
		
		logger.info(
			f"pipeline groups: {ppg_ranks}\n"
			f"data groups: {dpg_ranks}\n"
			f"tensor groups: {tpg_ranks}"
	      )
	elif args.mode.lower() == "cs":
		raise NotImplementedError(f"CS mode not support yet...")
	
	return ppg_ranks, dpg_ranks, tpg_ranks


def init_main_pg(backend, rank, world_size, dist_url, grp_name):
	"""
	main pg建群, pp grp和dp grp负责实际工作
	"""
	if dist.is_initialized():
		logger.info("MAIN PG with same name exists. Now destroy and rebuild it")
		dist.destroy_process_group()	# destroy main group & subgroup
	#
	grp_name = f"{grp_name}_mainPG_{world_size}:{rank}"
	logger.info(f"Global rank[{rank}] start building MAIN PG: [{grp_name}]...")
	dist.init_process_group(
		backend=backend, 							# 
		timeout=datetime.timedelta(seconds=2*60), 	# 2 min
		init_method=dist_url, 						# tcp init
		world_size=world_size,
		rank=rank,
		group_name=grp_name
	)

def init_pipeline_parallel_group(backend, ppg_ranks, global_rank):
	"""
	pipeline PG建群

	:param backend:
	"""
	global _PIPELINE_PARALLEL_RANKS
	global _PIPELINE_PARALLEL_GROUP
	assert dist.is_initialized()	# main group 
	for ranks in ppg_ranks:
		pg = dist.new_group(
			ranks=ranks,
			timeout=datetime.timedelta(seconds=2*60), 	# 
			backend=backend
		)
		if global_rank in ranks:
			_PIPELINE_PARALLEL_RANKS = ranks
			_PIPELINE_PARALLEL_GROUP = pg

def init_data_parallel_group(backend, dpg_ranks, global_rank):
	raise NotImplementedError()

def init_tensor_parallel_group(backend, tpg_ranks, global_rank):
	raise NotImplementedError()
