# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/06
Description   :
"""
import os
import argparse
from src.common.constants import CONFIG_PATH
from src.utils.file import read_yaml_file

def parse_args():
	parser = argparse.ArgumentParser(description='Warp Drive training engine')
	_add_distributed_arguments(parser)
	_model_arguments(parser)
	_add_training_args(parser)

	args = parser.parse_args()

	# load model configs
	if args.model_name.lower() == "pythia_7b":
		config_file = os.path.join(CONFIG_PATH, "pythia_7b.yaml")
		configs = read_yaml_file(config_file)
	else:
		raise NotImplementedError(f"LLM {args.model_name} not support YET!!!")

	#
	args.world_size = sum(args.gpus)
	args.pipeline_group_size = len(args.gpus)	# pipeline group size
	if args.pipeline_group_size < 2:
		raise ValueError(f"pipeline_group_size [{args.pipeline_group_size}] should be larger than 2")

	args.seq_length = configs.seq_length
	if args.global_batch_size is None:
		args.global_batch_size = args.batch_size * args.data_group_size
	else:
		assert args.global_batch_size == args.batch_size * args.data_group_size

	if args.batch_size % args.micro_batch_num != 0:
		raise ValueError(f"specify evenly divisible requirement: batch_size [{args.batch_size}] % micro_batch_num [{args.micro_batch_num}] != 0")
	args.micro_batch_size = args.batch_size // args.micro_batch_num

	return args, configs


def _add_distributed_arguments(parser):
	parser.add_argument("--group_name", type=str, default="user",
		     			help='process group name (default: user)')
	parser.add_argument("--mode", type=str, default="cluster",
		     			help="system mode: cluster, CS (default: cluster)")
	parser.add_argument('--pp_backend', type=str, default='nccl', metavar='S',
						help='backend type for pipeline parallel (default: nccl)')
	parser.add_argument('--dp_backend', type=str, default='nccl', metavar='S',
						help='backend type for data parallel (default: nccl)')
	parser.add_argument('--tp_backend', type=str, default='nccl', metavar='S',
						help='backend type for tensor parallel (default: nccl)')
	parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
						help='master ip:port for distributed env. (default: localhost, 9000)')
	parser.add_argument("--gpus", type=int, default=[1, 1], nargs="+",
						help="available gpus at each stage (default: [1,1])")
	parser.add_argument("--first_last_stage_merge", type=bool, default=False,
		     			help="merge first stage and last stage (default: False)")
	parser.add_argument("--stage", type=int, default=0,
		     			help="pipeline stage of current node (default: 0)")
	parser.add_argument('--data_group_size', type=int, default=1, metavar='D',
						help='data parallel group size (default: 1)')
	parser.add_argument('--tensor_group_size', type=int, default=1, metavar='D',
						help='tensor parallel group size (default: 1)')
	parser.add_argument('--local_rank', type=int, default=0, metavar='N',
	 					help='local rank of this gpu/process (default: 0)')
	parser.add_argument('--cuda_id', type=int, default=0, metavar='N',
						help='cuda indx, indicating which rank owns. (default: 0)')


def _model_arguments(parser):
	parser.add_argument('--model_name', type=str, default='pythia_7b', metavar='S',
						help='model name in lowercase (default: pythia_7b)')
	parser.add_argument("--training_mode", type=str, default="pretrain", choices=["train", "pretrain"],
						help="training mode: train from scratch, retrain (default: pretrain)")
	parser.add_argument('--optimizer', type=str, default='adamw',
					choices=['adam', 'sgd', "adamw"],
					help='Optimizer function name in lowercase (default: AdamW)')
	parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16", "int8"],
		     			help="model parameters precision (default: fp16)")
	parser.add_argument('--loss_scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
	parser.add_argument('--initial_loss_scale', type=float, default=2**32,
						help='Initial loss-scale for dynamic loss scaling.')
	parser.add_argument('--min_loss_scale', type=float, default=1.0,
						help='Minimum loss scale for dynamic loss scale.')
	parser.add_argument('--loss_scale_window', type=float, default=1000,
						help='Window over which to raise/lower dynamic scale.')
	parser.add_argument('--hysteresis', type=int, default=2,
						help='hysteresis for dynamic loss scaling')
	parser.add_argument("--num_layers", type=int,
		     			help="number of model layers at current stage")
	parser.add_argument('--seed', type=int, default=1234,
					help='Random seed used for python, numpy, '
					'pytorch, and cuda.')
	parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='learning rate ')


def _add_training_args(parser):
	parser.add_argument("--task_name", type=str, default="OIG/unified_chip2.jsonl:0.1",
		     			help="data files related to each task (default: OIG/unified_chip2.jsonl:0.1)")
	parser.add_argument('--batch_size', type=int, default=10,
						help='Batch size per rank (default: 10). '
						'global_batch_size = batch_size * data_group_size')
	parser.add_argument("--micro_batch_num", type=int, default=5,
		     			help="batch chunks for pipeline schedule (default: 5)")
	parser.add_argument('--global_batch_size', type=int, default=None,
						help='Training batch size summing all ranks in stage 0')
	parser.add_argument('--rampup-batch-size', nargs='*', default=None,
						help='Batch size ramp up with the following values:'
						'  --rampup-batch-size <start batch size> '
						'                      <batch size incerement> '
						'                      <ramp-up samples> '
						'For example:'
						'   --rampup-batch-size 16 8 300000 \ '
						'   --global-batch-size 1024'
						'will start with global batch size 16 and over '
						' (1024 - 16) / 8 = 126 intervals will increase'
						'the batch size linearly to 1024. In each interval'
						'we will use approximately 300000 / 126 = 2380 samples.')
	parser.add_argument('--recompute_activations', action='store_true',
						help='recompute activation to allow for training '
						'with larger models, sequences, and batch sizes.')
	parser.add_argument('--recompute_granularity', type=str, default=None,
						choices=['full', 'selective'],
						help='Checkpoint activations to allow for training '
						'with larger models, sequences, and batch sizes. '
						'It is supported at two granularities 1) full: '
						'whole transformer layer is recomputed, '
						'2) selective: core attention part of the transformer '
						'layer is recomputed.')
	parser.add_argument('--distribute-saved-activations',
						action='store_true',
						help='If set, distribute recomputed activations '
						'across model parallel group.')
	parser.add_argument("--warmup_steps", type=int, default=10,
		     			help="warmup period for LRScheduler (default: 10)")
	parser.add_argument("--total_steps", type=int, default=10000,
		     			help="total training steps (default: 10000)")
	parser.add_argument('--evaluation_steps', type=int, default=0, metavar='S',
                        help='every x steps, do evaluation. (0 means do not do evaluation)')


# def _add_mixed_precision_args(parser):

	# parser.add_argument('--fp32-residual-connection', action='store_true',
	# 					help='Move residual connections to fp32.')
	# parser.add_argument('--no-query-key-layer-scaling', action='store_false',
	# 					help='Do not scale Q * K^T by 1 / layer-number.',
	# 					dest='apply_query_key_layer_scaling')
	# parser.add_argument('--attention-softmax-in-fp32', action='store_true',
	# 					help='Run attention masking and softmax in fp32. '
	# 					'This flag is ignored unless '
	# 					'--no-query-key-layer-scaling is specified.')
	# parser.add_argument('--accumulate-allreduce-grads-in-fp32',
	# 					action='store_true',
	# 					help='Gradient accumulation and all-reduce in fp32.')
	# parser.add_argument('--fp16-lm-cross-entropy', action='store_true',
	# 					help='Move the cross entropy unreduced loss calculation'
	# 					'for lm head to fp16.')

