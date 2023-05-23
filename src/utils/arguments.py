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
	args.seq_length = configs.seq_length

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
	parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16", "int8"],
		     			help="model parameters precision (default: fp16)")
	parser.add_argument('--loss_scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
	parser.add_argument("--num_layers", type=int,
		     			help="number of model layers at current stage")
	parser.add_argument('--seed', type=int, default=1234,
					help='Random seed used for python, numpy, '
					'pytorch, and cuda.')


def _add_training_args(parser):
	parser.add_argument('--local_batch_size', type=int, default=10,
						help='Batch size per model instance (local batch size). '
						'Global batch size is local batch size times data parallel size times number of micro batches.')
	parser.add_argument('--global-batch-size', type=int, default=None,
						help='Training batch size. If set, it should be a '
						'multiple of micro-batch-size times data-parallel-size. '
						'If this value is None, then '
						'use micro-batch-size * data-parallel-size as the '
						'global batch size. This choice will result in 1 for '
						'number of micro-batches.')
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
	parser.add_argument('--recompute-activations', action='store_true',
						help='recompute activation to allow for training '
						'with larger models, sequences, and batch sizes.')
	parser.add_argument('--recompute-granularity', type=str, default=None,
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
	parser.add_argument('--recompute-method', type=str, default=None,
						choices=['uniform', 'block'],
						help='1) uniform: uniformly divide the total number of '
						'Transformer layers and recompute the input activation of '
						'each divided chunk at specified granularity, '
						'2) recompute the input activations of only a set number of '
						'individual Transformer layers per pipeline stage and do the '
						'rest without any recomputing at specified granularity'
						'default) do not apply activations recompute to any layers')
	parser.add_argument('--recompute-num-layers', type=int, default=1,
						help='1) uniform: the number of Transformer layers in each '
						'uniformly divided recompute unit, '
						'2) block: the number of individual Transformer layers '
						'to recompute within each pipeline stage.')

# def _add_mixed_precision_args(parser):
#     group = parser.add_argument_group(title='mixed precision')

#     group.add_argument('--fp16', action='store_true',
#                        help='Run model in fp16 mode.')
#     group.add_argument('--bf16', action='store_true',
#                        help='Run model in bfloat16 mode.')
#     group.add_argument('--loss-scale', type=float, default=None,
#                        help='Static loss scaling, positive power of 2 '
#                        'values can improve fp16 convergence. If None, dynamic'
#                        'loss scaling is used.')
#     group.add_argument('--initial-loss-scale', type=float, default=2**32,
#                        help='Initial loss-scale for dynamic loss scaling.')
#     group.add_argument('--min-loss-scale', type=float, default=1.0,
#                        help='Minimum loss scale for dynamic loss scale.')
#     group.add_argument('--loss-scale-window', type=float, default=1000,
#                        help='Window over which to raise/lower dynamic scale.')
#     group.add_argument('--hysteresis', type=int, default=2,
#                        help='hysteresis for dynamic loss scaling')
#     group.add_argument('--fp32-residual-connection', action='store_true',
#                        help='Move residual connections to fp32.')
#     group.add_argument('--no-query-key-layer-scaling', action='store_false',
#                        help='Do not scale Q * K^T by 1 / layer-number.',
#                        dest='apply_query_key_layer_scaling')
#     group.add_argument('--attention-softmax-in-fp32', action='store_true',
#                        help='Run attention masking and softmax in fp32. '
#                        'This flag is ignored unless '
#                        '--no-query-key-layer-scaling is specified.')
#     group.add_argument('--accumulate-allreduce-grads-in-fp32',
#                        action='store_true',
#                        help='Gradient accumulation and all-reduce in fp32.')
#     group.add_argument('--fp16-lm-cross-entropy', action='store_true',
#                        help='Move the cross entropy unreduced loss calculation'
#                        'for lm head to fp16.')

