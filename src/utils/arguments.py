# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/06
Description   :
"""
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Warp Drive training engine')
	_add_distributed_arguments(parser)
	_model_arguments(parser)

	args = parser.parse_args()

	return args

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
	parser.add_argument("--training_mode", type=str, default="pretrain", choices=["train", "pretrain"],
						help="training mode: train from scratch, retrain (default: pretrain)")
	parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16", "int8"],
		     			help="model parameters precision (default: fp16)")
	parser.add_argument('--loss_scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
	parser.add_argument('--model_name', type=str, default='gptneox', metavar='S',
						help='model name in lowercase (default: gptneox)')


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

