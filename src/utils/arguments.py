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

def _debug_arguments(parser):
	pass
