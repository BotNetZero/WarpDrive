import os, sys
sys.path.append(os.getcwd())

import argparse
import datetime
import traceback
import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.amp.autocast_mode import autocast
from src.parallel.schedule import SequenceSchedule
from src.utils.arguments import parse_args
from src.distributed.comm_utils import init_distributed_env, destroy_distributed_env, get_main_group_comm, get_pp_group
from src.distributed.comm_utils import get_pp_group_rank, get_pp_world_size, get_pp_prev_global_rank, get_pp_next_global_rank

class TestModel(nn.Module):
	def __init__(self, emb_size) -> None:
		super().__init__()
		self.fc1 = nn.Linear(emb_size, emb_size)
		self.fc2 = nn.Linear(emb_size, emb_size)
		self.act = nn.ReLU()

	def forward(self, x):
		hid = self.act(self.fc1(x))
		logits = self.fc2(hid)
		return logits



def main(args):
	comm = get_main_group_comm()
	pp_rank = get_pp_group_rank()
	# pp_world_size = get_pp_world_size()
	args.batch_size = 10
	args.micro_batch_num = 5
	args.micro_batch_size = 2
	#
	args.seq_len = 5
	args.emb_size = 5

	device = torch.device(args.cuda_id)
	cuda.set_device(device)

	scheduler = SequenceSchedule(args.micro_batch_num)
	if pp_rank == 0:
		input_micro_batchs = torch.rand(args.batch_size, args.seq_len, args.emb_size, device=device).chunk(args.micro_batch_num, dim=0)
	else:
		input_micro_batchs = [			# input tensor with grad from previous rank
			torch.zeros(
				(args.micro_batch_size, args.seq_len, args.emb_size),
				requires_grad=True,
				dtype=torch.float16,
				device=device,
			) for _ in range(args.micro_batch_num)
		]

	model = TestModel(args.emb_size).to(device)
	send_stream = cuda.Stream(device)
	recv_stream = cuda.Stream(device)
	#
	for action, micro_batch_idx in scheduler:
		print(f"in rank [{pp_rank}], current action: {action}_{micro_batch_idx}")
		if action == "wait":
			continue
		elif action == "fw":
			if pp_rank == 0:
				#
				with autocast(device_type="cuda", dtype=torch.float16):
					preds = model(input_micro_batchs[micro_batch_idx])
					print("cal preds:", preds)
				#
				send_stream.wait_stream(cuda.current_stream(device))
				comm.send(send_stream, preds, 1, None, None)
				print("send preds:", preds)
				#
				send_stream.synchronize()

			else:
				recv_stream.wait_stream(cuda.current_stream(device))
				comm.recv(																	# recv from prev rank
					recv_stream,
					input_micro_batchs[micro_batch_idx],
					0, None, None)
				recv_stream.synchronize()
				print("recv preds:", input_micro_batchs[micro_batch_idx])
				#
				with autocast(device_type="cuda", dtype=torch.float16):
					preds = model(input_micro_batchs[micro_batch_idx])
				cuda.current_stream(device).synchronize()
		else:
			pass

if __name__ == "__main__":
	try:
		args, configs = parse_args()
		init_distributed_env(args)

		main(args)

	except Exception as exc:
		end_time = datetime.datetime.now()
		print("fail!!! Init PG at: ", end_time)
		traceback.print_exc()
	#
	destroy_distributed_env()
	print("destroy PG.....")
