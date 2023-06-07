# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/05
Description   : auto mixed precision
"""
import os, sys
sys.path.append(os.getcwd())

import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import argparse

start_time = None

def start_timer():
	global start_time
	gc.collect()
	torch.cuda.empty_cache()
	torch.cuda.reset_max_memory_allocated()
	torch.cuda.synchronize()
	start_time = time.time()

def end_timer_and_print(local_msg):
	torch.cuda.synchronize()
	end_time = time.time()
	print("\n" + local_msg)
	print("Total execution time = {:.3f} sec".format(end_time - start_time))
	print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--fp16", action='store_true')
	parser.add_argument("--switch", type=int, default=0)
	parser.add_argument('--loss_scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
	parser.add_argument('--initial_loss_scale', type=float, default=2**16,
						help='Initial loss-scale for dynamic loss scaling.')
	parser.add_argument('--min_loss_scale', type=float, default=1.0,
						help='Minimum loss scale for dynamic loss scale.')
	parser.add_argument('--loss_scale_window', type=float, default=1000,
						help='Window over which to raise/lower dynamic scale.')
	parser.add_argument('--hysteresis', type=int, default=2,
						help='hysteresis for dynamic loss scaling')

	args = parser.parse_args()
	return args



class Net(nn.Module):
	def __init__(self, inp_dim, out_size, num_layers) -> None:
		super().__init__()
		layers = []
		layers.append(nn.Linear(inp_dim, 5000))
		layers.append(nn.ReLU())
		for _ in range(num_layers):
			layers.append(nn.Linear(5000, 5000))
			layers.append(nn.ReLU())
		layers.append(nn.Linear(5000, out_size))
		self.layers = nn.Sequential(*tuple(layers))

	def forward(self, x):
		logits = self.layers(x)
		return logits


batch_size = 512 # Try, for example, 128, 256, 513.
in_size = 4096
out_size = 4096
num_layers = 30
num_batches = 50
epochs = 100


data = [torch.randn(batch_size, in_size, device="cuda") for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size, device="cuda") for _ in range(num_batches)]
loss_fn = nn.MSELoss().cuda()

model = Net(in_size, out_size, num_layers).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)
scaler = GradScaler()

args = parse_args()

if args.switch == 0:
	print("<<<==========mixed pricision training================>>>")
	start_timer()
	for epoch in range(epochs):
		print("epoch:", epoch)
		for input, target in zip(data, targets):
			optimizer.zero_grad()

			with autocast(device_type='cuda', dtype=torch.float16):
				output = model(input)
				loss = loss_fn(output, target)
				print("loss:", loss.item())
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
	end_timer_and_print("mixed pricicion training")
	print()

elif args.switch == 1:
	print("<<<==========FP32 training================>>>")
	start_timer()
	for epoch in range(epochs):
		print("epoch:", epoch)
		for input, target in zip(data, targets):
			optimizer.zero_grad()

			output = model(input)
			loss = loss_fn(output, target)
			print("loss:", loss.item())
			loss.backward()
			optimizer.step()

	end_timer_and_print("FP32 training")
	print()
