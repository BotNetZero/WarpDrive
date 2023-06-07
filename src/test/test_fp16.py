# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/02
Description   : fp16 precision & optimizer
"""
import os, sys
sys.path.append(os.getcwd())

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from ml.optimizer import create_optimizer, get_fp16_optimizer


batch_size = 300
seq_len = 1000
vocab_size = 10
epochs = 2000

class Test(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		self.fc1 = nn.Linear(10, 100)
		self.fc2 = nn.Linear(100, vocab_size)
		self.act = nn.ReLU()

	def forward(self, inputs):
		hidden = self.fc1(inputs)		# [batch_size, seq_len, 100]
		hidden = self.act(hidden)
		logits = self.fc2(hidden)		# [batch_size, seq_len, 3]
		return logits


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--switch", type=int, default=0)
	parser.add_argument("--fp16", action='store_true')
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

if torch.cuda.is_available():
	device = torch.device(0)
	dtype = torch.float16
else:
	device = "cpu"
	dtype = torch.bfloat16
print(device)

args = parse_args()

# 对比fp32和fp16的训练效果
if args.switch == 0:
	dtype = torch.float32
	model = Test().to(device=device, dtype=dtype)
	optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-6)
else:
	model = Test().to(device=device, dtype=dtype)
	tmp_optimizer = create_optimizer(model, optimizer_type=getattr(args, 'optimizer', 'adamw'), learning_rate=1e-3)
	optimizer = get_fp16_optimizer(args, tmp_optimizer, device)

inputs = torch.rand(batch_size, seq_len, 10, dtype=dtype, device=device)
targets = torch.empty(batch_size, seq_len, dtype=torch.long).random_(vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss().to(device=device)

for epoch in range(epochs):
	print("epoch:", epoch)
	optimizer.zero_grad()
	#
	logits = model(inputs)
	# print(logits)
	# print("logits dtype:", logits.dtype)

	# print()
	# print("<========================>")
	fold_logits = logits.view(-1, logits.shape[-1])
	fold_targets = targets.view(-1)

	loss = loss_fn(fold_logits, fold_targets)
	print("loss:", loss.item(), loss.dtype)
	if args.switch == 0:
		loss.backward()
	else:
		optimizer.scale(loss).backward()
	optimizer.step()


