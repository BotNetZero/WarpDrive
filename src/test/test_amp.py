# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/05
Description   : auto mixed precision
"""
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler


class Net(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.fc1 = nn.Linear(10, 100)
		self.fc2 = nn.Linear(100, 3)
		self.act = nn.ReLU()

	def forward(self, x):
		hid = self.fc1(x)
		activation = self.act(hid)
		logits = self.fc2(activation)
		return logits


model = Net()
loss_fn = nn.CrossEntropyLoss()

x = torch.rand(3, 10)
target = torch.empty(3, dtype=torch.long).random_(3)

with autocast(device_type="cpu", dtype=torch.bfloat16):
	logits = model(x)
	print(logits.dtype)
	loss = loss_fn(logits, target)
	print(loss.dtype)
