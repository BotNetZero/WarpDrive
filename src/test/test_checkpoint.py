# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/29
Description   :
"""
import os, sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.ops.checkpoint import checkpoint

class Test(nn.Module):
	def __init__(self, checkpoint=True) -> None:
		super().__init__()
		self.layer1 = nn.Linear(5, 10)
		self.layer2 = nn.Linear(10,3)
		self.relu = nn.ReLU()
		self.checkpoint = checkpoint

	def _checkpoint_forward(self, x):
		def model_forward(x_):
			hiddens = self.layer1(x_)
			print("hidden grand_fn:", hiddens.grad_fn)
			return self.layer2(self.relu(hiddens))
		#
		preds = checkpoint(model_forward, x)

		return preds

	def forward(self, x):
		if self.checkpoint:
			preds = self._checkpoint_forward(x)
		else:
			hiddens = self.layer1(x)
			print("hidden grand_fn:", hiddens.grad_fn)
			preds = self.layer2(self.relu(hiddens))

		return preds


if __name__ == "__main__":
	torch.manual_seed(123)
	input = torch.rand(3, 5)
	print(input)
	targets = torch.tensor([1, 0, 2], dtype=torch.long)
	softmax = nn.Softmax(dim=1)
	loss_fn = nn.CrossEntropyLoss()
	#
	model = Test(checkpoint=False)

	logits = model(input)

	print("logits:", logits)
	preds = softmax(logits)
	print("preds:", preds)
	probs, labels = preds.max(dim=1)
	print(probs, labels)

	loss = loss_fn(preds, targets)
	print(loss)

	loss.backward()
	print(model.layer1.weight.grad)
	print()
	print("<<<======================>>>")
	print()

	model1 = Test(checkpoint=True)

	logits = model1(input)
	print(model1.layer1.weight.grad_fn)
	print("logits:", logits)
	preds = softmax(logits)
	print("preds:", preds)
	probs, labels = preds.max(dim=1)
	print(probs, labels)

	loss = loss_fn(preds, targets)
	print(loss)

	loss.backward()
	print(model1.layer1.weight.grad)
