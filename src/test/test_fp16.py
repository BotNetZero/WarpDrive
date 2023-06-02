# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/02
Description   : fp16 precision
"""
import torch
import torch.nn as nn

batch_size = 3
seq_len = 5
vocab_size = 10

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


if torch.cuda.is_available():
	device = torch.device(0)
	dtype = torch.float16
else:
	device = "cpu"
	dtype = torch.float
print(device)


inputs = torch.rand(batch_size, seq_len, 10, dtype=dtype, device=device)
targets = torch.empty(batch_size, seq_len, dtype=torch.long).random_(vocab_size).to(device)

print("inputs dtype:", inputs.dtype)

model = Test().to(device=device, dtype=dtype)
print("model weight dtype:", model.fc1.weight.dtype)

logits = model(inputs)
print(logits)
print("logits dtype:", logits.dtype)

print()
print("<========================>")
fold_logits = logits.view(-1, logits.shape[-1])
fold_targets = targets.view(-1)


loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(fold_logits, fold_targets)
print(loss)
print("loss dtype:", loss.dtype)

loss.backward()


