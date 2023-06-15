# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/14
Description   : quantization operator
"""
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd


class Quanz(Function):
	"""
	autocast embedding
	"""
	@staticmethod
	@custom_fwd
	def forward(ctx, emb):
		return emb

	@staticmethod
	@custom_bwd
	def backward(ctx, grad):
		return grad
