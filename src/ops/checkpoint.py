# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/29
Description   : checkpoint opration
"""
import torch
from torch.autograd import Function
from torch.utils.checkpoint import detach_variable


class CheckPointOperation(Function):
	"""
	activation recompute operation
	"""
	@staticmethod
	def forward(ctx, model_forward, *args):
		"""
		:param model_forward: model forward function
		:param args: args for model forward function
		"""
		ctx.model_forward = model_forward

		with torch.no_grad():
			preds = model_forward(*args)

		# saved_tensors
		ctx.save_for_backward(*args)

		return preds

	@staticmethod
	def backward(ctx, *args):
		inputs = ctx.saved_tensors

		# recompute
		detached_inputs = detach_variable(inputs)
		with torch.enable_grad():
			preds = ctx.model_forward(*detached_inputs)
		torch.autograd.backward(preds, args)
		grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
		return grads


def checkpoint(model_forward, *args):
	return CheckPointOperation.apply(model_forward, *args)
