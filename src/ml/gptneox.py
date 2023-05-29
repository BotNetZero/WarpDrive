# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/10
Description   : copy from openchatkit and revise
"""
import os
import gc
from copy import deepcopy
import torch
import torch.nn as nn
from src.common.constants import MODEL_PATH
from src.distributed.comm_utils import get_pp_group_rank
from src.accelerate.big_modeling import init_empty_weights, load_state_dict
from src.utils.conversion import normalize_precision
from src.ops.checkpoint import checkpoint


class GPTStageBase(nn.Module):
	def __init__(self, args, config, **kwargs):
		super(GPTStageBase, self).__init__()
		self._num_layers = args.num_layers
		full_mode = kwargs.get("full_mode", False)
		if full_mode:
			self._layer_begin = 0
			self._layer_end = min(self._layer_begin + args.num_layers, config.total_num_layers)
		else:
			self._layer_begin = get_pp_group_rank() * args.num_layers
			self._layer_end = min(self._layer_begin + args.num_layers, config.total_num_layers)

		self._task_type = getattr(args, 'task_type', 'language_model')

		self.load_pretrained_model = args.training_mode == "pretrain"
		self.model_name = args.model_name.lower()
		self.config = config
		self.dtype = normalize_precision(self.config.torch_dtype)		# pretrained model precision
		self.model_path = os.path.join(MODEL_PATH, self.model_name)

		if self.model_name == "pythia_7b":
			from src.ml.hf_gptneox_modules import GPTEmbeddings, GPTBlock, GPTLMHead
		else:
			raise NotImplementedError(f"{self.model_name} not support YET!!!")

		self._GPTEmbeddings = GPTEmbeddings
		self._GPTBlock = GPTBlock
		self._GPTLMHead = GPTLMHead

	def _create_first_layer(self, device):
		with init_empty_weights():
			layer = self._GPTEmbeddings(self.config)
		#
		if self.load_pretrained_model:
			print("load embedding")
			checkpoint = torch.load(f'{self.model_path}/pytorch_embs.pt')
			load_state_dict(layer, checkpoint, device, self.dtype)
			#
			del checkpoint
			gc.collect()
		return layer

	def _create_last_layer(self, device):
		with init_empty_weights():
			layer = self._GPTLMHead(self.config)
		if self.load_pretrained_model:
			print("load lm head")
			checkpoint = torch.load(f'{self.model_path}/pytorch_lm_head.pt')
			load_state_dict(layer, checkpoint, device, self.dtype)
			#
			del checkpoint
			gc.collect()
		return layer

	def _create_transformer_layer(self, device, layer_idx=0):
		with init_empty_weights():
			layer = self._GPTBlock(self.config, layer_id=layer_idx)
		if self.load_pretrained_model:
			print(f'loading layer {layer_idx}')
			checkpoint = torch.load(f'{self.model_path}/pytorch_{layer_idx}.pt')
			load_state_dict(layer, checkpoint, device, self.dtype)
			#
			del checkpoint
			gc.collect()
		return layer


class GPTStageFull(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageFull, self).__init__(args, config, full_mode=True)
		self.device = device
		self.model = nn.Sequential().to(device)
		#
		layer = self._create_first_layer(device)
		self.model.append(layer)
		#
		for layer_idx in range(self._layer_begin, self._layer_end):
			layer = self._create_transformer_layer(device, layer_idx=layer_idx)
			self.model.append(layer)
		#
		if hasattr(args, 'skip_lm_head') and args.skip_lm_head:
			pass
		else:
			layer = self._create_last_layer(device)
			self.model.append(layer)

	def _checkpoint_forward(self, x, **kwargs):
		"""
		"""
		def model_forward(x_, **kwargs):
			for module in self.model:
				x_ = module(x_, **kwargs)
			return x_
		checkpoint(model_forward, x, **kwargs)

	def forward(self, x, **kwargs):
		if self.args.recompute_activations:
			x = self._checkpoint_forward(x, **kwargs)
		else:
			for module in self.model:
				x = module(x, **kwargs)
		return x


class GPTStageFirst(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageFirst, self).__init__(args, config)
		self.device = device
		self.model = nn.Sequential().to(device)
		#
		layer = self._create_first_layer(device)
		self.model.append(layer)
		#
		for layer_idx in range(self._layer_begin, self._layer_end):
			layer = self._create_transformer_layer(device, layer_idx=layer_idx)
			self.model.append(layer)

	def forward(self, x, **kwargs):
		for module in self.model:
			x = module(x, **kwargs)
		return x


class GPTStageMiddle(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageMiddle, self).__init__(args, config)
		self.device = device
		self.model = nn.Sequential().to(device)
		for layer_idx in range(self._layer_begin, self._layer_end):
			layer = self._create_transformer_layer(device, layer_idx=layer_idx)
			self.model.append(layer)

	def forward(self, x, **kwargs):
		for module in self.model:
			x = module(x, **kwargs)
		return x


class GPTStageLast(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageLast, self).__init__(args, config)
		self.device = device
		self.model = nn.Sequential().to(device)
		for layer_idx in range(self._layer_begin, self._layer_end):
			layer = self._create_transformer_layer(device, layer_idx=layer_idx)
			self.model.append(layer)
		#
		if hasattr(args, 'skip_lm_head') and args.skip_lm_head:
			pass
		else:
			layer = self._create_last_layer(device)
			self.model.append(layer)

	def forward(self, x, **kwargs):
		for module in self.model:
			x = module(x, **kwargs)

		return x
