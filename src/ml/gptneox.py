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

class GPTStageBase(nn.Module):
	def __init__(self, args, config):
		super(GPTStageBase, self).__init__()
		self._num_layers = args.num_layers
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
			print("embed_in device:", layer.embed_in.weight.device)
			print("embed_in dtype:", layer.embed_in.weight.dtype)
			print()
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
			print("embed_out device:", layer.embed_out.weight.device)
			print("embed_out dtype:", layer.embed_out.weight.dtype)
			print()
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
			print("mlp.dense_4h_to_h device:", layer.mlp.dense_4h_to_h.weight.device)
			print("mlp.dense_4h_to_h dtype:", layer.mlp.dense_4h_to_h.weight.dtype)
			print()
			#
			del checkpoint
			gc.collect()
		return layer


class GPTStageFull(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageFull, self).__init__(args, config)
		self.device = device
		# module_list = [self._create_first_layer(device)]
		# for layer_idx in range(self._layer_begin, self._layer_end):
		# 	module_list.append(self._create_transformer_layer(device, layer_idx=layer_idx))
		# if hasattr(args, 'skip_lm_head') and args.skip_lm_head:
		# 	pass
		# else:
		# 	module_list.append(self._create_last_layer(device))
		# self.model = nn.Sequential(*module_list).to(device)

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

	def forward(self, x, **kargs):
		for module in self.model:
			x = module(x, **kargs)
		return x


class GPTStageFirst(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageFirst, self).__init__(args, config)
		self.device = device
		# module_list = [self._create_first_layer(device)]
		# for layer_idx in range(self._layer_begin, self._layer_end):
		# 	module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
		# self.model = nn.Sequential(*module_list).to(device)
		#
		self.model = nn.Sequential().to(device)
		#
		layer = self._create_first_layer(device)
		self.model.append(layer)
		#
		for layer_idx in range(self._layer_begin, self._layer_end):
			layer = self._create_transformer_layer(device, layer_idx=layer_idx)
			self.model.append(layer)

	def forward(self, x, **kargs):
		for module in self.model:
			x = module(x, **kargs)
		return x


class GPTStageMiddle(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageMiddle, self).__init__(args, config)
		self.device = device
		# module_list = []
		# for layer_idx in range(self._layer_begin, self._layer_end):
		# 	module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
		# self.model = nn.Sequential(*module_list).to(device)

		self.model = nn.Sequential().to(device)
		for layer_idx in range(self._layer_begin, self._layer_end):
			layer = self._create_transformer_layer(device, layer_idx=layer_idx)
			self.model.append(layer)

	def forward(self, x, **kargs):
		for module in self.model:
			x = module(x, **kargs)
		return x


class GPTStageLast(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageLast, self).__init__(args, config)
		self.device = device
		# module_list = []
		# for layer_idx in range(self._layer_begin, self._layer_end):
		# 	module_list.append(self._create_transformer_layer(layer_idx=layer_idx))

		# if hasattr(args, 'skip_lm_head') and args.skip_lm_head:
		# 	pass
		# else:
		# 	module_list.append(self._create_last_layer())

		# self.model = nn.Sequential(*module_list).to(device)

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

	def forward(self, x, **kargs):
		for module in self.model:
			x = module(x, **kargs)

		return x
