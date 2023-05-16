# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/10
Description   : copy from openchatkit and revise
"""
from copy import deepcopy
import torch
import torch.nn as nn
from src.distributed.comm_utils import get_pp_group_rank


class GPTStageBase(nn.Module):
	def __init__(self, args, config):
		super(GPTStageBase, self).__init__()
		self._embedding_dim = args.embedding_dim  # embedding dimension
		self._seq_length = args.seq_length
		# the dimension of the feedforward aws_network model in nn.TransformerEncoder
		self._feedforward_dim = args.embedding_dim * 4
		self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
		self._num_layers = args.num_layers
		self._layer_begin = get_pp_group_rank() * args.num_layers
		self._layer_end = min(self._layer_begin + args.num_layers, args.max_layers)

		self._task_type = getattr(args, 'task_type', 'language_model')

		self.load_pretrained_model = args.training_mode == "pretrain"
		self.model_name = args.model_name.lower()
		self.config = config

		if self.model_name == "gptneox":
			from src.ml.hf_gptneox_modules import GPTEmbeddings, GPTBlock, GPTLMHead
		else:
			raise NotImplementedError(f"{self.model_name} not support YET!!!")

		self._GPTEmbeddings = GPTEmbeddings
		self._GPTBlock = GPTBlock
		self._GPTLMHead = GPTLMHead

	def _create_first_layer(self, device):
		layer = self._GPTEmbeddings(deepcopy(self.config))
		if self.load_pretrained_model:
			layer.load_state_dict(
				torch.load(f'{self.model_name}/pytorch_embs.pt')
			)
		return layer

	def _create_last_layer(self):
		layer = self._GPTLMHead(deepcopy(self.config))
		if self.load_pretrained_model:
			layer.load_state_dict(
				torch.load(f'{self.model_name}/pytorch_lm_head.pt')
			)
		return layer

	def _create_transformer_layer(self, layer_idx=0):
		config = deepcopy(self.config)
		layer = self._GPTBlock(config, layer_id=layer_idx) # TODO: checkpoint
		if self.load_pretrained_model:
			print(f'loading layer {layer_idx}')
			layer.load_state_dict(
				torch.load(f'{self.model_name}/pytorch_{layer_idx}.pt')
			)
		return layer


class GPTStageFull(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageFull, self).__init__(args, config)
		self.device = device
		module_list = [self._create_first_layer()]
		for layer_idx in range(self._layer_begin, self._layer_end):
			module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
		if hasattr(args, 'skip_lm_head') and args.skip_lm_head:
			pass
		else:
			module_list.append(self._create_last_layer())
		self.model = nn.Sequential(*module_list).to(device)

	def forward(self, x, **kargs):
		for module in self.model:
			x = module(x, **kargs)
		return x


class GPTStageFirst(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageFirst, self).__init__(args, config)
		self.device = device
		module_list = [self._create_first_layer()]
		for layer_idx in range(self._layer_begin, self._layer_end):
			module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
		self.model = nn.Sequential(*module_list).to(device)

	def forward(self, x, **kargs):
		for module in self.model:
			x = module(x, **kargs)
		return x


class GPTStageMiddle(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageMiddle, self).__init__(args, config)
		self.device = device
		module_list = []
		for layer_idx in range(self._layer_begin, self._layer_end):
			module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
		self.model = nn.Sequential(*module_list).to(device)

	def forward(self, x, **kargs):
		for module in self.model:
			x = module(x, **kargs)
		return x


class GPTStageLast(GPTStageBase):
	def __init__(self, args, config, device):
		super(GPTStageLast, self).__init__(args, config)
		self.device = device
		module_list = []
		for layer_idx in range(self._layer_begin, self._layer_end):
			module_list.append(self._create_transformer_layer(layer_idx=layer_idx))

		if hasattr(args, 'skip_lm_head') and args.skip_lm_head:
			pass
		else:
			module_list.append(self._create_last_layer())

		self.model = nn.Sequential(*module_list).to(device)


	def forward(self, x, **kargs):
		for module in self.model:
			x = module(x, **kargs)

		return x

