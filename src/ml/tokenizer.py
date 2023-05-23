# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/19
Description   : wrapping AutoTokenizer
"""
import torch
from transformers import AutoTokenizer


class Tokenizer:
	def __init__(self, pretrain_path) -> None:
		self.tokenizer = self._build_tokenizer(pretrain_path)
		self.tokenizer.padding_side = "right"

	def _build_tokenizer(self, pretrain_path):
		tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
		if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
		return tokenizer

	def tokenize(self, batch_X):
		"""
		batch_X = [sentence1, sentence2, ...]
		BatchEncoding format:
		{
			"input_ids": tensor,
			"attention_mask": tensor
		}
		tensor shape: [batch, max_length]

		:param batch_X: list of string
		:return: BatchEncoding
		"""
		if not isinstance(batch_X, list) and not isinstance(batch_X[0], str):
			raise TypeError(f"specify parameter type: list of str")
		#
		batch_encoding = self.tokenizer(
			batch_X,
			padding=True,
			return_tensors="pt"
		)
		return batch_encoding

	def ids(self, text):
		"""
		获取text的token id
		:return: list of token ids
		"""
		if not isinstance(text, str):
			raise TypeError(f"specify parameter type: str")

		token_ids = self.tokenizer.encode(text)
		return token_ids

	def decode(self, input_ids):
		"""
		将input_ids decode成text
		tensor([2,4,7]) ==> "xxx"
		"""
		return self.tokenizer.decode(input_ids)

	def encoding(self, texts):
		"""
		利用LLM对text做编码, 获取句子表征

		:return: sentence feature, size = [batch, emb_dim]
		"""
		if isinstance(texts, str):
			texts = [texts]
		if not isinstance(texts, list):
			raise TypeError(f"specify parameter type: str or list")
		#
		batch = len(texts)
		encoding = self.tokenize(texts)				#
		token_ids = encoding["input_ids"]
		hidden_states = self.predict(**encoding)	# hidden_states: [batch, seq_len, emb_dim]
		if batch == 1:
			lengths = -1
		else:
			lengths = torch.ne(token_ids, self.pad_token_id).sum(-1) - 1

		feature = hidden_states[torch.arange(batch), lengths]	# feature: [batch, emb_dim]

		return feature

	def padding(self, ids):
		"""
		对list of token ids进行padding
		ids = [[token1_id, token2_id, ...], [], ...]
		BatchEncoding format:
		{
			"input_ids": tensor,
			"attention_mask": tensor
		}
		tensor shape: [batch, max_length]

		:param ids: list of token ids,
		:return: BatchEncoding in tensor format
		"""
		batch_encoding = self.tokenizer.prepare_for_model(ids, padding=True, return_tensors="pt")

		return batch_encoding

	@property
	def pad_token_id(self):
		return self.tokenizer.pad_token_id

