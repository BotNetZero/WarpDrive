# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2022/10/13
Description   : Dataset, DataLoader
"""
import json
import random
from math import ceil
from torch.utils.data import Dataset
from src.utils.file import ensure_path


class ClassDataset(Dataset):
	def __init__(self, labeled_data_path, data_type="train", reload=False) -> None:
		"""
		:param labeled_data_path: 标签数据path
		:param data_type: train, valid, test
		:param reload: 重加载
		"""
		super().__init__()
		if data_type.lower() not in ("train", "valid", "test"):
			raise ValueError(f"data_type value error: {data_type}")

		self._labeled_data_path = ensure_path(labeled_data_path)
		self.data_type = data_type

		self._train_data_path = self._labeled_data_path/'train_data.json'
		self._valid_data_path = self._labeled_data_path/'valid_data.json'
		self._test_data_path = self._labeled_data_path/'test_data.json'

		if not self._train_data_path.exists() or reload:
			self.reload_labeled_data()

		self.data = []				#
		self._length = 0
		self.load_data(data_type)

	def reload_labeled_data(self):
		"""
		重新加载数据, 按照train, valid, test类型存储数据
		"""
		raise NotImplementedError()

	def _random_split(self, labeled_data):
		"""
		随机分配train, valid, test数据集, 比例8:1:1
		:param labeled_data:
		:type labeled_data : list
		"""
		random.shuffle(labeled_data)
		length = len(labeled_data)
		train_data_len = round(length*0.8)
		valid_data_len = round(length*0.1)
		train_data = labeled_data[0:train_data_len]
		valid_data = labeled_data[train_data_len:train_data_len+valid_data_len]
		test_data = labeled_data[train_data_len+valid_data_len:]

		return train_data, valid_data, test_data

	def load_data(self, data_type):
		"""
		加载指定类型的数据
		"""
		#
		if data_type == 'train':
			self._load_data(self._train_data_path)
		elif data_type == 'valid':
			self._load_data(self._valid_data_path)
		elif data_type == 'test':
			self._load_data(self._test_data_path)

	def _load_data(self, data_path):
		raise NotImplementedError()

	def __len__(self):
		return self._length


class ClassDataLoader:
	"""
	Dataset wrapper
	"""
	def __init__(self, dataset, batch_size=1, shuffle=False) -> None:
		"""
		:param dataset:
		:param batch_size: batch
		"""
		if not isinstance(dataset, ClassDataset):
			raise TypeError("dataset type: troch.utils.data.Dataset")
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle

	def __iter__(self):
		self._idx = 0
		if self.shuffle:
			random.shuffle(self.dataset.data)
		return self

	def __next__(self):
		raise NotImplementedError()
		# if self._idx == len(self):
		# 	raise StopIteration
		# _start = self._idx * self.batch_size
		# _end = (self._idx+1) * self.batch_size
		# if _end > len(self.dataset):
		# 	_end = len(self.dataset)
		# self._idx += 1
		# #
		# y = []
		# x = []
		# for data in self.dataset.data[_start:_end]:
		# 	x.append(data[0])
		# 	y.append(data[1])
		# return x, y

	def __len__(self):
		"""
		batch个数
		"""
		return ceil(len(self.dataset)/self.batch_size)
