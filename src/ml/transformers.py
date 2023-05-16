# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/15
Description   :
"""
import torch
import torch.nn as nn


class Transformers(nn.Module):
	def __init__(self, *args, **kwargs)-> None:
		super().__init__(*args, **kwargs)
