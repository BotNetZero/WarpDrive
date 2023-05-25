# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/16
Description   :
"""
import psutil
import torch


def get_node_memory():
	"""
	Get the maximum memory available if nothing is passed, converts string to int otherwise.
	"""
	max_memory = {}
	if torch.cuda.is_available():
		max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}

	max_memory["cpu"] = psutil.virtual_memory().available

	return max_memory


def check_device_map(device):
	max_memory = get_node_memory()
