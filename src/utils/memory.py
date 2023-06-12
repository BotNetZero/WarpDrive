# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/16
Description   :
"""
import psutil
import torch
import torch.cuda as cuda

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


def memory_status(reset_max=False):
	"""
	record cuda memory status
	"""
	if reset_max:
		cuda.reset_max_memory_cached()
		cuda.reset_max_memory_allocated()

	new_alloced = cuda.memory_allocated()
	new_cached  = cuda.memory_reserved()

	max_alloced = cuda.max_memory_allocated()
	max_cached = cuda.max_memory_reserved()

	# convert to GB for printing
	new_alloced /= 1024**3
	new_cached /= 1024**3
	max_alloced /= 1024**3
	max_cached /= 1024**3

	print(
		f'current alloc={new_alloced:0.4f}GB  max allocated={max_alloced:0.4f}GB) '
		f'current cache={new_cached:0.4f}GB max cached={max_cached:0.4f}GB)')
