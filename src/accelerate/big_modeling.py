# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/18
Description   : copy from accelerate and revise
"""
from contextlib import contextmanager
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


@contextmanager
def init_empty_weights(include_buffers: bool = False):
	"""
	A context manager under which models are initialized with all parameters on the meta device, therefore creating an
	empty model. Useful when just initializing the model would blow the available RAM.

	Args:
		include_buffers (`bool`, *optional*, defaults to `False`):
			Whether or not to also put all buffers on the meta device while initializing.

	Example:

	```python
	import torch.nn as nn
	from accelerate import init_empty_weights

	# Initialize a model with 100 billions parameters in no time and without using any RAM.
	with init_empty_weights():
		tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
	```

	<Tip warning={true}>

	Any model created under this context manager has no weights. As such you can't do something like
	`model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

	</Tip>
	"""
    # if not is_torch_version(">=", "1.9.0"):
    #     raise NotImplementedError("Initializing empty weights to a meta device requires torch >= 1.9.0")
	with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
		yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
	"""
	A context manager under which models are initialized with all parameters on the specified device.

	Args:
		device (`torch.device`):
			Device to initialize all parameters on.
		include_buffers (`bool`, *optional*, defaults to `False`):
			Whether or not to also put all buffers on the meta device while initializing.

	Example:

	```python
	import torch.nn as nn
	from accelerate import init_on_device

	with init_on_device(device=torch.device("cuda")):
		tst = nn.Liner(100, 100)  # on `cuda` device
	```
	"""
	old_register_parameter = nn.Module.register_parameter
	if include_buffers:
		old_register_buffer = nn.Module.register_buffer

	def register_empty_parameter(module, name, param):
		old_register_parameter(module, name, param)
		if param is not None:
			param_cls = type(module._parameters[name])
			kwargs = module._parameters[name].__dict__
			module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

	def register_empty_buffer(module, name, buffer):
		old_register_buffer(module, name, buffer)
		if buffer is not None:
			module._buffers[name] = module._buffers[name].to(device)

	# Patch tensor creation
	if include_buffers:
		tensor_constructors_to_patch = {
			torch_function_name: getattr(torch, torch_function_name)
			for torch_function_name in ["empty", "zeros", "ones", "full"]
		}
	else:
		tensor_constructors_to_patch = {}

	def patch_tensor_constructor(fn):
		def wrapper(*args, **kwargs):
			kwargs["device"] = device
			return fn(*args, **kwargs)

		return wrapper

	try:
		nn.Module.register_parameter = register_empty_parameter
		if include_buffers:
			nn.Module.register_buffer = register_empty_buffer
		for torch_function_name in tensor_constructors_to_patch.keys():
			setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
		yield
	finally:
		nn.Module.register_parameter = old_register_parameter
		if include_buffers:
			nn.Module.register_buffer = old_register_buffer
		for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
			setattr(torch, torch_function_name, old_torch_function)


def load_state_dict(model, state_dict, device, dtype=None):
	"""
	revise model.load_state_dict for model in meta device

	:param model: model in meta device
	:param state_dict:
	:param device:
	:param dtype:
	"""
	for param_name, param in state_dict.items():
		_set_module_tensor_to_device(model, param_name, device, value=param, dtype=dtype)


def _set_module_tensor_to_device(
	module: nn.Module,
	tensor_name: str,
	device: Union[int, str, torch.device],
	value: Optional[torch.Tensor] = None,
	dtype: Optional[Union[str, torch.dtype]] = None,
	):
	"""
	A helper function to set a given tensor (parameter or buffer) of a module on a specific device (note that doing
	`param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

	Args:
		module (`torch.nn.Module`):
			The module in which the tensor we want to move lives.
		param_name (`str`):
			The full name of the parameter/buffer.
		device (`int`, `str` or `torch.device`):
			The device on which to set the tensor.
		value (`torch.Tensor`, *optional*):
			The value of the tensor (useful when going from the meta device to any other device).
		dtype (`torch.dtype`, *optional*):
			If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
			the dtype of the existing parameter in the model.
	"""
	# Recurse if needed
	if "." in tensor_name:
		splits = tensor_name.split(".")
		for split in splits[:-1]:
			new_module = getattr(module, split)
			if new_module is None:
				raise ValueError(f"{module} has no attribute {split}.")
			module = new_module
		tensor_name = splits[-1]

	if tensor_name not in module._parameters and tensor_name not in module._buffers:
		raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
	is_buffer = tensor_name in module._buffers
	old_value = getattr(module, tensor_name)

	if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
		raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

	if value is not None:
		if dtype is None:
			# For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
			value = value.to(old_value.dtype)
		elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
			value = value.to(dtype)

	with torch.no_grad():
		if value is None:
			new_value = old_value.to(device)
		elif isinstance(value, torch.Tensor):
			new_value = value.to(device)
		else:
			new_value = torch.tensor(value, device=device)

		if is_buffer:
			module._buffers[tensor_name] = new_value
		elif value is not None or torch.device(device) != module._parameters[tensor_name].device:
			param_cls = type(module._parameters[tensor_name])
			kwargs = module._parameters[tensor_name].__dict__
			if param_cls.__name__ == "Int8Params":
				# downcast to fp16 if any
				if new_value.dtype == torch.float32:
					new_value = new_value.to(torch.float16)
				new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
			else:
				new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)
			module._parameters[tensor_name] = new_value

			if module.__class__.__name__ == "Linear8bitLt" and getattr(module.weight, "SCB", None) is None:
				# quantize only if necessary
				device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
				if not getattr(module.weight, "SCB", None) and device_index is not None:
					if module.bias is not None and module.bias.device.type != "meta":
						# if a bias exists, we need to wait until the bias is set on the correct device
						module = module.cuda(device_index)
					elif module.bias is None:
						# if no bias exists, we can quantize right away
						module = module.cuda(device_index)
