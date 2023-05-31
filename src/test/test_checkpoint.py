# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/29
Description   :
"""
import os, sys
sys.path.append(os.getcwd())


import time
import torch
import torch.nn as nn
import torch.cuda as cuda
from src.ml.gptneox import GPTStageFull
from src.utils.arguments import parse_args


def main():
	args, configs = parse_args()
	print("recompute activations:", args.recompute_activations)
	# model_path = os.path.join(MODEL_PATH, "pythia_7b")

	# model
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = "cpu"

	micro_batch = 8
	micro_batch_num = 10

	input_ids = torch.empty(micro_batch, 1000, dtype=torch.long, device=device).random_(1000)
	model = GPTStageFull(args, configs, device)

	print()
	print("<<===========================>>")
	print("no grad forward...")
	for _ in range(micro_batch_num):
		with torch.no_grad():
			out = model(input_ids, ignore_checkpoint=False)
	print("In process, tensor requires_grad:", out.requires_grad)
	print("In process, tensor grad fn:", out.grad_fn)
	print("grad fn attributes:", dir(out.grad_fn))
	print(f"Memory after forward: {torch.cuda.memory_allocated()} bytes")
	del out
	cuda.empty_cache()
	print(f"Memory after del: {torch.cuda.memory_allocated()} bytes")


	print()
	print("<<===========================>>")
	print("checkpointing forward...")
	time.sleep(60)
	for _ in range(micro_batch_num):
		out = model(input_ids, ignore_checkpoint=False)
	print("In process, tensor requires_grad:", out.requires_grad)
	print("In process, tensor grad fn:", out.grad_fn)
	print("grad fn attributes:", dir(out.grad_fn))

	print(f"Memory after forward: {torch.cuda.memory_allocated()} bytes")
	del out
	cuda.empty_cache()
	print(f"Memory after del: {torch.cuda.memory_allocated()} bytes")

	print("<<===========================>>")
	print("normal forward...")
	time.sleep(60)
	for _ in range(micro_batch_num):
		out = model(input_ids, ignore_checkpoint=True)
	print("In process, tensor requires_grad:", out.requires_grad)
	print("In process, tensor grad fn:", out.grad_fn)
	print("grad fn attributes:", dir(out.grad_fn))

	print(f"Memory after forward: {torch.cuda.memory_allocated()} bytes")
	del out
	cuda.empty_cache()
	print(f"Memory after del: {torch.cuda.memory_allocated()} bytes")

if __name__ == "__main__":
	main()

