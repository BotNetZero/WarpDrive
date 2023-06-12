# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/20
Description   : tiny model的推理
"""
import os, sys
sys.path.append(os.getcwd())

import torch
from torch.amp.autocast_mode import autocast
from src.common.constants import MODEL_PATH
from src.ml.tokenizer import Tokenizer
from src.ml.gptneox import GPTStageFull
from src.ml.gptneox import GPTStageFirst
from src.ml.trainer import loss_func
from src.utils.arguments import parse_args
from src.utils.memory import memory_status
from src.common.constants import MODEL_PATH
from src.data.data_utils import get_train_data_loader

def main():
	args, configs = parse_args()
	model_path = os.path.join(MODEL_PATH, "pythia_7b")
	tokenizer = Tokenizer(model_path)

	train_dataloader = get_train_data_loader(args, tokenizer)

	# model
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = "cpu"
	model = GPTStageFull(args, configs, device).float()
	# model = GPTStageFirst(args, configs, device).float()

	# print(model)

	memory_status()

	#
	for _, global_batch_X in enumerate(train_dataloader, 1):
		input_ids = global_batch_X["input_ids"].to(device)
		targets = input_ids.clone().to(torch.long)
		with autocast(device_type="cuda", dtype=torch.float16):
			preds = model(input_ids)
			print(preds)
			loss = loss_func(preds, targets)
			print(loss)
		print()
		memory_status()
		break


if __name__ == "__main__":
	main()
