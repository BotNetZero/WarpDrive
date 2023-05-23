# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/20
Description   : tiny model的推理
"""
import os, sys
sys.path.append(os.getcwd())

import torch
from src.common.constants import MODEL_PATH
from src.ml.tokenizer import Tokenizer
from src.ml.gptneox import GPTStageFull
from src.utils.arguments import parse_args
from src.common.constants import MODEL_PATH


def main():
	args, configs = parse_args()
	model_path = os.path.join(MODEL_PATH, "pythia_7b")
	tokenizer = Tokenizer(model_path)
	text = "I don't want to go home!!!"

	# model
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = "cpu"
	model = GPTStageFull(args, configs, device)
	# print(model)

	#
	batch_encoding = tokenizer.tokenize([text]).to(device)
	print(batch_encoding)

	out = model(batch_encoding["input_ids"])
	print(out)
	print(out.shape)

if __name__ == "__main__":
	main()