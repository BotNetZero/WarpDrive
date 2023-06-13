import os, sys
sys.path.append(os.getcwd())

import os
import torch
import torch.cuda as cuda
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
from torch.amp.autocast_mode import autocast
from src.common.constants import MODEL_PATH
from src.ml.tokenizer import Tokenizer
from src.ml.gptneox import GPTStageFull
from src.ml.gptneox import GPTStageFirst
from src.ml.trainer import loss_func
from src.utils.arguments import parse_args
from src.utils.memory import memory_status
from src.data.data_utils import get_train_data_loader


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

memory_status()
#

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
				schedule=schedule(wait=1, warmup=1, active=2, repeat=2),
				on_trace_ready=tensorboard_trace_handler("./logs/test"),
				record_shapes=True,
				profile_memory=True,
				with_stack=True,
				with_flops=True,
				with_modules=True) as prof:
	for i, global_batch_X in enumerate(train_dataloader, 1):
		print("step:", i)
		input_ids = global_batch_X["input_ids"].to(device)
		targets = input_ids.clone().to(torch.long)
		with autocast(device_type="cuda", dtype=torch.float16):
			preds = model(input_ids)
			# print(preds)
			loss = loss_func(preds, targets)
			print(loss)

		prof.step()

		if i > 10:
			break
