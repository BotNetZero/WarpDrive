# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/29
Description   :
"""
import os, sys
sys.path.append(os.getcwd())

# import gc
# import time
# import torch
# import torch.nn as nn
# import torch.cuda as cuda
# from torch.utils.checkpoint import checkpoint
# from src.ml.gptneox import GPTStageFull
# from src.utils.arguments import parse_args
# from src.common.constants import MODEL_PATH


# def main():
# 	args, configs = parse_args()
# 	print("recompute activations:", args.recompute_activations)
# 	# model_path = os.path.join(MODEL_PATH, "pythia_7b")

# 	# model
# 	if torch.cuda.is_available():
# 		device = torch.device("cuda:0")
# 	else:
# 		device = "cpu"

# 	micro_batch = 8
# 	micro_batch_num = 100

# 	input_ids = torch.empty(micro_batch, 100, dtype=torch.long, device=device).random_(1000)

# 	print("<<===========================>>")
# 	print("normal forward...")
# 	model = GPTStageFull(args, configs, device)
# 	for _ in range(micro_batch_num):
# 		out = model(input_ids, ignore_checkpoint=True)

# 	del model
# 	gc.collect()
# 	torch.cuda.empty_cache()

# 	print("<<===========================>>")
# 	print("normal forward in eval mode...")
# 	time.sleep(60)

# 	model = GPTStageFull(args, configs, device)
# 	model.eval()
# 	for _ in range(micro_batch_num):
# 		out = model(input_ids, ignore_checkpoint=True)

# 	del model
# 	gc.collect()
# 	torch.cuda.empty_cache()

# 	print("<<===========================>>")
# 	print("checkpointing forward...")
# 	time.sleep(60)

# 	model = GPTStageFull(args, configs, device)
# 	for _ in range(micro_batch_num):
# 		out = model(input_ids, ignore_checkpoint=False)

# if __name__ == "__main__":
# 	main()

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# Define a sample neural network module
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        return self.checkpointed_forward(x)

    def checkpointed_forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)  # Using ReLU activation function
        x = self.conv2(x)
        x = torch.relu(x)  # Using ReLU activation function
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc(x)
        return x

# Create an instance of the model
model = MyModel().to("cuda:0")

# Create some random input
input_data = torch.randn(1, 3, 32, 32).to("cuda:0")

# Enable gradients for input_data
input_data.requires_grad_()

# Forward pass without checkpointing
output_no_checkpoint = model(input_data)

# Print memory usage without checkpointing
print(f"Memory used without checkpointing: {torch.cuda.memory_allocated()} bytes")

# Compute dummy loss
loss_no_checkpoint = torch.sum(output_no_checkpoint)

# Backward pass without checkpointing
loss_no_checkpoint.backward()


# Clear gradients
model.zero_grad()

# Enable checkpointing during forward pass
torch.backends.cudnn.enabled = False

# Forward pass with checkpointing
output_checkpoint = checkpoint.checkpoint(model, input_data)

# Print memory usage with checkpointing
print(f"Memory used with checkpointing: {torch.cuda.memory_allocated()} bytes")

# Compute dummy loss
loss_checkpoint = torch.sum(output_checkpoint)

# Backward pass with checkpointing
loss_checkpoint.backward()

