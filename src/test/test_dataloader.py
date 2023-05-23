# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/23
Description   : 测试dataloader
"""
import os, sys
sys.path.append(os.getcwd())

from src.data.data_utils import get_train_data_loader
from src.utils.arguments import parse_args
from src.ml.tokenizer import Tokenizer
from src.common.constants import MODEL_PATH

args, _ = parse_args()
args.task_name = "\
OIG/unified_chip2.jsonl:0.1,\
OIG/unified_conv_finqa.jsonl:0.1 \
"
model_path = os.path.join(MODEL_PATH, "pythia_7b")

tokenizer = Tokenizer(model_path)
train_dataloader = get_train_data_loader(args, tokenizer)

for i, data in enumerate(train_dataloader):
    print(data)

    break

