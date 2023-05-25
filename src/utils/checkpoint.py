# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/24
Description   : latest checkpoint saving & loading
"""
import os
from src.common.constants import MODEL_PATH

def load_checkpoint(args):
    ckp_file = args.checkpoint_file_name + "latest"
    file = os.path.join(MODEL_PATH, )

def save_checkpoint(args):
    pass
