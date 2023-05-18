# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/18
Description   :
"""
import torch


def normalize_precision(precision):
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:
        raise NotImplementedError(f"{precision} conversion not support YET!!!")
