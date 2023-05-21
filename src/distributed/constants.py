#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = []

_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True

try:
    from torch._C._distributed_c10d import ProcessGroupNCCL
    ProcessGroupNCCL.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupNCCL"]
except ImportError:
    _NCCL_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupGloo
    from torch._C._distributed_c10d import _ProcessGroupWrapper
    ProcessGroupGloo.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupGloo", "_ProcessGroupWrapper"]
except ImportError:
    _GLOO_AVAILABLE = False
