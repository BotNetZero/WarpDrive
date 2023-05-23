#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import pickle
import logging
import itertools
import collections.abc
import functools
import contextlib
from datetime import timedelta
from typing import Optional, Union, List, Callable, Any

import torch
from torch.autograd.profiler import record_function
from torch.distributed import supports_complex, Backend, default_pg_timeout
from torch._C._distributed_c10d import (
    ProcessGroup,
    Work,
    BroadcastOptions,
    AllreduceOptions,
    ReduceOptions,
    GatherOptions,
    ScatterOptions,
    ReduceScatterOptions,
    AllToAllOptions,
    BarrierOptions,
    ReduceOp,
)

from .group import Group
from .manager import _group_manager
from .helper import new_process_group_helper, store_based_barrier, \
    GLOO_AVAILABLE, NCCL_AVAILABLE, _ProcessGroupWrapper


logger = logging.getLogger(__name__)

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


def _check_group(group: Group):
    if group == None:
        raise ValueError(
            "Invalid group specified: group argument should not be None. "
            "Please specify a group or use the default process group."
        )
    if not _group_manager.exist_group(group):
        raise ValueError(
            "Invalid group specified: group argument should be a valid "
            "Group created via new_group, "
            "which is created at the first invocation of new_process_group."
        )


def isend(tensor: torch.Tensor, group: Group, dst: int, tag: int = 0) -> Work:
    """
    Sends a tensor asynchronously.

    .. warning::
        Modifying ``tensor`` before the request completes causes undefined
        behavior.

    Args:
        tensor (Tensor): Tensor to send.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        dst (int): Destination rank.
        tag (int, optional): Tag to match send with remote recv

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    _check_group(group)

    group_dst_rank = get_group_rank(group, dst)
    return group.pg.send([tensor], group_dst_rank, tag)


def irecv(tensor: torch.Tensor, group: Group, src: Optional[int] = None, tag: int = 0) -> Work:
    """
    Receives a tensor asynchronously.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        src (int, optional): Source rank. Will receive from any
            process if unspecified.
        tag (int, optional): Tag to match recv with remote send

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    _check_group(group)

    if src is None:
        return group.pg.recv_anysource([tensor], tag)
    else:
        group_src_rank = get_group_rank(group.pg, src)
        return group.pg.recv([tensor], group_src_rank, tag)


def send(tensor: torch.Tensor, group: Group, dst: int, tag: int = 0) -> None:
    """
    Sends a tensor synchronously.

    Args:
        tensor (Tensor): Tensor to send.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        dst (int): Destination rank. Destination rank should not be the same
        as the rank of the current process.
        tag (int, optional): Tag to match send with remote recv

    """
    _check_group(group)
    if group.rank == dst:
        raise ValueError(
            "Invalid destination rank: destination rank should not be the same as "
            "the rank of the current process."
        )

    _check_single_tensor(tensor, "tensor")

    group_dst_rank = get_group_rank(group, dst)
    group.pg.send([tensor], group_dst_rank, tag).wait()


def recv(tensor: torch.Tensor, group: Group, src: Optional[int] = None, tag: int = 0) -> int:
    """
    Receives a tensor synchronously.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        src (int, optional): Source rank. Will receive from any
            process if unspecified.
        tag (int, optional): Tag to match recv with remote send

    Returns:
        Sender rank
        -1, if not part of the group

    """
    _check_group(group)

    if src is None:
        work = group.pg.recv_anysource([tensor], tag)
        work.wait()
        src_rank = work._source_rank()
        # 从 group.rank_map 查找 src_rank 对应的 group_rank
        return get_group_rank(group, src_rank)
    else:
        group_src_rank = get_group_rank(group, src)
        group.pg.recv([tensor], group_src_rank, tag).wait()
        return src


class P2POp:
    """
    A class to build point-to-point operations for ``batch_isend_irecv``.

    This class builds the type of P2P operation, communication buffer, peer rank,
    Process Group group, and tag. Instances of this class will be passed to
    ``batch_isend_irecv`` for point-to-point communications.

    Args:
        op (Callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``torch.distributed.isend`` or
            ``torch.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int): Destination or source rank.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with recv.
    """

    def __init__(self, op: Callable, tensor: torch.Tensor, peer: int, group: Group, tag: int=0):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag

    def __new__(cls, op, tensor: torch.Tensor, peer: int, group: Group, tag=0):
        _check_op(op)
        _check_single_tensor(tensor, "tensor")
        return object.__new__(cls)


def _check_op(op):
    """
    Helper to check that the ``op`` is either isend or irecv.
    """
    if op not in [isend, irecv]:
        raise RuntimeError(
            "Invalid ``op``. Expected ``op`` "
            "to be of type ``torch.distributed.isend`` or "
            "``torch.distributed.irecv``."
        )


@contextlib.contextmanager
def _coalescing_manager(group: Group, device: Union[str, torch.device], reqs):
    _check_group(group)
    group.pg._start_coalescing(device)
    try:
        yield
    finally:
        group.pg._end_coalescing(device, reqs)


def batch_isend_irecv(p2p_op_list: List[P2POp]):
    """
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in ``p2p_op_list`` and return the corresponding
    requests. NCCL, Gloo, and UCC backend are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``torch.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed request objects returned by calling the corresponding
        op in the op_list.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> send_tensor = torch.arange(2) + 2 * rank
        >>> recv_tensor = torch.randn(2)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
        >>> recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1

    .. note:: Note that when this API is used with the NCCL PG backend, users must set
        the current GPU device with `torch.cuda.set_device`, otherwise it will
        lead to unexpected hang issues.

        In addition, if this API is the first collective call in the ``group``
        passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
        this API call; otherwise, the behavior is undefined. If this API call is
        not the first collective call in the ``group``, batched P2P operations
        involving only a subset of ranks of the ``group`` are allowed.
    """
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    device = p2p_op_list[0].tensor.device
    reqs = []
    with _coalescing_manager(group, device, reqs):
        for p2p_op in p2p_op_list:
            op = p2p_op.op
            tensor = p2p_op.tensor
            peer = p2p_op.peer
            curr_group = p2p_op.group
            tag = p2p_op.tag

            ret = op(tensor, peer, curr_group, tag)

            if ret is not None:
                reqs.append(ret)
    return reqs


def exception_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            error_msg_dict = {
                "func_name": f"{func.__name__}",
                "args": f"{args}, {kwargs}",
                "error": f"{error}",
            }
            logger.debug(error_msg_dict)
            raise
    return wrapper


# broadcast_multigpu will be deprecated: 
#     https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions


@exception_handler
def broadcast(tensor: torch.Tensor, group: Group, src: int, async_op: bool=False):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        src (int): Source rank.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    _check_group(group)

    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0

    group_src_rank = get_group_rank(group, src)
    opts.rootRank = group_src_rank
    work = group.pg.broadcast([tensor], opts)
    if async_op:
        return work
    else:
        work.wait()


# all_reduce_multigpu will be deprecated: 
#     https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions


@exception_handler
def all_reduce(tensor: torch.Tensor, group: Group, op=ReduceOp.SUM, async_op: bool=False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4, 6]) # Rank 0
        tensor([4, 6]) # Rank 1

        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j]) # Rank 0
        tensor([3.+3.j, 4.+4.j]) # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4.+4.j, 6.+6.j]) # Rank 0
        tensor([4.+4.j, 6.+6.j]) # Rank 1

    """
    _check_single_tensor(tensor, "tensor")
    _check_group(group)

    if tensor.is_complex():
        if not supports_complex(op):
            raise RuntimeError(f"all_reduce does not support {op} on complex tensors")
        tensor = torch.view_as_real(tensor)

    opts = AllreduceOptions()
    opts.reduceOp = op
    work = group.pg.allreduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


# all_reduce_coalesced will be deprecated:
#     https://pytorch.org/docs/master/distributed.html#collective-functions


# reduce_multigpu will be deprecated:
#     https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions


@exception_handler
def reduce(tensor: torch.Tensor, group: Group, dst: int, op=ReduceOp.SUM, async_op=False):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    _check_group(group)

    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst

    group_dst_rank = get_group_rank(group, dst)
    opts.rootRank = group_dst_rank
    work = group.pg.reduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


# all_gather_multigpu will be deprecated:
#     https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions


@exception_handler
def all_gather(tensor_list: List[torch.Tensor], tensor: torch.Tensor, group: Group, async_op: bool=False):
    """
    Gathers tensors from the whole group in a list.

    Complex tensors are supported.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(2)]
        >>> tensor_list
        [tensor([0, 0]), tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1, 2]), tensor([3, 4])] # Rank 0
        [tensor([1, 2]), tensor([3, 4])] # Rank 1

        >>> # All tensors below are of torch.cfloat dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zeros(2, dtype=torch.cfloat) for _ in range(2)]
        >>> tensor_list
        [tensor([0.+0.j, 0.+0.j]), tensor([0.+0.j, 0.+0.j])] # Rank 0 and 1
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j]) # Rank 0
        tensor([3.+3.j, 4.+4.j]) # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 0
        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 1

    """
    _check_tensor_list(tensor_list, "tensor_list")
    _check_single_tensor(tensor, "tensor")
    _ensure_all_tensors_same_dtype(tensor_list, tensor)
    _check_group(group)

    tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in tensor_list
    ]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    work = group.pg.allgather([tensor_list], [tensor])

    if async_op:
        return work
    else:
        work.wait()


def _check_for_nccl_backend(group: Group):
    _check_group(group)
    pg: ProcessGroup = group.pg
    # Gate PG wrapper check on Gloo availability.
    if GLOO_AVAILABLE:
        # It is not expected for PG to be wrapped many times, but support it just
        # in case
        while isinstance(pg, _ProcessGroupWrapper):
            pg = pg.wrapped_pg

    return (
        NCCL_AVAILABLE and
        pg.name() == Backend.NCCL
    )


def _validate_output_list_for_rank(my_rank: int, dst: int, gather_list: List[Any]):
    if dst == my_rank:
        if not gather_list:
            raise ValueError(
                "Argument ``gather_list`` must be specified on destination rank."
            )
    elif gather_list:
        raise ValueError(
            "Argument ``gather_list`` must NOT be specified "
            "on non-destination ranks."
        )


def _get_pg_device(group: Group) -> torch.device:
    """
    Returns the device to use with ``group``.
    This is cuda for NCCL and CPU for everything else
    """
    if _check_for_nccl_backend(group):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _object_to_tensor(obj: Any, device: torch.device):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


def _tensor_to_object(tensor: torch.Tensor, tensor_size: int):
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


@exception_handler
def all_gather_object(object_list: List[Any], obj: Any, group: Group):
    """
    Gathers picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the object
    must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        obj (Any): Pickable Python object to be broadcast from current process.
        group (Group): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.all_gather_object(output, gather_objects[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]
    """
    _check_group(group)

    current_device = _get_pg_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = group.pg.size()
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    all_gather(object_size_list, local_size, group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device
    )
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    all_gather(output_tensors, input_tensor, group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device("cpu"):
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


@exception_handler
def gather(tensor: torch.Tensor, group: Group, gather_list: list[torch.Tensor]=None, dst: int=0, async_op=False):
    """
    Gathers a list of tensors in a single process.

    Args:
        tensor (Tensor): Input tensor.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_single_tensor(tensor, "tensor")

    # Parameter ``gather_list`` may be left unspecified on non-dst ranks.
    if gather_list:
        _check_tensor_list(gather_list, "gather_list")
    else:
        gather_list = []
    _ensure_all_tensors_same_dtype(tensor, gather_list)

    _check_group(group)

    my_rank = group.pg.rank()
    _validate_output_list_for_rank(my_rank, dst, gather_list)
    output_tensors = [gather_list] if dst == my_rank else []
    input_tensors = [tensor]

    opts = GatherOptions()
    opts.rootRank = dst

    group_dst_rank = get_group_rank(group, dst)
    opts.rootRank = group_dst_rank
    work = group.pg.gather(output_tensors, input_tensors, opts)

    if async_op:
        return work
    else:
        work.wait()


@exception_handler
def gather_object(obj: Any, group: Group, object_gather_list: List[Any]=None, dst: int=0):
    """
    Gathers picklable objects from the whole group in a single process.
    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Args:
        obj (Any): Input object. Must be picklable.
        group: (Group): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it
            should be correctly sized as the size of the group for this
            collective and will contain the output. Must be ``None`` on non-dst
            ranks. (default is ``None``)
        dst (int, optional): Destination rank. (default is 0)

    Returns:
        None. On the ``dst`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API differs slightly from the gather collective
        since it does not provide an async_op handle and thus will be a blocking
        call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.gather_object(
        ...     gather_objects[dist.get_rank()],
        ...     output if dist.get_rank() == 0 else None,
        ...     dst=0
        ... )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]
    """
    _check_group(group)

    # Ensure object_gather_list is specified appopriately.
    my_rank = group.pg.rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    current_device = _get_pg_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = group.pg.size()
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_rank == dst:
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size, dtype=torch.uint8, device=current_device
        )
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    gather(
        input_tensor,
        gather_list=output_tensors if my_rank == dst else None,
        dst=dst,
        group=group,
    )
    if my_rank != dst:
        return
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)


@exception_handler
def broadcast_object_list(object_list: List[Any], group: Group, src: int=0, device: torch.device=None):
    """
    Broadcasts picklable objects in ``object_list`` to the whole group. Similar
    to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        group: (Group): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        src (int): Source rank from which to broadcast ``object_list``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> dist.broadcast_object_list(objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    _check_group(group)

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    current_device = device or _get_pg_device(group)
    my_rank = group.pg.rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(*[_object_to_tensor(obj, current_device) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long, device=current_device)

    # Broadcast object sizes
    broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=current_device
        )

    broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device("cpu"):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


@exception_handler
def scatter(tensor: torch.Tensor, scatter_list: List[torch.Tensor], group: Group, src: int=0, async_op: bool=False):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        group (Group): The process group to work on. If None,
            the default process group will be used.
        src (int): Source rank (default is 0)
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: Note that all Tensors in scatter_list must have the same size.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> tensor_size = 2
        >>> t_ones = torch.ones(tensor_size)
        >>> t_fives = torch.ones(tensor_size) * 5
        >>> output_tensor = torch.zeros(tensor_size)
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     # Only tensors, all of which must be the same size.
        >>>     scatter_list = [t_ones, t_fives]
        >>> else:
        >>>     scatter_list = None
        >>> dist.scatter(output_tensor, scatter_list, src=0)
        >>> # Rank i gets scatter_list[i]. For example, on rank 1:
        >>> output_tensor
        tensor([5., 5.])

    """
    _check_single_tensor(tensor, "tensor")

    # Parameter ``scatter_list`` may be left unspecified on non-src ranks.
    if scatter_list:
        _check_tensor_list(scatter_list, "scatter_list")
    else:
        scatter_list = []
    _ensure_all_tensors_same_dtype(tensor, scatter_list)

    _check_group(group)
    scatter_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in scatter_list
    ]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    my_rank = group.pg.rank()
    if src == my_rank:
        if not scatter_list:
            raise ValueError(
                "Argument ``scatter_list`` must be specified " "on source rank."
            )
        input_tensors = [scatter_list]
        output_tensors = [tensor]
    else:
        if scatter_list:
            raise ValueError(
                "Argument ``scatter_list`` must NOT be specified "
                "on non-source ranks."
            )
        input_tensors = []
        output_tensors = [tensor]

    opts = ScatterOptions()
    opts.rootRank = src

    group_src_rank = get_group_rank(group, src)
    opts.rootRank = group_src_rank
    work = group.pg.scatter(output_tensors, input_tensors, opts)

    if async_op:
        return work
    else:
        work.wait()


@exception_handler
def scatter_object_list(
    scatter_object_output_list: List[Any],
    scatter_object_input_list: List[Any],
    group: Group,
    src: int=0
):
    """
    Scatters picklable objects in ``scatter_object_input_list`` to the whole
    group. Similar to :func:`scatter`, but Python objects can be passed in. On
    each rank, the scattered object will be stored as the first element of
    ``scatter_object_output_list``. Note that all objects in
    ``scatter_object_input_list`` must be picklable in order to be scattered.

    Args:
        scatter_object_output_list (List[Any]): Non-empty list whose first
            element will store the object scattered to this rank.
        scatter_object_input_list (List[Any]): List of input objects to scatter.
            Each object must be picklable. Only objects on the ``src`` rank will
            be scattered, and the argument can be ``None`` for non-src ranks.
        group: (Group): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        src (int): Source rank from which to scatter
            ``scatter_object_input_list``.

    Returns:
        ``None``. If rank is part of the group, ``scatter_object_output_list``
        will have its first element set to the scattered object for this rank.

    .. note:: Note that this API differs slightly from the scatter collective
        since it does not provide an ``async_op`` handle and thus will be a
        blocking call.

    .. warning::
        :func:`scatter_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     # Can be any list on non-src ranks, elements are not used.
        >>>     objects = [None, None, None]
        >>> output_list = [None]
        >>> dist.scatter_object_list(output_list, objects, src=0)
        >>> # Rank i gets objects[i]. For example, on rank 2:
        >>> output_list
        [{1: 2}]
    """
    _check_group(group)

    if (
        not isinstance(scatter_object_output_list, list)
        or len(scatter_object_output_list) < 1
    ):
        raise RuntimeError(
            "Expected argument scatter_object_output_list to be a list of size at least 1."
        )

    my_rank = group.pg.rank()
    pg_device = _get_pg_device(group)
    if my_rank == src:
        tensor_list, tensor_sizes = zip(
            *[_object_to_tensor(obj, pg_device) for obj in scatter_object_input_list]
        )
        tensor_list, tensor_sizes = list(tensor_list), list(tensor_sizes)

    # Src rank broadcasts the maximum tensor size. This is because all ranks are
    # expected to call into scatter() with equal-sized tensors.
    if my_rank == src:
        max_tensor_size = max(tensor_sizes)
        for tensor in tensor_list:
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    broadcast(max_tensor_size, src=src, group=group)

    # Scatter actual serialized objects
    output_tensor = torch.empty(max_tensor_size.item(), dtype=torch.uint8, device=pg_device)
    scatter(
        output_tensor,
        scatter_list=None if my_rank != src else tensor_list,
        src=src,
        group=group,
    )

    # Scatter per-object sizes to trim tensors when deserializing back to object
    obj_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    scatter(
        obj_tensor_size,
        scatter_list=None if my_rank != src else tensor_sizes,
        src=src,
        group=group,
    )

    # Deserialize back to object
    scatter_object_output_list[0] = _tensor_to_object(output_tensor, obj_tensor_size)


@exception_handler
def all_gather(tensor_list: List[torch.Tensor], tensor: torch.Tensor, group: Group, async_op: bool=False):
    """
    Gathers tensors from the whole group in a list.

    Complex tensors are supported.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(2)]
        >>> tensor_list
        [tensor([0, 0]), tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1, 2]), tensor([3, 4])] # Rank 0
        [tensor([1, 2]), tensor([3, 4])] # Rank 1

        >>> # All tensors below are of torch.cfloat dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zeros(2, dtype=torch.cfloat) for _ in range(2)]
        >>> tensor_list
        [tensor([0.+0.j, 0.+0.j]), tensor([0.+0.j, 0.+0.j])] # Rank 0 and 1
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j]) # Rank 0
        tensor([3.+3.j, 4.+4.j]) # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 0
        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 1

    """
    _check_tensor_list(tensor_list, "tensor_list")
    _check_single_tensor(tensor, "tensor")
    _ensure_all_tensors_same_dtype(tensor_list, tensor)
    _check_group(group)

    tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in tensor_list
    ]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    work = group.pg.allgather([tensor_list], [tensor])

    if async_op:
        return work
    else:
        work.wait()


@exception_handler
def all_gather_into_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Group,
    async_op: bool=False
):
    """
    Gather tensors from all ranks and put them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor to accommodate tensor elements
            from all ranks. It must be correctly sized to have one of the
            following forms:
            (i) a concatenation of all the input tensors along the primary
            dimension; for definition of "concatenation", see ``torch.cat()``;
            (ii) a stack of all the input tensors along the primary dimension;
            for definition of "stack", see ``torch.stack()``.
            Examples below may better explain the supported output forms.
        input_tensor (Tensor): Tensor to be gathered from current rank.
            Different from the ``all_gather`` API, the input tensors in this
            API must have the same size across all ranks.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f'cuda:{rank}')
        >>> tensor_in = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor_in
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> # Output in concatenation form
        >>> tensor_out = torch.zeros(world_size * 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([1, 2, 3, 4], device='cuda:0') # Rank 0
        tensor([1, 2, 3, 4], device='cuda:1') # Rank 1
        >>> # Output in stack form
        >>> tensor_out2 = torch.zeros(world_size, 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out2, tensor_in)
        >>> tensor_out2
        tensor([[1, 2],
                [3, 4]], device='cuda:0') # Rank 0
        tensor([[1, 2],
                [3, 4]], device='cuda:1') # Rank 1

    .. warning::
        The Gloo backend does not support this API.

    """
    _check_single_tensor(input_tensor, "input_tensor")
    _check_single_tensor(output_tensor, "output_tensor")
    _check_group(group)

    output_tensor = (
        output_tensor
        if not output_tensor.is_complex()
        else torch.view_as_real(output_tensor)
    )
    input_tensor = (
        input_tensor
        if not input_tensor.is_complex()
        else torch.view_as_real(input_tensor)
    )

    work = group.pg._allgather_base(output_tensor, input_tensor)

    if async_op:
        return work
    else:
        work.wait()


# _all_gather_base is a private function and will be deprecated.
# Please use torch.distributed.all_gather_into_tensor instead.


# all_gather_coalesced will be deprecated:
#     https://pytorch.org/docs/master/distributed.html#collective-functions


# reduce_scatter_multigpu will be deprecated:
#     https://pytorch.org/docs/master/distributed.html#multi-gpu-collective-functions


@exception_handler
def reduce_scatter(
    output: torch.Tensor,
    input_list: List[torch.Tensor],
    group: Group,
    op=ReduceOp.SUM,
    async_op=False
):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    _check_single_tensor(output, "output")
    _check_tensor_list(input_list, "input_list")
    _ensure_all_tensors_same_dtype(output, input_list)
    _check_group(group)

    opts = ReduceScatterOptions()
    opts.reduceOp = op

    work = group.pg.reduce_scatter([output], [input_list], opts)

    if async_op:
        return work
    else:
        work.wait()


@exception_handler
def reduce_scatter_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    op=ReduceOp.SUM,
    async_op=False
):
    """
    Reduces, then scatters a tensor to all ranks in a group.

    Args:
        output (Tensor): Output tensor. It should have the same size across all
            ranks.
        input (Tensor): Input tensor to be reduced and scattered. Its size
            should be output tensor size times the world size. The input tensor
            can have one of the following shapes:
            (i) a concatenation of the output tensors along the primary
            dimension, or
            (ii) a stack of the output tensors along the primary dimension.
            For definition of "concatenation", see ``torch.cat()``.
            For definition of "stack", see ``torch.stack()``.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f'cuda:{rank}')
        >>> tensor_out = torch.zeros(2, dtype=torch.int64, device=device)
        >>> # Input in concatenation form
        >>> tensor_in = torch.arange(world_size * 2, dtype=torch.int64, device=device)
        >>> tensor_in
        tensor([0, 1, 2, 3], device='cuda:0') # Rank 0
        tensor([0, 1, 2, 3], device='cuda:1') # Rank 1
        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([0, 2], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1
        >>> # Input in stack form
        >>> tensor_in = torch.reshape(tensor_in, (world_size, 2))
        >>> tensor_in
        tensor([[0, 1],
                [2, 3]], device='cuda:0') # Rank 0
        tensor([[0, 1],
                [2, 3]], device='cuda:1') # Rank 1
        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([0, 2], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1

    .. warning::
        The Gloo backend does not support this API.

    """
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    _check_group(group)

    opts = ReduceScatterOptions()
    opts.reduceOp = op

    work = group.pg._reduce_scatter_base(output, input, opts)

    if async_op:
        return work
    else:
        work.wait()


# _reduce_scatter_base is a private function and will be deprecated.
# Please use torch.distributed.reduce_scatter_tensor instead.


@exception_handler
def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    output_split_sizes: List[int]=None,
    input_split_sizes: List[int]=None,
    async_op: bool=False,
):
    """
    Each process splits input tensor and then scatters the split list
    to all processes in a group. Then concatenate the received tensors from all
    the processes in the group and return single output tensor.

    Complex tensors are supported.

    Args:
        output (Tensor): Gathered cancatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all_single` is experimental and subject to change.

    Examples:
        >>> # xdoctest: +SKIP("Undefined rank")
        >>> input = torch.arange(4) + rank * 4
        >>> input
        tensor([0, 1, 2, 3])     # Rank 0
        tensor([4, 5, 6, 7])     # Rank 1
        tensor([8, 9, 10, 11])   # Rank 2
        tensor([12, 13, 14, 15]) # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([0, 4, 8, 12])    # Rank 0
        tensor([1, 5, 9, 13])    # Rank 1
        tensor([2, 6, 10, 14])   # Rank 2
        tensor([3, 7, 11, 15])   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = list(input.chunk(world_size))
        >>> gather_list  = list(output.chunk(world_size))
        >>> for i in range(world_size):
        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)

        >>> # Another example with uneven split
        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> output = ...
        >>> dist.all_to_all_single(output, input, output_splits, input_splits)
        >>> output
        tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])                     # Rank 0
        tensor([ 2,  3, 13, 14, 22, 32, 33])                             # Rank 1
        tensor([ 4, 15, 16, 23, 34, 35])                                 # Rank 2
        tensor([ 5, 17, 18, 24, 36])                                     # Rank 3


        >>> # Another example with tensors of torch.cfloat type.
        >>> input = torch.tensor([1+1j, 2+2j, 3+3j, 4+4j], dtype=torch.cfloat) + 4 * rank * (1+1j)
        >>> input
        tensor([1+1j, 2+2j, 3+3j, 4+4j])                                # Rank 0
        tensor([5+5j, 6+6j, 7+7j, 8+8j])                                # Rank 1
        tensor([9+9j, 10+10j, 11+11j, 12+12j])                          # Rank 2
        tensor([13+13j, 14+14j, 15+15j, 16+16j])                        # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([1+1j, 5+5j, 9+9j, 13+13j])                              # Rank 0
        tensor([2+2j, 6+6j, 10+10j, 14+14j])                            # Rank 1
        tensor([3+3j, 7+7j, 11+11j, 15+15j])                            # Rank 2
        tensor([4+4j, 8+8j, 12+12j, 16+16j])                            # Rank 3
    """
    _check_group(group)

    opts = AllToAllOptions()
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    _ensure_all_tensors_same_dtype(output, input)

    if input.is_complex():
        input = torch.view_as_real(input)
    if output.is_complex():
        output = torch.view_as_real(output)

    output_split_sizes = [] if output_split_sizes is None else output_split_sizes
    input_split_sizes = [] if input_split_sizes is None else input_split_sizes

    work = group.pg.alltoall_base(
            output, input, output_split_sizes, input_split_sizes, opts
        )

    if async_op:
        return work
    else:
        work.wait()


@exception_handler
def all_to_all(
    output_tensor_list: List[torch.Tensor],
    input_tensor_list: List[torch.Tensor],
    group: Group,
    async_op=False
):
    """
    Each process scatters list of input tensors to all processes in a group and
    return gathered list of tensors in output list.

    Complex tensors are supported.

    Args:
        output_tensor_list (list[Tensor]): List of tensors to be gathered one
            per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (Group): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all` is experimental and subject to change.

    Examples:
        >>> # xdoctest: +SKIP("Undefined rank")
        >>> input = torch.arange(4) + rank * 4
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
        [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
        [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
        [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
        [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
        [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
        [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = input
        >>> gather_list  = output
        >>> for i in range(world_size):
        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src=i)

        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> input = list(input.split(input_splits))
        >>> input
        [tensor([0, 1]), tensor([2, 3]), tensor([4]), tensor([5])]                   # Rank 0
        [tensor([10, 11, 12]), tensor([13, 14]), tensor([15, 16]), tensor([17, 18])] # Rank 1
        [tensor([20, 21]), tensor([22]), tensor([23]), tensor([24])]                 # Rank 2
        [tensor([30, 31]), tensor([32, 33]), tensor([34, 35]), tensor([36])]         # Rank 3
        >>> output = ...
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0, 1]), tensor([10, 11, 12]), tensor([20, 21]), tensor([30, 31])]   # Rank 0
        [tensor([2, 3]), tensor([13, 14]), tensor([22]), tensor([32, 33])]           # Rank 1
        [tensor([4]), tensor([15, 16]), tensor([23]), tensor([34, 35])]              # Rank 2
        [tensor([5]), tensor([17, 18]), tensor([24]), tensor([36])]                  # Rank 3

        >>> # Another example with tensors of torch.cfloat type.
        >>> input = torch.tensor([1+1j, 2+2j, 3+3j, 4+4j], dtype=torch.cfloat) + 4 * rank * (1+1j)
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([1+1j]), tensor([2+2j]), tensor([3+3j]), tensor([4+4j])]            # Rank 0
        [tensor([5+5j]), tensor([6+6j]), tensor([7+7j]), tensor([8+8j])]            # Rank 1
        [tensor([9+9j]), tensor([10+10j]), tensor([11+11j]), tensor([12+12j])]      # Rank 2
        [tensor([13+13j]), tensor([14+14j]), tensor([15+15j]), tensor([16+16j])]    # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([1+1j]), tensor([5+5j]), tensor([9+9j]), tensor([13+13j])]          # Rank 0
        [tensor([2+2j]), tensor([6+6j]), tensor([10+10j]), tensor([14+14j])]        # Rank 1
        [tensor([3+3j]), tensor([7+7j]), tensor([11+11j]), tensor([15+15j])]        # Rank 2
        [tensor([4+4j]), tensor([8+8j]), tensor([12+12j]), tensor([16+16j])]        # Rank 3

    """
    _check_group(group)

    opts = AllToAllOptions()
    _check_tensor_list(output_tensor_list, "output_tensor_list")
    _check_tensor_list(input_tensor_list, "input_tensor_list")
    _ensure_all_tensors_same_dtype(output_tensor_list, input_tensor_list)

    input_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in input_tensor_list
    ]
    output_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in output_tensor_list
    ]

    work = group.pg.alltoall(output_tensor_list, input_tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def barrier(group: Group, async_op: bool=False, device_ids: List[int]=None):

    """
    Synchronizes all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Args:
        group (Group): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        device_ids ([int], optional): List of device/GPU ids.
                                      Valid only for NCCL backend.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    _check_group(group)

    opts = BarrierOptions()
    if device_ids is not None:
        if group.backend != Backend.NCCL:
            raise RuntimeError(
                "Function argument device_ids not supported "
                "for the selected backend {}".format(group.backend)
            )
        if isinstance(device_ids, list):
            opts.device_ids = device_ids
        else:
            raise RuntimeError(
                "Invalid function argument: " "device_ids type should be List[int]"
            )

    work = group.pg.barrier(opts=opts)

    if async_op:
        return work
    else:
        work.wait()


def monitored_barrier(group: Group, timeout: timedelta=None, wait_all_ranks: bool=False):
    """
    Synchronizes all processes similar to ``torch.distributed.barrier``, but takes
    a configurable timeout and is able to report ranks that did not pass this
    barrier within that timeout. Specifically, for non-zero ranks, will block
    until a send/recv is processed from rank 0. Rank 0 will block until all send
    /recv from other ranks are processed, and will report failures for ranks
    that failed to respond in time. Note that if one rank does not reach the
    monitored_barrier (for example due to a hang), all other ranks would fail
    in monitored_barrier.

    This collective will block all processes/ranks in the group, until the
    whole group exits the function successfully, making it useful for debugging
    and synchronizing. However, it can have a performance impact and should only
    be used for debugging or scenarios that require full synchronization points
    on the host-side. For debugging purposes, this barrier can be inserted
    before the application's collective calls to check if any ranks are
    desynchronized.

    .. note:: Note that this collective is only supported with the GLOO backend.

    Args:
        group (ProcessGroup, optional): The process group to work on. If
            ``None``, the default process group will be used.
        timeout (datetime.timedelta, optional): Timeout for monitored_barrier.
            If ``None``, the default process group timeout will be used.
        wait_all_ranks (bool, optional): Whether to collect all failed ranks or
            not. By default, this is ``False`` and ``monitored_barrier`` on rank 0
            will throw on the first failed rank it encounters in order to fail
            fast. By setting ``wait_all_ranks=True`` ``monitored_barrier`` will
            collect all failed ranks and throw an error containing information
            about all failed ranks.

    Returns:
        ``None``.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() != 1:
        >>>     dist.monitored_barrier() # Raises exception indicating that
        >>> # rank 1 did not call into monitored_barrier.
        >>> # Example with wait_all_ranks=True
        >>> if dist.get_rank() == 0:
        >>>     dist.monitored_barrier(wait_all_ranks=True) # Raises exception
        >>> # indicating that ranks 1, 2, ... world_size - 1 did not call into
        >>> # monitored_barrier.
    """

    # Need to call rank not in group before using the group, otherwise
    # "Invalid process group" error is raised.
    _check_group(group)

    if group.backend != Backend.GLOO:
        raise RuntimeError("monitored_barrier is only implemented for GLOO backend.")

    if timeout is None:
        timeout = default_pg_timeout

    return group.pg.monitored_barrier(timeout, wait_all_ranks=wait_all_ranks)


def _check_p2p_op_list(p2p_op_list: List[P2POp]):
    """
    Helper to check that the ``p2p_op_list`` is a list of P2POp instances and
    all ops use the same group.
    """
    if not isinstance(p2p_op_list, list) or not all(
        isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list
    ):
        raise RuntimeError(
            "Invalid ``p2p_op_list``. Each op is expected to "
            "to be of type ``torch.distributed.P2POp``."
        )

    group = p2p_op_list[0].group
    if not all(group == p2p_op.group for p2p_op in p2p_op_list):
        raise RuntimeError("All ops need to use the same group.")


def _as_iterable(obj) -> collections.abc.Iterable:
    return obj if isinstance(obj, list) else (obj,)


def _ensure_all_tensors_same_dtype(*tensors) -> None:
    last_dtype = None
    for tensor in itertools.chain(*map(_as_iterable, tensors)):
        tensor_dtype = tensor.dtype
        # Mixing complex and its element type is allowed
        if tensor_dtype.is_complex:
            tensor_dtype = torch.float32 if tensor_dtype == torch.complex64 else torch.complex128

        if last_dtype is None:
            last_dtype = tensor_dtype
        else:
            if last_dtype != tensor_dtype:
                raise RuntimeError(
                    "Invalid usage of tensors with different dtypes"
                    f"Found {last_dtype} and  {tensor.dtype}"
                )


def _check_single_tensor(param, param_name):
    """
    Helper to check that the parameter ``param_name`` is a single tensor.
    """
    if not isinstance(param, torch.Tensor):
        raise RuntimeError(
            "Invalid function argument. Expected parameter `{}` "
            "to be of type torch.Tensor.".format(param_name)
        )


def _check_tensor_list(param, param_name):
    """
    Helper to check that the parameter ``param_name`` is a list of tensors.
    """
    if not isinstance(param, list) or not all(
        isinstance(p, torch.Tensor) for p in param
    ):
        raise RuntimeError(
            "Invalid function argument. Expected parameter `{}` "
            "to be of type List[torch.Tensor].".format(param_name)
        )


def get_group_rank(group: Group, rank: int) -> int:
    """
    Translate a global rank into a group rank.

    ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the relative rank.
        global_rank (int): Global rank to query.

    Returns:
        Group rank of ``global_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    if group is None or not _group_manager.exist_group(group):
        raise RuntimeError(f"Invalid group specified: {group}")
    
    group_rank = group.rank_map[rank]
    if group_rank is None:
        raise RuntimeError(f"Invalid rank {rank} for group {group.name}")
    
    return group_rank
