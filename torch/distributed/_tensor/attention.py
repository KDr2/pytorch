from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor

aten = torch.ops.aten


def sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    rank = dist.get_rank()
    size = dist.get_world_size()

    query, key, value = op_info.local_args

    chunks = []
    logsumexps = []
    for i in range(size):
        if i > 0:
            key, value = _ring_send_recv(key, value)
        local_results = op_call(query, key, value, **op_info.local_kwargs)
        chunks.append(local_results[0])
        logsumexps.append(local_results[1])

    softmax_lse = torch.empty_like(logsumexps[0])
    for lse in logsumexps:
        softmax_lse += lse.exp()
    softmax_lse = softmax_lse.log_()

    out = torch.empty_like(chunks[0])
    for chunk, chunk_lse in zip(chunks, logsumexps):
        softmax_lse_corrected = torch.exp(chunk_lse - softmax_lse)
        out_corrected = chunk * softmax_lse_corrected.unsqueeze(-1)
        out += out_corrected

    return dtensor.DTensor._op_dispatcher.wrap(
        (out, softmax_lse_corrected) + local_results[2:], output_sharding.output_spec
    )


def _ring_send_recv(k_send, v_send):
    # dist comms and reconstruct local input tensor
    k_recv = torch.empty_like(k_send)
    v_recv = torch.empty_like(v_send)

    rank = dist.get_rank()
    size = dist.get_world_size()

    right = (rank + 1) % size
    left = (rank - 1 + size) % size

    send_op_k = dist.P2POp(dist.isend, k_send, right)
    send_op_v = dist.P2POp(dist.isend, v_send, right)
    recv_op_k = dist.P2POp(dist.irecv, k_recv, left)
    recv_op_v = dist.P2POp(dist.irecv, v_recv, left)

    reqs = dist.batch_isend_irecv(
        [send_op_k, send_op_v, recv_op_k, recv_op_v],
    )
    for req in reqs:
        req.wait()

    return k_recv, v_recv
