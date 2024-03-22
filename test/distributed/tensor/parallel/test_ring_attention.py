# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed._tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    NUM_DEVICES,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional


class RingAttentionTest(DTensorTestBase):
    @with_comms
    @torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    )
    def test_ring_attention(self):
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        dtype = torch.bfloat16
        bs = 8
        query_tokens = 8
        context_tokens = 16
        dim = 32
        nheads = 8
        query = torch.randn(
            (bs, nheads, query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.randn(
            (bs, nheads, context_tokens, dim), device=self.device_type, dtype=dtype
        )
        value = torch.randn(
            (bs, nheads, context_tokens, dim), device=self.device_type, dtype=dtype
        )

        query_placement = [Replicate()]
        query = DTensor.from_local(query, device_mesh, query_placement)

        context_placement = [Shard(1)]
        key = DTensor.from_local(key, device_mesh, context_placement)
        value = DTensor.from_local(value, device_mesh, context_placement)

        with CommDebugMode() as comm_mode:
            out = torch.ops.aten._scaled_dot_product_flash_attention(query, key, value)

        self.assertEqual(out[0].shape, (bs, nheads, query_tokens, dim))

        # TODO backwards
        # out[0].sum().backward()

        # self.assertDictEqual(
        #    comm_mode.get_comm_counts(),
        #    {
        #        # no functional send/recv yet
        #        c10d_functional.batch_isend_irecv: self.world_size - 1,
        #    },
        # )


instantiate_parametrized_tests(DTensorTestBase)

if __name__ == "__main__":
    run_tests()
