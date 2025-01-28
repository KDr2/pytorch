# mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


class AOTICallDelegate(HigherOrderOperator):
    def __init__(self):
        super().__init__("aoti_call_delegate")

    def __call__(self, lowered_module, original_gm, weight_args, input_args):
        return super().__call__(lowered_module, original_gm, weight_args, input_args)


aoti_call_delegate = AOTICallDelegate()
aoti_call_delegate.fallthrough(torch._C.DispatchKey.PythonDispatcher)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.PythonTLSSnapshot)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.ADInplaceOrView)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.AutocastCPU)

LOWERED_BACKEND_MODULE_TYPE = "LoweredBackendModule"


@aoti_call_delegate.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def call_delegate_cpu(lowered_module, original_gm, weight_args, input_args):
    # FX creates this immutable_dict/list concept. Get rid of this.
    map_types = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_args = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        input_args,
        lambda a: isinstance(a, tuple(map_types.keys())),
    )

    has_fake_input_args = any(isinstance(arg, FakeTensor) for arg in new_args)
    has_fake_params = any(
        isinstance(param, FakeTensor) for param in original_gm.parameters()
    )
    has_fake_buffers = any(
        isinstance(buffer, FakeTensor) for buffer in original_gm.buffers()
    )

    if has_fake_input_args or has_fake_params or has_fake_buffers:
        # aoti lowered module doesn't support fake tensor
        return original_gm(*new_args)
    else:
        return lowered_module(new_args)


@aoti_call_delegate.py_impl(FakeTensorMode)
# pyre-ignore
def call_delegate_fake_tensor_mode(
    mode, lowered_module, original_gm, weight_args, input_args
):
    with mode:
        return call_delegate_cpu(lowered_module, original_gm, weight_args, input_args)
