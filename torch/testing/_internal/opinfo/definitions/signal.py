import random
import unittest
from functools import partial

from itertools import product
from typing import List

import torch
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
    DecorateInfo,
    ErrorInput,
    OpInfo,
    SampleInput,
)
from torch.testing._legacy import floating_types_and

if TEST_SCIPY:
    import scipy.signal


def sample_inputs_window(op_info, device, dtype, requires_grad, *args, **kwargs):
    r"""Base function used to create sample inputs for windows.

    For additional required args you should use *args, as well as **kwargs for
    additional keyword arguments.
    """

    # Test a window size of length zero and one.
    # If it's either symmetric or not doesn't matter in these sample inputs.
    for size in range(2):
        yield SampleInput(
            size,
            *args,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            **kwargs,
        )

    # For sizes larger than 1 we need to test both symmetric and non-symmetric windows.
    # Note: sample input tensors must be kept rather small.
    for size, sym in product(list(range(2, 6)), (True, False)):
        yield SampleInput(
            size,
            *args,
            sym=sym,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            **kwargs,
        )


def error_inputs_window(op_info, device, *args, **kwargs):
    # Tests for windows that have a negative size
    yield ErrorInput(
        SampleInput(-1, *args, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="requires non-negative window length, got M=-1",
    )

    # Tests for window tensors that are not torch.strided, for instance, torch.sparse_coo.
    yield ErrorInput(
        SampleInput(
            3,
            *args,
            layout=torch.sparse_coo,
            device=device,
            dtype=torch.float32,
            **kwargs,
        ),
        error_type=ValueError,
        error_regex="is implemented for strided tensors only, got: torch.sparse_coo",
    )

    # Tests for window tensors that are not floating point dtypes, for instance, torch.long.
    yield ErrorInput(
        SampleInput(3, *args, dtype=torch.long, device=device, **kwargs),
        error_type=ValueError,
        error_regex="expects floating point dtypes, got: torch.int64",
    )


def error_inputs_exponential_window(op_info, device, **kwargs):
    # Yield common error inputs
    yield from error_inputs_window(op_info, device, **kwargs)

    # Tests for negative decay values.
    yield ErrorInput(
        SampleInput(3, tau=-1, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="Tau must be positive, got: -1 instead.",
    )

    # Tests for non-symmetric windows and a given center value.
    yield ErrorInput(
        SampleInput(3, center=1, sym=False, dtype=torch.float32, device=device),
        error_type=ValueError,
        error_regex="Center must be 'None' for non-symmetric windows",
    )


def error_inputs_gaussian_window(op_info, device, **kwargs):
    # Yield common error inputs
    yield from error_inputs_window(op_info, device, std=0.5, **kwargs)

    # Tests for negative standard deviations
    yield ErrorInput(
        SampleInput(3, std=-1, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="Standard deviation must be positive, got: -1 instead.",
    )


def make_signal_windows_ref(fn):
    r"""Wrapper for signal window references.

    Particularly used for window references that don't have a matching signature with
    torch, e.g., gaussian window.
    """

    def _fn(*args, **kwargs):
        # Remove torch-specific kwargs
        for torch_key in {"device", "layout", "dtype", "requires_grad"}:
            if torch_key in kwargs:
                kwargs.pop(torch_key)
        return fn(*args, **kwargs)

    return _fn


def make_signal_windows_opinfo(
    name, ref, sample_inputs_func, error_inputs_func, *, skips=()
):
    r"""Helper function to create OpInfo objects related to different windows."""
    return OpInfo(
        name=name,
        ref=ref if TEST_SCIPY else None,
        dtypes=floating_types_and(torch.bfloat16, torch.float16),
        dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=sample_inputs_func,
        error_inputs_func=error_inputs_func,
        supports_out=False,
        supports_autograd=False,
        skips=(
            # TODO: same as this?
            # https://github.com/pytorch/pytorch/issues/81774
            # also see: arange, new_full
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # skip these tests since we have non tensor input
            DecorateInfo(
                unittest.skip("Skipped!"), "TestCommon", "test_noncontiguous_samples"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_conj_view"),
            DecorateInfo(
                unittest.skip("Skipped!"), "TestMathBits", "test_neg_conj_view"
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_neg_view"),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestVmapOperatorsOpInfo",
                "test_vmap_exhaustive",
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestVmapOperatorsOpInfo",
                "test_op_has_batch_rule",
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo"
                "test_schema_correctness",
                dtypes=[torch.float16]
            ),
            *skips,
        ),
    )


op_db: List[OpInfo] = [
    make_signal_windows_opinfo(
        name="signal.windows.cosine",
        ref=make_signal_windows_ref(scipy.signal.windows.cosine),
        sample_inputs_func=sample_inputs_window,
        error_inputs_func=error_inputs_window,
    ),
    make_signal_windows_opinfo(
        name="signal.windows.exponential",
        ref=make_signal_windows_ref(scipy.signal.windows.exponential),
        sample_inputs_func=partial(sample_inputs_window, tau=random.uniform(0, 10)),
        error_inputs_func=error_inputs_exponential_window,
    ),
    make_signal_windows_opinfo(
        name="signal.windows.gaussian",
        ref=make_signal_windows_ref(scipy.signal.windows.gaussian),
        sample_inputs_func=partial(sample_inputs_window, std=random.uniform(0, 3)),
        error_inputs_func=error_inputs_gaussian_window,
    ),
]
