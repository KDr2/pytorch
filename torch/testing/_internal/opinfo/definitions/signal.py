import unittest
from typing import List

import torch
from torch.testing._internal.common_dtype import all_types_and, floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
    DecorateInfo, ErrorInput,
)
from torch.testing._internal.opinfo.refs import (
    PythonRefInfo
)

if TEST_SCIPY:
    import scipy.signal


def _sample_input_windows(sample_input, *args, **kwargs):
    for size in [0, 1, 2, 5, 10, 50, 100, 1024, 2048]:
        for periodic in [True, False]:
            kwargs['periodic'] = periodic
            yield sample_input(size, args=args, kwargs=kwargs)


# TODO: Add gaussian and exponential windows sample inputs
def sample_inputs_cosine_window(op_info, *args, **kwargs):
    return _sample_input_windows(SampleInput, *args, **kwargs)


# TODO: Add gaussian and exponential windows error inputs
def error_inputs_cosine_window(op_info, device, **kwargs):
    _kwargs = {'device': device, 'dtype': torch.float32}

    yield ErrorInput(
        SampleInput(-1, args=(), kwargs=_kwargs),
        error_regex="cosine requires non-negative window_length, got window_length=-1"
    )

    # TODO: Add more error inputs


# TODO: Add gaussian and exponential windows OpInfo
op_db: List[OpInfo] = [
    OpInfo('signal.windows.cosine',
           ref=scipy.signal.windows.cosine if TEST_SCIPY else None,
           dtypes=all_types_and(torch.float, torch.double, torch.long),
           dtypesIfCUDA=all_types_and(torch.float, torch.double, torch.bfloat16, torch.half, torch.long),
           sample_inputs_func=sample_inputs_cosine_window,
           error_inputs_func=error_inputs_cosine_window,
           supports_out=False,
           supports_autograd=False,
           skips=(
               DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),
               # TODO: same as this?
               # https://github.com/pytorch/pytorch/issues/81774
               # also see: arange, new_full
               # fails to match any schemas despite working in the interpreter
               DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
               # fails to match any schemas despite working in the interpreter
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
               # skip these tests since we have non tensor input
               DecorateInfo(unittest.skip('Skipped!'), "TestCommon", "test_noncontiguous_samples"),
               DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_conj_view'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_conj_view'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_view'),
               # UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
           )
           ),
]

python_ref_db: List[OpInfo] = [
    PythonRefInfo(
        "_refs.signal.windows.cosine",
        op="torch.signal.windows.cosine",
        op_db=op_db,
        torch_opinfo_name="signal.windows.cosine",
        supports_nvfuser=False,
        skips=(
            # skip these tests since we have non tensor input
            DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_conj_view'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_conj_view'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_view'),
        ),
    ),
]
