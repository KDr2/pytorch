import torch
import unittest


from torch.testing._internal.common_utils import (
    TestCase, run_tests
)
from torch.testing._internal.common_device_type import (
    ops, dtypes, dtypesIfCUDA, instantiate_device_type_tests
)
from torch.testing._internal.common_methods_invocations import (
    op_db, precisionOverride, signal_funcs
)


class TestSignal(TestCase):
    exact_dtype = False

    supported_windows = {
        'cosine',
        'exponential',
        'gaussian'
    }

    @precisionOverride({torch.bfloat16: 5e-2, torch.half: 1e-3})
    @ops([op for op in signal_funcs if 'windows' in op.name],
         allowed_dtypes=(torch.float, torch.double, torch.long))
    def test_windows(self, device, dtype, op):
        if op.ref is None:
            raise unittest.SkipTest("No reference implementation")

        sample_inputs = op.sample_inputs(device, dtype)

        for sample_input in sample_inputs:
            window_input = sample_input.input

            expected = op.ref(window_input, *sample_input.args, **sample_input.kwargs)
            actual = op(window_input, **sample_input.kwargs)

            self.assertEqual(actual, expected, exact_dtype=self.exact_dtype)


instantiate_device_type_tests(TestSignal, globals())

if __name__ == '__main__':
    run_tests()
