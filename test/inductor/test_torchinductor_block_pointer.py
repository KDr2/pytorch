import functools
import importlib
import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import (
    HAS_GPU,
    GPU_TYPE,
    skip_windows_ci,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch._inductor.utils import run_and_get_code
from typing import Tuple
import unittest

requires_gpu = functools.partial(unittest.skipIf, not HAS_GPU, "requires gpu")

skip_windows_ci(__name__, __file__)

importlib.import_module("filelock")

@instantiate_parametrized_tests
class TritonBlockPointerTest(InductorTestCase):

        @requires_gpu()
        @config.patch("triton.use_block_ptr", True)
        @parametrize(
            "orig_size,view_size,require_block_ptr",
            [
                (orig_size, view_size, require_block_ptr)
                for orig_size, view_size, require_block_ptr in (
                    ((32, 32, 32), (32, 16, 8), True),
                    ((64, 32, 32), (32, 16, 8), True),
                    ((16, 8, 8, 8), (8, 8, 4, 2), True),
                    ((15, 9), (15, 3), False), # Non-power-of-2 dims
                    ((1, 1, 1), (1, 1, 1), False), # Scalar
                )
            ],
        )
        def test_strided_block_ptr(self, orig_size: Tuple[int], view_size: Tuple[int], require_block_ptr: bool):
            """
            Test generating strided ND block pointers.
            """
            def foo(x, y):
                return x + y

            device = torch.device(GPU_TYPE)
            args = []
            for arg_idx in range(2):
                orig = torch.randn(orig_size).to(device)
                view = torch.as_strided(orig, view_size, orig.stride())
                args.append(view)

            compiled = torch.compile(foo, backend="inductor")

            ref = foo(*args)
            actual, (code,) = run_and_get_code(compiled, *args)

            self.assertTrue(torch.allclose(ref, actual))

            # Optionally check for block pointers
            if require_block_ptr:
                num_block_ptrs = code.count("tl.make_block_ptr")
                self.assertEqual(num_block_ptrs, 3)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
