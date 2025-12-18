# Owner(s): ["module: inductor"]
"""
Tests for useful PyTorch ops under inductor with various compilation modes.

Tests three properties:
1. Batch invariance - output shouldn't change based on batch size
2. Run-to-run determinism - same input should give same output
3. Bitwise equivalence with torch eager mode

Tests three compilation backends:
1. aot_eager_decomp_partition - AOT autograd with eager execution
2. inductor_default - Standard inductor compilation
3. inductor_numerics - Inductor with strict numerics flags

Focuses on ops commonly used in LLMs from unary_ufuncs, binary_ufuncs, and op_db.
"""

import unittest

import pytest

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    op_db,
    unary_ufuncs,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    skipIfTorchDynamo,
    TEST_WITH_ASAN,
)
from torch.testing._internal.inductor_utils import HAS_GPU


# LLM-useful op names to filter from unary_ufuncs
LLM_UNARY_OP_NAMES = {
    # Basic math
    "abs",
    "neg",
    "reciprocal",
    # Exponential/log
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    # Power/root
    "sqrt",
    "rsqrt",
    # Trigonometric
    "sin",
    "cos",
    "tan",
    "tanh",
    # Activations
    "sigmoid",
    "nn.functional.relu6",
}

# LLM-useful op names to filter from binary_ufuncs
LLM_BINARY_OP_NAMES = {
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "remainder",
    "fmod",
    "maximum",
    "minimum",
}

# LLM-useful op names from op_db (nn.functional and others)
LLM_OP_DB_NAMES = {
    # Activations
    "nn.functional.gelu",
    "nn.functional.silu",
    "nn.functional.leaky_relu",
    "nn.functional.hardswish",
    # Normalization
    "nn.functional.layer_norm",
    "nn.functional.rms_norm",
    # Attention/linear
    "nn.functional.linear",
    "matmul",
    "bmm",
    # Softmax
    "softmax",
    "log_softmax",
}

# Filter OpInfos for LLM-useful ops
llm_unary_ops = [op for op in unary_ufuncs if op.name in LLM_UNARY_OP_NAMES]
llm_binary_ops = [op for op in binary_ufuncs if op.name in LLM_BINARY_OP_NAMES]
llm_op_db_ops = [op for op in op_db if op.name in LLM_OP_DB_NAMES]

# Combine all ops, avoiding duplicates by name
_seen_names = set()
llm_ops = []
for op in llm_unary_ops + llm_binary_ops + llm_op_db_ops:
    if op.name not in _seen_names:
        _seen_names.add(op.name)
        llm_ops.append(op)

# Backends to test
BACKENDS = [
    "aot_eager_decomp_partition",
    "inductor_default",
    "inductor_numerics",
]

# Dtypes to test - common LLM dtypes
DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
]

# Inductor options for strict numerics matching eager behavior
INDUCTOR_NUMERICS_OPTIONS = {
    "deterministic": True,
    "fallback_random": True,
    "emulate_precision_casts": True,
    # Note: config key has typo "divison" (missing 'i')
    "emulate_divison_rounding": True,
}


def slice_tensors_to_batch_size(sample_input, batch_size):
    """Slice all tensors in a SampleInput to the given batch size.

    For batch invariance testing, we need to slice all tensor inputs that have
    matching batch dimensions. This function slices:
    - sample_input.input if it's a tensor with dim > 0
    - All tensor args that have the same size in dim 0 as the input (not broadcast dims)
    - All tensor kwargs that have the same size in dim 0 as the input (not broadcast dims)

    Tensors with size 1 in dim 0 are broadcast dimensions and should not be sliced.

    Returns a tuple (sliced_input, sliced_args, sliced_kwargs), or None if slicing is not possible.
    """
    inp = sample_input.input
    if not isinstance(inp, torch.Tensor) or inp.dim() == 0:
        return None

    original_batch_size = inp.shape[0]
    if batch_size > original_batch_size:
        return None

    # Slice the input
    sliced_input = inp[:batch_size]

    # Slice args - only slice tensors that have matching batch dimension (not broadcast)
    sliced_args = []
    for arg in sample_input.args:
        if (
            isinstance(arg, torch.Tensor)
            and arg.dim() > 0
            and arg.shape[0] == original_batch_size
            and arg.shape[0] > 1  # Don't slice broadcast dimensions
        ):
            sliced_args.append(arg[:batch_size])
        else:
            sliced_args.append(arg)
    sliced_args = tuple(sliced_args)

    # Slice kwargs - only slice tensors that have matching batch dimension (not broadcast)
    sliced_kwargs = {}
    for key, val in sample_input.kwargs.items():
        if (
            isinstance(val, torch.Tensor)
            and val.dim() > 0
            and val.shape[0] == original_batch_size
            and val.shape[0] > 1  # Don't slice broadcast dimensions
        ):
            sliced_kwargs[key] = val[:batch_size]
        else:
            sliced_kwargs[key] = val

    return sliced_input, sliced_args, sliced_kwargs


def sample_operates_on_batch_dim(op_name, sample_input):
    """Check if a sample input operates on the batch dimension (dim 0).

    For ops that normalize/reduce over a dimension, if that dimension is 0,
    slicing the batch will change the result and batch invariance doesn't apply.
    """
    # Ops that take a 'dim' argument and normalize/reduce over it
    dim_based_ops = {
        "softmax",
        "log_softmax",
        "nn.functional.softmax",
        "nn.functional.log_softmax",
    }

    if op_name not in dim_based_ops:
        return False

    # Get dim from args or kwargs
    dim = None
    if sample_input.args:
        dim = sample_input.args[0]
    if "dim" in sample_input.kwargs:
        dim = sample_input.kwargs["dim"]

    # If dim is 0 or -ndim (equivalent to 0), the op operates on the batch dimension
    if dim is not None:
        inp = sample_input.input
        if isinstance(inp, torch.Tensor):
            ndim = inp.dim()
            # Normalize negative dim
            if dim < 0:
                dim = dim + ndim
            return dim == 0

    return False


# Expected failures for bitwise equivalence tests.
# Maps (device_type, op_name, backend, test_type, dtype) -> reason for expected failure.
# test_type is one of: "batch_invariance", "determinism", "eager_equivalence"
# dtype can be None to match all dtypes, or a specific torch.dtype
#
# These track known numerical differences between eager and compiled execution.
# The goal is to eventually fix these and remove entries from this dict.
EXPECTED_FAILURES = {
    # div has numerical differences on CUDA due to Triton's division implementation
    (
        "cuda",
        "div",
        "inductor_default",
        "eager_equivalence",
        None,
    ): "div has ~6e-8 numerical differences on CUDA",
    # reciprocal has numerical differences on CUDA
    (
        "cuda",
        "reciprocal",
        "inductor_default",
        "eager_equivalence",
        None,
    ): "reciprocal has ~6e-8 numerical differences on CUDA",
    (
        "cuda",
        "reciprocal",
        "inductor_numerics",
        "eager_equivalence",
        None,
    ): "reciprocal has ~6e-8 numerical differences on CUDA even with numerics flags",
    # sigmoid has numerical differences on CUDA
    (
        "cuda",
        "sigmoid",
        "inductor_default",
        "eager_equivalence",
        None,
    ): "sigmoid has ~6e-8 numerical differences on CUDA",
    (
        "cuda",
        "sigmoid",
        "inductor_numerics",
        "eager_equivalence",
        None,
    ): "sigmoid has ~6e-8 numerical differences on CUDA even with numerics flags",
    # gelu has numerical differences on CUDA
    (
        "cuda",
        "nn.functional.gelu",
        "inductor_default",
        "eager_equivalence",
        None,
    ): "gelu has numerical differences on CUDA",
    (
        "cuda",
        "nn.functional.gelu",
        "inductor_numerics",
        "eager_equivalence",
        None,
    ): "gelu has numerical differences on CUDA even with numerics flags",
    # rms_norm decomposition has numerical differences on CUDA
    (
        "cuda",
        "nn.functional.rms_norm",
        "aot_eager_decomp_partition",
        "eager_equivalence",
        None,
    ): "rms_norm decomposition has numerical differences on CUDA",
}


def is_expected_failure(device_type, op_name, backend, test_type, dtype=None):
    """Check if a test is expected to fail.

    First checks for dtype-specific failure, then falls back to dtype=None (all dtypes).
    """
    # Check dtype-specific failure first
    if (device_type, op_name, backend, test_type, dtype) in EXPECTED_FAILURES:
        return True
    # Fall back to dtype=None (matches all dtypes)
    return (device_type, op_name, backend, test_type, None) in EXPECTED_FAILURES


def get_expected_failure_reason(device_type, op_name, backend, test_type, dtype=None):
    """Get the reason for an expected failure."""
    # Check dtype-specific failure first
    key = (device_type, op_name, backend, test_type, dtype)
    if key in EXPECTED_FAILURES:
        return EXPECTED_FAILURES[key]
    # Fall back to dtype=None
    key = (device_type, op_name, backend, test_type, None)
    return EXPECTED_FAILURES.get(key, "Unknown")


def compile_fn(fn, backend):
    """Compile a function with the given backend."""
    if backend == "aot_eager_decomp_partition":
        return torch.compile(fn, backend="aot_eager_decomp_partition")
    elif backend == "inductor_default":
        return torch.compile(fn, backend="inductor")
    elif backend == "inductor_numerics":
        return torch.compile(fn, backend="inductor", options=INDUCTOR_NUMERICS_OPTIONS)
    else:
        raise ValueError(f"Unknown backend: {backend}")


@unittest.skipIf(IS_WINDOWS, "Skipped on Windows")
@unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
@unittest.skipIf(not HAS_GPU, "Requires GPU")
class TestOpInfoProperties(TestCase):
    """Test op properties under various inductor modes using OpInfo on CUDA."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def _get_sample_inputs(self, op, device, dtype):
        """Get sample inputs from OpInfo using reference_inputs for comprehensive coverage."""
        # Use reference_inputs for more comprehensive test coverage
        # Falls back to sample_inputs if reference_inputs_func is not defined
        try:
            samples = list(op.reference_inputs(device, dtype, requires_grad=False))
        except Exception:
            samples = list(op.sample_inputs(device, dtype, requires_grad=False))
        return samples

    # =========================================================================
    # Batch Invariance Tests
    # =========================================================================

    @onlyCUDA
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=DTYPES)
    @parametrize("backend", BACKENDS)
    def test_batch_invariance(self, device, dtype, op, backend):
        """Test batch invariance with exponentially decreasing batch sizes.

        For each sample input, this test:
        1. Runs the compiled op at full batch size
        2. Runs at size/2, size/4, etc. (exponentially decreasing)
        3. Verifies the sliced output matches the corresponding slice of the full output

        All tensor inputs with matching batch dimensions are sliced together.
        """
        torch._dynamo.reset()
        device_type = torch.device(device).type

        samples = self._get_sample_inputs(op, device, dtype)
        if not samples:
            self.skipTest(f"No samples for {op.name}")

        fn = op.get_op()

        for sample in samples:
            # Skip if input is not a tensor or is 0-dim
            if not isinstance(sample.input, torch.Tensor) or sample.input.dim() == 0:
                continue

            # Need at least size 4 in first dimension for meaningful test
            full_size = sample.input.shape[0]
            if full_size < 4:
                continue

            # Skip samples where the op normalizes/reduces over dim 0 (batch dimension)
            # because slicing the batch changes the normalization result
            if sample_operates_on_batch_dim(op.name, sample):
                continue

            compiled_fn = compile_fn(fn, backend)

            # Get reference output at full size
            full_args = (sample.input,) + tuple(sample.args)
            full_kwargs = sample.kwargs
            try:
                full_out = compiled_fn(*full_args, **full_kwargs)
            except Exception:
                continue  # Skip if compilation fails for this sample

            if not isinstance(full_out, torch.Tensor):
                continue

            # Skip if output is 0-dim (scalar) - can't slice it
            if full_out.dim() == 0:
                continue

            # Test with exponentially decreasing sizes: size, size/2, size/4, ...
            size = full_size
            try:
                while size >= 1:
                    # Slice all tensor inputs with matching batch dimensions
                    sliced = slice_tensors_to_batch_size(sample, size)
                    if sliced is None:
                        break
                    sliced_input, sliced_args, sliced_kwargs = sliced

                    out = compiled_fn(sliced_input, *sliced_args, **sliced_kwargs)

                    # Verify output matches the corresponding slice of full output (bitwise)
                    self.assertEqual(out, full_out[:size], rtol=0, atol=0)

                    # Step down exponentially
                    size = size // 2
            except AssertionError:
                if is_expected_failure(
                    device_type, op.name, backend, "batch_invariance", dtype
                ):
                    pytest.xfail(
                        f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'batch_invariance', dtype)}"
                    )
                raise

            # Only need one successful sample to pass
            return

        self.skipTest("No suitable samples found for batch invariance test")

    # =========================================================================
    # Run-to-Run Determinism Tests
    # =========================================================================

    @onlyCUDA
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=DTYPES)
    @parametrize("backend", BACKENDS)
    def test_determinism(self, device, dtype, op, backend):
        """Test run-to-run determinism."""
        torch._dynamo.reset()
        device_type = torch.device(device).type

        samples = self._get_sample_inputs(op, device, dtype)
        if not samples:
            self.skipTest(f"No samples for {op.name}")

        fn = op.get_op()

        for sample in samples:
            args = (sample.input,) + tuple(sample.args)
            kwargs = sample.kwargs

            compiled_fn = compile_fn(fn, backend)

            try:
                out1 = compiled_fn(*args, **kwargs)
                out2 = compiled_fn(*args, **kwargs)
                out3 = compiled_fn(*args, **kwargs)
            except Exception:
                continue  # Skip if compilation fails for this sample

            # Bitwise identical
            try:
                self.assertEqual(out1, out2, rtol=0, atol=0)
                self.assertEqual(out2, out3, rtol=0, atol=0)
            except AssertionError:
                if is_expected_failure(
                    device_type, op.name, backend, "determinism", dtype
                ):
                    pytest.xfail(
                        f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'determinism', dtype)}"
                    )
                raise

            # Only need one successful sample to pass
            return

        self.skipTest("No suitable samples found for determinism test")

    # =========================================================================
    # Bitwise Equivalence with Eager Mode Tests
    # =========================================================================

    @onlyCUDA
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=DTYPES)
    @parametrize("backend", BACKENDS)
    def test_eager_equivalence(self, device, dtype, op, backend):
        """Test bitwise equivalence with eager execution."""
        torch._dynamo.reset()
        device_type = torch.device(device).type

        samples = self._get_sample_inputs(op, device, dtype)
        if not samples:
            self.skipTest(f"No samples for {op.name}")

        fn = op.get_op()

        for sample in samples:
            args = (sample.input,) + tuple(sample.args)
            kwargs = sample.kwargs

            # Eager reference
            try:
                eager_out = fn(*args, **kwargs)
            except Exception:
                continue  # Skip if eager fails for this sample

            # Compiled output
            compiled_fn = compile_fn(fn, backend)
            try:
                compiled_out = compiled_fn(*args, **kwargs)
            except Exception:
                continue  # Skip if compilation fails for this sample

            # Bitwise identical
            try:
                self.assertEqual(eager_out, compiled_out, rtol=0, atol=0)
            except AssertionError:
                if is_expected_failure(
                    device_type, op.name, backend, "eager_equivalence", dtype
                ):
                    pytest.xfail(
                        f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'eager_equivalence', dtype)}"
                    )
                raise

            # Only need one successful sample to pass
            return

        self.skipTest("No suitable samples found for eager equivalence test")


instantiate_device_type_tests(TestOpInfoProperties, globals(), except_for=["cpu"])

if __name__ == "__main__":
    run_tests()
