"""TPUTensor: A tensor subclass wrapping a JAX TPU array.

Provides PyTorch-compatible metadata (shape, stride, dtype) while storing
actual data as a JAX array on TPU. Must be used within torch.compile --
eager compute ops produce wrong results (they operate on the CPU backing
storage, not the JAX array).

How it works with torch.compile:
  - Dynamo tracing: Only reads metadata (shape, dtype). Creates a FakeTensor
    for symbolic execution. The JAX array is never accessed during compilation.
    Because __torch_dispatch__ is NOT overridden, Dynamo's builder enters
    wrap_tensor() and wraps as TensorWithTFOverrideVariable.
  - AOTAutograd: Passes through without subclass decomposition (no
    __tensor_flatten__), so compiled code receives TPUTensor directly.
  - Inductor codegen: With pallas_tpu_native=True, generates code that
    accesses ._jax_array at runtime instead of DLPack conversion.
  - Runtime: Generated _main() extracts ._jax_array from inputs, runs
    Pallas kernel natively on TPU, stores result back to ._jax_array.
"""

from typing import Any, Tuple

import torch

_reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


def _contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute contiguous (row-major) strides for a given shape."""
    if not shape:
        return ()
    strides = [1]
    for s in reversed(shape[1:]):
        strides.append(strides[-1] * max(s, 1))
    strides.reverse()
    return tuple(strides)


class TPUTensor(torch.Tensor):
    """Tensor subclass wrapping a JAX TPU array.

    Provides correct PyTorch metadata (shape, stride, dtype, device='cpu')
    with zero CPU memory allocation. The actual compute data lives in
    ._jax_array as a JAX array on TPU.

    Uses _reinterpret_tensor on a zero-byte storage to create a tensor with
    correct metadata but no backing memory. This avoids allocating
    shape-proportional CPU memory (e.g. 4GB for a 1024^3 float32 tensor)
    that would never be read.

    Design choices:
      - No __torch_dispatch__ override: Dynamo's builder checks
        type(value).__torch_dispatch__ is torch.Tensor.__torch_dispatch__
        (builder.py:764). If overridden, Dynamo skips wrap_tensor() and
        cannot trace through the subclass. By NOT overriding, Dynamo wraps
        as TensorWithTFOverrideVariable and traces ops on FakeTensor.
      - No __tensor_flatten__/__tensor_unflatten__: Prevents AOTAutograd
        from decomposing the subclass. The compiled function receives
        TPUTensor directly at runtime.
      - device='cpu' metadata: 'tpu' is not a registered PyTorch device.
        Inductor routes to Pallas backend via cpu_backend="pallas" config.
    """

    _jax_array: Any

    @staticmethod
    def __new__(cls, jax_array: Any, dtype: torch.dtype) -> "TPUTensor":
        shape = tuple(jax_array.shape)
        strides = _contiguous_strides(shape)

        # Zero-memory backing: a 0-byte CPU storage reinterpreted with the
        # desired shape/strides gives Dynamo correct metadata without
        # allocating shape-proportional CPU memory.
        base = torch.empty(0, dtype=dtype, device="cpu")
        instance = _reinterpret_tensor(base, shape, strides).as_subclass(cls)
        instance._jax_array = jax_array
        return instance

    def as_subclass(self, cls: type) -> "torch.Tensor":
        # Dynamo's output reconstruction calls to_subclass(tensor, cls) which
        # invokes tensor.as_subclass(cls). The base as_subclass creates a new
        # Python object sharing the same TensorImpl but with an empty __dict__,
        # dropping _jax_array. Override to preserve it.
        result = super().as_subclass(cls)
        result._jax_array = self._jax_array
        return result

    def __repr__(self) -> str:
        devices = self._jax_array.devices()
        device_str = ", ".join(str(d) for d in devices) if devices else "unknown"
        return (
            f"TPUTensor(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"jax_dtype={self._jax_array.dtype}, jax_device={device_str})"
        )

    # NOTE: __torch_dispatch__ is intentionally NOT overridden.
    # Dynamo's builder.py:764 checks:
    #   type(value).__torch_dispatch__ is torch.Tensor.__torch_dispatch__
    # If this is True (not overridden), Dynamo calls wrap_tensor() which
    # creates TensorWithTFOverrideVariable, allowing FakeTensor tracing.
    # If we overrode __torch_dispatch__, Dynamo would skip wrap_tensor()
    # and fail to trace operations on TPUTensor inputs.

    # NOTE: __tensor_flatten__ / __tensor_unflatten__ are intentionally
    # NOT implemented. This prevents AOTAutograd from decomposing the
    # subclass (requires_subclass_dispatch() returns False), so the
    # compiled Runner.call() receives actual TPUTensor instances at
    # runtime, giving generated code access to ._jax_array.
