# mypy: allow-untyped-defs
import contextlib
import functools
import traceback
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_all, tree_map
from torch.utils._traceback import CapturedTraceback


if TYPE_CHECKING:
    from torch.distributed._tools.mod_tracker import ModTracker


__all__ = ["DebugMode", "get_active_debug_mode", "_TritonKernelCall"]


REDISTRIBUTE_FUNC = "redistribute_input"
_DISPATCH_RECORD_HOOKS: list[Callable] = []
_DISPATCH_LOG_HOOKS: list[Callable] = []
_TRITON_KERNEL_HOOKS: list[Callable] = []

# Thread-local flag to skip recording during hash computation
import threading
_in_triton_hash_computation = threading.local()


@contextlib.contextmanager
def _skip_triton_hash_logging():
    """Context manager to skip logging operations during hash computation."""
    global _in_triton_hash_computation
    old_value = getattr(_in_triton_hash_computation, 'value', False)
    _in_triton_hash_computation.value = True
    try:
        yield
    finally:
        _in_triton_hash_computation.value = old_value


def _is_in_triton_hash_computation():
    """Check if we're currently computing hashes for Triton kernels."""
    global _in_triton_hash_computation
    return getattr(_in_triton_hash_computation, 'value', False)


def _stringify_shape(shape) -> str:
    return f"[{', '.join([str(x) for x in shape])}]"


def _stringify_device_mesh(mesh) -> str:
    return f"DM({', '.join([str(s) for s in mesh.shape])})"


def _stringify_placement(placement) -> str:
    return f"[{', '.join([str(p) for p in placement])}]"


def _stringify_attributes(tensor, attributes) -> str:
    pairs = {}
    for attr in attributes:
        if hasattr(tensor, attr):
            pairs[attr] = getattr(tensor, attr)
    if len(pairs) == 0:
        return ""
    return f"{{{', '.join([f'{k}={v}' for k, v in pairs.items()])}}}"


def _stringify_dtensor_spec(spec) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    return DTensorSpec.format_shard_order_str(spec.placements, spec.shard_order)


def _tensor_debug_string(tensor, attributes) -> str:
    """Convert tensor to debug string representation."""

    if isinstance(tensor, torch.Tensor):
        tensor_debug_str = f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}{_stringify_attributes(tensor, attributes)}"

        if isinstance(tensor, torch.distributed.tensor.DTensor):
            # omitted device mesh
            return f"dt: {tensor_debug_str}| {_stringify_dtensor_spec(tensor._spec)}"
        elif isinstance(tensor, FakeTensor):
            return f"ft: {tensor_debug_str}"
        else:
            return f"t: {tensor_debug_str}"
    else:
        raise RuntimeError(f"Unsupported tensor type: {type(tensor)}")


def _arg_to_str(arg, attributes) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    def to_str(x):
        if isinstance(x, torch.Tensor):
            return _tensor_debug_string(x, attributes)
        elif isinstance(x, DTensorSpec):
            return _stringify_dtensor_spec(x)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


def default_hash_fn(t: torch.Tensor, use_scalar: bool = False) -> torch.Tensor:
    """
    from Observer. Computes a hash for a tensor by converting it to float (if needed), making it contiguous,
    replacing NaN/inf values with fixed numbers, and then computing the L1 norm in float64 or complex128.
    This is used to generate a deterministic summary value for tensor comparison.
    """
    if not (t.is_floating_point() or t.is_complex()):
        t = t.float()
    t = t.contiguous()
    # Clean the tensor to handle NaN/inf values, then compute norm
    t_clean = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)

    dtype = torch.complex128 if t.is_complex() else torch.float64
    out = t_clean.norm(p=1, dtype=dtype)
    if use_scalar:
        return out.item()
    return out


def _get_stack_trace() -> str:
    from torch.fx.experimental.symbolic_shapes import uninteresting_files

    summary = CapturedTraceback.extract().summary()
    summary = summary[:-4]  # filter out DebugMode frames
    summary = [
        frame for frame in summary if frame.filename not in uninteresting_files()
    ]
    summary = traceback.StackSummary.from_list(summary)
    return "".join(summary.format())


class _DebugCall:
    """Base class for tracking operator calls in DebugMode"""

    def __init__(
        self,
        call_depth: int,
        record: Optional[dict[str, Any]] = None,
        log: Optional[dict[str, Any]] = None,
        stack: bool = False,
    ):
        self.call_depth = call_depth
        if stack:
            self.stack_trace = _get_stack_trace()

        # results from dispatch hooks
        self.record = record
        self.log = log

    def stringify_args(self, attributes: list[str]) -> None:
        """
        To reduce memory consumption, this method stringifies args/kwargs, stores the result, and deletes original args/kwargs.
        """
        raise NotImplementedError(
            "Subclasses must implement stringify_args(), even if no-op"
        )

    def render(self, attributes: list[str]) -> str:
        raise NotImplementedError("Subclasses must implement string render()")

    def __repr__(self) -> str:
        return self.render([])


class _OpCall(_DebugCall):
    """Normal operator call"""

    def __init__(
        self,
        op,
        args: tuple,
        kwargs: dict,
        call_depth: int,
        stack: bool = False,
    ):
        super().__init__(call_depth, stack=stack)
        self.op = op
        self.args = args
        self.kwargs = kwargs

        self.args_str: Optional[str] = None
        self.kwargs_str: Optional[str] = None

    def stringify_args(self, attributes: list[str]) -> None:
        self.args_str = ", ".join(_arg_to_str(arg, attributes) for arg in self.args)
        if self.kwargs:
            self.kwargs_str = ", " + ", ".join(
                f"{k}={_arg_to_str(v, attributes)}" for k, v in self.kwargs.items()
            )
        else:
            self.kwargs_str = ""
        del self.args
        del self.kwargs

    def render(self, attributes: list[str]) -> str:
        if self.args_str is not None:
            args_str = self.args_str
        else:
            args_str = ", ".join(_arg_to_str(arg, attributes) for arg in self.args)

        if self.kwargs_str is not None:
            kwargs_str = self.kwargs_str
        else:
            if self.kwargs:
                kwargs_str = ", " + ", ".join(
                    f"{k}={_arg_to_str(v, attributes)}" for k, v in self.kwargs.items()
                )
            else:
                kwargs_str = ""

        if isinstance(self.op, torch._ops.OpOverload):
            op_name = self.op.__qualname__
        elif hasattr(self.op, "__module__") and hasattr(self.op, "__name__"):
            op_name = f"{self.op.__module__}.{self.op.__name__}"
        else:
            op_name = str(self.op)

        base_str = f"{op_name}({args_str}{kwargs_str})"

        if self.log:
            base_str += f"  # {self.log}"
        return base_str

    def __iter__(self):
        # for BC; tuple(self) returns (op, args, kwargs, call_depth)
        if self.args_str is not None:
            yield from [self.op, self.args_str, self.kwargs_str, self.call_depth]
        else:
            yield from [self.op, self.args, self.kwargs, self.call_depth]


class _RedistributeCall(_DebugCall):
    """Redistribute call from DTensor dispatch"""

    def __init__(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str,
        call_depth,
        stack=False,
    ):
        super().__init__(call_depth, stack=stack)
        self.arg = arg
        self.src_placement = src_placement
        self.dst_placement = dst_placement
        self.transform_info_str = transform_info_str

        self.arg_str: Optional[str] = None

    def stringify_args(self, attributes: list[str]) -> None:
        self.arg_str = f"{_arg_to_str(self.arg, attributes)}"
        del self.arg

    def render(self, attributes: list[str]) -> str:
        if self.arg_str is not None:
            arg_str = self.arg_str
        else:
            arg_str = f"{_arg_to_str(self.arg, attributes)}"

        if self.transform_info_str is not None:  # prioritize over src/dst placements
            placement_str = f"trace: {self.transform_info_str}"
        else:
            src_placement_str = _arg_to_str(self.src_placement, attributes)
            dst_placement_str = _arg_to_str(self.dst_placement, attributes)
            placement_str = f"{src_placement_str} -> {dst_placement_str}"
        return f"{REDISTRIBUTE_FUNC}({arg_str}, {placement_str})"

    def __iter__(self):
        # for BC; tuple(self) returns (op, placement info, kwargs, call_depth)
        if self.arg_str is not None:
            arg = self.arg_str
        else:
            arg = self.arg

        yield REDISTRIBUTE_FUNC
        if self.transform_info_str:
            yield [arg, self.transform_info_str]
        else:
            yield [arg, self.src_placement, self.dst_placement]
        yield {}
        yield self.call_depth


class _NNModuleCall(_DebugCall):
    """Designates entering an nn.Module's forward method"""

    def __init__(self, module_name: str, call_depth: int, stack: bool = False):
        super().__init__(call_depth, stack=stack)
        self.module_name = module_name

    def stringify_args(self, attributes: list[str]) -> None:
        pass  # nothing to stringify

    def render(self, attributes: list[str]) -> str:
        return f"[nn.Mod] {self.module_name}"

    def __iter__(self):
        yield from [
            f"[nn.Mod] {self.module_name}",
            (),
            {},
            self.call_depth,
        ]


class _TritonKernelCall(_DebugCall):
    """Triton kernel launch from inductor"""

    def __init__(
        self,
        kernel_name: str,
        grid: tuple,
        args: tuple,
        kwargs: dict,
        call_depth: int,
        stack: bool = False,
        arg_names: Optional[tuple[str, ...]] = None,
    ):
        super().__init__(call_depth, stack=stack)
        self.kernel_name = kernel_name
        self.grid = grid
        self.args = args
        self.kwargs = kwargs
        self.arg_names = arg_names  # Names of the arguments

        self.args_str: Optional[str] = None
        self.kwargs_str: Optional[str] = None
        self.grid_str: Optional[str] = None

        # For tracking pre/post kernel state
        self.pre_hashes: Optional[dict] = None  # Hashes before kernel execution
        self.post_hashes: Optional[dict] = None  # Hashes after kernel execution
        self.modified_args: Optional[set[int]] = None  # Indices of modified arguments (outputs)

    def stringify_args(self, attributes: list[str]) -> None:
        self.args_str = ", ".join(_arg_to_str(arg, attributes) for arg in self.args)
        if self.kwargs:
            self.kwargs_str = ", " + ", ".join(
                f"{k}={_arg_to_str(v, attributes)}" for k, v in self.kwargs.items()
            )
        else:
            self.kwargs_str = ""
        self.grid_str = str(self.grid)
        del self.args
        del self.kwargs

    def render(self, attributes: list[str]) -> str:
        if self.args_str is not None:
            args_str = self.args_str
            kwargs_str = self.kwargs_str
            grid_str = self.grid_str
        else:
            # Build args string with names and input/output annotations
            args_parts = []
            for i, arg in enumerate(self.args):
                arg_str = _arg_to_str(arg, attributes)

                # Add argument name if available
                if self.arg_names and i < len(self.arg_names):
                    name = self.arg_names[i]
                    arg_str = f"{name}={arg_str}"

                # Mark as output if it was modified
                if self.modified_args and i in self.modified_args:
                    arg_str = f"[OUT] {arg_str}"

                args_parts.append(arg_str)

            args_str = ", ".join(args_parts)
            kwargs_str = (
                ", " + ", ".join(
                    f"{k}={_arg_to_str(v, attributes)}" for k, v in self.kwargs.items()
                )
                if self.kwargs
                else ""
            )
            grid_str = str(self.grid)

        base_str = f"[Triton] {self.kernel_name}[grid={grid_str}]({args_str}{kwargs_str})"

        if self.log:
            base_str += f"  # {self.log}"
        return base_str

    def __iter__(self):
        # for BC; tuple(self) returns (kernel_name, args, kwargs, call_depth)
        if self.args_str is not None:
            yield from [self.kernel_name, self.args_str, self.kwargs_str, self.call_depth]
        else:
            yield from [self.kernel_name, self.args, self.kwargs, self.call_depth]


def _run_hook(hook, *args):
    out = hook(*args)
    assert out is None or isinstance(out, dict)
    return out


def _call_triton_kernel_hooks_pre(kernel_name: str, grid: tuple, arg_names: tuple, *args, **kwargs):
    """
    Hook function called from generated inductor code before Triton kernel execution.
    Returns a token that can be used for post-execution hooks.
    """
    global _TRITON_KERNEL_HOOKS

    if not _TRITON_KERNEL_HOOKS:
        return None

    # Store pre-execution state for each hook
    hook_tokens = []
    for hook in _TRITON_KERNEL_HOOKS:
        try:
            token = hook(kernel_name, grid, arg_names, args, kwargs, pre_execution=True)
            hook_tokens.append(token)
        except Exception as e:
            # Don't let hooks break kernel execution
            import warnings
            warnings.warn(f"Triton kernel pre-hook failed: {e}")
            hook_tokens.append(None)

    return hook_tokens


def _call_triton_kernel_hooks_post(hook_tokens, *args, **kwargs):
    """
    Hook function called from generated inductor code after Triton kernel execution.
    """
    global _TRITON_KERNEL_HOOKS

    if not hook_tokens or not _TRITON_KERNEL_HOOKS:
        return

    for hook, token in zip(_TRITON_KERNEL_HOOKS, hook_tokens):
        try:
            hook(None, None, None, args, kwargs, pre_execution=False, token=token)
        except Exception as e:
            # Don't let hooks break kernel execution
            import warnings
            warnings.warn(f"Triton kernel post-hook failed: {e}")


def _run_dispatch_hooks(call: _DebugCall, func, types, args, kwargs, result) -> None:
    global _DISPATCH_RECORD_HOOKS, _DISPATCH_LOG_HOOKS
    if _DISPATCH_RECORD_HOOKS:
        record = {}
        for hook in _DISPATCH_RECORD_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, result)
            if hook_out is not None:
                record.update(hook_out)
        if record:
            call.record = record

    if _DISPATCH_LOG_HOOKS:
        log = {}
        for hook in _DISPATCH_LOG_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, result)
            if hook_out is not None:
                log.update(hook_out)
        if log:
            call.log = log


class DebugMode(TorchDispatchMode):
    def __init__(
        self,
        *,
        record_torchfunction=False,
        record_faketensor=False,
        record_realtensor=True,
        record_tensor_attributes=None,
        record_nn_module=False,
        store_original_args=False,
        record_stack_trace=False,
        record_triton_kernels=False,
    ):
        super().__init__()
        import torch.distributed.tensor  # noqa: F401

        self.supports_higher_order_operators = True

        # Pushes DebugMode onto the torchfunction stack, and records __torch_function__ calls as well.
        # WARNING: currently incompatible with torch.compile due to dynamo guard failures.
        self.record_torchfunction = record_torchfunction

        # Records __torch_dispatch__ calls on FakeTensors.
        self.record_faketensor = record_faketensor

        # Records __torch_dispatch__ calls on real tensors.
        self.record_realtensor = record_realtensor

        # Optional list[str] of tensor attributes, to be annotated in the string dump.
        self.record_tensor_attributes = record_tensor_attributes or []

        # Uses ModTracker to record nn.Module entrances, as _NNModuleCall entries.
        # This flag currently has no effect on torch.compiled-regions.
        self.record_nn_module = record_nn_module

        self.module_tracker: Optional[ModTracker] = None
        if self.record_nn_module:
            self.module_tracker_setup()

        # If True, stores call args/kwargs in logs, without immediately stringifying.
        # Defaults to False for memory concerns.
        self.store_original_args = store_original_args

        # For stack trace recording, stores log call stack traces in .stack_trace.
        self.record_stack_trace = record_stack_trace

        # Records Triton kernel calls from compiled inductor code.
        self.record_triton_kernels = record_triton_kernels

        self.operators = []
        self.call_depth = 0

    # Without this override, running torch.compile under DebugMode
    # will force torch.compile to always use the “eager” backend
    # With this, DebugMode will not take effect on torch.compile
    @classmethod
    def ignore_compile_internals(cls):
        return True

    def _record_call(self, call):
        if not self.store_original_args:
            call.stringify_args(self.record_tensor_attributes)
        self.operators.append(call)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        self._record_call(
            _OpCall(func, args, kwargs, self.call_depth, stack=self.record_stack_trace)
        )

        try:
            self.call_depth += 1
            return func(*args, **kwargs)
        finally:
            self.call_depth -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Skip recording if we're computing hashes for Triton kernels
        if _is_in_triton_hash_computation():
            return func(*args, **kwargs)

        # Record the operation with its call depth
        call = None
        if torch.distributed.tensor.DTensor in types:
            call = _OpCall(
                func, args, kwargs, self.call_depth, stack=self.record_stack_trace
            )
            self._record_call(call)
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    call = _OpCall(
                        func,
                        args,
                        kwargs,
                        self.call_depth + 1,
                        stack=self.record_stack_trace,
                    )
                    self._record_call(call)
        elif len(types) == 0:
            if self.record_realtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)

        result = func(*args, **kwargs)
        if call:
            _run_dispatch_hooks(call, func, types, args, kwargs, result)

        return result

    def __enter__(self):
        self.operators = []
        self.call_depth = 0

        if self.record_torchfunction:
            torch._C._push_on_torch_function_stack(self)

        if self.record_triton_kernels:
            self._triton_kernel_hook = self._make_triton_kernel_hook()
            global _TRITON_KERNEL_HOOKS
            _TRITON_KERNEL_HOOKS.append(self._triton_kernel_hook)

        super().__enter__()
        if self.record_nn_module:
            self.module_tracker.__enter__()  # type: ignore[attribute, union-attr]
        return self

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args):
        super().__exit__(*args)
        if self.record_nn_module:
            self.module_tracker.__exit__()  # type: ignore[attribute, union-attr]
        if self.record_torchfunction:
            torch._C._pop_torch_function_stack()
        if self.record_triton_kernels:
            global _TRITON_KERNEL_HOOKS
            _TRITON_KERNEL_HOOKS.remove(self._triton_kernel_hook)

    def module_tracker_setup(self):
        from torch.distributed._tools.mod_tracker import ModTracker

        self.module_tracker = ModTracker()

        # module pre-fw hook: record module call
        def pre_fw_hook(module, input):
            fqn = self.module_tracker._get_mod_name(module)  # type: ignore[attribute, union-attr]
            self.operators.append(_NNModuleCall(fqn, self.call_depth + 1))
            self.call_depth += 1

        # module post-fw hook: decrement call depth
        def post_fw_hook(module, input, output):
            self.call_depth -= 1

        self.module_tracker.register_user_hooks(pre_fw_hook, post_fw_hook)

    def _make_triton_kernel_hook(self):
        """Create a hook function that records Triton kernel calls."""
        # Track whether hash logging is enabled
        hash_logging_enabled = False
        for mode_stack_item in _get_current_dispatch_mode_stack():
            if isinstance(mode_stack_item, DebugMode):
                # Check if hash logging context is active
                # We'll detect this by checking if _DISPATCH_LOG_HOOKS is non-empty
                break

        def triton_kernel_hook(
            kernel_name: Optional[str],
            grid: Optional[tuple],
            arg_names: Optional[tuple],
            args: tuple,
            kwargs: dict,
            pre_execution: bool = True,
            token: Optional[Any] = None
        ):
            if pre_execution:
                # Pre-execution: create and record the call
                call = _TritonKernelCall(
                    kernel_name=kernel_name,
                    grid=grid,
                    args=args,
                    kwargs=kwargs,
                    arg_names=arg_names,
                    call_depth=1,  # Triton kernels are at a fixed depth in compiled code
                    stack=self.record_stack_trace,
                )

                # Compute pre-execution hashes if hash logging is enabled
                if _DISPATCH_LOG_HOOKS:
                    call.pre_hashes = {}
                    for i, arg in enumerate(args):
                        if isinstance(arg, torch.Tensor):
                            try:
                                # Skip logging hash computation operations
                                with _skip_triton_hash_logging():
                                    with torch._C._DisablePythonDispatcher():
                                        hash_val = default_hash_fn(arg, use_scalar=True)
                                        call.pre_hashes[i] = hash_val
                            except Exception:
                                pass

                self._record_call(call)
                return call  # Return the call object as the token

            else:
                # Post-execution: update the call with post-execution state
                if token is None:
                    return

                call = token

                # Compute post-execution hashes and detect modified arguments
                if _DISPATCH_LOG_HOOKS:
                    call.post_hashes = {}
                    call.modified_args = set()

                    for i, arg in enumerate(args):
                        if isinstance(arg, torch.Tensor):
                            try:
                                # Skip logging hash computation operations
                                with _skip_triton_hash_logging():
                                    with torch._C._DisablePythonDispatcher():
                                        hash_val = default_hash_fn(arg, use_scalar=True)
                                        call.post_hashes[i] = hash_val

                                        # Check if this argument was modified
                                        if i in call.pre_hashes and call.pre_hashes[i] != hash_val:
                                            call.modified_args.add(i)
                            except Exception:
                                pass

                    # Add hash info to log
                    if call.modified_args:
                        output_names = []
                        for i in call.modified_args:
                            if call.arg_names and i < len(call.arg_names):
                                output_names.append(call.arg_names[i])
                            else:
                                output_names.append(f"arg{i}")

                        if not call.log:
                            call.log = {}
                        call.log["outputs"] = output_names
                        call.log["pre_hashes"] = {
                            (call.arg_names[i] if call.arg_names and i < len(call.arg_names) else f"arg{i}"): call.pre_hashes.get(i)
                            for i in range(len(args)) if isinstance(args[i], torch.Tensor)
                        }
                        call.log["post_hashes"] = {
                            (call.arg_names[i] if call.arg_names and i < len(call.arg_names) else f"arg{i}"): call.post_hashes.get(i)
                            for i in range(len(args)) if isinstance(args[i], torch.Tensor)
                        }

        return triton_kernel_hook

    @contextlib.contextmanager
    def record_redistribute_calls(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str: Optional[str] = None,
    ):
        try:
            self._record_call(
                _RedistributeCall(
                    arg,
                    src_placement=src_placement,
                    dst_placement=dst_placement,
                    transform_info_str=transform_info_str,
                    call_depth=self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def debug_string(self) -> str:
        with torch._C.DisableTorchFunction():
            result = ""
            result += "\n".join(
                "  " + "  " * op.call_depth + op.render(self.record_tensor_attributes)
                for op in self.operators
            )
        return result

    @staticmethod
    @contextlib.contextmanager
    def dispatch_hooks(
        record_hook: Optional[Callable] = None,
        log_hook: Optional[Callable] = None,
    ):
        """
        Allows installing post-hooks on arguments to intercepted __torch_dispatch__ calls;
        hook signatures are expected as (func, types, args, kwargs, result),
        i.e. __torch_dispatch__ args + return value.

        Logging hook outputs are stored in call.log and annotate calls in debug_string(),
        while recording hook outputs are just stored in call.record.
        For now hooks are expected to return dictionaries.
        """
        global _DISPATCH_RECORD_HOOKS, _DISPATCH_LOG_HOOKS

        if record_hook:
            _DISPATCH_RECORD_HOOKS.append(record_hook)
        if log_hook:
            _DISPATCH_LOG_HOOKS.append(log_hook)
        try:
            yield
        finally:
            if record_hook:
                _DISPATCH_RECORD_HOOKS.pop()
            if log_hook:
                _DISPATCH_LOG_HOOKS.pop()

    @staticmethod
    @contextlib.contextmanager
    def record_outputs():
        """
        Hook for storing cloned output tensors in .record["output"].
        """

        def dispatch_hook(func, types, args, kwargs, result):
            with torch._C._DisablePythonDispatcher():
                out = tree_map(
                    lambda x: x.clone() if isinstance(x, torch.Tensor) else x, result
                )
            return {"output": out}

        with DebugMode.dispatch_hooks(record_hook=dispatch_hook):
            yield

    @staticmethod
    @contextlib.contextmanager
    def log_tensor_hashes(
        hash_fn: Optional[Callable] = None, hash_inputs: bool = False
    ):
        """
        Installs hook for tensor hash logging.

        hash_fn: optional function for custom hashing
        hash_inputs: if True, also hashes tensors in (args, kwargs), storing them in "input_hash".
        NOTE: this is currently a post-hook, so e.g. inplace ops will log the "output" hashes.
        """
        if hash_fn is None:
            hash_fn = functools.partial(default_hash_fn, use_scalar=True)

        def _tree_hash(obj):
            with torch._C._DisablePythonDispatcher():
                return tree_map(
                    lambda x: hash_fn(x) if isinstance(x, torch.Tensor) else None, obj
                )

        def _dispatch_hash_hook(func, types, args, kwargs, result):
            if "empty" in str(func) or "profiler" in str(func):
                return None

            out = {}
            out["hash"] = _tree_hash(result)
            if hash_inputs:
                out["input_hash"] = _tree_hash((args, kwargs))

            if tree_all(lambda x: x is None, out.values()):
                return None
            return out

        with DebugMode.dispatch_hooks(log_hook=_dispatch_hash_hook):
            yield


def get_active_debug_mode() -> Optional[DebugMode]:
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
