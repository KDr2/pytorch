# pyre-strict

"""
Custom encoder functions for use with PersistentMemoizer.

This module provides reusable encoder functions that convert function parameters
into JSON-serializable dictionaries for caching purposes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, TYPE_CHECKING

from typing_extensions import NotRequired, ParamSpec, TypedDict

import torch
from torch import Tensor
from torch._inductor.runtime.caching.interfaces import DeferredRecording


if TYPE_CHECKING:
    from torch._inductor.ir import Buffer, Layout, MultiTemplateBuffer, TensorBox
    from torch._inductor.kernel_inputs import MMKernelInputs
    from torch._inductor.kernel_template_choice import KernelTemplateChoice
    from torch._inductor.pattern_matcher import Match
    from torch._inductor.select_algorithm import ChoiceCaller


# Type variable for function parameters
_P = ParamSpec("_P")


# =============================================================================
# Encoded Types
# =============================================================================


class EncodedTensor(TypedDict):
    """TypedDict for encoded tensor metadata."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


class EncodedNode(TypedDict):
    """TypedDict for encoded input node information (dtype, shape, stride)."""

    dtype: str
    shape: list[int]
    stride: list[int]


class EncodedChoice(TypedDict, total=False):
    """TypedDict for an encoded choice from a KernelTemplateChoice.

    Fields:
        template_id: identifies which template (e.g., "aten::mm", "mm")
        params: serialized params from ktc.params.to_serializeable_dict()
        rank: optional rank for multi-choice encoding (lower is better)
    """

    template_id: str
    params: dict[str, Any]
    rank: int


class TunedKernelEncodedParams(TypedDict):
    """TypedDict for encoded tuned kernel parameters (mm, addmm, etc.).

    This structure mirrors the key format from LookupTableChoices._generate_kernel_inputs_key:
    - nodes: list of (dtype, shape, stride) for each input node
    - scalars: optional dict of scalar key=value pairs (e.g., alpha, beta for addmm)
    """

    nodes: list[EncodedNode]
    scalars: NotRequired[dict[str, float | int]]


class TunedKernelEncodedResult(TypedDict):
    """TypedDict for encoded tuned kernel result (mm, addmm, etc.).

    The `_type` field determines which other fields are present:
    - "single_choice": `choice` is present
    - "multi_template_buffer": `choices` is present
    - "unknown": encoding failed, decoder should recompute
    """

    _type: NotRequired[str]
    choice: NotRequired[EncodedChoice]
    choices: NotRequired[list[EncodedChoice]]


class ShouldPadEncodedParams(TypedDict):
    """TypedDict for encoded should_pad parameters."""

    mat1: EncodedTensor
    mat2: EncodedTensor
    op: str
    input: EncodedTensor | None
    mat1_exclude_padding_time: bool
    mat2_exclude_padding_time: bool
    tf32: bool


# =============================================================================
# Encoder Helper Functions
# =============================================================================


def _encode_tensor(t: Tensor) -> EncodedTensor:
    """Encode a tensor's metadata into a JSON-serializable dict."""
    return EncodedTensor(
        shape=tuple(t.shape),
        stride=tuple(t.stride()),
        dtype=str(t.dtype),
    )


def _encode_kernel_inputs(kernel_inputs: MMKernelInputs) -> TunedKernelEncodedParams:
    """Encode MMKernelInputs into a human-readable dict."""
    dtypes = kernel_inputs.dtypes()
    shapes = kernel_inputs.shapes_hinted()
    strides = kernel_inputs.strides_hinted()

    nodes: list[EncodedNode] = [
        EncodedNode(
            dtype=str(dtype),
            shape=list(shape),
            stride=list(stride),
        )
        for dtype, shape, stride in zip(dtypes, shapes, strides)
    ]

    result = TunedKernelEncodedParams(nodes=nodes)

    if kernel_inputs._scalars:
        result["scalars"] = dict(kernel_inputs._scalars)

    return result


def _encode_choice_from_ktc(
    ktc: KernelTemplateChoice,
    rank: int | None = None,
) -> EncodedChoice | None:
    """Encode a choice from a KernelTemplateChoice."""
    if not hasattr(ktc, "template") or ktc.template is None:
        return None
    if not hasattr(ktc, "params") or ktc.params is None:
        return None

    result = EncodedChoice(
        template_id=ktc.template.uid,
        params=ktc.params.to_serializeable_dict(),
    )

    if rank is not None:
        result["rank"] = rank

    return result


def _encode_choice_from_caller_or_node(
    obj: ChoiceCaller | Buffer,
    rank: int | None = None,
) -> EncodedChoice | None:
    """Encode a choice from a ChoiceCaller or buffer node by extracting its KTC annotation.

    This function is general and works with any object that has an annotations dict
    containing a "ktc" key. This includes:
    - ChoiceCaller instances (from autotune_select_algorithm)
    - Buffer nodes like TemplateBuffer, ExternKernelOut, etc. (from output_node())

    Args:
        obj: A ChoiceCaller, buffer node, or any object with annotations["ktc"]
        rank: Optional rank for multi-choice encoding (lower is better)

    Returns:
        EncodedChoice if encoding succeeded, None otherwise
    """
    annotations = getattr(obj, "annotations", None)
    if not annotations or "ktc" not in annotations:
        return None

    ktc = annotations["ktc"]
    return _encode_choice_from_ktc(ktc, rank=rank)


def _encode_multi_template_buffer(
    buffer: MultiTemplateBuffer,
    fn: Callable[..., TensorBox],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> DeferredRecording[TensorBox, TunedKernelEncodedResult]:
    """Encode a MultiTemplateBuffer using deferred recording."""
    from torch._inductor.ir import MultiTemplateBuffer, StorageBox, TensorBox

    deferred: DeferredRecording[TensorBox, TunedKernelEncodedResult] = DeferredRecording()

    original_choice_timings = buffer.choice_timings  # type: ignore[union-attr]
    finalized = False

    def wrapped_choice_timings(
        hint_override: int | None = None,
    ) -> dict[object, float]:
        nonlocal finalized
        timings_result = original_choice_timings(hint_override)

        if hint_override is None and not finalized:
            finalized = True

            sorted_choices = sorted(timings_result.items(), key=lambda x: x[1])
            encoded_choices: list[EncodedChoice] = []
            encoding_failed = False

            for rank, (choice_caller, _timing) in enumerate(sorted_choices):
                encoded = _encode_choice_from_caller_or_node(choice_caller, rank=rank)
                if encoded is None:
                    encoding_failed = True
                    break
                encoded_choices.append(encoded)

            if encoding_failed or not encoded_choices:
                encoded_result = TunedKernelEncodedResult(_type="unknown")
            else:
                encoded_result = TunedKernelEncodedResult(
                    _type="multi_template_buffer",
                    choices=encoded_choices,
                )

            deferred.finalize(encoded_result)

        return timings_result

    buffer.choice_timings = wrapped_choice_timings  # type: ignore[union-attr]

    def get_interim() -> TensorBox:
        new_result = fn(*args, **kwargs)

        if isinstance(new_result, TensorBox) and isinstance(new_result.data, StorageBox):
            new_buffer = new_result.data.data
            if isinstance(new_buffer, MultiTemplateBuffer):
                new_buffer.choice_timings = wrapped_choice_timings

        return new_result

    deferred.get_interim_result = get_interim

    return deferred


# =============================================================================
# Encoders
# =============================================================================


def should_pad_params_encoder(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> ShouldPadEncodedParams:
    """Encode parameters for _should_pad into a human-readable dict.

    This encoder extracts only the information needed for caching:
    - Tensor shape, stride, and dtype (not the actual data)
    - Whether padding time should be excluded for mat1 and mat2
    - The operation as a string

    Args:
        match: The pattern match object
        mat1: First matrix tensor
        mat2: Second matrix tensor
        op: The operation being performed
        input: Optional input tensor for addmm

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    # Import here to avoid circular dependency
    from torch._inductor.fx_passes.pad_mm import should_exclude_padding_time

    return ShouldPadEncodedParams(
        mat1=_encode_tensor(mat1),
        mat2=_encode_tensor(mat2),
        op=str(op),
        input=_encode_tensor(input) if input is not None else None,
        mat1_exclude_padding_time=should_exclude_padding_time(match, "mat1"),
        mat2_exclude_padding_time=should_exclude_padding_time(match, "mat2"),
        tf32=False
        if mat1.dtype != torch.float32
        else cast(
            bool,
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.mkldnn.allow_tf32,
        ),
    )


def tuned_mm_params_encoder(
    mat1: Buffer,
    mat2: Buffer,
    out_dtype: torch.dtype | None = None,
    *,
    layout: Layout | None = None,
) -> TunedKernelEncodedParams:
    """Encode parameters for tuned_mm into a human-readable dict.

    This encoder mirrors the behavior of tuned_mm:
    1. First calls mm_args to realize the matrices (just like tuned_mm does)
    2. Creates MMKernelInputs with the realized matrices
    3. Extracts the same information used by _generate_kernel_inputs_key

    The encoding includes:
    - nodes: dtype, shape (hinted), and stride (hinted) for each input node
    - scalars: any scalar values (not used in basic mm, but used in addmm)

    Args:
        mat1: First matrix buffer
        mat2: Second matrix buffer
        out_dtype: Optional output dtype
        layout: Optional layout

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    from torch._inductor.kernel_inputs import MMKernelInputs
    from torch._inductor.kernel.mm_common import mm_args

    # First call mm_args to realize the matrices, exactly as done in tuned_mm (mm.py:375-377)
    _m, _n, _k, _layout, mat1_realized, mat2_realized = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )

    # Create MMKernelInputs with the realized matrices (mm.py:381-382)
    kernel_inputs = MMKernelInputs([mat1_realized, mat2_realized], out_dtype=out_dtype)

    return _encode_kernel_inputs(kernel_inputs)


def tuned_addmm_params_encoder(
    inp: Buffer,
    mat1: Buffer,
    mat2: Buffer,
    *,
    alpha: float | int = 1,
    beta: float | int = 1,
    layout: Layout | None = None,
) -> TunedKernelEncodedParams:
    """Encode parameters for tuned_addmm into a human-readable dict.

    This encoder mirrors the behavior of tuned_addmm:
    1. First calls mm_args to realize the matrices (just like tuned_addmm does)
    2. Creates MMKernelInputs with the realized matrices and scalars
    3. Extracts the same information used by _generate_kernel_inputs_key

    The encoding includes:
    - nodes: dtype, shape (hinted), and stride (hinted) for each input node
    - scalars: alpha and beta values

    Args:
        inp: Input bias buffer
        mat1: First matrix buffer
        mat2: Second matrix buffer
        alpha: Scalar multiplier for mat1 @ mat2
        beta: Scalar multiplier for inp
        layout: Optional layout

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    from torch._inductor.kernel_inputs import MMKernelInputs
    from torch._inductor.kernel.mm_common import mm_args

    # First call mm_args to realize the matrices, exactly as done in tuned_addmm (mm.py:627)
    _m, _n, _k, _layout, mat1_realized, mat2_realized, inp_expanded = mm_args(
        mat1, mat2, inp, layout=layout
    )

    # Create MMKernelInputs with the realized matrices (mm.py:632-634)
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1_realized, mat2_realized],
        scalars=dict(alpha=alpha, beta=beta),
    )

    return _encode_kernel_inputs(kernel_inputs)


def tuned_kernel_result_encoder(
    fn: Callable[_P, TensorBox],
) -> Callable[
    _P,
    Callable[
        [TensorBox],
        TunedKernelEncodedResult | DeferredRecording[TensorBox, TunedKernelEncodedResult],
    ],
]:
    """Factory factory that returns a params-to-encoder factory for tuned kernel results.

    This is a generic result encoder that works with any tuned kernel function
    (tuned_mm, tuned_addmm, etc.). It encodes choices using the KernelTemplateChoice
    (KTC) annotations on ChoiceCallers and/or the buffer's annotations dict.

    Encoding strategy:
    1. MultiTemplateBuffer → deferred recording (choice_timings is expensive)
       - When choice_timings completes, encode each ChoiceCaller via its KTC annotation
    2. Single output node (TemplateBuffer or ExternKernelOut) → extract KTC from
       buffer.annotations["ktc"]
    3. Unknown types → return "unknown" for recomputation on decode

    Args:
        fn: The underlying unwrapped function (passed by the memoizer)

    Returns:
        A factory that takes (*args, **kwargs) and returns an encoder function
    """

    def params_to_encoder(
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Callable[
        [TensorBox],
        TunedKernelEncodedResult | DeferredRecording[TensorBox, TunedKernelEncodedResult],
    ]:
        """Factory that returns an encoder function for the given params."""

        def encode_result(
            result: TensorBox,
        ) -> TunedKernelEncodedResult | DeferredRecording[
            TensorBox, TunedKernelEncodedResult
        ]:
            # Import at runtime to avoid circular imports
            from torch._inductor.ir import (
                MultiTemplateBuffer,
                StorageBox,
                TensorBox as TensorBoxType,
            )

            # Cases 1-2: TensorBox containing a StorageBox
            if isinstance(result, TensorBoxType) and isinstance(result.data, StorageBox):
                buffer = result.data.data

                # Case 1: MultiTemplateBuffer - use deferred recording
                if isinstance(buffer, MultiTemplateBuffer):
                    return _encode_multi_template_buffer(buffer, fn, args, kwargs)

                # Case 2: Single output node - encode via annotations["ktc"] if available
                encoded = _encode_choice_from_caller_or_node(buffer)
                if encoded is not None:
                    return TunedKernelEncodedResult(
                        _type="single_choice",
                        choice=encoded,
                    )

            # Fallback for unknown types - mark as unknown
            return TunedKernelEncodedResult(_type="unknown")

        return encode_result

    return params_to_encoder
