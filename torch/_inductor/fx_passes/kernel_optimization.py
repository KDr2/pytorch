import logging
from typing import Optional

import torch
from torch._dynamo.utils import counters, is_node_meta_valid

from .. import config


if config.is_fbcode():
    from ads_mkl.ops.helion.layer_norm import helion_layernorm
    from ads_mkl.ops.helion.rms_norm import helion_rmsnorm

from ..pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    MULTIPLE,
    register_graph_pattern,
)
from .split_cat import construct_pattern_matcher_pass


log = logging.getLogger(__name__)


def check_device(inputs: list[torch.Tensor], device: str = "cuda") -> bool:
    return all(input.device.type == device for input in inputs)


def should_replace_norm(inputs: list[Optional[torch.fx.Node]]) -> bool:
    inputs = [input for input in inputs if input is not None]

    if not all(is_node_meta_valid(input) for input in inputs):
        return False

    return check_device([input.meta["example_value"] for input in inputs])


def print_norm_pattern(match: Match, inputs: list[Optional[torch.fx.Node]]) -> None:
    node = match.nodes[-1]
    log.debug(
        "replace norm node %s with input shape: %s",
        node.target,
        ", ".join(
            str(input.meta["example_value"].shape) if input is not None else "None"
            for input in inputs
        ),
    )


@register_graph_pattern(
    CallFunctionVarArgs([torch.nn.functional.rms_norm, torch.rms_norm], users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("use_custom_rmsnorm_kernel_pass"),
    extra_check=lambda match: config.is_fbcode(),
)
def rms_norm_replacement(
    match: Match,
    input: torch.fx.Node,
    normalized_shape: list[int],
    weight: Optional[torch.fx.Node] = None,
    eps: Optional[float] = None,
) -> None:
    def repl(
        input: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        if config.pre_grad_fusion_options["use_custom_rmsnorm_kernel_pass"].get(
            "helion", False
        ):
            out, _ = helion_rmsnorm(input, weight, eps=eps)
            return out
        return

    if should_replace_norm([input, weight]):
        counters["inductor"]["use_custom_rmsnorm_kernel_pass"] += 1
        if eps is None:
            eps = torch.finfo(input.meta["example_value"].dtype).eps
        match.replace_by_example(repl, [input, weight, eps])
        print_norm_pattern(match, [input, weight])


@register_graph_pattern(
    CallFunctionVarArgs(torch.nn.functional.layer_norm, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("use_custom_layernorm_kernel_pass"),
    extra_check=lambda match: config.is_fbcode(),
)
def layer_norm_replacement(
    match: Match,
    input: torch.fx.Node,
    normalized_shape: list[int],
    weight: Optional[torch.fx.Node] = None,
    bias: Optional[torch.fx.Node] = None,
    eps: Optional[float] = 1e-5,
) -> None:
    def repl(
        input: torch.Tensor,
        normalized_shape: list[int],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: Optional[float] = 1e-5,
    ) -> Optional[torch.Tensor]:
        if config.pre_grad_fusion_options["use_custom_layernorm_kernel_pass"].get(
            "helion", False
        ):
            out, _, _ = helion_layernorm(input, normalized_shape, weight, bias, eps)
            return out
        return

    if should_replace_norm([input, weight, bias]):
        counters["inductor"]["use_custom_layernorm_kernel_pass"] += 1
        match.replace_by_example(repl, [input, normalized_shape, weight, bias, eps])
        print_norm_pattern(match, [input, weight, bias])
