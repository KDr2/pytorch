"""
Adversarial input generation for DTensor strategy validation.

This module provides infrastructure for using AI-generated adversarial inputs
alongside the default generators in strategy discovery.

Usage:
    1. Generate adversarial inputs using the prompt in _adversarial_inputs.md
    2. Save generated code to a file or paste into this module
    3. Use discover_strategies_with_adversarial() for enhanced validation
"""

import torch
from typing import Callable

from torch.distributed.tensor._strategy_validator import (
    DEFAULT_GENERATORS,
    discover_strategies,
    validate_strategy_multi_input,
)


# Registry of adversarial generators per op
_ADVERSARIAL_REGISTRY: dict[str, list[Callable]] = {}


def register_adversarial_generators(op_name: str, generators: list[Callable]):
    """Register adversarial generators for an operator."""
    _ADVERSARIAL_REGISTRY[op_name] = generators


def get_adversarial_generators(op_name: str) -> list[Callable]:
    """Get registered adversarial generators for an operator."""
    return _ADVERSARIAL_REGISTRY.get(op_name, [])


def discover_strategies_with_adversarial(
    op: Callable,
    op_name: str,
    input_ndims: list[int],
    output_ndims: list[int],
    input_shape: tuple[int, ...],
    device: torch.device,
    world_size: int = 2,
    kwargs_list: list[dict] | None = None,
    include_partial: bool = True,
    use_defaults: bool = True,
) -> dict:
    """
    Discover strategies using both default and adversarial generators.

    Args:
        op: The operator to test
        op_name: Name for looking up adversarial generators
        ... (same as discover_strategies)
        use_defaults: Whether to include DEFAULT_GENERATORS
    """
    generators = []
    if use_defaults:
        generators.extend(DEFAULT_GENERATORS)

    adversarial = get_adversarial_generators(op_name)
    generators.extend(adversarial)

    if not generators:
        generators = DEFAULT_GENERATORS

    return discover_strategies(
        op=op,
        input_ndims=input_ndims,
        output_ndims=output_ndims,
        input_shape=input_shape,
        device=device,
        world_size=world_size,
        kwargs_list=kwargs_list,
        include_partial=include_partial,
        generators=generators,
    )


# =============================================================================
# Example adversarial generators (can be AI-generated and pasted here)
# =============================================================================

# --- torch.cross ---
def gen_cross_orthogonal(shape, dtype, device, input_idx=0):
    """Orthogonal vectors - maximum cross product magnitude."""
    t = torch.zeros(shape, dtype=dtype, device=device)
    if input_idx == 0:
        t[:, 0] = torch.arange(shape[0], dtype=dtype, device=device) + 1
    else:
        t[:, 1] = torch.arange(shape[0], dtype=dtype, device=device) + 1
    return t

def gen_cross_parallel_scaled(shape, dtype, device, input_idx=0):
    """Parallel vectors with different scales - tests linearity."""
    scale = 1.0 + input_idx * 0.5
    t = torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape)
    return t * scale

def gen_cross_unit_basis(shape, dtype, device, input_idx=0):
    """Unit vectors along different axes."""
    t = torch.zeros(shape, dtype=dtype, device=device)
    axis = input_idx % 3
    t[:, axis] = 1.0
    return t

register_adversarial_generators("torch.cross", [
    gen_cross_orthogonal,
    gen_cross_parallel_scaled,
    gen_cross_unit_basis,
])


# --- torch.outer ---
def gen_outer_basis(shape, dtype, device, input_idx=0):
    """One-hot vectors - produces sparse outer product."""
    t = torch.zeros(shape, dtype=dtype, device=device)
    pos = (input_idx * 3) % shape[0]
    t[pos] = 1.0
    return t

def gen_outer_alternating(shape, dtype, device, input_idx=0):
    """Alternating sign pattern."""
    t = torch.ones(shape, dtype=dtype, device=device)
    t[::2] = -1
    return t * (input_idx + 1)

register_adversarial_generators("torch.outer", [
    gen_outer_basis,
    gen_outer_alternating,
])


# --- torch.kron ---
def gen_kron_identity_like(shape, dtype, device, input_idx=0):
    """Identity-like matrix - special structure in Kronecker product."""
    t = torch.eye(min(shape), dtype=dtype, device=device)
    if shape[0] != shape[1]:
        t = t[:shape[0], :shape[1]]
    return t * (input_idx + 1)

def gen_kron_rank_one(shape, dtype, device, input_idx=0):
    """Rank-one matrix (outer product of vectors)."""
    u = torch.arange(shape[0], dtype=dtype, device=device) + input_idx + 1
    v = torch.arange(shape[1], dtype=dtype, device=device) + input_idx + 1
    return u.unsqueeze(1) * v.unsqueeze(0)

register_adversarial_generators("torch.kron", [
    gen_kron_identity_like,
    gen_kron_rank_one,
])


# =============================================================================
# Prompt for AI generation
# =============================================================================

GENERATION_PROMPT = '''
Read the file torch/distributed/tensor/_adversarial_inputs.md for full instructions.

Generate adversarial input generators for the following operator:

OPERATOR: {op_name}
SIGNATURE: {signature}
SEMANTICS: {semantics}
SHAPES: {shapes}
KWARGS: {kwargs}

Follow the output format in the instructions. Each generator should:
1. Have signature: def gen_name(shape, dtype, device, input_idx=0)
2. Include docstring explaining what edge case it targets
3. Use input_idx to produce different values per input
4. Be deterministic

Output Python code that can be added to _adversarial_inputs_generated.py
'''


def get_generation_prompt(
    op_name: str,
    signature: str,
    semantics: str,
    shapes: list[tuple],
    kwargs: dict | None = None,
) -> str:
    """Generate a prompt for AI to create adversarial inputs."""
    return GENERATION_PROMPT.format(
        op_name=op_name,
        signature=signature,
        semantics=semantics,
        shapes=shapes,
        kwargs=kwargs or {},
    )
