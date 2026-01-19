"""
Conditional sharding rules synthesis for DTensor.

This module provides infrastructure for:
1. Running kwargs sweeps to discover dim-dependent strategies
2. Synthesizing conditional rules from sweep results
3. Expressing rules in a structured format

Usage:
    1. AI proposes kwargs to sweep (see _conditional_rules.md)
    2. Run sweeps with discover_strategies_with_kwargs_sweep()
    3. Use synthesize_conditional_rules() to derive patterns
"""

import torch
from typing import Callable
from collections import defaultdict

from torch.distributed.tensor._strategy_validator import (
    discover_strategies_with_kwargs_sweep,
    discover_strategies,
    DEFAULT_GENERATORS,
)


def run_dim_sweep(
    op: Callable,
    input_ndims: list[int],
    output_ndims: list[int],
    input_shape: tuple[int, ...],
    device: torch.device,
    world_size: int = 2,
    dim_param: str = "dim",
    dim_values: list[int] | None = None,
) -> dict:
    """
    Run a sweep over dim values and return results.

    If dim_values is None, automatically determines valid dims from input_ndims.
    """
    if dim_values is None:
        max_ndim = max(input_ndims)
        dim_values = list(range(max_ndim))

    return discover_strategies_with_kwargs_sweep(
        op=op,
        input_ndims=input_ndims,
        output_ndims=output_ndims,
        input_shape=input_shape,
        device=device,
        world_size=world_size,
        kwarg_name=dim_param,
        kwarg_values=dim_values,
    )


def synthesize_conditional_rules(sweep_results: dict) -> list[dict]:
    """
    Synthesize conditional rules from kwargs sweep results.

    Takes output from discover_strategies_with_kwargs_sweep and identifies
    patterns like "S(d) valid when d != dim".

    Returns list of synthesized rules with conditions.
    """
    strategies = sweep_results.get("strategies", {})
    kwarg_name = sweep_results.get("kwarg_name", "dim")
    kwarg_values = sweep_results.get("kwarg_values_tested", [])

    if not strategies:
        return []

    synthesized = []

    # Group strategies by pattern
    for (input_pls, output_pls), valid_kwargs in strategies.items():
        rule = {
            "inputs": list(input_pls),
            "output": list(output_pls),
            "valid_when": {kwarg_name: valid_kwargs},
            "condition": None,
            "condition_type": None,
        }

        # Try to synthesize a condition
        condition = _infer_condition(input_pls, output_pls, kwarg_name, valid_kwargs, kwarg_values)
        if condition:
            rule["condition"] = condition["expression"]
            rule["condition_type"] = condition["type"]

        synthesized.append(rule)

    return synthesized


def _infer_condition(
    input_pls: tuple[str, ...],
    output_pls: tuple[str, ...],
    kwarg_name: str,
    valid_kwargs: list,
    all_kwargs: list,
) -> dict | None:
    """
    Try to infer a symbolic condition from observed valid kwargs.

    Returns dict with 'expression' and 'type', or None if no pattern found.
    """
    # Check if valid for all kwargs (unconditional)
    if set(valid_kwargs) == set(all_kwargs):
        return {"expression": "always", "type": "unconditional"}

    # Check if valid for no kwargs (should not happen for discovered strategies)
    if not valid_kwargs:
        return None

    # Look for "d != dim" pattern
    # If S(d) appears and valid when dim != d
    for pl in input_pls:
        if pl.startswith("S("):
            shard_dim = int(pl[2:-1])  # Extract dim from "S(0)"
            # Check if valid exactly when kwarg_name != shard_dim
            expected_valid = [k for k in all_kwargs if k != shard_dim]
            if set(valid_kwargs) == set(expected_valid):
                return {
                    "expression": f"shard_dim != {kwarg_name}",
                    "type": "non_op_dim",
                }

    # Look for "d == dim" pattern
    for pl in input_pls:
        if pl.startswith("S("):
            shard_dim = int(pl[2:-1])
            if valid_kwargs == [shard_dim]:
                return {
                    "expression": f"shard_dim == {kwarg_name}",
                    "type": "op_dim",
                }

    # No pattern found - just report the specific values
    return {
        "expression": f"{kwarg_name} in {valid_kwargs}",
        "type": "specific_values",
    }


def format_synthesized_rules(rules: list[dict], op_name: str) -> str:
    """Format synthesized rules for display."""
    lines = [f"Synthesized conditional rules for {op_name}:", ""]

    for rule in rules:
        inputs = rule["inputs"]
        outputs = rule["output"]
        condition = rule.get("condition", "always")
        cond_type = rule.get("condition_type", "unknown")

        lines.append(f"  {inputs} -> {outputs}")
        lines.append(f"    condition: {condition}")
        lines.append(f"    type: {cond_type}")
        lines.append("")

    return "\n".join(lines)


def analyze_and_synthesize(
    op: Callable,
    op_name: str,
    input_ndims: list[int],
    output_ndims: list[int],
    input_shape: tuple[int, ...],
    device: torch.device,
    world_size: int = 2,
    dim_param: str = "dim",
    dim_values: list[int] | None = None,
) -> dict:
    """
    Full pipeline: run sweep, synthesize rules, format output.

    Returns dict with sweep results and synthesized rules.
    """
    # Run the sweep
    sweep_results = run_dim_sweep(
        op=op,
        input_ndims=input_ndims,
        output_ndims=output_ndims,
        input_shape=input_shape,
        device=device,
        world_size=world_size,
        dim_param=dim_param,
        dim_values=dim_values,
    )

    # Synthesize rules
    synthesized = synthesize_conditional_rules(sweep_results)

    # Separate unconditional from conditional
    unconditional = [r for r in synthesized if r.get("condition_type") == "unconditional"]
    conditional = [r for r in synthesized if r.get("condition_type") != "unconditional"]

    return {
        "operator": op_name,
        "sweep_results": sweep_results,
        "unconditional_rules": unconditional,
        "conditional_rules": conditional,
        "formatted": format_synthesized_rules(synthesized, op_name),
    }


# =============================================================================
# Prompt generation for AI analysis
# =============================================================================

ANALYSIS_PROMPT = '''
Read the file torch/distributed/tensor/_conditional_rules.md for full instructions.

Analyze the following operator and propose conditional sharding rules:

OPERATOR: {op_name}
SIGNATURE: {signature}
SEMANTICS: {semantics}
INPUT NDIMS: {input_ndims}

Based on the operator semantics:
1. What kwargs should be swept? (e.g., dim values)
2. What conditional rules do you expect? (e.g., "S(d) valid when d != dim")
3. What is the mathematical justification?

Output in YAML format as shown in _conditional_rules.md
'''


def get_analysis_prompt(
    op_name: str,
    signature: str,
    semantics: str,
    input_ndims: list[int],
) -> str:
    """Generate prompt for AI to analyze op and propose conditional rules."""
    return ANALYSIS_PROMPT.format(
        op_name=op_name,
        signature=signature,
        semantics=semantics,
        input_ndims=input_ndims,
    )
