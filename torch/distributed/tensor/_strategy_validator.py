"""
Harness for discovering DTensor op strategies via pure tensor math (no DTensor dependency).

An "op strategy" is a valid (input_placements, output_placements) combination that works
without redistribution - the op can be executed purely locally on each device.
"""
from itertools import product
from typing import Any, Callable

import torch
from torch.distributed.tensor import Replicate, Shard, Partial


# Input generators: callable(shape, dtype, device) -> Tensor
def make_arange(shape, dtype, device, input_idx=0):
    """Ordered values: 0, 1, 2, ... (offset by input_idx)"""
    offset = input_idx * 100
    return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape) + offset

def make_randn(shape, dtype, device, input_idx=0):
    """Random normal values (seeded for reproducibility)."""
    torch.manual_seed(42 + input_idx)
    return torch.randn(shape, dtype=dtype, device=device)

def make_zeros(shape, dtype, device, input_idx=0):
    """All zeros - catches Partial false positives."""
    return torch.zeros(shape, dtype=dtype, device=device)

def make_ones(shape, dtype, device, input_idx=0):
    """All ones."""
    return torch.ones(shape, dtype=dtype, device=device)

def make_negative(shape, dtype, device, input_idx=0):
    """All negative values."""
    return torch.ones(shape, dtype=dtype, device=device) * (-1.5 - input_idx)

def make_mixed(shape, dtype, device, input_idx=0):
    """Mixed positive/negative values."""
    t = torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape)
    return t - t.mean() + input_idx

# Default set of generators for thorough testing
DEFAULT_GENERATORS = [make_arange, make_randn, make_zeros, make_ones, make_negative]


def validate_strategy(
    op: Callable,
    input_placements: list,
    output_placements: list,
    per_device_inputs: list[list[torch.Tensor]],
    kwargs: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Validate an op strategy using pure tensor math (no DTensor).
    Returns (is_valid, error_message).
    """
    kwargs = kwargs or {}
    world_size = len(per_device_inputs)

    # Reconstruct full inputs from per-device inputs
    full_inputs = []
    for i, placement in enumerate(input_placements):
        tensors = [per_device_inputs[r][i] for r in range(world_size)]
        full_inputs.append(_combine_tensors(tensors, placement))

    # Run op on full inputs to get reference output
    try:
        full_output = op(*full_inputs, **kwargs)
    except Exception as e:
        return False, f"Full op failed: {e}"

    full_outputs = full_output if isinstance(full_output, (list, tuple)) else [full_output]

    if len(full_outputs) != len(output_placements):
        return False, f"Output count mismatch: {len(full_outputs)} vs {len(output_placements)}"

    # Collect local results from all ranks
    all_local_results = []
    for rank in range(world_size):
        local_inputs = per_device_inputs[rank]
        try:
            local_result = op(*local_inputs, **kwargs)
        except Exception as e:
            return False, f"Local op failed on rank {rank}: {e}"
        local_results = local_result if isinstance(local_result, (list, tuple)) else [local_result]
        all_local_results.append(local_results)

    # Validate each output
    for out_idx, (full_out, out_placement) in enumerate(zip(full_outputs, output_placements)):
        locals_for_output = [all_local_results[r][out_idx] for r in range(world_size)]
        is_valid, err = _validate_output_placement(locals_for_output, full_out, out_placement)
        if not is_valid:
            return False, f"Output {out_idx}: {err}"

    return True, ""


def validate_strategy_multi_input(
    op: Callable,
    input_placements: list,
    output_placements: list,
    input_specs: list[tuple],  # [(shape, dtype), ...]
    device: torch.device,
    world_size: int,
    generators: list[Callable] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[bool, str, int]:
    """
    Validate strategy with multiple sample inputs.
    Returns (is_valid, error_message, num_inputs_tested).
    Strategy is valid only if ALL sample inputs pass.
    """
    generators = generators or DEFAULT_GENERATORS
    kwargs = kwargs or {}

    num_tested = 0
    for gen in generators:
        per_device = generate_per_device_inputs(
            world_size, input_specs, input_placements, device,
            input_generators=[gen] * len(input_specs)
        )

        is_valid, error = validate_strategy(
            op, input_placements, output_placements, per_device, kwargs
        )
        num_tested += 1

        if not is_valid:
            return False, f"Failed with {gen.__name__}: {error}", num_tested

    return True, "", num_tested


def _validate_output_placement(
    local_results: list[torch.Tensor],
    full_output: torch.Tensor,
    placement,
) -> tuple[bool, str]:
    """Validate that local results match the expected output placement."""
    if isinstance(placement, Replicate):
        for r, local in enumerate(local_results):
            if local.shape != full_output.shape:
                return False, f"Replicate shape mismatch on rank {r}"
            if not torch.allclose(local, full_output, atol=1e-5, rtol=1e-5):
                return False, f"Replicate value mismatch on rank {r}"
        return True, ""

    elif isinstance(placement, Shard):
        try:
            combined = torch.cat(local_results, dim=placement.dim)
        except Exception as e:
            return False, f"Shard concat failed: {e}"
        if combined.shape != full_output.shape:
            return False, f"Shard shape mismatch: {combined.shape} vs {full_output.shape}"
        if not torch.allclose(combined, full_output, atol=1e-5, rtol=1e-5):
            return False, "Shard value mismatch"
        return True, ""

    elif isinstance(placement, Partial):
        if placement.reduce_op == "sum":
            combined = sum(local_results)
        elif placement.reduce_op == "max":
            combined = local_results[0].clone()
            for local in local_results[1:]:
                combined = torch.maximum(combined, local)
        elif placement.reduce_op == "min":
            combined = local_results[0].clone()
            for local in local_results[1:]:
                combined = torch.minimum(combined, local)
        else:
            return False, f"Unknown reduce_op: {placement.reduce_op}"

        if combined.shape != full_output.shape:
            return False, f"Partial shape mismatch"
        if not torch.allclose(combined, full_output, atol=1e-5, rtol=1e-5):
            return False, "Partial value mismatch"
        return True, ""

    return False, f"Unknown placement: {placement}"


def _combine_tensors(tensors: list[torch.Tensor], placement) -> torch.Tensor:
    """Combine per-device tensors into full tensor based on placement."""
    if isinstance(placement, Replicate):
        return tensors[0].clone()
    elif isinstance(placement, Shard):
        return torch.cat(tensors, dim=placement.dim)
    elif isinstance(placement, Partial):
        if placement.reduce_op == "sum":
            return sum(tensors)
        elif placement.reduce_op == "max":
            result = tensors[0].clone()
            for t in tensors[1:]:
                result = torch.maximum(result, t)
            return result
        elif placement.reduce_op == "min":
            result = tensors[0].clone()
            for t in tensors[1:]:
                result = torch.minimum(result, t)
            return result
        else:
            raise NotImplementedError(f"reduce_op {placement.reduce_op}")
    else:
        raise ValueError(f"Unknown placement: {placement}")


# Enumeration
def enumerate_placements(
    ndims: list[int],
    include_partial: bool = True,
    partial_ops: tuple[str, ...] = ("sum",),
) -> list[tuple]:
    """Enumerate all placement combinations for tensors with given ndims."""
    placement_options = []
    for ndim in ndims:
        options = [Replicate()]
        options.extend(Shard(d) for d in range(ndim))
        if include_partial:
            options.extend(Partial(op) for op in partial_ops)
        placement_options.append(options)
    return list(product(*placement_options))


def enumerate_strategies(
    input_ndims: list[int],
    output_ndims: list[int],
    include_partial_inputs: bool = True,
    include_partial_outputs: bool = True,
    partial_ops: tuple[str, ...] = ("sum",),
):
    """Enumerate all (input_placements, output_placements) combinations."""
    input_combos = enumerate_placements(input_ndims, include_partial_inputs, partial_ops)
    output_combos = enumerate_placements(output_ndims, include_partial_outputs, partial_ops)
    for inp, out in product(input_combos, output_combos):
        yield list(inp), list(out)


def generate_per_device_inputs(
    world_size: int,
    input_specs: list[tuple],  # [(shape, dtype), ...]
    input_placements: list,
    device: torch.device,
    input_generators: list[Callable] | None = None,
) -> list[list[torch.Tensor]]:
    """
    Generate per-device inputs for validation.
    Returns per_device_inputs[rank][input_idx].
    """
    if input_generators is None:
        input_generators = [make_arange] * len(input_specs)

    per_device = [[] for _ in range(world_size)]

    for input_idx, ((shape, dtype), placement, gen) in enumerate(zip(input_specs, input_placements, input_generators)):
        full_t = gen(torch.Size(shape), dtype, device, input_idx=input_idx)

        if isinstance(placement, Replicate):
            for r in range(world_size):
                per_device[r].append(full_t.clone())
        elif isinstance(placement, Shard):
            chunks = torch.chunk(full_t, world_size, dim=placement.dim)
            for r in range(world_size):
                idx = min(r, len(chunks) - 1)
                per_device[r].append(chunks[idx].clone())
        elif isinstance(placement, Partial):
            for r in range(world_size):
                if placement.reduce_op == "sum":
                    per_device[r].append(full_t / world_size)
                else:
                    per_device[r].append(full_t.clone())

    return per_device


def discover_strategies(
    op: Callable,
    input_ndims: list[int],
    output_ndims: list[int],
    input_shape: tuple[int, ...],
    device: torch.device,
    world_size: int = 2,
    kwargs_list: list[dict] | None = None,
    include_partial: bool = True,
    generators: list[Callable] | None = None,
) -> dict:
    """
    Discover all valid op strategies for an operator.

    Tests each strategy against multiple sample inputs (generators).
    A strategy is valid only if it passes ALL sample inputs.

    Returns dict mapping kwargs_key -> {
        "valid": [(input_placements, output_placements), ...],
        "generators_used": [gen_names],
    }
    """
    kwargs_list = kwargs_list or [{}]
    generators = generators or DEFAULT_GENERATORS
    dtype = torch.float32

    results = {}
    for kwargs in kwargs_list:
        kwargs_key = str(kwargs) if kwargs else "default"
        valid_strategies = []

        for input_placements, output_placements in enumerate_strategies(
            input_ndims, output_ndims,
            include_partial_inputs=include_partial,
            include_partial_outputs=include_partial,
        ):
            input_specs = [(input_shape, dtype)] * len(input_ndims)

            is_valid, _, _ = validate_strategy_multi_input(
                op, input_placements, output_placements,
                input_specs, device, world_size,
                generators=generators, kwargs=kwargs,
            )

            if is_valid:
                valid_strategies.append((
                    [str(p) for p in input_placements],
                    [str(p) for p in output_placements],
                ))

        results[kwargs_key] = {
            "valid": valid_strategies,
            "generators_used": [g.__name__ for g in generators],
        }

    return results


def discover_strategies_with_kwargs_sweep(
    op: Callable,
    input_ndims: list[int],
    output_ndims: list[int],
    input_shape: tuple[int, ...],
    device: torch.device,
    world_size: int = 2,
    kwarg_name: str = "dim",
    kwarg_values: list = None,
    include_partial: bool = True,
    generators: list[Callable] | None = None,
) -> dict:
    """
    Discover strategies while sweeping over a kwarg (e.g., dim).
    Reports which strategies are valid for which kwarg values.

    Returns dict mapping strategy -> list of valid kwarg values.
    """
    generators = generators or DEFAULT_GENERATORS
    dtype = torch.float32

    if kwarg_values is None:
        # Default: sweep dim from 0 to max input ndim - 1
        max_ndim = max(input_ndims)
        kwarg_values = list(range(max_ndim))

    # strategy -> list of kwarg values where it's valid
    strategy_conditions: dict[tuple, list] = {}

    for kwarg_val in kwarg_values:
        kwargs = {kwarg_name: kwarg_val}

        for input_placements, output_placements in enumerate_strategies(
            input_ndims, output_ndims,
            include_partial_inputs=include_partial,
            include_partial_outputs=include_partial,
        ):
            input_specs = [(input_shape, dtype)] * len(input_ndims)

            is_valid, _, _ = validate_strategy_multi_input(
                op, input_placements, output_placements,
                input_specs, device, world_size,
                generators=generators, kwargs=kwargs,
            )

            if is_valid:
                key = (
                    tuple(str(p) for p in input_placements),
                    tuple(str(p) for p in output_placements),
                )
                if key not in strategy_conditions:
                    strategy_conditions[key] = []
                strategy_conditions[key].append(kwarg_val)

    return {
        "strategies": strategy_conditions,
        "kwarg_name": kwarg_name,
        "kwarg_values_tested": kwarg_values,
        "generators_used": [g.__name__ for g in generators],
    }


def print_discovery_report(
    op_name: str,
    results: dict,
    input_shape: tuple,
    world_size: int,
):
    """Print a formatted report of discovered strategies."""
    print(f"Op: {op_name}")
    print(f"Shape: {input_shape}, Dtype: float32, Mesh: 1D size {world_size}")

    if "generators_used" in results:
        print(f"Generators: {results['generators_used']}")

    print()

    if "strategies" in results:
        # Kwargs sweep format
        print(f"Kwarg: {results['kwarg_name']}")
        print(f"Values tested: {results['kwarg_values_tested']}")
        print(f"\nDiscovered strategies ({len(results['strategies'])}):")
        for (inp, out), conditions in sorted(results['strategies'].items()):
            cond_str = f"{results['kwarg_name']} in {conditions}"
            print(f"  {list(inp)} -> {list(out)}")
            print(f"    valid when: {cond_str}")
    else:
        # Simple format
        for kwargs_key, data in results.items():
            print(f"Kwargs: {kwargs_key}")
            strategies = data["valid"] if isinstance(data, dict) else data
            print(f"  Valid strategies ({len(strategies)}):")
            for inp, out in strategies:
                print(f"    {inp} -> {out}")
        print()
