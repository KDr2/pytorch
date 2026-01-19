# AI-Generated Adversarial Inputs for DTensor Strategy Validation

This document provides instructions for an AI to generate adversarial test inputs for validating DTensor op strategies. Read this entire document before generating inputs.

## Task Overview

Given a PyTorch operator, generate a set of adversarial input generators that stress-test edge cases specific to that operator. The goal is to catch false positive sharding strategies that only work for certain input values.

## Input Format

You will be given:
1. **Operator name**: e.g., `torch.gather`, `torch.where`, `aten.index.Tensor`
2. **Operator signature**: parameter names and types
3. **Operator semantics**: brief description of what the op does
4. **Input shapes**: shapes of tensor inputs being tested
5. **Kwargs**: any keyword arguments being tested

## Output Format

Generate Python code defining adversarial input generators. Each generator must:
1. Follow the signature: `def gen_name(shape, dtype, device, input_idx=0) -> torch.Tensor`
2. Be deterministic (use fixed seeds for any randomness)
3. Produce different values for different `input_idx` values
4. Target a specific edge case with a comment explaining what it tests

**Output template:**
```python
# Adversarial generators for: {op_name}
# Generated for shapes: {shapes}, kwargs: {kwargs}

def gen_edge_case_1(shape, dtype, device, input_idx=0):
    """
    Tests: {what edge case this targets}
    Why: {why this might cause false positives}
    """
    # implementation
    return tensor

def gen_edge_case_2(shape, dtype, device, input_idx=0):
    """
    Tests: {what edge case this targets}
    Why: {why this might cause false positives}
    """
    # implementation
    return tensor

# ... more generators ...

ADVERSARIAL_GENERATORS = [gen_edge_case_1, gen_edge_case_2, ...]
```

## Edge Case Categories

When analyzing an operator, consider these categories:

### 1. Value-Based Edge Cases
- **Zeros**: Many ops have special behavior with zeros (division, log, cross product)
- **Ones**: Identity for multiplication, special for power ops
- **Negatives**: Sign-sensitive ops (abs, relu, sqrt of negative)
- **Infinities/NaN**: Boundary behavior
- **Very large/small values**: Overflow, underflow, precision loss
- **Mixed signs**: Positive and negative in same tensor

### 2. Pattern-Based Edge Cases
- **Constant tensors**: All same value - may trivially satisfy placements
- **Monotonic**: Strictly increasing/decreasing
- **Alternating**: Alternating patterns that might align with sharding boundaries
- **Sparse**: Mostly zeros with few non-zeros
- **Boundary values**: Values at dtype min/max

### 3. Relationship-Based Edge Cases (for multi-input ops)
- **Identical inputs**: `op(x, x)` often produces degenerate results
- **One input is subset of other**: Overlapping values
- **Disjoint ranges**: No overlapping values
- **One dominates**: One input much larger magnitude than other
- **Inverse relationship**: `x` and `1/x`, or `x` and `-x`

### 4. Structure-Based Edge Cases
- **Alignment with shard boundaries**: Values that change at chunk boundaries
- **Uniform within chunks**: Same value per shard, different across shards
- **Single non-zero per chunk**: Sparse patterns aligned with sharding

### 5. Op-Specific Edge Cases

**Reduction ops** (sum, mean, max, min, prod):
- All same value (max/min becomes ambiguous across shards)
- Extreme value in one position only
- Values that sum to zero

**Index ops** (gather, scatter, index_select):
- Indices at boundaries (0, size-1)
- Duplicate indices
- Out-of-order indices
- Indices that would be invalid if tensor is sharded

**Comparison ops** (eq, ne, lt, gt, le, ge):
- Inputs designed to produce mixed True/False
- Inputs at equality boundary
- All-True and all-False cases

**Arithmetic ops** (add, sub, mul, div, pow):
- Division by values close to zero
- Negative base with fractional exponent
- Subtraction of nearly-equal values (cancellation)

**Matrix ops** (matmul, mm, bmm):
- Orthogonal matrices
- Identity-like matrices
- Rank-deficient matrices
- Values that produce exact integer results vs floating point

**Normalization ops** (softmax, layer_norm, batch_norm):
- Constant input (gradient issues)
- Very large variance
- Single outlier value

## Analysis Process

Before generating inputs, analyze the operator:

1. **What does this op compute mathematically?**
   - Write the formula if possible
   - Identify any special cases in the math

2. **What makes a sharding strategy valid for this op?**
   - Which dims can be sharded independently?
   - What communication would be needed for invalid shardings?

3. **What input values could mask an invalid strategy?**
   - Could zeros make an invalid strategy appear valid?
   - Could identical inputs hide issues?
   - Could specific patterns align with shard boundaries?

4. **What are the known edge cases for this op?**
   - Check PyTorch documentation for special behaviors
   - Consider numerical stability issues

## Examples

### Example 1: torch.cross

**Operator**: `torch.cross(input, other, dim=-1)`
**Semantics**: Cross product of 3D vectors
**Key insight**: `cross(x, x) = 0` for any x; cross is linear in each argument

```python
# Adversarial generators for: torch.cross
# Generated for shapes: [(8, 3), (8, 3)], kwargs: {'dim': 1}

def gen_orthogonal_vectors(shape, dtype, device, input_idx=0):
    """
    Tests: Orthogonal input vectors
    Why: Cross product of orthogonal vectors has maximum magnitude,
         stress-tests numeric precision in sharding
    """
    t = torch.zeros(shape, dtype=dtype, device=device)
    if input_idx == 0:
        t[:, 0] = torch.arange(shape[0], dtype=dtype, device=device) + 1
    else:
        t[:, 1] = torch.arange(shape[0], dtype=dtype, device=device) + 1
    return t

def gen_parallel_vectors(shape, dtype, device, input_idx=0):
    """
    Tests: Parallel vectors (cross product = 0)
    Why: Zero output could mask invalid strategies, but with different
         magnitudes per input, scaling issues become visible
    """
    scale = 1.0 + input_idx * 0.5
    t = torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape)
    return t * scale

def gen_unit_basis(shape, dtype, device, input_idx=0):
    """
    Tests: Unit vectors along different axes
    Why: Cross product of basis vectors gives clean results,
         catches any index/ordering issues in sharding
    """
    t = torch.zeros(shape, dtype=dtype, device=device)
    axis = input_idx % 3
    t[:, axis] = 1.0
    return t

def gen_mixed_magnitude(shape, dtype, device, input_idx=0):
    """
    Tests: Vectors with varying magnitudes across batch
    Why: Different magnitudes per batch element stress-tests
         whether sharding preserves per-element behavior
    """
    torch.manual_seed(123 + input_idx)
    t = torch.randn(shape, dtype=dtype, device=device)
    # Scale each row differently
    scales = torch.arange(1, shape[0] + 1, dtype=dtype, device=device).unsqueeze(1)
    return t * scales

ADVERSARIAL_GENERATORS = [
    gen_orthogonal_vectors,
    gen_parallel_vectors,
    gen_unit_basis,
    gen_mixed_magnitude,
]
```

### Example 2: torch.gather

**Operator**: `torch.gather(input, dim, index)`
**Semantics**: Gather values along dim using index tensor
**Key insight**: Index values determine which elements are accessed; sharding input may make indices invalid

```python
# Adversarial generators for: torch.gather
# Generated for shapes: input=(8, 16), index=(8, 4), kwargs: {'dim': 1}

def gen_sequential_indices(shape, dtype, device, input_idx=0):
    """
    Tests: Sequential index pattern
    Why: Indices 0,1,2,3 would be valid only for first shard if input is sharded
    """
    if input_idx == 0:  # input tensor
        return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape)
    else:  # index tensor
        # Indices in range [0, 4) - only valid for first chunk if sharded
        idx = torch.arange(shape[1], device=device).unsqueeze(0).expand(shape[0], -1)
        return idx.to(torch.int64)

def gen_boundary_indices(shape, dtype, device, input_idx=0):
    """
    Tests: Indices at shard boundaries
    Why: Index value 3 and 4 would cross shard boundary for dim=1 with world_size=4
    """
    if input_idx == 0:
        return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape)
    else:
        # Indices specifically at boundaries: 3, 4, 7, 8, etc.
        idx = torch.tensor([3, 4, 7, 8], device=device).unsqueeze(0).expand(shape[0], -1)
        return idx.to(torch.int64)

def gen_duplicate_indices(shape, dtype, device, input_idx=0):
    """
    Tests: Same index repeated
    Why: Duplicate gathering might mask issues if all ranks get same value
    """
    if input_idx == 0:
        return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape)
    else:
        # All same index
        idx = torch.zeros(shape, device=device, dtype=torch.int64)
        return idx

def gen_reversed_indices(shape, dtype, device, input_idx=0):
    """
    Tests: Indices in reverse order
    Why: Tests whether gather preserves order correctly under sharding
    """
    if input_idx == 0:
        return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape)
    else:
        # Reversed indices
        idx = torch.arange(shape[1] - 1, -1, -1, device=device).unsqueeze(0).expand(shape[0], -1)
        return idx.to(torch.int64)

ADVERSARIAL_GENERATORS = [
    gen_sequential_indices,
    gen_boundary_indices,
    gen_duplicate_indices,
    gen_reversed_indices,
]
```

### Example 3: torch.where

**Operator**: `torch.where(condition, input, other)`
**Semantics**: Element-wise selection based on condition
**Key insight**: Result mixes values from both inputs based on condition pattern

```python
# Adversarial generators for: torch.where
# Generated for shapes: [(8, 16), (8, 16), (8, 16)], kwargs: {}

def gen_alternating_condition(shape, dtype, device, input_idx=0):
    """
    Tests: Checkerboard True/False pattern
    Why: Alternating selection from both inputs; catches if sharding
         disrupts the interleaving
    """
    if input_idx == 0:  # condition
        t = torch.zeros(shape, dtype=torch.bool, device=device)
        t.view(-1)[::2] = True
        return t
    else:
        return torch.full(shape, input_idx * 10.0, dtype=dtype, device=device)

def gen_row_condition(shape, dtype, device, input_idx=0):
    """
    Tests: Condition True for even rows, False for odd
    Why: Row-aligned pattern - tests sharding along dim=0
    """
    if input_idx == 0:
        t = torch.zeros(shape, dtype=torch.bool, device=device)
        t[::2, :] = True
        return t
    else:
        return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape) + input_idx * 100

def gen_col_condition(shape, dtype, device, input_idx=0):
    """
    Tests: Condition True for even cols, False for odd
    Why: Column-aligned pattern - tests sharding along dim=1
    """
    if input_idx == 0:
        t = torch.zeros(shape, dtype=torch.bool, device=device)
        t[:, ::2] = True
        return t
    else:
        return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape) + input_idx * 100

def gen_single_true(shape, dtype, device, input_idx=0):
    """
    Tests: Only one True in condition
    Why: Single selection point - catches if it falls in wrong shard
    """
    if input_idx == 0:
        t = torch.zeros(shape, dtype=torch.bool, device=device)
        t[shape[0]//2, shape[1]//2] = True  # Middle element
        return t
    else:
        return torch.full(shape, input_idx * 5.0, dtype=dtype, device=device)

ADVERSARIAL_GENERATORS = [
    gen_alternating_condition,
    gen_row_condition,
    gen_col_condition,
    gen_single_true,
]
```

## Prompt Template

When asked to generate adversarial inputs for an operator, use this template:

---

**TASK**: Generate adversarial input generators for DTensor strategy validation.

**OPERATOR**: {op_name}
**SIGNATURE**: {signature}
**SEMANTICS**: {description}
**SHAPES**: {input_shapes}
**KWARGS**: {kwargs}

**ANALYSIS**:
1. Mathematical behavior: {analysis}
2. Edge cases to target: {list}
3. Potential false positive patterns: {list}

**GENERATORS**:
```python
{generated code following the output format}
```

---

## Integration with Harness

The generated adversarial inputs should be used alongside the default generators:

```python
from torch.distributed.tensor._strategy_validator import DEFAULT_GENERATORS

# Load or generate adversarial inputs for specific op
adversarial_gens = get_adversarial_generators(op_name, shapes, kwargs)

# Combine with defaults
all_generators = DEFAULT_GENERATORS + adversarial_gens

# Run validation
results = discover_strategies(
    op=op,
    generators=all_generators,
    ...
)
```

## Quality Checklist

Before finalizing generated inputs, verify:

- [ ] Each generator has a clear docstring explaining what it tests and why
- [ ] Generators use `input_idx` to produce different values per input
- [ ] Generators are deterministic (fixed seeds for any randomness)
- [ ] At least one generator targets each relevant edge case category
- [ ] Generators handle the correct dtypes (e.g., int64 for indices, bool for conditions)
- [ ] The set of generators would catch the false positives identified in analysis
