AI-generating more sharding rule coverage for DTensor

---

## Quick Start

**To run discovery for an operator, see:** `torch/distributed/tensor/DISCOVERY_PROMPT.md`

This contains copy-paste prompts for a new Claude process to execute the full pipeline.

---

There's a long-tail of ops that DTensor doesn't have sharding rule coverage for, see the xfails in test_dtensor_ops.py. While operator decompositions can help a bit with this, there a lot of ops with no sharding rule & no decomposition. This proposes an AI-driven harness for proposing & validating AI-generated sharding rules.

**Terminology**:
- **Op strategy**: A valid (input_placements, output_placements) combination that works without redistribution. The op can be executed purely locally on each device.
- **Sharding rule**: A broader term that may include rules requiring redistribution before/after the op.

The harness focuses on discovering **op strategies** - the no-redistribution cases.


The harness should consist of the following parts:


1) A function for programatically validating a proposed (operator, op strategy: input -> output placements, sample inputs) under a single mesh-dim.

**Critical: No DTensor dependency for validation**. The validation must NOT use DTensor to run the op, because:
- Ops without existing sharding rules would fail or fall back to all-Replicate
- Using DTensor would leak knowledge from existing rules, defeating the purpose of discovery
- We want to discover mathematically valid strategies independent of what DTensor knows

**Mesh setup**: Use a 1D mesh with size 2 (or 4). This is sufficient to catch most sharding bugs while being accessible on common multi-GPU machines. Multi-mesh-dim expansion can be handled separately using existing DTensor infrastructure.

**Per-device sample inputs**: Sample inputs should be specified per-device (one tensor set per device, not a single global tensor). This gives full control over the local tensor values and naturally handles all placement types:
- For **Replicate**: each device gets the same tensor
- For **Shard(dim)**: each device gets its chunk of the full tensor along dim
- For **Partial(reduce_op)**: each device gets a portion that reduces (e.g., sums) to the full tensor

This per-device specification is important because it allows testing scenarios where input tensors don't match across devices (which is the common case for Shard and Partial).

**Validation algorithm** (pure tensor math, no DTensor):
```
def validate_strategy(op, input_placements, output_placement, per_device_inputs, kwargs):
    for each rank r:
        local_inputs = per_device_inputs[r]

        # 1. Run op on local tensors directly
        local_result = op(*local_inputs, **kwargs)

        # 2. Compute expected local output:
        #    a. Reconstruct full inputs from per-device inputs
        full_inputs = [combine(per_device_inputs, placement) for placement in input_placements]
        #    b. Run op on full inputs
        full_output = op(*full_inputs, **kwargs)
        #    c. Extract local chunk based on output_placement
        expected_local = extract_chunk(full_output, output_placement, rank=r)

        # 3. Compare
        if local_result != expected_local:
            return False, "mismatch"

    return True, ""
```

This validates whether "running the op locally produces the correct chunk of the global result" without ever calling DTensor.


2) A set of adversarially-generated sample inputs for each operator, given the validation process.

The validation process in 1) can lead to a lot of false positives; numeric equivalence for one sample input might not mean the proposed rule is valid across all sample inputs/function configs, or may only be valid under certain input conditions.

**For comprehensive instructions on AI-generated adversarial inputs, see:**
`torch/distributed/tensor/_adversarial_inputs.md`

**For the implementation and registry of adversarial generators, see:**
`torch/distributed/tensor/_adversarial_inputs.py`

### Critical: Input Differentiation for Multi-Input Ops

**IMPORTANT**: For ops with multiple tensor inputs, each input MUST have different values. Using identical values for all inputs causes false positives because many ops have degenerate behavior when inputs are equal.

**Examples of false positives from identical inputs:**
- `torch.cross(x, x) = 0` for any x → zero output passes any placement validation
- `torch.sub(x, x) = 0` → same issue
- `torch.eq(x, x) = True` → boolean outputs mask numeric issues
- `torch.div(x, x) = 1` → constant output passes trivially

**Solution**: Generators take an `input_idx` parameter to produce different values per input:
```python
def make_arange(shape, dtype, device, input_idx=0):
    offset = input_idx * 100  # Different offset per input
    return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape) + offset

def make_randn(shape, dtype, device, input_idx=0):
    torch.manual_seed(42 + input_idx)  # Different seed per input
    return torch.randn(shape, dtype=dtype, device=device)

def make_negative(shape, dtype, device, input_idx=0):
    return torch.ones(shape, dtype=dtype, device=device) * (-1.5 - input_idx)
```

**Usage in validation**:
```python
full_inputs = [
    gen(torch.Size(shape), dtype, device, input_idx=i)
    for i, shape in enumerate(input_shapes)
]
```

### Multi-Input Validation

A strategy is only considered valid if it passes ALL sample inputs. This filters out false positives that only work with specific values.

**Default generator set**:
```python
DEFAULT_GENERATORS = [
    make_arange,   # 0, 1, 2, ... - catches index/ordering issues
    make_randn,    # random normal - general case
    make_zeros,    # all zeros - catches Partial false positives
    make_ones,     # all ones - catches mul/div edge cases
    make_negative, # all negative - catches sign-sensitive ops
]
```

**Why each generator matters**:
| Generator | What it catches |
|-----------|-----------------|
| `make_arange` | Index-dependent ops, ordering issues, non-commutativity |
| `make_randn` | General numeric correctness with varied values |
| `make_zeros` | Partial placement false positives (0 + 0 = 0 trivially) |
| `make_ones` | Multiplicative identity issues, div-by-one edge cases |
| `make_negative` | Sign-sensitive ops (abs, relu, max vs min) |

**Real example - torch.cross false positive filtering**:
```
Before (same values for both inputs):
  9 strategies including FALSE POSITIVES:
    ['P(sum)', 'P(sum)'] -> ['R']  ❌ cross(a/n, b/n) ≠ cross(a,b)
    ['R', 'P(sum)'] -> ['R']       ❌ cross(a, b/n) ≠ cross(a,b)

After (different values via input_idx):
  4 strategies, all CORRECT:
    ['R', 'R'] -> ['R']
    ['S(0)', 'S(0)'] -> ['S(0)']
    ['R', 'P(sum)'] -> ['P(sum)']   ✓ cross(a, b/n) = cross(a,b)/n
    ['P(sum)', 'R'] -> ['P(sum)']   ✓ cross(a/n, b) = cross(a,b)/n
```

### Validation API

```python
validate_strategy_multi_input(
    op, input_placements, output_placements,
    input_specs, device, world_size,
    generators=DEFAULT_GENERATORS, kwargs=None
) -> (is_valid, error_message, num_inputs_tested)
```


3) A routine for exhaustively enumerating all possible input -> output placements (i.e. op strategies), considering multi-inputs/outputs and non-tensor input/outputs, over Replicate(), Shard(), and all Partial() placements (all reduction types, but not NormPartial).

**Placement space for 1D mesh**: For each tensor argument, enumerate:
- `Replicate()`
- `Shard(dim)` for each valid dim in `range(tensor.ndim)`
- `Partial(reduce_op)` for reduce_op in `{SUM, MAX, MIN, ...}` (excluding NormPartial)

**Enumeration scope**: Given an op with N tensor inputs and M tensor outputs, enumerate all combinations of (input_placements, output_placements) where:
- `input_placements` is a tuple of N placements
- `output_placements` is a tuple of M placements
- Non-tensor inputs/outputs (scalars, dims, etc.) are ignored for placement purposes

**Kwargs enumeration**: For ops with input-dependent behavior (e.g., reduction dim), enumerate over relevant kwargs:
- `torch.sum(x, dim=...)`: test `dim=0`, `dim=1`, `dim=None`, with `keepdim=True/False`
- `torch.gather(x, dim, index)`: test different `dim` values
- Discovered strategies become conditional: `"S(0) -> S(0) when dim=1"`

**Kwargs sweep API**:
```python
discover_strategies_with_kwargs_sweep(
    op, input_ndims, output_ndims, input_shape, device, world_size,
    kwarg_name="dim", kwarg_values=[0, 1, ...]
) -> {
    "strategies": {strategy -> [valid_kwarg_values]},
    "kwarg_name": "dim",
    "kwarg_values_tested": [0, 1, ...],
}
```

Example output for torch.sum:
```
Discovered strategies (6):
  ['S(0)'] -> ['P(sum)']
    valid when: dim in [0]
  ['S(0)'] -> ['S(0)']
    valid when: dim in [1]
  ['S(1)'] -> ['P(sum)']
    valid when: dim in [1]
  ['S(1)'] -> ['S(0)']
    valid when: dim in [0]
```

With this, we can enumerate all op strategies, and validate them across the sample input set one-by-one.

**Reporting format**: When reporting discovered strategies for an op, include:
```
Op: torch.sum
Shape: (8, 16), Dtype: float32, Mesh: 1D size 2

Input generators:
  - make_arange: torch.arange(n).reshape(shape)
  - make_randn: torch.randn(shape) [seed=42]

Kwargs tested: dim in [0, 1, None], keepdim in [True, False]

Discovered strategies:
  dim=0, keepdim=False: [S(1)] -> [S(1)]
  dim=1, keepdim=False: [S(0)] -> [S(0)]
  dim=None: [R] -> [R]
```


4) The ability to, on top of the exhaustive enumeration, AI-analyze operator semantics, and propose additional input-dependent sharding rules that aren't included in the above. A lot of sharding rules are only valid under certain input-conditions, e.g. non-reduction/indexing dims are shardable - this depends on the dim arg; some rules are valid only tensor shape equality, or under tensor shape 1. Plenty of examples are in the DTensor codebase.

**For comprehensive instructions on AI-proposed conditional rules, see:**
`torch/distributed/tensor/_conditional_rules.md`

**For the implementation and synthesis logic, see:**
`torch/distributed/tensor/_conditional_rules.py`

**Key APIs:**
- `analyze_and_synthesize()` - runs dim sweep and synthesizes rules
- `synthesize_conditional_rules()` - infers patterns like "S(d) valid when d != dim"
- `get_analysis_prompt()` - generates prompt for AI analysis

**Example output for torch.sum:**
```
Conditional (valid for specific dim values):
  ['S(0)'] -> ['P(sum)']
    when: shard_dim == dim     # sharding reduced dim produces Partial
    type: op_dim
  ['S(0)'] -> ['S(0)']
    when: shard_dim != dim     # sharding non-reduced dim is preserved
    type: non_op_dim
```

We cannot enumerate these programatically, and the best hope is to AI-generate them given operator semantics. They can then be validated under the subset of sample inputs where the condition holds true.


5) A validation set: the existing DTensor codebase.

For validating how well this process works, we can blindly run it against ops that have existing DTensor coverage, and see how what rules are lost/found. For each operator, this also gives the chance to create a fuzzy "DSL", which does not need to be executable or complete, but serves as an intermediate validation step for proposed sharding rules.

An example for aten.gather.default:

 "aten.gather.default": {
    "strategies": [
      {
        "inputs": [
          "R",
          "R"
        ],
        "output": "R",
        "condition": null,
      },
      {
        "inputs": [
          "R",
          "S(dim)"
        ],
        "output": "S(dim)",
        "condition": null,
      },
      {
        "inputs": [
          "S(d)",
          "S(d)"
        ],
        "output": "S(d)",
        "condition": "d != dim AND input.ndim == index.ndim",
      },
      {
        "inputs": [
          "S(dim)",
          "_MaskPartial"
        ],
        "output": "_MaskPartial",
        "condition": "dim < index.ndim AND index.shape[dim] == 1",
      }
    ],
    "source_file": "_tensor_ops.py",
    "source_line": 660,
 }


---

## Appendix: Comprehensive Input Generation Checklist

When implementing or extending the harness, follow this checklist to avoid common pitfalls:

### Must-Have Properties

1. **Different values per input** (input_idx parameter)
   - Binary ops like `sub(x,x)=0`, `div(x,x)=1`, `cross(x,x)=0` produce degenerate outputs
   - Solution: `gen(shape, dtype, device, input_idx=i)` with offset/seed based on `input_idx`

2. **Multiple generator types** (not just random)
   - Random alone misses edge cases
   - Must include: ordered values, zeros, ones, negatives, mixed

3. **Fixed seeds for reproducibility**
   - `torch.manual_seed(42 + input_idx)` before random generation
   - Same values across all ranks in distributed setting

4. **ALL generators must pass** (not just one)
   - A strategy is valid only if it passes every generator
   - Early-exit on first failure for efficiency

### Common False Positive Patterns

| Pattern | Cause | Detection |
|---------|-------|-----------|
| `P, P -> R` | Zero inputs: `0/n + 0/n = 0` | Use `make_arange` with non-zero values |
| `P, R -> R` | Zero inputs or identity ops | Use different values per input |
| `S(d), S(d) -> R` | Same values cancel out | Use `input_idx` offset |
| Any -> R for cross/sub | `op(x, x) = 0` or constant | Different seeds per input |

### Op-Specific Considerations

**Reduction ops** (sum, mean, max, min):
- Test all valid `dim` values via kwargs sweep
- Output ndim changes based on `keepdim`
- Sharding reduced dim → Partial output

**Index ops** (gather, scatter, index_select):
- Index tensor often must be Replicate (sharding causes out-of-bounds)
- Consider restricting placement enumeration for index args

**Comparison ops** (eq, lt, gt):
- Boolean outputs - use `make_arange` to ensure varied True/False patterns
- Same inputs give all-True, masking issues

**Normalization ops** (batch_norm, layer_norm):
- Statistics computed over specific dims
- Sharding those dims requires allreduce

### Shape Considerations

- Use shapes divisible by world_size (e.g., 8 for world_size=4)
- Test both square and non-square for matmul-like ops
- Consider dimension-specific behavior (dim=0 vs dim=1)

### Extending Generators

To add a new generator, ensure it:
1. Takes `(shape, dtype, device, input_idx=0)` signature
2. Produces deterministic output (no unseeded random)
3. Produces different values for different `input_idx`
4. Tests a specific edge case not covered by existing generators

Example - sparse pattern generator:
```python
def make_sparse(shape, dtype, device, input_idx=0):
    t = torch.zeros(shape, dtype=dtype, device=device)
    t.view(-1)[::3 + input_idx] = 1.0  # Every 3rd/4th element
    return t
```
