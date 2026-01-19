# DTensor Sharding Rule Discovery - Entry Point

## Quick Start Prompt

Copy and paste this to a new Claude process:

---

**Prompt:**

```
I want you to discover sharding rules for a PyTorch operator using the DTensor strategy validation harness.

Read these files first:
1. ai_generated_sharding_rules.md - Overview of the process
2. torch/distributed/tensor/_adversarial_inputs.md - How to generate adversarial test inputs
3. torch/distributed/tensor/_conditional_rules.md - How to propose dim-dependent rules
4. torch/distributed/tensor/_ops/_tensor_ops.py - See searchsorted_single_dim_strategy for code format

Then run the full discovery process for the requested operator.

The process is:
1. Analyze the operator semantics
2. Propose kwargs to sweep (e.g., dim values)
3. Generate adversarial input generators specific to this op
4. Run the harness with discover_strategies() or analyze_and_synthesize()
5. Synthesize conditional rules from results
6. Report findings in structured format
7. Write the sharding rule code (using @register_single_dim_strategy)
8. Write minimal tests for the new sharding rule
```

---

## Detailed Prompt Template

For any operator, use this template:

```
I want you to discover sharding rules for: {OP_NAME}

Operator signature: {SIGNATURE}
Operator semantics: {BRIEF_DESCRIPTION}
Input shapes to test: {SHAPES}

Read these instruction files:
- ai_generated_sharding_rules.md
- torch/distributed/tensor/_adversarial_inputs.md
- torch/distributed/tensor/_conditional_rules.md

Then execute the full discovery pipeline:

1. **Analyze**: What category is this op? What dims matter?

2. **Propose adversarial inputs**: Generate anywhere from 5-20 op-specific input generators
   that target edge cases for this op, depending on operator complexity and argument variability.

3. **Propose kwargs sweep**: Which kwargs should be varied? What values?

4. **Run discovery**: Use the harness to find valid strategies.
   ```python
   from torch.distributed.tensor._conditional_rules import analyze_and_synthesize
   result = analyze_and_synthesize(
       op=torch.{op},
       op_name="torch.{op}",
       input_ndims=[...],
       output_ndims=[...],
       input_shape=(...),
       device=torch.device("cpu"),
       dim_values=[...],
   )
   ```

5. **Synthesize**: What conditional rules emerged? Express as:
   - "S(d) -> S(d) WHEN d != dim"
   - Mathematical justification

6. **Report**: Output in this format:
   ```yaml
   operator: torch.{op}

   sample_inputs_tested:
     shapes: [(8, 3), (8, 3)]
     dtypes: [float32]
     generators:
       - make_arange: "Ordered values with input_idx offset"
       - make_randn: "Random normal with seed=42+input_idx"
       - make_zeros: "All zeros"
       - make_ones: "All ones"
       - make_negative: "All negative values"
     kwargs_swept:
       dim: [0, 1]

   unconditional_rules:
     - inputs: [...], output: [...]
   conditional_rules:
     - inputs: [...], output: [...], condition: "..."

   adversarial_generators:
     - name: "description of what edge case it targets"
   ```

7. **Write sharding rule code**: Based on the discovered strategies, write the actual
   DTensor sharding rule code using `@register_single_dim_strategy`. Read:
   - `torch/distributed/tensor/_ops/*_ops.py` - example sharding rules
   - `torch/distributed/tensor/_ops/single_dim_strategy.py` - infrastructure

8. **Write minimal tests**: Add tests for the new sharding rule in
   `test/distributed/tensor/test_tensor_ops.py`. Test each discovered strategy.
```

---

## Example: Complete Prompt for torch.cross

```
I want you to discover sharding rules for: torch.cross

Operator signature: torch.cross(input, other, dim=-1) -> Tensor
Operator semantics: Cross product of two tensors of 3D vectors along specified dim.
                    The dim must have size 3.
Input shapes to test: [(8, 3), (8, 3)] with dim=1

Read these instruction files first:
- ai_generated_sharding_rules.md
- torch/distributed/tensor/_adversarial_inputs.md
- torch/distributed/tensor/_conditional_rules.md

Then execute the full discovery pipeline:

1. Analyze the operator - what makes cross product special for sharding?

2. Generate adversarial inputs - what edge cases matter for cross product?
   (hint: orthogonal vectors, parallel vectors, unit vectors)

3. Propose kwargs sweep - which dim values can we test?

4. Run discovery using the harness

5. Synthesize conditional rules - when is S(d) valid?

6. Report findings in structured YAML format
```

---

## What Claude Should Do

When Claude receives this prompt, it should:

### Step 1: Read the instruction files
```python
# Claude reads:
# - ai_generated_sharding_rules.md (overview)
# - _adversarial_inputs.md (input generation instructions)
# - _conditional_rules.md (conditional rule instructions)
```

### Step 2: Analyze the operator
```
torch.cross computes cross product along dim.
- Category: Element-wise along batch dims, cross-dim operation along dim
- Dim parameter: dim (must have size 3)
- Mathematical property: bilinear - cross(a+b, c) = cross(a,c) + cross(b,c)
```

### Step 3: Generate adversarial inputs
```python
def gen_orthogonal(shape, dtype, device, input_idx=0):
    """Orthogonal vectors maximize cross product magnitude."""
    t = torch.zeros(shape, dtype=dtype, device=device)
    t[:, input_idx % 3] = torch.arange(shape[0], dtype=dtype, device=device) + 1
    return t

def gen_parallel_scaled(shape, dtype, device, input_idx=0):
    """Parallel vectors (cross = 0) with different scales."""
    return torch.arange(shape.numel(), dtype=dtype, device=device).reshape(shape) * (1 + input_idx)

# ... more generators
```

### Step 4: Run discovery
```python
from torch.distributed.tensor._conditional_rules import analyze_and_synthesize

result = analyze_and_synthesize(
    op=torch.cross,
    op_name="torch.cross",
    input_ndims=[2, 2],
    output_ndims=[2],
    input_shape=(8, 3),
    device=torch.device("cpu"),
    world_size=2,
    dim_param="dim",
    dim_values=[1],  # Only dim=1 valid for shape (8,3)
)
print(result["formatted"])
```

### Step 5: Synthesize rules
```
Observed: S(0), S(0) -> S(0) valid when dim=1
Synthesized: S(d), S(d) -> S(d) WHEN d != dim

Justification: Cross product operates along dim. Other dimensions are
batch dimensions where each element is independent. Sharding batch dims
means each shard computes cross product of its vectors independently.
```

### Step 6: Report
```yaml
operator: torch.cross
signature: torch.cross(input, other, dim=-1)

analysis:
  category: cross-dim operation
  dim_parameter: dim
  mathematical_properties:
    - bilinear in both arguments
    - cross(x, x) = 0

sample_inputs_tested:
  shapes: [(8, 3), (8, 3)]
  dtypes: [float32]
  world_size: 2
  generators:
    - make_arange: "Ordered values [0,1,2,...] + input_idx*100"
    - make_randn: "Random normal, seed=42+input_idx"
    - make_zeros: "All zeros"
    - make_ones: "All ones"
    - make_negative: "All -1.5 - input_idx"
  kwargs_swept:
    dim: [1]  # Only dim=1 valid for shape (8,3)

unconditional_rules:
  - inputs: [R, R], output: [R]
  - inputs: [R, P(sum)], output: [P(sum)]
  - inputs: [P(sum), R], output: [P(sum)]

conditional_rules:
  - inputs: [S(d), S(d)]
    output: [S(d)]
    condition: "d != dim"
    justification: "Batch dims are independent"

adversarial_generators:
  - gen_orthogonal: "Orthogonal vectors for max magnitude"
  - gen_parallel_scaled: "Parallel vectors with different scales"
  - gen_unit_basis: "Unit vectors along axes"
```

### Step 7: Write sharding rule code

Read existing sharding rule implementations for reference:
- `torch/distributed/tensor/_ops/_tensor_ops.py` - see `searchsorted_single_dim_strategy` for format
- `torch/distributed/tensor/_ops/single_dim_strategy.py` - infrastructure

**Use `@register_single_dim_strategy`** - the simpler single-mesh-dim format:
- Returns `list[list[Placement | _ShardingPlaceholder]]`
- Each inner list is: `[output_placement, input1_placement, input2_placement, ...]`
- Uses `_ShardingPlaceholder(dim)` for generic "shard on this dim"
- R, R -> R is added automatically (don't include it)
- Uses `Replicate()`, `Partial()`, `_ShardingPlaceholder(d)` for placements

Then write the sharding rule for torch.cross:

```python
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor.placement_types import Partial, Placement, Replicate


@register_single_dim_strategy(
    aten.linalg_cross.default,
    schema_info=RuntimeSchemaInfo(1),  # dim is static arg at index 1
)
def cross_single_dim_strategy(
    op: torch._ops.OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """
    Cross product single-dim sharding strategy.

    Discovered rules:
    - R, R -> R (added automatically, not listed)
    - S(d), S(d) -> S(d) when d != cross_dim (batch dims are independent)
    - P(sum), R -> P(sum) (bilinear in first arg)
    - R, P(sum) -> P(sum) (bilinear in second arg)
    """
    input_meta = args_schema[0]
    other_meta = args_schema[1]

    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")

    ndim = len(input_meta.shape)

    # Get the cross product dim (default -1)
    cross_dim = args_schema[2] if len(args_schema) > 2 else -1
    if cross_dim < 0:
        cross_dim = ndim + cross_dim

    single_dim_strategies: list[list[Placement | _ShardingPlaceholder]] = []

    # Strategy 1: S(d), S(d) -> S(d) for each d != cross_dim
    for d in range(ndim):
        if d == cross_dim:
            continue  # Cannot shard along cross product dim
        single_dim_strategies.append([
            _ShardingPlaceholder(d),  # output
            _ShardingPlaceholder(d),  # input
            _ShardingPlaceholder(d),  # other
        ])

    # Strategy 2: P(sum), R -> P(sum) (bilinear in first arg)
    single_dim_strategies.append([
        Partial(),    # output
        Partial(),    # input
        Replicate(),  # other
    ])

    # Strategy 3: R, P(sum) -> P(sum) (bilinear in second arg)
    single_dim_strategies.append([
        Partial(),    # output
        Replicate(),  # input
        Partial(),    # other
    ])

    return single_dim_strategies
```

**Key patterns for `register_single_dim_strategy`:**
1. Returns `list[list[Placement | _ShardingPlaceholder]]`
2. Each inner list: `[output, input1, input2, ...]` in order
3. Use `_ShardingPlaceholder(d)` for "shard on dim d" (gets expanded to Shard/StridedShard)
4. Use `Replicate()` and `Partial()` directly
5. Don't include `R, R -> R` - it's added automatically
6. Args are `TensorMeta`, not `OpStrategy` - access shape via `len(meta.shape)`

### Step 8: Write minimal tests

Add a test for the new sharding rule in `test/distributed/tensor/test_tensor_ops.py`:

```python
from torch.distributed.tensor.debug import CommDebugMode

@with_comms
def test_cross(self):
    """Test torch.cross with DTensor sharding."""
    mesh = self.build_device_mesh()

    # Test inputs - shape (batch, 3) with cross product on dim=1
    a = torch.randn(8, 3, device=self.device_type)
    b = torch.randn(8, 3, device=self.device_type)
    expected = torch.cross(a, b, dim=1)

    # Test S(0), S(0) -> S(0) strategy (batch dim sharding)
    # Batch dims are independent - each shard computes cross product of its vectors
    a_dt = distribute_tensor(a.clone(), mesh, [Shard(0)])
    b_dt = distribute_tensor(b.clone(), mesh, [Shard(0)])

    with CommDebugMode() as comm_mode:
        result_dt = torch.cross(a_dt, b_dt, dim=1)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(result_dt.placements, (Shard(0),))

    self.assertEqual(result_dt.full_tensor(), expected)

    # Test P(sum), R -> P(sum) strategy (bilinear in first arg)
    # Create partial input by dividing by world_size on each rank
    a_local = a.clone().to(self.device_type) / self.world_size
    a_partial = DTensor.from_local(a_local, mesh, [Partial()])
    b_replicate = distribute_tensor(b.clone(), mesh, [Replicate()])

    with CommDebugMode() as comm_mode:
        result_dt = torch.cross(a_partial, b_replicate, dim=1)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(result_dt.placements, (Partial(),))

    self.assertEqual(result_dt.full_tensor(), expected)

    # Test R, P(sum) -> P(sum) strategy (bilinear in second arg)
    a_replicate = distribute_tensor(a.clone(), mesh, [Replicate()])
    b_local = b.clone().to(self.device_type) / self.world_size
    b_partial = DTensor.from_local(b_local, mesh, [Partial()])

    with CommDebugMode() as comm_mode:
        result_dt = torch.cross(a_replicate, b_partial, dim=1)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(result_dt.placements, (Partial(),))

    self.assertEqual(result_dt.full_tensor(), expected)
```

**Key patterns for DTensor tests:**
1. Inherit from `DTensorTestBase` and use `@with_comms` decorator
2. Use `self.build_device_mesh()` to get the test mesh
3. Compute expected result with regular tensors first
4. Use `distribute_tensor(tensor, mesh, [placement])` to create DTensors
5. **Use `CommDebugMode` to verify no communication is required** - this is critical!
   - Wrap the op call in `with CommDebugMode() as comm_mode:`
   - Check `comm_mode.get_total_counts() == 0` to verify no comms
6. **Verify output placements match the expected rule**
   - Check `result_dt.placements == (expected_placement,)`
7. **Assert full tensor numerics are correct**
   - Use `.full_tensor()` to gather the result and compare with expected
   - This verifies the local computation produced the correct global result
8. **Test ALL non-full-replicate strategies** - every discovered strategy except R, R -> R must be tested
9. **Don't test the R, R -> R case** - it's obvious/automatic and doesn't need explicit testing

**The three assertions every strategy test needs:**
```python
# 1. No communication required (strategy is truly local)
self.assertEqual(comm_mode.get_total_counts(), 0)

# 2. Output placement matches expected rule
self.assertEqual(result_dt.placements, (expected_out_placement,))

# 3. Numerical correctness (full tensor matches expected)
self.assertEqual(result_dt.full_tensor(), expected)
```

**Why CommDebugMode matters:**
The whole point of discovering sharding strategies is to find (input, output) placement
combinations where the op can run **locally without communication**. If comms are required,
the strategy is not truly valid - DTensor would need to redistribute inputs first.
CommDebugMode catches cases where the strategy appears correct but actually triggers
implicit redistributions.

---

## Files Reference

| File | What Claude reads it for |
|------|-------------------------|
| `ai_generated_sharding_rules.md` | Overall process, terminology, validation algorithm |
| `_adversarial_inputs.md` | How to generate op-specific test inputs |
| `_conditional_rules.md` | How to analyze ops and propose conditional rules |
| `_strategy_validator.py` | Implementation of validation harness |
| `_adversarial_inputs.py` | Implementation of adversarial input registry |
| `_conditional_rules.py` | Implementation of conditional rule synthesis |
| `_ops/single_dim_strategy.py` | **Code format**: `register_single_dim_strategy`, `_ShardingPlaceholder` |
| `_ops/_tensor_ops.py` | **Code examples**: see `searchsorted_single_dim_strategy` |
| `_op_schema.py` | `RuntimeSchemaInfo`, `ArgsType`, `KwargsType` |
| `test/.../test_tensor_ops.py` | **Test examples**: DTensor op tests with `@with_comms` |
