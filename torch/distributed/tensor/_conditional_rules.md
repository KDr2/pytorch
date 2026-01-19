# AI-Proposed Conditional Sharding Rules

This document provides instructions for an AI to analyze operator semantics and propose **conditional sharding rules** - rules that are valid only under certain input conditions or kwargs values.

## Task Overview

Given a PyTorch operator, analyze its semantics and propose:
1. Which kwargs should be swept (e.g., `dim` values)
2. Conditional rules in the form: `"placement -> placement WHEN condition"`
3. The mathematical justification for each rule

## Input Format

You will be given:
1. **Operator name**: e.g., `torch.sum`, `torch.gather`, `torch.cross`
2. **Operator signature**: parameter names and types
3. **Operator semantics**: description of what the op computes
4. **Input tensor ndims**: number of dimensions for each input
5. **Discovered unconditional strategies**: strategies found to work for all tested kwargs

## Output Format

```yaml
operator: {op_name}
analysis:
  description: |
    {Brief description of the op's mathematical behavior}
  dim_parameter: {name of dim-like parameter, or null}
  reduction_dims: {list of dims that are reduced/indexed, or null}
  preserves_shape: {true/false}

kwargs_to_sweep:
  - name: dim
    values: [0, 1, ...]  # all valid values
    reason: "Sharding validity depends on which dim the op operates on"

conditional_rules:
  - placements:
      inputs: ["S(d)", "S(d)"]
      output: "S(d)"
    condition: "d != dim"
    justification: |
      The op operates along `dim`, so other dimensions are independent.
      Each shard can compute its portion without cross-shard communication.

  - placements:
      inputs: ["S(dim)"]
      output: "P(sum)"
    condition: null  # always true when sharding along dim
    justification: |
      Sharding along the reduction dim means each shard has partial results
      that must be summed to get the full result.

synthesis:
  # After running kwargs sweep, synthesize rules like:
  - observed: "S(0) -> S(0) valid when dim=1; S(1) -> S(1) valid when dim=0"
    synthesized: "S(d) -> S(d) when d != dim"
```

## Analysis Framework

### Step 1: Identify the Op Category

| Category | Examples | Key Insight |
|----------|----------|-------------|
| Reduction | sum, mean, max, prod | Reduces along dim → S(dim) produces Partial |
| Indexing | gather, scatter, index_select | Operates along dim → indices must be valid |
| Elementwise | add, mul, relu | No dim dependency → all shardings work equally |
| Cross-dim | matmul, conv | Contracts specific dims → those dims need care |
| Normalization | batch_norm, layer_norm | Computes stats over dims → those dims need allreduce |

### Step 2: Identify Dim-Dependent Parameters

Common patterns:
- `dim`: The dimension the op operates on
- `dims`: Multiple dimensions (e.g., `torch.norm`)
- `keepdim`: Whether output retains the reduced dim
- Implicit dims: Some ops have fixed dim semantics (e.g., batch_norm always normalizes over batch)

### Step 3: Derive Sharding Rules

For each dimension `d` in the input tensor:

**If `d == dim` (the operation dimension):**
- Sharding here typically produces `Partial` output for reductions
- Sharding here may be invalid for indexing ops
- Sharding here requires special handling

**If `d != dim` (independent dimensions):**
- Sharding here is usually safe
- Each shard computes its portion independently
- Output is sharded the same way

### Step 4: Express as Conditional Rules

Use this grammar:
```
CONDITION :=
  | "d != dim"           # d is not the operation dimension
  | "d == dim"           # d is the operation dimension
  | "d < input.ndim"     # dimension exists
  | "input.shape[d] > 1" # dimension is not trivial
  | CONDITION "AND" CONDITION
  | CONDITION "OR" CONDITION
```

## Examples

### Example 1: torch.sum(input, dim, keepdim=False)

```yaml
operator: torch.sum
analysis:
  description: |
    Reduces tensor along dim by summing elements.
    Output has one fewer dimension (unless keepdim=True).
  dim_parameter: dim
  reduction_dims: [dim]
  preserves_shape: false (true if keepdim)

kwargs_to_sweep:
  - name: dim
    values: [0, 1]  # for 2D input
    reason: "Which dimension is reduced determines valid shardings"
  - name: keepdim
    values: [true, false]
    reason: "Affects output dimensionality and thus output placement mapping"

conditional_rules:
  - placements:
      inputs: ["S(d)"]
      output: "S(d')"  # d' may differ if keepdim=False
    condition: "d != dim"
    justification: |
      Sharding along non-reduced dim: each shard sums its elements independently.
      Output is sharded on the corresponding dim (adjusted for dim removal if !keepdim).

  - placements:
      inputs: ["S(dim)"]
      output: "P(sum)"
    condition: null
    justification: |
      Sharding along reduced dim: each shard has partial sums.
      Full result requires summing across shards.

  - placements:
      inputs: ["P(sum)"]
      output: "P(sum)"
    condition: null
    justification: |
      Partial input summed along any dim is still partial.
```

### Example 2: torch.cross(input, other, dim)

```yaml
operator: torch.cross
analysis:
  description: |
    Computes cross product of 3D vectors along dim.
    input[..., dim] must have size 3.
    Cross product is bilinear: cross(a+b, c) = cross(a,c) + cross(b,c)
  dim_parameter: dim
  reduction_dims: null  # cross product doesn't reduce
  preserves_shape: true

kwargs_to_sweep:
  - name: dim
    values: [0, 1]  # depends on which dim has size 3
    reason: "Cross product operates along dim, other dims are batch dims"

conditional_rules:
  - placements:
      inputs: ["S(d)", "S(d)"]
      output: "S(d)"
    condition: "d != dim"
    justification: |
      Sharding along batch dim: each shard computes cross product
      of its vectors independently. No cross-shard communication needed.

  - placements:
      inputs: ["R", "P(sum)"]
      output: "P(sum)"
    condition: null
    justification: |
      Cross product is linear in second argument:
      cross(a, b/n) = cross(a, b)/n on each shard
      Sum across shards: Σ cross(a, b/n) = cross(a, b)

  - placements:
      inputs: ["P(sum)", "R"]
      output: "P(sum)"
    condition: null
    justification: |
      Cross product is linear in first argument:
      cross(a/n, b) = cross(a, b)/n on each shard

  - placements:
      inputs: ["S(dim)", "S(dim)"]
      output: "INVALID"
    condition: null
    justification: |
      Cannot shard along the cross product dimension.
      Cross product needs all 3 components to compute.
```

### Example 3: torch.gather(input, dim, index)

```yaml
operator: torch.gather
analysis:
  description: |
    Gathers values from input along dim using index tensor.
    out[i][j][k] = input[index[i][j][k]][j][k] for dim=0
    Index values must be valid for the input size along dim.
  dim_parameter: dim
  reduction_dims: null
  preserves_shape: false  # output shape matches index shape

kwargs_to_sweep:
  - name: dim
    values: [0, 1]
    reason: "Determines which input dim indices refer to"

conditional_rules:
  - placements:
      inputs: ["R", "S(d)"]
      output: "S(d)"
    condition: "d != dim"
    justification: |
      Index sharded on non-gather dim: each shard gathers from full input
      using its portion of indices. Valid because indices don't cross shards.

  - placements:
      inputs: ["S(d)", "S(d)"]
      output: "S(d)"
    condition: "d != dim AND input.ndim == index.ndim"
    justification: |
      Both input and index sharded on same non-gather dim.
      Each shard gathers from its input portion using its index portion.

  - placements:
      inputs: ["S(dim)", "R"]
      output: "INVALID without communication"
    condition: null
    justification: |
      Input sharded along gather dim: index values may refer to elements
      on other shards. Would need all-to-all communication.

  - placements:
      inputs: ["R", "S(dim)"]
      output: "S(dim)"
    condition: null
    justification: |
      Index sharded along gather dim: each shard's indices gather from
      full input. Output is sharded same as index.
```

## Synthesis Process

After running kwargs sweep with the harness, synthesize conditional rules:

**Input:** Results from `discover_strategies_with_kwargs_sweep`
```
dim=0: S(1),S(1) -> S(1) valid
dim=1: S(0),S(0) -> S(0) valid
```

**Synthesis:**
1. Identify pattern: S(d) valid when d ≠ dim
2. Express as: `"S(d), S(d) -> S(d) WHEN d != dim"`
3. Verify: For 2D tensor, d ∈ {0,1}, dim ∈ {0,1}, rule predicts correctly

**Output format:**
```python
{
    "operator": "torch.cross",
    "conditional_rules": [
        {
            "inputs": ["S(d)", "S(d)"],
            "output": "S(d)",
            "condition": "d != dim",
            "observed_from": {"dim=0": "S(1)", "dim=1": "S(0)"}
        }
    ]
}
```

## Prompt Template

When asked to analyze an operator for conditional rules:

---

**TASK**: Analyze operator semantics and propose conditional sharding rules.

**OPERATOR**: {op_name}
**SIGNATURE**: {signature}
**SEMANTICS**: {description}
**INPUT NDIMS**: {ndims}

**DISCOVERED UNCONDITIONAL STRATEGIES** (valid for all tested kwargs):
{list of strategies}

**ANALYSIS**:
1. Op category: {category}
2. Dim-dependent parameters: {params}
3. Mathematical properties: {properties}

**KWARGS TO SWEEP**:
{list with reasons}

**CONDITIONAL RULES**:
{rules in yaml format}

**SYNTHESIS** (after sweep results are available):
{synthesized rules}

---

## Integration with Harness

1. AI analyzes op and proposes kwargs to sweep
2. Harness runs `discover_strategies_with_kwargs_sweep()` for each proposed kwarg
3. AI receives sweep results
4. AI synthesizes conditional rules from observed patterns
5. Rules are validated against additional edge cases
