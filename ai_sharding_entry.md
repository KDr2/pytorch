# AI Sharding Rules Discovery - Entry Point

**The canonical instructions are in:** `torch/distributed/tensor/DISCOVERY_PROMPT.md`

That file contains the complete 8-step discovery process:
1. Analyze the operator semantics
2. Propose kwargs to sweep (e.g., dim values)
3. Generate adversarial input generators specific to this op
4. Run the harness with discover_strategies() or analyze_and_synthesize()
5. Synthesize conditional rules from results
6. Report findings in structured format
7. Write the sharding rule code (using @register_single_dim_strategy)
8. Write minimal tests for the new sharding rule

See also:
- `ai_generated_sharding_rules.md` - Main design doc with terminology and validation algorithm
- `torch/distributed/tensor/_adversarial_inputs.md` - How to generate adversarial test inputs
- `torch/distributed/tensor/_conditional_rules.md` - How to propose dim-dependent rules
