import torch


def fuzzed_program(arg_0):
    _x_nz = torch.zeros((), dtype=torch.bool).reshape(-1)
    _x_nz[:20] = True
    return torch.add(arg_0, torch.nonzero(_x_nz))


arg_0 = torch.as_strided(torch.randint(5, 30, (20,)).to(torch.int64), (20, 0), (1, 20))
result_original = fuzzed_program(arg_0)
print("✅ eager success")
print(result_original)
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(arg_0)
print("✅ compile success")
print(result_compiled)
assert torch.equal(result_original, result_compiled)
print("✅ results match!")


def fuzzed_program_2():
    _uniq_wide = torch.unique(torch.arange(1)).float()
    return torch.matmul(_uniq_wide, torch.full((1, 18), 0.5))


result_original = fuzzed_program_2()
print("✅ eager success 2")
print(result_original)
compiled_program_2 = torch.compile(fuzzed_program_2, fullgraph=True, dynamic=True)
result_compiled = compiled_program_2()
print("✅ compile success 2")
print(result_compiled)
assert torch.equal(result_original, result_compiled)
print("✅ results match!")
