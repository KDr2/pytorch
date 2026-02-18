"""
Benchmark: Conv1d & Conv2d – Triton template vs cuDNN.

8 variants:
  1. triton_conv1d_CF  – torch.compile inductor triton, conv1d NCL (channels-first)
  2. triton_conv1d_CL  – torch.compile inductor triton, conv1d NLC (channels-last)
  3. triton_conv2d_CF  – torch.compile inductor triton, conv2d NCHW (channels-first)
  4. triton_conv2d_CL  – torch.compile inductor triton, conv2d NHWC (channels-last)
  5. cudnn_conv1d_CF   – cuDNN conv1d NCL (eager)
  6. cudnn_conv1d_CL   – cuDNN conv1d NLC (eager)
  7. cudnn_conv2d_CF   – cuDNN conv2d NCHW (eager)
  8. cudnn_conv2d_CL   – cuDNN conv2d NHWC (eager)

Usage:
    buck2 run @fbcode//mode/dev-nosan fbcode//caffe2/benchmarks:conv_template_benchmark
"""

import gc

import torch
import torch._dynamo
import torch.nn.functional as F
from triton.testing import do_bench


def conv1d_fn(x, w, stride, padding):
    return F.conv1d(x, w, None, stride=stride, padding=padding)


def conv2d_fn(x, w, stride, padding):
    return F.conv2d(x, w, None, stride=stride, padding=padding)


def compile_with_triton(fn):
    torch._dynamo.reset()
    gc.collect()
    return torch.compile(
        fn,
        options={
            "max_autotune": True,
            "max_autotune_conv_backends": "TRITON",
        },
    )


def print_tensor_info(label, t):
    print(f"    {label:10s}: shape={list(t.shape)}  stride={list(t.stride())}")


VARIANT_ORDER = [
    "triton_conv1d_CF",
    "triton_conv1d_CL",
    "triton_conv2d_CF",
    "triton_conv2d_CL",
    "cudnn_conv1d_CF",
    "cudnn_conv1d_CL",
    "cudnn_conv2d_CF",
    "cudnn_conv2d_CL",
]


def make_channels_last_1d(src):
    """Create a channels-last (NLC) view of a 3D NCL tensor.

    Returns a tensor with shape (N, C, L) and stride (C*L, 1, C).
    """
    assert src.ndim == 3
    N, C, L = src.shape
    storage = torch.empty(N, L, C, device=src.device, dtype=src.dtype)
    cl = torch.as_strided(storage, size=(N, C, L), stride=(L * C, 1, C))
    cl.copy_(src)
    return cl


def verify_correctness(ref, out, name, atol=0.1):
    """Check output matches reference within tolerance. Returns max abs diff."""
    diff = (out.float() - ref.float()).abs().max().item()
    status = "PASS" if diff <= atol else "FAIL"
    print(f"    {status}  max_diff={diff:.6f}  (atol={atol})  [{name}]")
    return diff


def benchmark(warmup=25, rep=100):
    device = "cuda"
    dtype = torch.float16
    atol = 0.1  # tolerance for fp16 conv correctness

    N, Cin, L, Cout, kL = 3072, 128, 202, 384, 3
    stride_1d, padding_1d = 1, 0
    stride_2d, padding_2d = (1, 1), (0, 0)

    # --- 1D tensors (NCL) ---
    x1d_cf = torch.randn(N, Cin, L, device=device, dtype=dtype)
    w1d_cf = torch.randn(Cout, Cin, kL, device=device, dtype=dtype)

    x1d_cl = make_channels_last_1d(x1d_cf)
    w1d_cl = make_channels_last_1d(w1d_cf)

    # --- 2D tensors (H=1) ---
    x2d_cf = x1d_cf.unsqueeze(2)
    w2d_cf = w1d_cf.unsqueeze(2)

    x2d_cl = x2d_cf.to(memory_format=torch.channels_last)
    w2d_cl = w2d_cf.to(memory_format=torch.channels_last)

    print(f"  Input tensors:")
    print(f"    --- Conv1d ---")
    print(f"    x1d_CF : shape={list(x1d_cf.shape)}  stride={list(x1d_cf.stride())}")
    print(f"    x1d_CL : shape={list(x1d_cl.shape)}  stride={list(x1d_cl.stride())}")
    print(f"    w1d_CF : shape={list(w1d_cf.shape)}  stride={list(w1d_cf.stride())}")
    print(f"    w1d_CL : shape={list(w1d_cl.shape)}  stride={list(w1d_cl.stride())}")
    print(f"    --- Conv2d ---")
    print(f"    x2d_CF : shape={list(x2d_cf.shape)}  stride={list(x2d_cf.stride())}")
    print(f"    x2d_CL : shape={list(x2d_cl.shape)}  stride={list(x2d_cl.stride())}")
    print(f"    w2d_CF : shape={list(w2d_cf.shape)}  stride={list(w2d_cf.stride())}")
    print(f"    w2d_CL : shape={list(w2d_cl.shape)}  stride={list(w2d_cl.stride())}")
    print()

    # ---- Correctness verification of inputs ----
    print(f"  Input verification:")
    # CL tensors must contain the same data as CF tensors
    x1d_cl_diff = (x1d_cl.float() - x1d_cf.float()).abs().max().item()
    w1d_cl_diff = (w1d_cl.float() - w1d_cf.float()).abs().max().item()
    x2d_cl_diff = (x2d_cl.float() - x2d_cf.float()).abs().max().item()
    w2d_cl_diff = (w2d_cl.float() - w2d_cf.float()).abs().max().item()
    print(f"    x1d CF vs CL data match: max_diff={x1d_cl_diff:.6f}  {'PASS' if x1d_cl_diff == 0 else 'FAIL'}")
    print(f"    w1d CF vs CL data match: max_diff={w1d_cl_diff:.6f}  {'PASS' if w1d_cl_diff == 0 else 'FAIL'}")
    print(f"    x2d CF vs CL data match: max_diff={x2d_cl_diff:.6f}  {'PASS' if x2d_cl_diff == 0 else 'FAIL'}")
    print(f"    w2d CF vs CL data match: max_diff={w2d_cl_diff:.6f}  {'PASS' if w2d_cl_diff == 0 else 'FAIL'}")
    # Verify 2D tensors are just unsqueezed 1D tensors
    x_1d_2d_diff = (x2d_cf.squeeze(2).float() - x1d_cf.float()).abs().max().item()
    print(f"    x2d_cf.squeeze(2) == x1d_cf: max_diff={x_1d_2d_diff:.6f}  {'PASS' if x_1d_2d_diff == 0 else 'FAIL'}")
    print()

    # ---- Compute references (eager, CF) ----
    ref_1d = F.conv1d(x1d_cf, w1d_cf, None, stride=stride_1d, padding=padding_1d)
    ref_2d = F.conv2d(x2d_cf, w2d_cf, None, stride=stride_2d, padding=padding_2d)

    # Cross-check: conv1d ref == conv2d ref squeezed
    cross_diff = (ref_1d.float() - ref_2d.squeeze(2).float()).abs().max().item()
    print(f"  Reference cross-check:")
    print(f"    conv1d_ref vs conv2d_ref.squeeze(2): max_diff={cross_diff:.6f}  {'PASS' if cross_diff < 1e-6 else 'FAIL'}")
    print()

    # ---- Correctness verification of all variants ----
    print(f"  Output correctness (vs eager CF reference, atol={atol}):")
    all_pass = True
    checks = {}

    # cuDNN variants
    out = F.conv1d(x1d_cf, w1d_cf, None, stride=stride_1d, padding=padding_1d)
    checks["cudnn_conv1d_CF"] = verify_correctness(ref_1d, out, "cudnn_conv1d_CF", atol)

    out = F.conv1d(x1d_cl, w1d_cl, None, stride=stride_1d, padding=padding_1d)
    checks["cudnn_conv1d_CL"] = verify_correctness(ref_1d, out, "cudnn_conv1d_CL", atol)

    out = F.conv2d(x2d_cf, w2d_cf, None, stride=stride_2d, padding=padding_2d)
    checks["cudnn_conv2d_CF"] = verify_correctness(ref_2d, out, "cudnn_conv2d_CF", atol)

    out = F.conv2d(x2d_cl, w2d_cl, None, stride=stride_2d, padding=padding_2d)
    checks["cudnn_conv2d_CL"] = verify_correctness(ref_2d, out, "cudnn_conv2d_CL", atol)

    # Triton variants
    compiled = compile_with_triton(conv1d_fn)
    compiled(x1d_cf, w1d_cf, stride_1d, padding_1d)
    out = compiled(x1d_cf, w1d_cf, stride_1d, padding_1d)
    checks["triton_conv1d_CF"] = verify_correctness(ref_1d, out, "triton_conv1d_CF", atol)

    compiled = compile_with_triton(conv1d_fn)
    compiled(x1d_cl, w1d_cl, stride_1d, padding_1d)
    out = compiled(x1d_cl, w1d_cl, stride_1d, padding_1d)
    checks["triton_conv1d_CL"] = verify_correctness(ref_1d, out, "triton_conv1d_CL", atol)

    compiled = compile_with_triton(conv2d_fn)
    compiled(x2d_cf, w2d_cf, stride_2d, padding_2d)
    out = compiled(x2d_cf, w2d_cf, stride_2d, padding_2d)
    checks["triton_conv2d_CF"] = verify_correctness(ref_2d, out, "triton_conv2d_CF", atol)

    compiled = compile_with_triton(conv2d_fn)
    compiled(x2d_cl, w2d_cl, stride_2d, padding_2d)
    out = compiled(x2d_cl, w2d_cl, stride_2d, padding_2d)
    checks["triton_conv2d_CL"] = verify_correctness(ref_2d, out, "triton_conv2d_CL", atol)

    for name, diff in checks.items():
        if diff > atol:
            all_pass = False

    print()
    print(f"  Correctness: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    # ---- Performance benchmarks ----
    print(f"  Performance benchmarks:")
    results = {}

    # --- 1) Triton conv1d CF ---
    print(f"  [triton_conv1d_CF]")
    compiled = compile_with_triton(conv1d_fn)
    compiled(x1d_cf, w1d_cf, stride_1d, padding_1d)
    out = compiled(x1d_cf, w1d_cf, stride_1d, padding_1d)
    print_tensor_info("output", out)
    diff = checks["triton_conv1d_CF"]
    t = do_bench(lambda: compiled(x1d_cf, w1d_cf, stride_1d, padding_1d),
                 warmup=warmup, rep=rep)
    results["triton_conv1d_CF"] = (t, diff)

    # --- 2) Triton conv1d CL (NLC) ---
    print(f"  [triton_conv1d_CL]")
    compiled = compile_with_triton(conv1d_fn)
    compiled(x1d_cl, w1d_cl, stride_1d, padding_1d)
    out = compiled(x1d_cl, w1d_cl, stride_1d, padding_1d)
    print_tensor_info("output", out)
    diff = checks["triton_conv1d_CL"]
    t = do_bench(lambda: compiled(x1d_cl, w1d_cl, stride_1d, padding_1d),
                 warmup=warmup, rep=rep)
    results["triton_conv1d_CL"] = (t, diff)

    # --- 3) Triton conv2d CF ---
    print(f"  [triton_conv2d_CF]")
    compiled = compile_with_triton(conv2d_fn)
    compiled(x2d_cf, w2d_cf, stride_2d, padding_2d)
    out = compiled(x2d_cf, w2d_cf, stride_2d, padding_2d)
    print_tensor_info("output", out)
    diff = checks["triton_conv2d_CF"]
    t = do_bench(lambda: compiled(x2d_cf, w2d_cf, stride_2d, padding_2d),
                 warmup=warmup, rep=rep)
    results["triton_conv2d_CF"] = (t, diff)

    # --- 4) Triton conv2d CL ---
    print(f"  [triton_conv2d_CL]")
    compiled = compile_with_triton(conv2d_fn)
    compiled(x2d_cl, w2d_cl, stride_2d, padding_2d)
    out = compiled(x2d_cl, w2d_cl, stride_2d, padding_2d)
    print_tensor_info("output", out)
    diff = checks["triton_conv2d_CL"]
    t = do_bench(lambda: compiled(x2d_cl, w2d_cl, stride_2d, padding_2d),
                 warmup=warmup, rep=rep)
    results["triton_conv2d_CL"] = (t, diff)

    # --- 5) cuDNN conv1d CF ---
    print(f"  [cudnn_conv1d_CF]")
    for _ in range(3):
        F.conv1d(x1d_cf, w1d_cf, None, stride=stride_1d, padding=padding_1d)
    out = F.conv1d(x1d_cf, w1d_cf, None, stride=stride_1d, padding=padding_1d)
    print_tensor_info("output", out)
    diff = checks["cudnn_conv1d_CF"]
    t = do_bench(lambda: F.conv1d(x1d_cf, w1d_cf, None, stride=stride_1d, padding=padding_1d),
                 warmup=warmup, rep=rep)
    results["cudnn_conv1d_CF"] = (t, diff)

    # --- 6) cuDNN conv1d CL (NLC) ---
    print(f"  [cudnn_conv1d_CL]")
    for _ in range(3):
        F.conv1d(x1d_cl, w1d_cl, None, stride=stride_1d, padding=padding_1d)
    out = F.conv1d(x1d_cl, w1d_cl, None, stride=stride_1d, padding=padding_1d)
    print_tensor_info("output", out)
    diff = checks["cudnn_conv1d_CL"]
    t = do_bench(lambda: F.conv1d(x1d_cl, w1d_cl, None, stride=stride_1d, padding=padding_1d),
                 warmup=warmup, rep=rep)
    results["cudnn_conv1d_CL"] = (t, diff)

    # --- 7) cuDNN conv2d CF ---
    print(f"  [cudnn_conv2d_CF]")
    for _ in range(3):
        F.conv2d(x2d_cf, w2d_cf, None, stride=stride_2d, padding=padding_2d)
    out = F.conv2d(x2d_cf, w2d_cf, None, stride=stride_2d, padding=padding_2d)
    print_tensor_info("output", out)
    diff = checks["cudnn_conv2d_CF"]
    t = do_bench(lambda: F.conv2d(x2d_cf, w2d_cf, None, stride=stride_2d, padding=padding_2d),
                 warmup=warmup, rep=rep)
    results["cudnn_conv2d_CF"] = (t, diff)

    # --- 8) cuDNN conv2d CL ---
    print(f"  [cudnn_conv2d_CL]")
    for _ in range(3):
        F.conv2d(x2d_cl, w2d_cl, None, stride=stride_2d, padding=padding_2d)
    out = F.conv2d(x2d_cl, w2d_cl, None, stride=stride_2d, padding=padding_2d)
    print_tensor_info("output", out)
    diff = checks["cudnn_conv2d_CL"]
    t = do_bench(lambda: F.conv2d(x2d_cl, w2d_cl, None, stride=stride_2d, padding=padding_2d),
                 warmup=warmup, rep=rep)
    results["cudnn_conv2d_CL"] = (t, diff)

    return results


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print(f"{'='*80}")
    print(f"  Conv1d(3072, 128, 202) -> Conv(384, 128, k=3, s=1, p=0)")
    print(f"  Conv2d(3072, 128, 1, 202) -> Conv(384, 128, k=(1,3), s=(1,1), p=(0,0))")
    print(f"{'='*80}")

    results = benchmark()

    best_time = min(t for t, _ in results.values())

    print(f"  {'Variant':25s} {'Time (ms)':>10s} {'Relative':>10s} {'Max Diff':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for name in VARIANT_ORDER:
        t, diff = results[name]
        rel = best_time / t * 100
        print(f"  {name:25s} {t:10.4f} {rel:9.1f}% {diff:10.6f}")
    print()


if __name__ == "__main__":
    main()
