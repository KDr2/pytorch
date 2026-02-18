"""
Benchmark: Depthwise Conv1d forward – hand-written Triton, Inductor, cuDNN.

Triton kernels support variable kernel_size, padding, and stride via tl.constexpr.
All variants operate on native 3D conv1d tensors (no 4D unsqueeze).

Shape: x=[3072, 128, 202], w=[128, 1, K].

6 variants (default k=3, p=1, s=1):
  1. triton_ncl_CF    – Hand-written Triton NCL (channels-first)
  2. triton_nlc_CL    – Hand-written Triton NLC (channels-last)
  3. inductor_dw1d_CF – torch.compile triton template, conv1d CF
  4. inductor_dw1d_CL – torch.compile triton template, conv1d CL
  5. cudnn_dw1d_CF    – cuDNN conv1d eager, CF
  6. cudnn_dw1d_CL    – cuDNN conv1d eager, CL

Variable kernel size sweep: k=3, k=5, k=7 (Triton + cuDNN only).

Usage:
    buck2 run @fbcode//mode/dev-nosan fbcode//caffe2/benchmarks:depthwise_conv1d_benchmark
"""

import gc

import torch
import torch._dynamo
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench


# =============================================================================
# NCL (channels-first) depthwise conv1d kernel
# =============================================================================

_ncl_autotune_configs = [
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128}, num_warps=2, num_stages=5),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 64}, num_warps=4, num_stages=5),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32}, num_warps=4, num_stages=5),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32}, num_warps=2, num_stages=4),
]


@triton.autotune(configs=_ncl_autotune_configs, key=["batch_size", "channels", "out_length", "kernel_size"])
@triton.jit
def depthwise_conv1d_ncl_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size,
    channels: tl.constexpr, in_length: tl.constexpr, out_length: tl.constexpr,
    kernel_size: tl.constexpr, padding: tl.constexpr, conv_stride: tl.constexpr,
    stride_in_n: tl.constexpr, stride_in_c: tl.constexpr, stride_in_l: tl.constexpr,
    stride_out_n: tl.constexpr, stride_out_c: tl.constexpr, stride_out_l: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    l_start = pid_l * BLOCK_L

    n_offs = n_start + tl.arange(0, BLOCK_N)
    l_offs = l_start + tl.arange(0, BLOCK_L)
    n_mask = n_offs < batch_size
    l_mask = l_offs < out_length

    n2 = n_offs[:, None]
    nm = n_mask[:, None]
    in_base = n2 * stride_in_n + pid_c * stride_in_c

    acc = tl.zeros((BLOCK_N, BLOCK_L), dtype=tl.float32)
    w_base = weight_ptr + pid_c * kernel_size

    for k in range(kernel_size):
        wk = tl.load(w_base + k).to(tl.float32)
        l_in = l_offs * conv_stride - padding + k
        mask_in = (l_in >= 0) & (l_in < in_length)
        acc += tl.load(input_ptr + in_base + l_in[None, :] * stride_in_l,
                       mask=nm & mask_in[None, :], other=0.0).to(tl.float32) * wk

    out_base = n2 * stride_out_n + pid_c * stride_out_c
    tl.store(output_ptr + out_base + l_offs[None, :] * stride_out_l,
             acc.to(tl.float16), mask=nm & l_mask[None, :])


def triton_ncl_forward(x, weight, kernel_size, padding, stride):
    N, C, L_in = x.shape
    L_out = (L_in + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N, C, L_out, device=x.device, dtype=x.dtype)
    wf = weight.view(C, kernel_size).contiguous()

    def grid(META):
        return (triton.cdiv(L_out, META["BLOCK_L"]), C, triton.cdiv(N, META["BLOCK_N"]))

    depthwise_conv1d_ncl_kernel[grid](
        x, wf, output, N, C, L_in, L_out,
        kernel_size, padding, stride,
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
    )
    return output


# =============================================================================
# NLC (channels-last) depthwise conv1d kernel
# =============================================================================

_nlc_autotune_configs = [
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256, "BLOCK_C": 32}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256, "BLOCK_C": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256, "BLOCK_C": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256, "BLOCK_C": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 256, "BLOCK_C": 32}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 256, "BLOCK_C": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 32, "BLOCK_C": 32}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 32, "BLOCK_C": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 32, "BLOCK_C": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 32, "BLOCK_C": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 32, "BLOCK_C": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32, "BLOCK_C": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32, "BLOCK_C": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32, "BLOCK_C": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 64, "BLOCK_C": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 64, "BLOCK_C": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 64, "BLOCK_C": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 64, "BLOCK_C": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 64, "BLOCK_C": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128, "BLOCK_C": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 128, "BLOCK_C": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 128, "BLOCK_C": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32, "BLOCK_C": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32, "BLOCK_C": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 32, "BLOCK_C": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 64, "BLOCK_C": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32, "BLOCK_C": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 32, "BLOCK_C": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_L": 64, "BLOCK_C": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 32, "BLOCK_L": 64, "BLOCK_C": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_nlc_autotune_configs, key=["batch_size", "out_length", "channels", "kernel_size"])
@triton.jit
def depthwise_conv1d_nlc_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size: tl.constexpr, in_length: tl.constexpr, out_length: tl.constexpr,
    channels: tl.constexpr,
    kernel_size: tl.constexpr, padding: tl.constexpr, conv_stride: tl.constexpr,
    stride_in_n, stride_in_l, stride_in_c,
    stride_out_n, stride_out_l, stride_out_c,
    BLOCK_N: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_c = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    l_start = pid_l * BLOCK_L
    c_start = pid_c * BLOCK_C

    n_offs = n_start + tl.arange(0, BLOCK_N)
    l_offs = l_start + tl.arange(0, BLOCK_L)
    c_offs = c_start + tl.arange(0, BLOCK_C)
    n_mask = n_offs < batch_size
    l_mask = l_offs < out_length
    c_mask = c_offs < channels

    n3 = n_offs[:, None, None]
    nm3 = n_mask[:, None, None]
    c3 = c_offs[None, None, :]
    cm3 = c_mask[None, None, :]

    in_base = n3 * stride_in_n + c3 * stride_in_c

    acc = tl.zeros((BLOCK_N, BLOCK_L, BLOCK_C), dtype=tl.float32)

    for k in range(kernel_size):
        wk = tl.load(weight_ptr + c_offs * kernel_size + k,
                      mask=c_mask, other=0.0).to(tl.float32)
        l_in = l_offs * conv_stride - padding + k
        mask_in = (l_in >= 0) & (l_in < in_length)
        acc += tl.load(input_ptr + in_base + l_in[None, :, None] * stride_in_l,
                       mask=nm3 & mask_in[None, :, None] & cm3, other=0.0).to(tl.float32) * wk[None, None, :]

    out_idx = n3 * stride_out_n + l_offs[None, :, None] * stride_out_l + c3 * stride_out_c
    tl.store(output_ptr + out_idx, acc.to(tl.float16), mask=nm3 & l_mask[None, :, None] & cm3)


def triton_nlc_forward(x_nlc, weight, kernel_size, padding, stride):
    """x_nlc: [N, L, C] contiguous NLC tensor."""
    N, L_in, C = x_nlc.shape
    L_out = (L_in + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N, L_out, C, device=x_nlc.device, dtype=x_nlc.dtype)
    wf = weight.view(C, kernel_size).contiguous()

    def grid(META):
        return (
            triton.cdiv(N, META["BLOCK_N"]),
            triton.cdiv(L_out, META["BLOCK_L"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )

    depthwise_conv1d_nlc_kernel[grid](
        x_nlc, wf, output, N, L_in, L_out, C,
        kernel_size, padding, stride,
        x_nlc.stride(0), x_nlc.stride(1), x_nlc.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
    )
    return output


# =============================================================================
# Inductor (torch.compile) helpers
# =============================================================================

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


# =============================================================================
# Helpers
# =============================================================================

def make_channels_last_1d(src):
    assert src.ndim == 3
    N, C, L = src.shape
    storage = torch.empty(N, L, C, device=src.device, dtype=src.dtype)
    cl = torch.as_strided(storage, size=(N, C, L), stride=(L * C, 1, C))
    cl.copy_(src)
    return cl


def print_tensor_info(label, t):
    print(f"    {label:15s}: shape={list(t.shape)}  stride={list(t.stride())}")


def verify_correctness(ref, out, name, atol=0.1):
    diff = (out.float() - ref.float()).abs().max().item()
    status = "PASS" if diff <= atol else "FAIL"
    print(f"    {status}  max_diff={diff:.6f}  (atol={atol})  [{name}]")
    return diff


VARIANT_ORDER = [
    "triton_ncl_CF",
    "triton_nlc_CL",
    "inductor_dw1d_CF",
    "inductor_dw1d_CL",
    "cudnn_dw1d_CF",
    "cudnn_dw1d_CL",
]


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(warmup=25, rep=100):
    device = "cuda"
    dtype = torch.float16
    atol = 0.1

    N, C, L = 3072, 128, 202
    K, PAD, STRIDE = 3, 1, 1

    w = torch.randn(C, 1, K, device=device, dtype=dtype)
    x_cf = torch.randn(N, C, L, device=device, dtype=dtype)
    x_cl = make_channels_last_1d(x_cf)
    w_cl = make_channels_last_1d(w)
    x_nlc = x_cf.permute(0, 2, 1).contiguous()

    print(f"  Input tensors:")
    print_tensor_info("x_CF (NCL)", x_cf)
    print_tensor_info("x_CL (NLC)", x_cl)
    print_tensor_info("x_NLC", x_nlc)
    print_tensor_info("w", w)
    print()

    ref = F.conv1d(x_cf, w, None, stride=STRIDE, padding=PAD, groups=C)

    print(f"  Output correctness (vs cuDNN conv1d CF eager, atol={atol}):")
    checks = {}

    out = F.conv1d(x_cf, w, None, stride=STRIDE, padding=PAD, groups=C)
    checks["cudnn_dw1d_CF"] = verify_correctness(ref, out, "cudnn_dw1d_CF", atol)
    out = F.conv1d(x_cl, w_cl, None, stride=STRIDE, padding=PAD, groups=C)
    checks["cudnn_dw1d_CL"] = verify_correctness(ref, out, "cudnn_dw1d_CL", atol)

    out = triton_ncl_forward(x_cf, w, K, PAD, STRIDE)
    checks["triton_ncl_CF"] = verify_correctness(ref, out, "triton_ncl_CF", atol)

    out_nlc = triton_nlc_forward(x_nlc, w, K, PAD, STRIDE)
    out_ncl = out_nlc.permute(0, 2, 1).contiguous()
    checks["triton_nlc_CL"] = verify_correctness(ref, out_ncl, "triton_nlc_CL", atol)

    def dw_conv1d_fn(x, w, groups):
        return F.conv1d(x, w, None, stride=1, padding=1, groups=groups)

    compiled = compile_with_triton(dw_conv1d_fn)
    compiled(x_cf, w, C)
    out = compiled(x_cf, w, C)
    checks["inductor_dw1d_CF"] = verify_correctness(ref, out, "inductor_dw1d_CF", atol)

    compiled = compile_with_triton(dw_conv1d_fn)
    compiled(x_cl, w_cl, C)
    out = compiled(x_cl, w_cl, C)
    checks["inductor_dw1d_CL"] = verify_correctness(ref, out, "inductor_dw1d_CL", atol)

    all_pass = all(d <= atol for d in checks.values())
    print(f"  Correctness: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    print(f"  Performance benchmarks:")
    results = {}

    print(f"  [triton_ncl_CF]")
    triton_ncl_forward(x_cf, w, K, PAD, STRIDE)
    t = do_bench(lambda: triton_ncl_forward(x_cf, w, K, PAD, STRIDE), warmup=warmup, rep=rep)
    results["triton_ncl_CF"] = (t, checks["triton_ncl_CF"])

    print(f"  [triton_nlc_CL]")
    triton_nlc_forward(x_nlc, w, K, PAD, STRIDE)
    t = do_bench(lambda: triton_nlc_forward(x_nlc, w, K, PAD, STRIDE), warmup=warmup, rep=rep)
    results["triton_nlc_CL"] = (t, checks["triton_nlc_CL"])

    print(f"  [inductor_dw1d_CF]")
    compiled = compile_with_triton(dw_conv1d_fn)
    compiled(x_cf, w, C)
    t = do_bench(lambda: compiled(x_cf, w, C), warmup=warmup, rep=rep)
    results["inductor_dw1d_CF"] = (t, checks["inductor_dw1d_CF"])

    print(f"  [inductor_dw1d_CL]")
    compiled = compile_with_triton(dw_conv1d_fn)
    compiled(x_cl, w_cl, C)
    t = do_bench(lambda: compiled(x_cl, w_cl, C), warmup=warmup, rep=rep)
    results["inductor_dw1d_CL"] = (t, checks["inductor_dw1d_CL"])

    print(f"  [cudnn_dw1d_CF]")
    for _ in range(3):
        F.conv1d(x_cf, w, None, stride=STRIDE, padding=PAD, groups=C)
    t = do_bench(lambda: F.conv1d(x_cf, w, None, stride=STRIDE, padding=PAD, groups=C),
                 warmup=warmup, rep=rep)
    results["cudnn_dw1d_CF"] = (t, checks["cudnn_dw1d_CF"])

    print(f"  [cudnn_dw1d_CL]")
    for _ in range(3):
        F.conv1d(x_cl, w_cl, None, stride=STRIDE, padding=PAD, groups=C)
    t = do_bench(lambda: F.conv1d(x_cl, w_cl, None, stride=STRIDE, padding=PAD, groups=C),
                 warmup=warmup, rep=rep)
    results["cudnn_dw1d_CL"] = (t, checks["cudnn_dw1d_CL"])

    return results


def benchmark_variable_kernel(kernel_sizes=(3, 5, 7), warmup=25, rep=100):
    """Benchmark Triton flex kernels across multiple kernel sizes."""
    device = "cuda"
    dtype = torch.float16

    N, C, L = 3072, 128, 202
    STRIDE = 1

    all_results = {}
    for K in kernel_sizes:
        PAD = K // 2
        L_out = (L + 2 * PAD - K) // STRIDE + 1

        print(f"\n  {'─'*70}")
        print(f"  kernel_size={K}  padding={PAD}  stride={STRIDE}  L_out={L_out}")
        print(f"  {'─'*70}")

        w = torch.randn(C, 1, K, device=device, dtype=dtype)
        x_cf = torch.randn(N, C, L, device=device, dtype=dtype)
        x_nlc = x_cf.permute(0, 2, 1).contiguous()

        ref = F.conv1d(x_cf, w, None, stride=STRIDE, padding=PAD, groups=C)

        out_ncl = triton_ncl_forward(x_cf, w, K, PAD, STRIDE)
        d_ncl = (out_ncl.float() - ref.float()).abs().max().item()
        print(f"    triton_NCL correctness: max_diff={d_ncl:.6f}  {'PASS' if d_ncl <= 0.1 else 'FAIL'}")

        out_nlc = triton_nlc_forward(x_nlc, w, K, PAD, STRIDE)
        out_nlc_ncl = out_nlc.permute(0, 2, 1).contiguous()
        d_nlc = (out_nlc_ncl.float() - ref.float()).abs().max().item()
        print(f"    triton_NLC correctness: max_diff={d_nlc:.6f}  {'PASS' if d_nlc <= 0.1 else 'FAIL'}")

        print(f"    Benchmarking ...")

        triton_ncl_forward(x_cf, w, K, PAD, STRIDE)
        t_ncl = do_bench(lambda: triton_ncl_forward(x_cf, w, K, PAD, STRIDE), warmup=warmup, rep=rep)

        triton_nlc_forward(x_nlc, w, K, PAD, STRIDE)
        t_nlc = do_bench(lambda: triton_nlc_forward(x_nlc, w, K, PAD, STRIDE), warmup=warmup, rep=rep)

        for _ in range(3):
            F.conv1d(x_cf, w, None, stride=STRIDE, padding=PAD, groups=C)
        t_cudnn = do_bench(lambda: F.conv1d(x_cf, w, None, stride=STRIDE, padding=PAD, groups=C),
                           warmup=warmup, rep=rep)

        all_results[K] = {
            "triton_NCL": (t_ncl, d_ncl),
            "triton_NLC": (t_nlc, d_nlc),
            "cuDNN_CF": (t_cudnn, 0.0),
        }

    return all_results


# =============================================================================
# Main
# =============================================================================

def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print(f"{'='*80}")
    print(f"  Depthwise conv1d  x=(3072,128,202)  w=(128,1,K)  groups=128")
    print(f"  default: kernel=3  stride=1  padding=1")
    print(f"{'='*80}")

    results = benchmark()

    best_time = min(t for t, _ in results.values())
    print()
    print(f"  {'Variant':25s} {'Time (ms)':>10s} {'Relative':>10s} {'Max Diff':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for name in VARIANT_ORDER:
        t, diff = results[name]
        rel = best_time / t * 100
        print(f"  {name:25s} {t:10.4f} {rel:9.1f}% {diff:10.6f}")

    print()
    print(f"  {'='*80}")
    print(f"  Variable kernel size benchmark: k=3, k=5, k=7")
    print(f"  {'='*80}")
    vk_results = benchmark_variable_kernel(kernel_sizes=(3, 5, 7))

    print()
    print(f"  {'='*80}")
    print(f"  Summary: Triton kernel performance across kernel sizes")
    print(f"  {'='*80}")
    print(f"  {'k':>4s} {'triton_NCL (ms)':>16s} {'triton_NLC (ms)':>16s} {'cuDNN_CF (ms)':>14s} {'NCL/cuDNN':>10s} {'NLC/cuDNN':>10s}")
    print(f"  {'-'*4} {'-'*16} {'-'*16} {'-'*14} {'-'*10} {'-'*10}")
    for K in (3, 5, 7):
        r = vk_results[K]
        t_ncl = r["triton_NCL"][0]
        t_nlc = r["triton_NLC"][0]
        t_cud = r["cuDNN_CF"][0]
        print(f"  {K:4d} {t_ncl:16.4f} {t_nlc:16.4f} {t_cud:14.4f} {t_ncl/t_cud:9.2f}x {t_nlc/t_cud:9.2f}x")
    print()


if __name__ == "__main__":
    main()
