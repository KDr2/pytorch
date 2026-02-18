"""
Benchmark: Depthwise Conv2d forward – hand-written Triton, Inductor, cuDNN.

Triton kernels support variable kernel_h, padding_h, and stride_h via tl.constexpr.
Width dimension folded into grid for W>1 support.

Shape: x=[3072, 128, 202, 1], w=[128, 1, K, 1].

6 variants (default k=3, p=1, s=1):
  1. triton_nchw_CF    – Hand-written Triton NCHW (channels-first)
  2. triton_nhwc_CL    – Hand-written Triton NHWC (channels-last)
  3. inductor_dw_CF    – torch.compile inductor triton template, CF
  4. inductor_dw_CL    – torch.compile inductor triton template, CL
  5. cudnn_dw_CF        – cuDNN depthwise conv2d eager, CF
  6. cudnn_dw_CL        – cuDNN depthwise conv2d eager, CL

Variable kernel size sweep: k=3, k=5, k=7 (Triton + cuDNN only).

Usage:
    buck2 run @fbcode//mode/dev-nosan fbcode//caffe2/benchmarks:depthwise_conv2d_benchmark
"""

import gc

import torch
import torch._dynamo
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench


# =============================================================================
# NCHW depthwise conv2d kernel – W>1 capable
# Grid dim 0 = cdiv(out_H, BLOCK_HEIGHT) * W
# =============================================================================

_nchw_autotune_configs = [
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128}, num_warps=2, num_stages=5),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 64}, num_warps=4, num_stages=5),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_nchw_autotune_configs, key=["batch_size", "channels", "out_height", "width", "kernel_h"])
@triton.jit
def depthwise_conv2d_nchw_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size,
    channels: tl.constexpr, in_height: tl.constexpr, out_height: tl.constexpr, width: tl.constexpr,
    kernel_h: tl.constexpr, padding_h: tl.constexpr, conv_stride_h: tl.constexpr,
    stride_in_n: tl.constexpr, stride_in_c: tl.constexpr,
    stride_in_h: tl.constexpr, stride_in_w: tl.constexpr,
    stride_out_n: tl.constexpr, stride_out_c: tl.constexpr,
    stride_out_h: tl.constexpr, stride_out_w: tl.constexpr,
    BLOCK_BATCH: tl.constexpr, BLOCK_HEIGHT: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    pid_h = pid_hw // width
    w = pid_hw % width

    batch_start = pid_b * BLOCK_BATCH
    height_start = pid_h * BLOCK_HEIGHT
    batch_offs = batch_start + tl.arange(0, BLOCK_BATCH)
    height_offs = height_start + tl.arange(0, BLOCK_HEIGHT)
    batch_mask = batch_offs < batch_size
    height_mask = height_offs < out_height

    b2 = batch_offs[:, None]
    bm2 = batch_mask[:, None]
    in_base = b2 * stride_in_n + pid_c * stride_in_c + w * stride_in_w

    acc = tl.zeros((BLOCK_BATCH, BLOCK_HEIGHT), dtype=tl.float32)
    w_base = weight_ptr + pid_c * kernel_h

    for k in range(kernel_h):
        wk = tl.load(w_base + k).to(tl.float32)
        h_in = height_offs * conv_stride_h - padding_h + k
        mask_in = (h_in >= 0) & (h_in < in_height)
        acc += tl.load(input_ptr + in_base + h_in[None, :] * stride_in_h,
                       mask=bm2 & mask_in[None, :], other=0.0).to(tl.float32) * wk

    out_idx = b2 * stride_out_n + pid_c * stride_out_c + height_offs[None, :] * stride_out_h + w * stride_out_w
    tl.store(output_ptr + out_idx, acc.to(tl.float16), mask=bm2 & height_mask[None, :])


def triton_nchw_forward(x, weight, kernel_h, padding_h, stride_h):
    N, C, H_in, W = x.shape
    H_out = (H_in + 2 * padding_h - kernel_h) // stride_h + 1
    output = torch.empty(N, C, H_out, W, device=x.device, dtype=x.dtype)
    wf = weight.view(C, kernel_h).contiguous()

    def grid(META):
        return (
            triton.cdiv(H_out, META["BLOCK_HEIGHT"]) * W,
            C,
            triton.cdiv(N, META["BLOCK_BATCH"]),
        )

    depthwise_conv2d_nchw_kernel[grid](
        x, wf, output, N, C, H_in, H_out, W,
        kernel_h, padding_h, stride_h,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    return output


# =============================================================================
# NHWC depthwise conv2d kernel – W>1 capable
# Grid dim 1 = cdiv(out_H, BLOCK_HEIGHT) * W
# =============================================================================

_nhwc_autotune_configs = [
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 256, "BLOCK_CHANNELS": 32}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 256, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 64, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 64, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 64, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 64, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 64, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 64, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 128, "BLOCK_CHANNELS": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 128, "BLOCK_CHANNELS": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_BATCH": 32, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 64, "BLOCK_CHANNELS": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BATCH": 16, "BLOCK_HEIGHT": 32, "BLOCK_CHANNELS": 128}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_nhwc_autotune_configs, key=["batch_size", "out_height", "width", "channels", "kernel_h"])
@triton.jit
def depthwise_conv2d_nhwc_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size: tl.constexpr, in_height: tl.constexpr, out_height: tl.constexpr,
    width: tl.constexpr, channels: tl.constexpr,
    kernel_h: tl.constexpr, padding_h: tl.constexpr, conv_stride_h: tl.constexpr,
    stride_in_n, stride_in_h, stride_in_w, stride_in_c,
    stride_out_n, stride_out_h, stride_out_w, stride_out_c,
    BLOCK_BATCH: tl.constexpr, BLOCK_HEIGHT: tl.constexpr, BLOCK_CHANNELS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)

    pid_h = pid_hw // width
    w = pid_hw % width

    batch_start = pid_b * BLOCK_BATCH
    height_start = pid_h * BLOCK_HEIGHT
    channel_start = pid_c * BLOCK_CHANNELS

    batch_offs = batch_start + tl.arange(0, BLOCK_BATCH)
    height_offs = height_start + tl.arange(0, BLOCK_HEIGHT)
    channel_offs = channel_start + tl.arange(0, BLOCK_CHANNELS)
    batch_mask = batch_offs < batch_size
    height_mask = height_offs < out_height
    channel_mask = channel_offs < channels

    b3 = batch_offs[:, None, None]
    bm3 = batch_mask[:, None, None]
    c3 = channel_offs[None, None, :]
    cm3 = channel_mask[None, None, :]

    in_base = b3 * stride_in_n + w * stride_in_w + c3 * stride_in_c

    acc = tl.zeros((BLOCK_BATCH, BLOCK_HEIGHT, BLOCK_CHANNELS), dtype=tl.float32)

    for k in range(kernel_h):
        wk = tl.load(weight_ptr + channel_offs * kernel_h + k,
                      mask=channel_mask, other=0.0).to(tl.float32)
        h_in = height_offs * conv_stride_h - padding_h + k
        mask_in = (h_in >= 0) & (h_in < in_height)
        acc += tl.load(input_ptr + in_base + h_in[None, :, None] * stride_in_h,
                       mask=bm3 & mask_in[None, :, None] & cm3, other=0.0).to(tl.float32) * wk[None, None, :]

    out_idx = b3 * stride_out_n + height_offs[None, :, None] * stride_out_h + w * stride_out_w + c3 * stride_out_c
    tl.store(output_ptr + out_idx, acc.to(tl.float16), mask=bm3 & height_mask[None, :, None] & cm3)


def triton_nhwc_forward(x_nhwc, weight, kernel_h, padding_h, stride_h):
    """x_nhwc: [N, H, W, C] contiguous NHWC tensor."""
    N, H_in, W, C = x_nhwc.shape
    H_out = (H_in + 2 * padding_h - kernel_h) // stride_h + 1
    output = torch.empty(N, H_out, W, C, device=x_nhwc.device, dtype=x_nhwc.dtype)
    wf = weight.view(C, kernel_h).contiguous()

    def grid(META):
        return (
            triton.cdiv(N, META["BLOCK_BATCH"]),
            triton.cdiv(H_out, META["BLOCK_HEIGHT"]) * W,
            triton.cdiv(C, META["BLOCK_CHANNELS"]),
        )

    depthwise_conv2d_nhwc_kernel[grid](
        x_nhwc, wf, output, N, H_in, H_out, W, C,
        kernel_h, padding_h, stride_h,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
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

def print_tensor_info(label, t):
    print(f"    {label:15s}: shape={list(t.shape)}  stride={list(t.stride())}")


def verify_correctness(ref, out, name, atol=0.1):
    diff = (out.float() - ref.float()).abs().max().item()
    status = "PASS" if diff <= atol else "FAIL"
    print(f"    {status}  max_diff={diff:.6f}  (atol={atol})  [{name}]")
    return diff


VARIANT_ORDER = [
    "triton_nchw_CF",
    "triton_nhwc_CL",
    "inductor_dw_CF",
    "inductor_dw_CL",
    "cudnn_dw_CF",
    "cudnn_dw_CL",
]


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(warmup=25, rep=100):
    device = "cuda"
    dtype = torch.float16
    atol = 0.1

    N, C, H, W = 3072, 128, 202, 1
    K_H, PAD_H, STRIDE_H = 3, 1, 1

    w_cf = torch.randn(C, 1, K_H, 1, device=device, dtype=dtype)
    x_cf = torch.randn(N, C, H, W, device=device, dtype=dtype)
    x_cl = x_cf.to(memory_format=torch.channels_last)
    w_cl = w_cf.to(memory_format=torch.channels_last)
    x_nhwc = x_cf.permute(0, 2, 3, 1).contiguous()

    print(f"  Input tensors:")
    print_tensor_info("x_CF (NCHW)", x_cf)
    print_tensor_info("x_CL (NHWC)", x_cl)
    print_tensor_info("x_NHWC", x_nhwc)
    print_tensor_info("w_CF", w_cf)
    print()

    ref = F.conv2d(x_cf, w_cf, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C)

    print(f"  Output correctness (vs cuDNN CF eager, atol={atol}):")
    checks = {}

    out = F.conv2d(x_cf, w_cf, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C)
    checks["cudnn_dw_CF"] = verify_correctness(ref, out, "cudnn_dw_CF", atol)

    out = F.conv2d(x_cl, w_cl, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C)
    checks["cudnn_dw_CL"] = verify_correctness(ref, out, "cudnn_dw_CL", atol)

    out = triton_nchw_forward(x_cf, w_cf, K_H, PAD_H, STRIDE_H)
    checks["triton_nchw_CF"] = verify_correctness(ref, out, "triton_nchw_CF", atol)

    out_nhwc = triton_nhwc_forward(x_nhwc, w_cf, K_H, PAD_H, STRIDE_H)
    out_nchw = out_nhwc.permute(0, 3, 1, 2).contiguous()
    checks["triton_nhwc_CL"] = verify_correctness(ref, out_nchw, "triton_nhwc_CL", atol)

    def dw_conv2d_fn(x, w, groups):
        return F.conv2d(x, w, None, stride=(1, 1), padding=(1, 0), groups=groups)

    compiled_cf = compile_with_triton(dw_conv2d_fn)
    compiled_cf(x_cf, w_cf, C)
    out = compiled_cf(x_cf, w_cf, C)
    checks["inductor_dw_CF"] = verify_correctness(ref, out, "inductor_dw_CF", atol)

    compiled_cl = compile_with_triton(dw_conv2d_fn)
    compiled_cl(x_cl, w_cl, C)
    out = compiled_cl(x_cl, w_cl, C)
    checks["inductor_dw_CL"] = verify_correctness(ref, out, "inductor_dw_CL", atol)

    all_pass = all(d <= atol for d in checks.values())
    print(f"  Correctness: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    print(f"  Performance benchmarks:")
    results = {}

    print(f"  [triton_nchw_CF]")
    triton_nchw_forward(x_cf, w_cf, K_H, PAD_H, STRIDE_H)
    t = do_bench(lambda: triton_nchw_forward(x_cf, w_cf, K_H, PAD_H, STRIDE_H), warmup=warmup, rep=rep)
    results["triton_nchw_CF"] = (t, checks["triton_nchw_CF"])

    print(f"  [triton_nhwc_CL]")
    triton_nhwc_forward(x_nhwc, w_cf, K_H, PAD_H, STRIDE_H)
    t = do_bench(lambda: triton_nhwc_forward(x_nhwc, w_cf, K_H, PAD_H, STRIDE_H), warmup=warmup, rep=rep)
    results["triton_nhwc_CL"] = (t, checks["triton_nhwc_CL"])

    print(f"  [inductor_dw_CF]")
    compiled_cf = compile_with_triton(dw_conv2d_fn)
    compiled_cf(x_cf, w_cf, C)
    t = do_bench(lambda: compiled_cf(x_cf, w_cf, C), warmup=warmup, rep=rep)
    results["inductor_dw_CF"] = (t, checks["inductor_dw_CF"])

    print(f"  [inductor_dw_CL]")
    compiled_cl = compile_with_triton(dw_conv2d_fn)
    compiled_cl(x_cl, w_cl, C)
    t = do_bench(lambda: compiled_cl(x_cl, w_cl, C), warmup=warmup, rep=rep)
    results["inductor_dw_CL"] = (t, checks["inductor_dw_CL"])

    print(f"  [cudnn_dw_CF]")
    for _ in range(3):
        F.conv2d(x_cf, w_cf, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C)
    t = do_bench(lambda: F.conv2d(x_cf, w_cf, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C),
                 warmup=warmup, rep=rep)
    results["cudnn_dw_CF"] = (t, checks["cudnn_dw_CF"])

    print(f"  [cudnn_dw_CL]")
    for _ in range(3):
        F.conv2d(x_cl, w_cl, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C)
    t = do_bench(lambda: F.conv2d(x_cl, w_cl, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C),
                 warmup=warmup, rep=rep)
    results["cudnn_dw_CL"] = (t, checks["cudnn_dw_CL"])

    return results


def benchmark_variable_kernel(kernel_sizes=(3, 5, 7), warmup=25, rep=100):
    """Benchmark Triton kernels across multiple kernel sizes."""
    device = "cuda"
    dtype = torch.float16

    N, C, H, W = 3072, 128, 202, 1
    STRIDE_H = 1

    all_results = {}
    for K_H in kernel_sizes:
        PAD_H = K_H // 2
        H_out = (H + 2 * PAD_H - K_H) // STRIDE_H + 1

        print(f"\n  {'─'*70}")
        print(f"  kernel_h={K_H}  padding_h={PAD_H}  stride_h={STRIDE_H}  H_out={H_out}")
        print(f"  {'─'*70}")

        w_cf = torch.randn(C, 1, K_H, 1, device=device, dtype=dtype)
        x_cf = torch.randn(N, C, H, W, device=device, dtype=dtype)
        x_nhwc = x_cf.permute(0, 2, 3, 1).contiguous()

        ref = F.conv2d(x_cf, w_cf, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C)

        out_nchw = triton_nchw_forward(x_cf, w_cf, K_H, PAD_H, STRIDE_H)
        d_nchw = (out_nchw.float() - ref.float()).abs().max().item()
        print(f"    triton_NCHW correctness: max_diff={d_nchw:.6f}  {'PASS' if d_nchw <= 0.1 else 'FAIL'}")

        out_nhwc = triton_nhwc_forward(x_nhwc, w_cf, K_H, PAD_H, STRIDE_H)
        out_nhwc_nchw = out_nhwc.permute(0, 3, 1, 2).contiguous()
        d_nhwc = (out_nhwc_nchw.float() - ref.float()).abs().max().item()
        print(f"    triton_NHWC correctness: max_diff={d_nhwc:.6f}  {'PASS' if d_nhwc <= 0.1 else 'FAIL'}")

        print(f"    Benchmarking ...")

        triton_nchw_forward(x_cf, w_cf, K_H, PAD_H, STRIDE_H)
        t_nchw = do_bench(lambda: triton_nchw_forward(x_cf, w_cf, K_H, PAD_H, STRIDE_H), warmup=warmup, rep=rep)

        triton_nhwc_forward(x_nhwc, w_cf, K_H, PAD_H, STRIDE_H)
        t_nhwc = do_bench(lambda: triton_nhwc_forward(x_nhwc, w_cf, K_H, PAD_H, STRIDE_H), warmup=warmup, rep=rep)

        for _ in range(3):
            F.conv2d(x_cf, w_cf, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C)
        t_cudnn = do_bench(lambda: F.conv2d(x_cf, w_cf, None, stride=(STRIDE_H, 1), padding=(PAD_H, 0), groups=C),
                           warmup=warmup, rep=rep)

        all_results[K_H] = {
            "triton_NCHW": (t_nchw, d_nchw),
            "triton_NHWC": (t_nhwc, d_nhwc),
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
    print(f"  Depthwise conv2d  x=(3072,128,202,1)  w=(128,1,K,1)  groups=128")
    print(f"  default: kernel=(3,1)  stride=(1,1)  padding=(1,0)")
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
    print(f"  {'k':>4s} {'triton_NCHW (ms)':>17s} {'triton_NHWC (ms)':>17s} {'cuDNN_CF (ms)':>14s} {'NCHW/cuDNN':>11s} {'NHWC/cuDNN':>11s}")
    print(f"  {'-'*4} {'-'*17} {'-'*17} {'-'*14} {'-'*11} {'-'*11}")
    for K in (3, 5, 7):
        r = vk_results[K]
        t_nchw = r["triton_NCHW"][0]
        t_nhwc = r["triton_NHWC"][0]
        t_cud = r["cuDNN_CF"][0]
        print(f"  {K:4d} {t_nchw:17.4f} {t_nhwc:17.4f} {t_cud:14.4f} {t_nchw/t_cud:10.2f}x {t_nhwc/t_cud:10.2f}x")
    print()


if __name__ == "__main__":
    main()
