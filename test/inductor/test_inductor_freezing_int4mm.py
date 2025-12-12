# Owner(s): ["module: inductor"]
"""
Tests for INT4 GPU matmul fusion in freezing_patterns.py

This tests the pattern:
    (int4_mm(x, w1, gs, sz1), int4_mm(x, w2, gs, sz2), int4_mm(x, w3, gs, sz3))
    -> split(int4_mm(x, cat(w1,w2,w3), gs, cat(sz1,sz2,sz3)), sizes)
"""

import unittest

import torch
from torch import nn
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import skipIfRocm
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

# Require torchao - fail if not installed
from torchao.quantization import Int4WeightOnlyConfig, quantize_


class Int4MMFusionTests(InductorTestCase):
    """Tests for INT4 GPU matmul fusion during freezing."""

    device = GPU_TYPE

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        counters.clear()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @skipIfRocm
    def test_int4_mm_fusion_three_way(self):
        """Test 3-way INT4 matmul fusion (like Q/K/V projections)."""

        class QKVProjection(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.q_proj(x), self.k_proj(x), self.v_proj(x)

        model = QKVProjection(dim=512).eval().to(self.device).to(torch.bfloat16)

        # Quantize to INT4
        int4_config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(model, int4_config)

        x = torch.randn(2, 128, 512, device=self.device, dtype=torch.bfloat16)

        # Get reference output
        with torch.no_grad():
            ref = model(x)

        # Compile with freezing
        with config.patch({
            "freezing": True,
            "freezing_discard_parameters": False,
            "implicit_fallbacks": True,
        }):

            @torch.compile()
            def foo(mod, inp):
                return mod(inp)

            with torch.no_grad():
                out, code = run_and_get_code(foo, model, x)

        # Verify output correctness
        for i in range(3):
            self.assertEqual(ref[i], out[i])

        # Count INT4 matmul calls - should be 1 (fused) not 3
        lines = code[0].split('\n')
        int4_mm_calls = [
            l for l in lines
            if 'torch.ops.aten._weight_int4pack_mm.default' in l and '=' in l
        ]

        self.assertEqual(
            len(int4_mm_calls), 1,
            f"Expected 1 fused INT4 matmul, got {len(int4_mm_calls)}. "
            f"Fusion should combine 3 matmuls into 1."
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @skipIfRocm
    def test_int4_mm_fusion_two_way(self):
        """Test 2-way INT4 matmul fusion (like K/V in cross-attention)."""

        class KVProjection(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.k_proj(x), self.v_proj(x)

        model = KVProjection(dim=512).eval().to(self.device).to(torch.bfloat16)

        # Quantize to INT4
        int4_config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(model, int4_config)

        x = torch.randn(2, 128, 512, device=self.device, dtype=torch.bfloat16)

        # Get reference output
        with torch.no_grad():
            ref = model(x)

        # Compile with freezing
        with config.patch({
            "freezing": True,
            "freezing_discard_parameters": False,
            "implicit_fallbacks": True,
        }):

            @torch.compile()
            def foo(mod, inp):
                return mod(inp)

            with torch.no_grad():
                out, code = run_and_get_code(foo, model, x)

        # Verify output correctness
        for i in range(2):
            self.assertEqual(ref[i], out[i])

        # Count INT4 matmul calls - should be 1 (fused) not 2
        lines = code[0].split('\n')
        int4_mm_calls = [
            l for l in lines
            if 'torch.ops.aten._weight_int4pack_mm.default' in l and '=' in l
        ]

        self.assertEqual(
            len(int4_mm_calls), 1,
            f"Expected 1 fused INT4 matmul, got {len(int4_mm_calls)}. "
            f"Fusion should combine 2 matmuls into 1."
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @skipIfRocm
    def test_int4_mm_fusion_different_group_sizes(self):
        """Test INT4 fusion works with different group_size values."""

        class TwoLinear(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.linear1(x), self.linear2(x)

        # Test all supported group sizes
        for group_size in [32, 64, 128, 256]:
            model = TwoLinear(dim=512).eval().to(self.device).to(torch.bfloat16)

            int4_config = Int4WeightOnlyConfig(
                group_size=group_size,
                int4_packing_format="tile_packed_to_4d",
            )
            quantize_(model, int4_config)

            x = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)

            with torch.no_grad():
                ref = model(x)

            torch._dynamo.reset()

            with config.patch({
                "freezing": True,
                "freezing_discard_parameters": False,
                "implicit_fallbacks": True,
            }):

                @torch.compile()
                def foo(mod, inp):
                    return mod(inp)

                with torch.no_grad():
                    out, code = run_and_get_code(foo, model, x)

            # Verify correctness
            for i in range(2):
                self.assertEqual(ref[i], out[i])

            # Verify fusion
            lines = code[0].split('\n')
            int4_mm_calls = [
                l for l in lines
                if 'torch.ops.aten._weight_int4pack_mm.default' in l and '=' in l
            ]

            self.assertEqual(
                len(int4_mm_calls), 1,
                f"group_size={group_size}: Expected 1 fused INT4 matmul, got {len(int4_mm_calls)}"
            )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @skipIfRocm
    def test_int4_mm_no_fusion_different_inputs(self):
        """Test that INT4 matmuls with different inputs are NOT fused."""

        class DifferentInputs(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x, y):
                # Different inputs - should NOT be fused
                return self.linear1(x), self.linear2(y)

        model = DifferentInputs(dim=512).eval().to(self.device).to(torch.bfloat16)

        int4_config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(model, int4_config)

        x = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)
        y = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            ref = model(x, y)

        with config.patch({
            "freezing": True,
            "freezing_discard_parameters": False,
            "implicit_fallbacks": True,
        }):

            @torch.compile()
            def foo(mod, inp1, inp2):
                return mod(inp1, inp2)

            with torch.no_grad():
                out, code = run_and_get_code(foo, model, x, y)

        # Verify correctness
        for i in range(2):
            self.assertEqual(ref[i], out[i])

        # Should NOT be fused - expect 2 separate calls
        lines = code[0].split('\n')
        int4_mm_calls = [
            l for l in lines
            if 'torch.ops.aten._weight_int4pack_mm.default' in l and '=' in l
        ]

        self.assertEqual(
            len(int4_mm_calls), 2,
            f"Expected 2 separate INT4 matmuls (different inputs), got {len(int4_mm_calls)}"
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @skipIfRocm
    def test_int4_mm_no_fusion_different_k_dims(self):
        """Test that INT4 matmuls with different K dimensions are NOT fused."""

        class DifferentKDims(nn.Module):
            def __init__(self):
                super().__init__()
                # Different input dimensions (K) - should NOT be fused
                self.linear1 = nn.Linear(512, 256, bias=False)
                self.linear2 = nn.Linear(256, 256, bias=False)

            def forward(self, x):
                # Apply linear1, then use its output for linear2
                # Both share same output dim but different input dims
                out1 = self.linear1(x)
                out2 = self.linear2(out1)
                return out1, out2

        model = DifferentKDims().eval().to(self.device).to(torch.bfloat16)

        int4_config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(model, int4_config)

        x = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            ref = model(x)

        torch._dynamo.reset()

        with config.patch({
            "freezing": True,
            "freezing_discard_parameters": False,
            "implicit_fallbacks": True,
        }):

            @torch.compile()
            def foo(mod, inp):
                return mod(inp)

            with torch.no_grad():
                out, code = run_and_get_code(foo, model, x)

        # Verify correctness
        for i in range(2):
            self.assertEqual(ref[i], out[i])

        # Should NOT be fused due to different K dimensions - expect 2 separate calls
        lines = code[0].split('\n')
        int4_mm_calls = [
            l for l in lines
            if 'torch.ops.aten._weight_int4pack_mm.default' in l and '=' in l
        ]

        self.assertEqual(
            len(int4_mm_calls), 2,
            f"Expected 2 separate INT4 matmuls (different K dims), got {len(int4_mm_calls)}"
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @skipIfRocm
    def test_int4_mm_fusion_different_output_dims(self):
        """Test INT4 fusion with different output dimensions (asymmetric projections)."""

        class AsymmetricProjection(nn.Module):
            def __init__(self):
                super().__init__()
                # Different output dimensions - should still fuse
                self.proj1 = nn.Linear(512, 256, bias=False)
                self.proj2 = nn.Linear(512, 256, bias=False)
                self.proj3 = nn.Linear(512, 256, bias=False)

            def forward(self, x):
                return self.proj1(x), self.proj2(x), self.proj3(x)

        model = AsymmetricProjection().eval().to(self.device).to(torch.bfloat16)

        int4_config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(model, int4_config)

        x = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            ref = model(x)

        torch._dynamo.reset()

        with config.patch({
            "freezing": True,
            "freezing_discard_parameters": False,
            "implicit_fallbacks": True,
        }):

            @torch.compile()
            def foo(mod, inp):
                return mod(inp)

            with torch.no_grad():
                out, code = run_and_get_code(foo, model, x)

        # Verify correctness
        for i in range(3):
            self.assertEqual(ref[i], out[i])

        # Should be fused into 1 call
        lines = code[0].split('\n')
        int4_mm_calls = [
            l for l in lines
            if 'torch.ops.aten._weight_int4pack_mm.default' in l and '=' in l
        ]

        self.assertEqual(
            len(int4_mm_calls), 1,
            f"Expected 1 fused INT4 matmul for asymmetric dims, got {len(int4_mm_calls)}"
        )

    # Note: Testing float16 or mixed dtypes is not possible with torchao's
    # Int4TilePackedTo4dTensor which only supports bfloat16. The dtype validation
    # in check_int4_gpu_concat_weights is defensive for future format support.
    #
    # Note: Testing non-constant weights is not possible because freezing always
    # converts weights to constants (get_attr). The get_attr check is still important
    # to guard against edge cases or future changes to the freezing behavior.


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
