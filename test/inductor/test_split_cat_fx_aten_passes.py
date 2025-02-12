# Owner(s): ["module: inductor"]

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_cuda


try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False


class TestSplitCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        cat = torch.ops.aten.cat.default([x, y], 1)
        split = torch.ops.aten.split.Tensor(cat, 32, 1)
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6]
        getitem_7 = split[7]
        cat_1 = torch.ops.aten.cat.default(
            [
                getitem,
                getitem_1,
                getitem_2,
                getitem_3,
                getitem_4,
                getitem_5,
                getitem_6,
                getitem_7,
            ],
            1,
        )
        return cat_1


class TestSelectCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        select = torch.ops.aten.select.int(x, 1, 0)
        select_1 = torch.ops.aten.select.int(x, 1, 1)
        select_2 = torch.ops.aten.select.int(x, 1, 2)
        select_3 = torch.ops.aten.select.int(x, 1, 3)
        select_4 = torch.ops.aten.select.int(x, 1, 4)
        select_5 = torch.ops.aten.select.int(x, 1, 5)
        cat = torch.ops.aten.cat.default(
            [select, select_1, select_2, select_3, select_4, select_5], 1
        )
        return cat


class TestFullSliceCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        full = torch.ops.aten.full.default(
            [4096, 128],
            0,
            dtype=torch.float32,
            layout=torch.strided,
            device=x.device,
            pin_memory=False,
        )
        slice_12 = torch.ops.aten.slice.Tensor(x, 1, 12800, 15872)
        slice_14 = torch.ops.aten.slice.Tensor(slice_12, 1, 1792, 1920)
        slice_15 = torch.ops.aten.slice.Tensor(slice_12, 1, 1920, 2048)
        slice_16 = torch.ops.aten.slice.Tensor(slice_12, 1, 2048, 2176)
        slice_17 = torch.ops.aten.slice.Tensor(slice_12, 1, 2176, 2304)
        slice_18 = torch.ops.aten.slice.Tensor(slice_12, 1, 2304, 2432)
        slice_19 = torch.ops.aten.slice.Tensor(slice_12, 1, 2432, 2560)
        slice_20 = torch.ops.aten.slice.Tensor(slice_12, 1, 2560, 2688)
        slice_21 = torch.ops.aten.slice.Tensor(slice_12, 1, 2688, 2816)
        slice_22 = torch.ops.aten.slice.Tensor(slice_12, 1, 2816, 2944)
        slice_23 = torch.ops.aten.slice.Tensor(slice_12, 1, 2944, 3072)
        cat = torch.ops.aten.cat.default(
            [
                full,
                full,
                full,
                full,
                full,
                slice_14,
                full,
                slice_15,
                slice_16,
                full,
                full,
                full,
                full,
                full,
                slice_17,
                full,
                full,
                full,
                full,
                full,
                full,
                slice_18,
                slice_19,
                slice_20,
                slice_21,
                slice_22,
                slice_23,
                full,
                full,
                full,
                full,
                full
            ],
            dim = 1,
        )
        return cat


class TestSelectViewCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        slice_node = torch.ops.aten.slice.Tensor(x, 1, 1034, 1044)
        select_64 = torch.ops.aten.select.int(slice_node, 1, 0)
        select_65 = torch.ops.aten.select.int(slice_node, 1, 1)
        select_66 = torch.ops.aten.select.int(slice_node, 1, 2)
        select_67 = torch.ops.aten.select.int(slice_node, 1, 3)
        select_68 = torch.ops.aten.select.int(slice_node, 1, 4)
        select_69 = torch.ops.aten.select.int(slice_node, 1, 5)
        select_70 = torch.ops.aten.select.int(slice_node, 1, 6)
        select_71 = torch.ops.aten.select.int(slice_node, 1, 7)
        select_72 = torch.ops.aten.select.int(slice_node, 1, 8)
        select_73 = torch.ops.aten.select.int(slice_node, 1, 9)
        view_522 = torch.ops.aten.reshape.default(select_64, [1, 4096, 128])
        view_521 = torch.ops.aten.reshape.default(select_65, [1, 4096, 128])
        view_520 = torch.ops.aten.reshape.default(select_66, [1, 4096, 128])
        view_519 = torch.ops.aten.reshape.default(select_67, [1, 4096, 128])
        view_518 = torch.ops.aten.reshape.default(select_68, [1, 4096, 128])
        view_517 = torch.ops.aten.reshape.default(select_69, [1, 4096, 128])
        view_516 = torch.ops.aten.reshape.default(select_70, [1, 4096, 128])
        view_515 = torch.ops.aten.reshape.default(select_71, [1, 4096, 128])
        view_514 = torch.ops.aten.reshape.default(select_72, [1, 4096, 128])
        view_513 = torch.ops.aten.reshape.default(select_73, [1, 4096, 128])
        cat = torch.ops.aten.cat.default(
            [view_522, view_521, view_520, view_519, view_518, view_517, view_516, view_515, view_514, view_513], dim=0
        )
        return cat


class TestSplitCatAten(TestCase):
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict.keys():
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @requires_cuda
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "split_cat_aten_pass": {},
        },
    )
    def test_split_cat_post_grad(self):
        counters.clear()
        inputs = [
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestSplitCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 3)
        self.assertEqual(counters["inductor"]["split_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_cuda
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "select_cat_aten_pass": {},
        },
    )
    def test_select_cat_post_grad(self):
        counters.clear()
        inputs = [
            torch.randn(1024, 6, 128, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestSelectCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 1)
        self.assertEqual(counters["inductor"]["select_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_cuda
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "full_slice_cat_aten_pass": {},
        },
    )
    def test_full_slice_cat_post_grad(self):
        counters.clear()
        inputs = [
            torch.randn(4096, 15872, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestFullSliceCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 2)
        self.assertEqual(counters["inductor"]["full_slice_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()


    @requires_cuda
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "select_view_cat_aten_pass": {},
        },
    )
    def test_select_view_cat_post_grad(self):
        counters.clear()
        inputs = [
            torch.randn(4096, 1154, 128, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestSelectViewCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 1)
        self.assertEqual(counters["inductor"]["select_view_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()


if __name__ == "__main__":
    run_tests()
