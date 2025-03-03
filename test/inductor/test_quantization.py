# Owner(s): ["module: inductor"]

import torch
import torch._inductor
import torch._inductor.fx_passes.group_batch_fusion
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


class TargetCPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        relued = torch.relu(x1)
        tanhed = torch.tanh(relued)
        tensor = torch.matmul(
            tanhed,
            x2,
        )
        return tensor


class TestQuantization(TestCase):
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

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "activation_quantization_aten_pass": {"quant_type": torch.float8_e5m2}
        },
    )
    def test_activation_quantization_aten(self):
        counters.clear()
        module = TargetCPModule().to(GPU_TYPE)
        input = [
            torch.rand((16, 10), requires_grad=True, device=GPU_TYPE, dtype=torch.bfloat16),
            torch.rand((10, 16), requires_grad=True, device=GPU_TYPE, dtype=torch.bfloat16),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(counters["inductor"]["activation_quantization_aten_pass"], 2)
        self.assertTrue(torch.allclose(ref, res))
        counters.clear()


if __name__ == "__main__":
    run_tests()
