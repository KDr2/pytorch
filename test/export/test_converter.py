# Owner(s): ["oncall: export"]

import torch
from torch._dynamo.test_case import TestCase
from torch._export.converter import TS2EPConverter

from torch.testing._internal.common_utils import run_tests


class TestConverter(TestCase):
    def test_ts2ep_converter_basic(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        m = Module()
        inp = (torch.ones(1, 3), torch.ones(1, 3))

        ts_model = torch.jit.script(m)
        ep = TS2EPConverter(ts_model, inp).convert()

        torch.testing.assert_close(ep.module()(*inp)[0], m(*inp))

    def test_convert_if_basic(self):
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                if x:
                    return y * y
                else:
                    return y + y

        sample_inputs = (torch.tensor(True), torch.tensor(4))

        scripted = torch.jit.script(M(), example_inputs=[sample_inputs])

        ep = TS2EPConverter(scripted, sample_inputs).convert()

        torch.testing.assert_close(ep.module()(*sample_inputs)[0], M()(*sample_inputs))
        torch.testing.assert_close(
            ep.module()(torch.tensor(False), torch.tensor(4))[0],
            M()(torch.tensor(False), torch.tensor(4)),
        )


if __name__ == "__main__":
    run_tests()
