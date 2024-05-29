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

    def test_ts2ep_converter_unpack(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x, y = torch.split(x, 2)
                return x + y

        inp = (torch.ones(1, 4))

        mod = M()
        ts_model = torch.jit.script(mod)
        print(ts_model.graph)

        ep = TS2EPConverter(ts_model, inp).convert()

        torch.testing.assert_close(ep.module()(*inp)[0], mod(*inp))
        print(mod(*inp))
        print(ep.module()(*inp)[0])


if __name__ == "__main__":
    run_tests()
