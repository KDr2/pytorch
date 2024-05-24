# Owner(s): ["oncall: distributed"]

import os
import time
from datetime import timedelta

# Disable rethrowing errors in the watchdog thread to avoid crashes on teardown.
os.environ["TORCH_NCCL_RETHROW_CUDA_ERRORS"] = "0"

import torch
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class HealthcheckNCCLMultiprocessTest(MultiProcessTestCase):
    store_path: str = "/tmp/test_healthcheck.filestore"

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self) -> None:
        if os.path.exists(self.store_path):
            print("removing file store!")
            os.remove(self.store_path)

        from torch._C._distributed_c10d import FileStore

        store = FileStore(self.store_path, self.world_size)
        store.set("warm", "up")

        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_healthcheck_success(self) -> None:
        from torch._C._distributed_c10d import FileStore, HealthcheckNCCL

        torch.cuda.set_device(self.rank)

        store = FileStore(self.store_path, self.world_size)

        healthcheck = HealthcheckNCCL(
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=1,
            abort_on_error=False,
            interval=timedelta(seconds=1),
            timeout=timedelta(seconds=60),
        )
        while healthcheck.num_failures == -1:
            time.sleep(0.01)
        self.assertEqual(healthcheck.num_failures, 0)

        healthcheck.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_healthcheck_timeout(self) -> None:
        from torch._C._distributed_c10d import FileStore, HealthcheckNCCL

        torch.cuda.set_device(self.rank)

        store = FileStore(self.store_path, self.world_size)

        healthcheck = HealthcheckNCCL(
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=1,
            abort_on_error=False,
            interval=timedelta(seconds=1),
            timeout=timedelta(milliseconds=1),
        )
        while healthcheck.num_failures == -1:
            time.sleep(0.01)
        self.assertEqual(healthcheck.num_failures, 2)

        # NCCL may be in a bad state -- force a clean exit
        os._exit(0)


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
