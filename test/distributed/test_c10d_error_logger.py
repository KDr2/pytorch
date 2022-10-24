# Owner(s): ["oncall: distributed"]

import logging
import os
import sys
from functools import partial, wraps
from unittest.mock import patch

import torch
import torch.distributed as dist

from torch.distributed.c10d_error_logger import _get_or_create_logger
from torch.distributed.distributed_c10d import exception_handler

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO
WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))


def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        self.dist_init()
        func(self)
        self.destroy_comms()

    return wrapper


class C10dErrorLoggerTest(MultiProcessTestCase):
    def setUp(self):
        super(C10dErrorLoggerTest, self).setUp()
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["BACKEND"] = BACKEND
        self._spawn_processes()

    @property
    def device(self):
        return (
            torch.device(self.rank)
            if BACKEND == dist.Backend.NCCL
            else torch.device("cpu")
        )

    @property
    def world_size(self):
        return WORLD_SIZE

    @property
    def process_group(self):
        return dist.group.WORLD

    def destroy_comms(self):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def dist_init(self):
        dist.init_process_group(
            backend=BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # set device for nccl pg for collectives
        if BACKEND == "nccl":
            torch.cuda.set_device(self.rank)

    @patch("torch.distributed.c10d_error_logger._get_logging_handler")
    def test_get_or_create_logger(self, logging_handler_mock):
        logging_handler_mock.return_value = logging.NullHandler(), "NullHandler"
        logger = _get_or_create_logger()
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)

    @with_comms
    @exception_handler
    def failed_broadcast(self):
        tensor = torch.arange(2, dtype=torch.int64)
        print(f"{tensor}")
        if self.rank == 0:
            dist.broadcast(tensor, 0)

    @with_comms
    def test_exception_handler(self):
        with self.assertRaises(RuntimeError):
            self.failed_broadcast()



if __name__ == "__main__":
    run_tests()



# Owner(s): ["oncall: distributed"]

# import logging
# import unittest
# from unittest.mock import patch

# from torch.distributed.c10d_error_logger import _get_or_create_logger
# from torch.distributed.distributed_c10d import exception_handler



# class C10dErrorLoggerTest(unittest.TestCase):

#     @patch("torch.distributed.c10d_error_logger.get_logging_handler")
#     def test_get_or_create_logger(self, logging_handler_mock):
#         logging_handler_mock.return_value = logging.NullHandler(), "NullHandler"
#         logger = _get_or_create_logger()
#         self.assertIsNotNone(logger)
#         self.assertEqual(1, len(logger.handlers))
#         self.assertIsInstance(logger.handlers[0], logging.NullHandler)


#     @exception_handler
#     def raise_runtime_error(self):
#         print("raise_runtime_error")
#         raise RuntimeError()
