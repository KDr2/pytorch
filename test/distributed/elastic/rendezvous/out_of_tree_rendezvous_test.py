import importlib
import pathlib
import subprocess
import sys
import unittest

import torch.distributed.elastic.rendezvous as rdvz


BACKEND_NAME = "testbackend"
TEST_PACKAGE_PATH = "/out_of_tree_test_package/"
TEST_MODULE_PATH = TEST_PACKAGE_PATH + "src/testbackend/__init__.py"


class OutOfTreeRendezvousTest(unittest.TestCase):
    def test_out_of_tree_handler_loading(self):
        current_path = str(pathlib.Path(__file__).parent.resolve())
        rdvz._register_out_of_tree_handlers()
        registry_dict = rdvz.rendezvous_handler_registry._registry

        # test backend should not be registered as a backend
        self.assertFalse(BACKEND_NAME in registry_dict)

        # Installing test_backend package
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                current_path + TEST_PACKAGE_PATH,
            ]
        )

        # Manually loading 'testbackend' module as we are trying to install and access it in the same process.
        # This manual loading is not required during actual usage as we usually install the package in one
        # process (using pip) and torch elastic jobs as another process.
        # This is required ONLY for this unit test.
        spec = importlib.util.spec_from_file_location(
            BACKEND_NAME, current_path + TEST_MODULE_PATH
        )
        testbackend = importlib.util.module_from_spec(spec)
        sys.modules[BACKEND_NAME] = testbackend
        spec.loader.exec_module(testbackend)

        # Registering the out of tree handlers again
        rdvz._register_out_of_tree_handlers()

        # test backend should be registered as a backend
        self.assertTrue(BACKEND_NAME in registry_dict)

        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", BACKEND_NAME])
