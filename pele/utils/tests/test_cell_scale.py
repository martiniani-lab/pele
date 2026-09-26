import os
import unittest
from types import SimpleNamespace
from unittest import mock

import pele
from pele.utils.cell_scale import _omp_thread_count


class TestOmpThreadCount(unittest.TestCase):
    def test_env_var(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "3"}):
            self.assertEqual(_omp_thread_count(), 3)

    def test_env_var_nested_list(self):
        # OMP_NUM_THREADS may list one value per nesting level
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "5,2"}):
            self.assertEqual(_omp_thread_count(), 5)

    def test_default_is_usable_cores(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": ""}):
            n = _omp_thread_count()
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, os.cpu_count())

    def test_default_without_affinity(self):
        # e.g. macOS, where os.sched_getaffinity does not exist
        fake_os = SimpleNamespace(environ={}, cpu_count=lambda: 7)
        with mock.patch("pele.utils.cell_scale.os", fake_os):
            self.assertEqual(_omp_thread_count(), 7)


class TestGetInclude(unittest.TestCase):
    def test_headers_present(self):
        self.assertTrue(os.path.isfile(os.path.join(pele.get_include(), "pele", "array.hpp")))


if __name__ == "__main__":
    unittest.main()
