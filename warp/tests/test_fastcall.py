# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

wp.init()


class TestFastcall(unittest.TestCase):
    """Tests for the _warp_fastcall METH_FASTCALL extension module.

    These tests verify the call chain (importlib loading, METH_FASTCALL dispatch,
    argument marshalling, return value wrapping). The underlying native functions
    are already exercised by test_fp16, test_types, test_scalar_ops, etc.
    """

    def test_module_loads(self):
        self.assertIsNotNone(wp._src.context.runtime.fastcall, "_warp_fastcall module failed to load")

    def test_float_to_half_bits(self):
        result = wp._src.context.runtime.fastcall.float_to_half_bits(1.0)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0x3C00)

    def test_half_bits_to_float(self):
        result = wp._src.context.runtime.fastcall.half_bits_to_float(0x3C00)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 1.0)

    def test_consistency_with_ctypes(self):
        """Verify METH_FASTCALL results match the ctypes path."""
        fastcall = wp._src.context.runtime.fastcall
        core = wp._src.context.runtime.core
        for v in [0.0, 1.0, -1.0, 3.14]:
            self.assertEqual(fastcall.float_to_half_bits(v), core.wp_float_to_half_bits(v))

    def test_wrong_arg_count(self):
        fastcall = wp._src.context.runtime.fastcall
        with self.assertRaises(TypeError):
            fastcall.float_to_half_bits()
        with self.assertRaises(TypeError):
            fastcall.float_to_half_bits(1.0, 2.0)
        with self.assertRaises(TypeError):
            fastcall.half_bits_to_float()

    def test_wrong_arg_type(self):
        fastcall = wp._src.context.runtime.fastcall
        with self.assertRaises(TypeError):
            fastcall.float_to_half_bits("not a float")

    def test_vec3h_construction(self):
        """Verify the fastcall path works through the public API (wp.vec3h)."""
        v = wp.vec3h(1.0, 2.0, 3.0)
        self.assertEqual(float(v[0]), 1.0)
        self.assertEqual(float(v[1]), 2.0)
        self.assertEqual(float(v[2]), 3.0)


if __name__ == "__main__":
    unittest.main()
