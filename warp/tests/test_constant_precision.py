# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for constant precision preservation (GH-485).

This module tests that floating-point literal constants maintain their precision
when explicitly cast to higher precision types like wp.float64.

Scope:
- Literal constants: wp.float64(3.14159...) - weak typing preserves full precision
- Negative literals: wp.float64(-3.14) - UnaryOp folding preserves full precision
- Module constants: wp.float64(wp.PI) - resolved to weak float, adapts to target type

IEEE-754 Compliance Assumptions:
All tests assume strict IEEE-754 compliance on both CPU and GPU:

1. Literal Assignment: Python float (float64) → wp.float64 is exact (no rounding)
   Consequence: Identical float64 literals compare exactly equal (error = 0.0)

2. Float32 Rounding: Python float → wp.float32 is deterministic (same input → same output)
   Consequence: Same conversion always produces identical bit pattern

3. Widening Conversions: float32 → float64 is exact (no rounding, all bits preserved)
   Consequence: Widened value equals original float32 (when both viewed as float64)

4. Basic Arithmetic: +, -, *, / produce deterministic IEEE-754 rounding
   Consequence: Same operation on same inputs always produces same output
   Note: FMA (Fused Multiply-Add) may cause ±1 ULP differences on some platforms

If exact assertions fail, it indicates:
- Non-IEEE-754 hardware (extremely rare)
- Different rounding modes
- Compiler optimization differences
- Platform-specific FMA behavior (document and adjust tolerance)
"""

import math
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_float64_cast_precision_kernel(result: wp.array(dtype=wp.float64)):
    """Test that wp.float64(constant literal) preserves full precision."""
    # Literal constant — weak float adapts to float64
    pi = wp.float64(3.141592653589793)
    expected = wp.float64(3.141592653589793)
    error = wp.abs(pi - expected)
    result[0] = error


@wp.kernel
def test_float32_cast_kernel(result: wp.array(dtype=wp.float32)):
    """Test that wp.float32 still works correctly (backward compatibility)."""
    pi = wp.float32(wp.PI)
    # When truncated to float32, we expect the value to be approximately this
    expected = wp.float32(3.141592741012573)
    error = wp.abs(pi - expected)
    result[0] = error


@wp.kernel
def test_literal_precision_kernel(result: wp.array(dtype=wp.float64)):
    """Test that literal constants preserve precision when cast to float64."""
    # Use a value with many decimal places that would be truncated in float32
    val = wp.float64(3.141592653589793)
    expected = wp.float64(3.141592653589793)
    error = wp.abs(val - expected)
    result[0] = error


@wp.kernel
def test_large_precision_value_kernel(result: wp.array(dtype=wp.float64)):
    """Test precision preservation with a larger value."""
    # Test with a value that loses significant digits if truncated to float32
    val = wp.float64(1234567.89012345)
    expected = wp.float64(1234567.89012345)
    error = wp.abs(val - expected)
    result[0] = error


@wp.kernel
def test_int64_cast_kernel(result: wp.array(dtype=wp.int64)):
    """Test that int64 casts work correctly."""
    # Test that large integers preserve precision
    val = wp.int64(9223372036854775807)  # Max int64 value
    result[0] = val


@wp.kernel
def test_float64_arithmetic_kernel(result: wp.array(dtype=wp.float64)):
    """Test that float64 arithmetic maintains precision."""
    # Test that arithmetic operations on float64 constants maintain precision
    a = wp.float64(1.0)
    b = wp.float64(3.0)
    c = a / b
    expected = wp.float64(0.3333333333333333)
    error = wp.abs(c - expected)
    result[0] = error


@wp.kernel
def test_nested_constructor_kernel(result: wp.array(dtype=wp.float64)):
    """Test nested type constructors."""
    # This should work: cast float32 to float64
    # The inner literal becomes float32, then gets cast to float64
    val = wp.float64(wp.float32(3.141592653589793))
    result[0] = val


@wp.kernel
def test_runtime_value_kernel(x: float, result: wp.array(dtype=wp.float64)):
    """Test that runtime values still behave normally."""
    # This should convert the runtime float32 to float64 (not optimized)
    val = wp.float64(x)
    result[0] = val


@wp.kernel
def test_zero_literal_kernel(result: wp.array(dtype=wp.float64)):
    """Test edge case with zero."""
    val = wp.float64(0.0)
    result[0] = val


@wp.kernel
def test_negative_literal_kernel(result: wp.array(dtype=wp.float64)):
    """Test negative literal."""
    val = wp.float64(-3.141592653589793)
    result[0] = val


@wp.kernel
def test_float32_mul_literal_kernel(result_f32: wp.array(dtype=wp.float32), result_f64: wp.array(dtype=wp.float64)):
    """Test that float32 * literal is computed in single precision."""
    # 1.00000005 rounds to 1.0 in float32, but stays 1.00000005 in float64
    x = wp.float32(2.0)

    # This should be single-precision: 2.0f * 1.0f = 2.0f (literal rounds to 1.0f)
    result_single = x * 1.00000005

    # This should be double-precision: 2.0 * 1.00000005 = 2.0000001
    # Need to explicitly cast the literal to float64 to match types
    result_double = wp.float64(x) * wp.float64(1.00000005)

    result_f32[0] = result_single
    result_f64[0] = result_double


@wp.kernel
def test_float32_mul_variable_kernel(result_f32: wp.array(dtype=wp.float32), result_f64: wp.array(dtype=wp.float64)):
    """Test that float32 * variable (assigned from literal) uses appropriate precision."""
    x = wp.float32(2.0)
    # Literal constant defaults to float64, but automatically casts to float32 when used with x
    c = 1.00000005

    # This should be single-precision (c is float64 constant, auto-cast to float32)
    result_single = x * c

    # This should be double-precision
    result_double = wp.float64(x) * 1.00000005

    result_f32[0] = result_single
    result_f64[0] = result_double


@wp.kernel
def test_float32_mul_expression_kernel(
    x: wp.float32, result_f32: wp.array(dtype=wp.float32), result_f64: wp.array(dtype=wp.float64)
):
    """Test that literal expressions preserve precision and cast automatically."""
    # Literal expression (1.0 + 0.00000005) computed in double precision
    # Automatically casts to float32 when used with float32 variable x
    # No explicit cast needed - codegen.py injects the cast
    result_single = x * (1.0 + 0.00000005)

    # This should be double-precision
    result_double = wp.float64(x) * (1.0 + 0.00000005)

    result_f32[0] = result_single
    result_f64[0] = result_double


@wp.kernel
def test_chained_addition_precision_kernel(x: wp.float32, result: wp.array(dtype=wp.float32)):
    """Test that chained additions preserve double precision until automatic cast.

    Verifies that (1.0 + 0.00000005 + 0.00000005) doesn't collapse to 1.0 in float32.
    Both additions should happen in double precision, then automatically cast to float32.
    """
    # Chain of additions in double precision
    # In float32: 1.0 + 0.00000005 = 1.0 (lost precision)
    # In float64: 1.0 + 0.00000005 + 0.00000005 = 1.0000001 (preserved)
    # Automatic cast to float32 for multiplication with x
    result[0] = x * (1.0 + 0.00000005 + 0.00000005)


@wp.kernel
def test_vec3d_precision_kernel(result: wp.array(dtype=wp.float64)):
    """Test vec3d constructor preserves double precision."""
    v = wp.vec3d(3.141592653589793, 2.718281828459045, 1.414213562373095)
    result[0] = v[0]


@wp.kernel
def test_vec3_precision_kernel(result: wp.array(dtype=wp.float32)):
    """Test vec3 constructor uses single precision."""
    v = wp.vec3(3.141592653589793, 2.718281828459045, 1.414213562373095)
    result[0] = v[0]


@wp.kernel
def test_mat22d_precision_kernel(result: wp.array(dtype=wp.float64)):
    """Test mat22d constructor preserves double precision."""
    m = wp.mat22d(1.111111111111111, 2.222222222222222, 3.333333333333333, 4.444444444444444)
    result[0] = m[0, 0]


@wp.kernel
def test_quatd_precision_kernel(result: wp.array(dtype=wp.float64)):
    """Test quatd constructor preserves double precision."""
    q = wp.quatd(0.707106781186547, 0.0, 0.707106781186547, 0.0)
    result[0] = q[0]


@wp.kernel
def test_transformd_precision_kernel(result: wp.array(dtype=wp.float64)):
    """Test transformd constructor preserves double precision."""
    p = wp.vec3d(1.234567890123456, 2.345678901234567, 3.456789012345678)
    q = wp.quatd(1.0, 0.0, 0.0, 0.0)
    t = wp.transformd(p, q)
    result[0] = wp.transform_get_translation(t)[0]


@wp.kernel
def test_scalar_int_literal_kernel(result: wp.array(dtype=wp.float64)):
    """Test wp.float64(int_literal) works correctly."""
    x = wp.float64(42)
    result[0] = x


@wp.kernel
def test_vec3d_int_literals_kernel(result: wp.array(dtype=wp.float64)):
    """Test wp.vec3d(int, int, int) works correctly."""
    v = wp.vec3d(1, 2, 3)
    result[0] = v[0] + v[1] + v[2]


@wp.kernel
def test_vec3_int_literals_kernel(result: wp.array(dtype=wp.float32)):
    """Test wp.vec3(int, int, int) works correctly (single precision)."""
    v = wp.vec3(1, 2, 3)
    result[0] = v[0] + v[1] + v[2]


# TODO(GH-485): Enable once weak typing of int literals is implemented.
# @wp.kernel
# def test_float64_arithmetic_int_literal_kernel(result: wp.array(dtype=wp.float64)):
#     """Test float64_var * int_literal preserves double precision."""
#     x = wp.float64(3.141592653589793)
#     y = x * 2  # int literal should be converted to 2.0 (double precision)
#     result[0] = y
#
#
# @wp.kernel
# def test_float32_arithmetic_int_literal_kernel(result: wp.array(dtype=wp.float32)):
#     """Test float32_var * int_literal uses single precision."""
#     x = wp.float32(3.141592653589793)
#     y = x * 2  # int literal should be converted to 2.0f (single precision)
#     result[0] = y
#
#
# @wp.kernel
# def test_int_literal_arithmetic_float64_kernel(result: wp.array(dtype=wp.float64)):
#     """Test int_literal * float64_var preserves double precision (commutative)."""
#     x = wp.float64(3.141592653589793)
#     y = 2 * x  # int literal should be converted to 2.0 (double precision)
#     result[0] = y


@wp.kernel
def test_float16_constructor_kernel(result: wp.array(dtype=wp.float16)):
    """Test wp.float16(literal) works correctly."""
    x = wp.float16(3.14159)
    result[0] = x


@wp.kernel
def test_float16_int_literal_kernel(result: wp.array(dtype=wp.float16)):
    """Test wp.float16(int) works correctly."""
    x = wp.float16(42)
    result[0] = x


@wp.kernel
def test_float16_arithmetic_precision_kernel(result: wp.array(dtype=wp.float16)):
    """Test float16_var * float_literal uses half precision."""
    x = wp.float16(1.0)
    y = x * 1.0005  # Should convert to float16 precision
    result[0] = y


@wp.kernel
def test_vec3h_int_literals_kernel(result: wp.array(dtype=wp.float16)):
    """Test wp.vec3h(int, int, int) works correctly."""
    v = wp.vec3h(1, 2, 3)
    result[0] = v[0] + v[1] + v[2]


@wp.kernel
def test_vec3d_mixed_literals_kernel(result: wp.array(dtype=wp.float64)):
    """Test wp.vec3d(float, int, int) works - mixed literals accepted."""
    v = wp.vec3d(3.14, 1, 2)  # Mixed float and int literals
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


@wp.kernel
def test_vec3_mixed_literals_kernel(result: wp.array(dtype=wp.float32)):
    """Test wp.vec3(int, float, int) works - mixed literals accepted."""
    v = wp.vec3(1, 2.5, 3)  # Mixed int and float literals
    result[0] = v[0] + v[1] + v[2]


@wp.kernel
def test_mat22d_mixed_literals_kernel(result: wp.array(dtype=wp.float64)):
    """Test wp.mat22d with mixed int/float literals works correctly."""
    m = wp.mat22d(1.5, 2, 3, 4.5)  # Mixed literals
    result[0] = m[0, 0] + m[1, 1]


@wp.kernel
def test_literal_literal_preserves_double_precision_kernel(result: wp.array(dtype=wp.float64)):
    """Test that operations between float literals use double precision."""
    # Operations between literals should preserve double precision
    # CUDA compiler will constant-fold these at compile time
    a = 3.141592653589793 * 1.00000005
    result[0] = a


@wp.kernel
def test_literal_literal_cast_to_float32_preserves_precision_kernel(result: wp.array(dtype=wp.float32)):
    """Test that float32(literal * literal) computes in double precision first."""
    # Multiply in double precision, then cast to float32
    # Use a larger factor to ensure measurable difference
    a = wp.float32(3.141592653589793 * 1.0001)
    result[0] = a


@wp.kernel
def test_literal_literal_int_stays_int_kernel(result: wp.array(dtype=int)):
    """Test that int * int stays as int, not converted to float64."""
    a = 2 * 3
    result[0] = a


@wp.kernel
def test_literal_literal_assignment_preserves_precision_kernel(result: wp.array(dtype=wp.float64)):
    """Test that assigning literal * literal to a variable preserves double precision."""
    # Assignment from literal expression should preserve double precision
    a = 3.141592653589793 * 1.00000005
    # Using the variable in another expression should preserve precision
    b = a + 0.0
    result[0] = b


@wp.kernel
def test_literal_literal_with_single_precision_var_kernel(result: wp.array(dtype=wp.float64)):
    """Test that literal expression assigned to var preserves precision."""
    # Literal-only expression preserves double precision
    a = 3.141592653589793 * 1.00000005

    # Cast to float32 and back to see the variable maintains its type
    b = wp.float32(a)
    c = wp.float64(b) + 1.0

    result[0] = a
    result[1] = c


@wp.kernel
def test_literal_literal_complex_expression_kernel(result: wp.array(dtype=wp.float64)):
    """Test complex expressions with only literals preserve double precision."""
    # Complex expression with only literals
    a = (3.141592653589793 * 1.00000005 + 2.718281828459045) / 1.5
    result[0] = a


# TODO(GH-485): Enable once weak typing of int literals is implemented.
# @wp.kernel
# def test_literal_literal_mixed_int_float_kernel(result: wp.array(dtype=wp.float64)):
#     """Test that int literal * float literal uses double precision."""
#     a = 2 * 3.141592653589793
#     result[0] = a


@wp.kernel
def test_constant_fold_div_by_zero_kernel(result: wp.array(dtype=wp.float32)):
    """Test that 1.0 / 0.0 doesn't crash codegen (constant fold skipped).

    Python raises ZeroDivisionError for 1.0/0.0, so the fold is caught and
    skipped.  C++ evaluates it at runtime and produces +inf.
    """
    result[0] = 1.0 / 0.0


@wp.kernel
def test_constant_fold_neg_div_by_zero_kernel(result: wp.array(dtype=wp.float32)):
    """Test that -1.0 / 0.0 doesn't crash codegen (constant fold skipped).

    Same ZeroDivisionError catch as the positive case; C++ produces -inf.
    """
    result[0] = -1.0 / 0.0


@wp.kernel
def test_constant_fold_zero_div_by_zero_kernel(result: wp.array(dtype=wp.float32)):
    """Test that 0.0 / 0.0 doesn't crash codegen (constant fold skipped).

    Python raises ZeroDivisionError for 0.0/0.0; C++ produces NaN.
    """
    result[0] = 0.0 / 0.0


@wp.kernel
def test_constant_fold_overflow_kernel(result: wp.array(dtype=wp.float32)):
    """Test that 1.0e308 * 2.0 doesn't crash codegen (overflow fold skipped).

    Python evaluates 1.0e308 * 2.0 = inf.  The isfinite check catches it and
    skips the fold.  C++ produces inf at runtime.
    """
    result[0] = 1.0e308 * 2.0


@wp.kernel
def test_constant_fold_pow_kernel(result: wp.array(dtype=wp.float64)):
    """Test that 3.14 ** 2.0 is constant-folded in double precision."""
    result[0] = 3.141592653589793**2.0


@wp.kernel
def test_constant_fold_neg_base_pow_kernel(result: wp.array(dtype=wp.float32)):
    """Test that (-1.0) ** 0.5 doesn't crash codegen (complex result skipped).

    Python computes (-1.0)**0.5 as a complex number.  The fold is skipped and
    C++ evaluates it at runtime (producing NaN for real-valued pow).
    """
    result[0] = (-1.0) ** 0.5


@wp.kernel
def test_typed_constructor_accepts_literals_kernel(result: wp.array(dtype=wp.vec3d)):
    """Test that float literals are accepted by typed constructors via weak typing."""
    v = wp.vec3d(1.0, 2.0, 3.0)
    result[0] = v


@wp.kernel
def test_sametypes_accepts_literal_with_variable_kernel(result: wp.array(dtype=wp.float32)):
    """Test that sametypes functions accept a literal alongside a typed variable."""
    x = wp.float32(2.0)
    result[0] = wp.pow(x, 3.0)


@wp.func
def _user_double(x: wp.float64) -> wp.float64:
    return x * 2.0


@wp.kernel
def test_func_overload_resolution_kernel(result: wp.array(dtype=wp.float64)):
    # wp.float64 literal preserves full precision through @wp.func call
    result[0] = _user_double(wp.float64(3.141592653589793))


@wp.func
def _user_identity_f32(x: wp.float32) -> wp.float32:
    return x


@wp.func
def _user_identity_f64(x: wp.float64) -> wp.float64:
    return x


@wp.kernel
def test_func_overload_default_kernel(result_f32: wp.array(dtype=wp.float32)):
    # Bare literal should resolve to float32 overload (default)
    result_f32[0] = _user_identity_f32(1.5)


@wp.kernel
def test_builtin_default_resolution_kernel(
    result_f32: wp.array(dtype=wp.float32), result_f64: wp.array(dtype=wp.float64)
):
    # Bare literal sin should use float32 (default resolution)
    a = wp.sin(1.0)
    result_f32[0] = a
    # Explicit float64 should use float64
    b = wp.sin(wp.float64(1.0))
    result_f64[0] = b


@wp.kernel
def test_array_store_literal_kernel(arr: wp.array(dtype=wp.float64)):
    # Direct assignment: bare literal adapts to float64 array element type
    arr[0] = 3.141592653589793


@wp.kernel
def test_sibling_generic_retyping_kernel(result: wp.array(dtype=wp.float64)):
    x = wp.float64(3.141592653589793)
    # The literal 1.0 should be retyped to float64 to match sibling x's type
    result[0] = wp.max(x, 1.0)


@wp.kernel
def test_warp_constant_float64_kernel(result: wp.array(dtype=wp.float64)):
    """Test that wp.float64(wp.PI) preserves full precision."""
    val = wp.float64(wp.PI)
    result[0] = val


@wp.kernel
def test_negative_warp_constant_kernel(result: wp.array(dtype=wp.float64)):
    """Test that wp.float64(-wp.PI) preserves full precision."""
    val = wp.float64(-wp.PI)
    result[0] = val


@wp.kernel
def test_warp_constant_constructor_kernel(result: wp.array(dtype=wp.float64)):
    """Test that wp.vec3d(wp.PI, wp.E, wp.TAU) preserves precision of each component."""
    v = wp.vec3d(wp.PI, wp.E, wp.TAU)
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


@wp.kernel
def test_warp_constant_arithmetic_kernel(result: wp.array(dtype=wp.float64)):
    """Test that wp.PI * 2.0 matches wp.TAU in precision."""
    pi_times_two = wp.PI * 2.0
    tau = wp.TAU
    result[0] = wp.abs(wp.float64(pi_times_two) - wp.float64(tau))


@wp.kernel
def test_int_literal_float64_constructor_kernel(result: wp.array(dtype=wp.float64)):
    """Test that int literals in float64 constructors preserve exact values."""
    v = wp.vec3d(1, 2, 3)
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


@wp.kernel
def test_adjoint_warp_constant_kernel(
    inp: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    """Test that weak-typed constants preserve float64 precision in adjoints."""
    i = wp.tid()
    out[i] = wp.PI * inp[i]


@wp.kernel
def test_scalar_mul_literal_vec3d_kernel(result: wp.array(dtype=wp.vec3d)):
    """Literal * vec3d works (scalar_mul value_func accepts weak scalar)."""
    v = wp.vec3d(1.0, 2.0, 3.0)
    result[0] = 2.0 * v


@wp.kernel
def test_sametypes_literal_first_kernel(result: wp.array(dtype=wp.float32)):
    """wp.pow(literal, float32_var) works when literal is the first arg."""
    x = wp.float32(2.0)
    result[0] = wp.pow(3.0, x)


@wp.kernel
def test_builtin_result_preserves_const_expr_kernel(result: wp.array(dtype=wp.float16)):
    """Builtin result stays weakly typed and auto-casts to float16."""
    r = wp.float16(2.0)
    result[0] = r * wp.sqrt(3.0) * 0.5


@wp.kernel
def test_float_annotation_is_strongly_typed_float32_kernel(result: wp.array(dtype=wp.float32)):
    """float annotation means float32, not weakly-typed float."""
    x = float(1.00000005)
    result[0] = x


@wp.kernel
def test_vector_dtype_float_is_float32_kernel(result: wp.array(dtype=wp.vec3)):
    """wp.types.vector(..., dtype=float) produces float32 elements."""
    v = wp.types.vector(1.0, 2.0, 3.0, dtype=float)
    result[0] = v


def test_float64_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_float64_cast_precision_kernel, dim=1, inputs=[result], device=device)
    error = result.numpy()[0]
    test.assertEqual(error, 0.0, f"Float64 precision lost: error = {error}")


def test_float32_backward_compat(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_float32_cast_kernel, dim=1, inputs=[result], device=device)
    error = result.numpy()[0]
    test.assertEqual(error, 0.0, f"Float32 behavior changed: error = {error}")


def test_literal_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_literal_precision_kernel, dim=1, inputs=[result], device=device)
    error = result.numpy()[0]
    test.assertEqual(error, 0.0, f"Literal precision lost: error = {error}")


def test_large_precision_value(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_large_precision_value_kernel, dim=1, inputs=[result], device=device)
    error = result.numpy()[0]
    test.assertEqual(error, 0.0, f"Large value precision lost: error = {error}")


def test_int64_cast(test, device):
    result = wp.zeros(1, dtype=wp.int64, device=device)
    wp.launch(test_int64_cast_kernel, dim=1, inputs=[result], device=device)
    value = result.numpy()[0]
    expected = 9223372036854775807
    test.assertEqual(value, expected)


def test_float64_arithmetic(test, device):
    """Note: If this fails with small error, may indicate FMA or platform-specific rounding."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_float64_arithmetic_kernel, dim=1, inputs=[result], device=device)
    error = result.numpy()[0]
    test.assertEqual(error, 0.0, f"Float64 arithmetic precision lost: error = {error}")


def test_float32_mul_literal(test, device):
    """Tests: 2.0f * 1.00000005 (literal rounds to 1.0f) = 2.0f vs
    2.0 * 1.00000005 (preserved) = 2.0000001
    """
    result_f32 = wp.zeros(1, dtype=wp.float32, device=device)
    result_f64 = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_float32_mul_literal_kernel, dim=1, inputs=[result_f32, result_f64], device=device)

    val_f32 = float(result_f32.numpy()[0])
    val_f64 = float(result_f64.numpy()[0])

    test.assertEqual(val_f32, 2.0)
    test.assertEqual(val_f64, 2.0 * 1.00000005)
    test.assertNotEqual(val_f32, val_f64)


def test_float32_mul_variable(test, device):
    result_f32 = wp.zeros(1, dtype=wp.float32, device=device)
    result_f64 = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_float32_mul_variable_kernel, dim=1, inputs=[result_f32, result_f64], device=device)

    val_f32 = float(result_f32.numpy()[0])
    val_f64 = float(result_f64.numpy()[0])

    test.assertEqual(val_f32, 2.0)
    test.assertEqual(val_f64, 2.0 * 1.00000005)
    test.assertNotEqual(val_f32, val_f64)


def test_float32_mul_expression(test, device):
    """Expression (1.0 + 0.00000005) stays float64, then casts to float32 when used with float32 var."""
    result_f32 = wp.zeros(1, dtype=wp.float32, device=device)
    result_f64 = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(
        test_float32_mul_expression_kernel, dim=1, inputs=[wp.float32(2.0), result_f32, result_f64], device=device
    )

    val_f32 = float(result_f32.numpy()[0])
    val_f64 = float(result_f64.numpy()[0])

    # Addition happens in float64, then casts to float32 for multiplication
    expected_f32 = float(np.float32(2.0) * np.float32(1.0 + 0.00000005))
    test.assertEqual(val_f32, expected_f32)
    expected_f64 = 2.0 * 1.00000005
    test.assertEqual(val_f64, expected_f64)
    test.assertNotEqual(val_f32, val_f64)


def test_chained_addition_precision(test, device):
    """Tests that (1.0 + 0.00000005 + 0.00000005) doesn't collapse to 1.0 due to float32 rounding.

    The additions should happen in double precision, preserving the sum as 1.0000001,
    which remains > 1.0 even after casting to float32 before multiplication.
    """
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_chained_addition_precision_kernel, dim=1, inputs=[wp.float32(2.0), result], device=device)

    val = float(result.numpy()[0])

    # If additions happened in float32, result would be 2.0 (no change)
    # With double precision additions: 2.0 * float32(1.0000001) > 2.0
    # Expected: 2.0 * 1.0000001 ≈ 2.0000002 in float32
    expected = float(np.float32(2.0) * np.float32(1.0 + 0.00000005 + 0.00000005))
    test.assertGreater(val, 2.0, "Chained additions lost precision - collapsed to 1.0 in float32")
    test.assertAlmostEqual(val, expected, places=6)


def test_nested_constructor(test, device):
    """Inner float32 conversion is deterministic, widening to float64 is exact."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_nested_constructor_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])

    expected = float(np.float32(3.141592653589793))
    test.assertEqual(val, expected)


def test_runtime_value(test, device):
    """Runtime float32 -> float64 is a widening conversion (exact - no rounding)."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    input_val = np.float32(3.141592653589793)
    wp.launch(test_runtime_value_kernel, dim=1, inputs=[input_val, result], device=device)
    val = float(result.numpy()[0])

    expected = float(input_val)
    test.assertEqual(val, expected)


def test_zero_literal(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_zero_literal_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 0.0)


def test_negative_literal(test, device):
    """Python AST represents -3.14 as UnaryOp(USub, Constant(3.14)). Constant
    folding preserves the full-precision Python value, which then adapts to
    the target type (float64) via weak typing.
    """
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_negative_literal_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])

    expected = -3.141592653589793
    test.assertEqual(val, expected)


def test_vec3d_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_precision_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 3.141592653589793)


def test_vec3_backward_compat(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_vec3_precision_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    expected = float(np.float32(3.141592653589793))
    test.assertEqual(val, expected)


def test_mat22d_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_mat22d_precision_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 1.111111111111111)


def test_quatd_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_quatd_precision_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 0.707106781186547)


def test_transformd_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_transformd_precision_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 1.234567890123456)


def test_scalar_int_literal(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_scalar_int_literal_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 42.0)


def test_vec3d_int_literals(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_int_literals_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 6.0)


def test_vec3_int_literals(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_vec3_int_literals_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 6.0)


# TODO(GH-485): Enable once weak typing of int literals is implemented.
# def test_float64_arithmetic_int_literal(test, device):
#     result = wp.zeros(1, dtype=wp.float64, device=device)
#     wp.launch(test_float64_arithmetic_int_literal_kernel, dim=1, inputs=[result], device=device)
#     val = float(result.numpy()[0])
#     test.assertEqual(val, 6.283185307179586)
#
#
# def test_float32_arithmetic_int_literal(test, device):
#     result = wp.zeros(1, dtype=wp.float32, device=device)
#     wp.launch(test_float32_arithmetic_int_literal_kernel, dim=1, inputs=[result], device=device)
#     val = float(result.numpy()[0])
#     expected = float(np.float32(3.141592653589793) * np.float32(2.0))
#     test.assertEqual(val, expected)
#
#
# def test_int_literal_arithmetic_float64(test, device):
#     result = wp.zeros(1, dtype=wp.float64, device=device)
#     wp.launch(test_int_literal_arithmetic_float64_kernel, dim=1, inputs=[result], device=device)
#     val = float(result.numpy()[0])
#     test.assertEqual(val, 6.283185307179586)


def test_float16_constructor(test, device):
    result = wp.zeros(1, dtype=wp.float16, device=device)
    wp.launch(test_float16_constructor_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    expected = float(np.float16(3.14159))
    test.assertEqual(val, expected)


def test_float16_int_literal(test, device):
    result = wp.zeros(1, dtype=wp.float16, device=device)
    wp.launch(test_float16_int_literal_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 42.0)


def test_float16_arithmetic_precision(test, device):
    result = wp.zeros(1, dtype=wp.float16, device=device)
    wp.launch(test_float16_arithmetic_precision_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    expected = float(np.float16(1.0) * np.float16(1.0005))
    test.assertEqual(val, expected)


def test_vec3h_int_literals(test, device):
    result = wp.zeros(1, dtype=wp.float16, device=device)
    wp.launch(test_vec3h_int_literals_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 6.0)


def test_vec3d_mixed_literals(test, device):
    result = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_mixed_literals_kernel, dim=1, inputs=[result], device=device)
    vals = result.numpy()
    test.assertEqual(vals[0], 3.14)
    test.assertEqual(vals[1], 1.0)
    test.assertEqual(vals[2], 2.0)


def test_vec3_mixed_literals(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_vec3_mixed_literals_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 6.5)


def test_mat22d_mixed_literals(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_mat22d_mixed_literals_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 6.0)  # 1.5 + 4.5


def test_literal_literal_preserves_double_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_literal_literal_preserves_double_precision_kernel, dim=1, inputs=[result], device=device)

    # Calculate expected value in Python (double precision)
    expected = 3.141592653589793 * 1.00000005
    test.assertAlmostEqual(result.numpy()[0], expected, places=15)


def test_literal_literal_cast_to_float32_preserves_precision(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_literal_literal_cast_to_float32_preserves_precision_kernel, dim=1, inputs=[result], device=device)

    # Expected: double precision multiplication, then cast to float32
    expected_fp64_then_fp32 = np.float32(3.141592653589793 * 1.0001)

    # Wrong: float32 multiplication
    wrong_fp32 = np.float32(np.float32(3.141592653589793) * np.float32(1.0001))

    # Result should match the correct version
    test.assertAlmostEqual(result.numpy()[0], expected_fp64_then_fp32, places=6)

    # Verify we're actually using double precision (results should differ)
    if abs(expected_fp64_then_fp32 - wrong_fp32) > 1e-8:
        test.assertNotAlmostEqual(result.numpy()[0], wrong_fp32, places=7)


def test_literal_literal_int_stays_int(test, device):
    result = wp.zeros(1, dtype=int, device=device)
    wp.launch(test_literal_literal_int_stays_int_kernel, dim=1, inputs=[result], device=device)

    test.assertEqual(result.numpy()[0], 6)


def test_literal_literal_assignment_preserves_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_literal_literal_assignment_preserves_precision_kernel, dim=1, inputs=[result], device=device)

    expected = 3.141592653589793 * 1.00000005
    test.assertAlmostEqual(result.numpy()[0], expected, places=15)


def test_literal_literal_with_single_precision_var(test, device):
    result = wp.zeros(2, dtype=wp.float64, device=device)
    wp.launch(test_literal_literal_with_single_precision_var_kernel, dim=1, inputs=[result], device=device)

    # 'a' should be double precision
    expected_a = 3.141592653589793 * 1.00000005
    test.assertAlmostEqual(result.numpy()[0], expected_a, places=15)

    # 'c' should be float32(a) converted back to float64, plus 1.0
    expected_c = np.float64(np.float32(expected_a)) + 1.0
    test.assertAlmostEqual(result.numpy()[1], expected_c, places=15)


def test_literal_literal_complex_expression(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_literal_literal_complex_expression_kernel, dim=1, inputs=[result], device=device)

    # Calculate expected value in Python (double precision)
    expected = (3.141592653589793 * 1.00000005 + 2.718281828459045) / 1.5
    test.assertAlmostEqual(result.numpy()[0], expected, places=14)


# TODO(GH-485): Enable once weak typing of int literals is implemented.
# def test_literal_literal_mixed_int_float(test, device):
#     result = wp.zeros(1, dtype=wp.float64, device=device)
#     wp.launch(test_literal_literal_mixed_int_float_kernel, dim=1, inputs=[result], device=device)
#
#     expected = 2 * 3.141592653589793
#     test.assertAlmostEqual(result.numpy()[0], expected, places=15)


def test_constant_fold_div_by_zero(test, device):
    """Verify 1.0 / 0.0 compiles and produces +inf (constant fold edge case)."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_constant_fold_div_by_zero_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertTrue(math.isinf(val), f"Expected inf, got {val}")
    test.assertGreater(val, 0.0, f"Expected +inf, got {val}")


def test_constant_fold_neg_div_by_zero(test, device):
    """Verify -1.0 / 0.0 compiles and produces -inf (constant fold edge case)."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_constant_fold_neg_div_by_zero_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertTrue(math.isinf(val), f"Expected inf, got {val}")
    test.assertLess(val, 0.0, f"Expected -inf, got {val}")


def test_constant_fold_zero_div_by_zero(test, device):
    """Verify 0.0 / 0.0 compiles and produces NaN (constant fold edge case)."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_constant_fold_zero_div_by_zero_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertTrue(math.isnan(val), f"Expected NaN, got {val}")


def test_constant_fold_overflow(test, device):
    """Verify 1.0e308 * 2.0 compiles and produces inf (overflow edge case)."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_constant_fold_overflow_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertTrue(math.isinf(val), f"Expected inf, got {val}")
    test.assertGreater(val, 0.0, f"Expected +inf, got {val}")


def test_constant_fold_pow(test, device):
    """Verify 3.14 ** 2.0 is constant-folded in double precision."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_constant_fold_pow_kernel, dim=1, inputs=[result], device=device)
    expected = 3.141592653589793**2.0
    test.assertEqual(float(result.numpy()[0]), expected)


def test_constant_fold_neg_base_pow(test, device):
    """Verify (-1.0) ** 0.5 compiles (complex fold skipped, C++ produces NaN)."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_constant_fold_neg_base_pow_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertTrue(math.isnan(val), f"Expected NaN, got {val}")


def test_typed_constructor_accepts_literals(test, device):
    result = wp.zeros(1, dtype=wp.vec3d, device=device)
    wp.launch(test_typed_constructor_accepts_literals_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], [1.0, 2.0, 3.0])


def test_sametypes_accepts_literal_with_variable(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_sametypes_accepts_literal_with_variable_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], 8.0)


def test_func_overload_resolution(test, device):
    """Verify literal arguments dispatch correctly to @wp.func overloaded functions."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_func_overload_resolution_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 3.141592653589793 * 2.0)


def test_func_overload_default(test, device):
    """Verify bare literal resolves to float32 overload (default) when both f32/f64 exist."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_func_overload_default_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 1.5)


def test_builtin_default_resolution(test, device):
    """Verify wp.sin(1.0) with bare literal defaults to float32."""
    result_f32 = wp.zeros(1, dtype=wp.float32, device=device)
    result_f64 = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_builtin_default_resolution_kernel, dim=1, inputs=[result_f32, result_f64], device=device)

    # Bare literal: sin(float32(1.0))
    expected_f32 = float(np.float32(np.sin(np.float32(1.0))))
    test.assertAlmostEqual(float(result_f32.numpy()[0]), expected_f32, places=6)
    # Explicit float64: sin(float64(1.0))
    expected_f64 = np.sin(1.0)
    test.assertAlmostEqual(float(result_f64.numpy()[0]), expected_f64, places=14)


def test_array_store_literal(test, device):
    """Verify storing a literal to a wp.float64 array preserves precision."""
    arr = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_array_store_literal_kernel, dim=1, inputs=[arr], device=device)
    val = float(arr.numpy()[0])
    test.assertEqual(val, 3.141592653589793)


def test_sibling_generic_retyping(test, device):
    """Verify a literal paired with a typed arg in a generic function adapts to its type."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_sibling_generic_retyping_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 3.141592653589793)


def test_warp_constant_float64(test, device):
    """Module constants (wp.PI) are resolved to weak floats that adapt to float64."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_warp_constant_float64_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 3.141592653589793)


def test_negative_warp_constant(test, device):
    """Negated module constants (-wp.PI) preserve full precision via UnaryOp folding."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_negative_warp_constant_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, -3.141592653589793)


def test_warp_constant_constructor(test, device):
    """Module constants in typed constructors (vec3d) preserve precision of each component."""
    result = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(test_warp_constant_constructor_kernel, dim=1, inputs=[result], device=device)
    vals = result.numpy()
    test.assertEqual(float(vals[0]), 3.141592653589793)  # wp.PI
    test.assertEqual(float(vals[1]), 2.718281828459045)  # wp.E
    test.assertEqual(float(vals[2]), 6.283185307179586)  # wp.TAU


def test_warp_constant_arithmetic(test, device):
    """Arithmetic on module constants (wp.PI * 2.0) matches wp.TAU."""
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_warp_constant_arithmetic_kernel, dim=1, inputs=[result], device=device)
    error = float(result.numpy()[0])
    test.assertEqual(error, 0.0)


def test_int_literal_float64_constructor(test, device):
    """Int literals in float64 constructors (vec3d(1, 2, 3)) produce exact values."""
    result = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(test_int_literal_float64_constructor_kernel, dim=1, inputs=[result], device=device)
    vals = result.numpy()
    test.assertEqual(float(vals[0]), 1.0)
    test.assertEqual(float(vals[1]), 2.0)
    test.assertEqual(float(vals[2]), 3.0)


def test_adjoint_warp_constant(test, device):
    """Adjoint of out = wp.PI * inp must use float64 PI, not float32."""
    inp = wp.array([1.0], dtype=wp.float64, device=device, requires_grad=True)
    out = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(test_adjoint_warp_constant_kernel, dim=1, inputs=[inp, out], device=device)

    tape.backward(loss=out)

    grad_val = float(inp.grad.numpy()[0])
    # If the adjoint incorrectly used float32 PI, this would be
    # 3.1415927410125732 (float32 PI widened to float64).
    test.assertEqual(grad_val, 3.141592653589793)


def test_scalar_mul_literal_vec3d(test, device):
    result = wp.zeros(1, dtype=wp.vec3d, device=device)
    wp.launch(test_scalar_mul_literal_vec3d_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], [2.0, 4.0, 6.0])


def test_sametypes_literal_first(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_sametypes_literal_first_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], 9.0)


def test_builtin_result_preserves_const_expr(test, device):
    """wp.sqrt(3.0) * 0.5 stays weakly typed and auto-casts to float16."""
    result = wp.zeros(1, dtype=wp.float16, device=device)
    wp.launch(test_builtin_result_preserves_const_expr_kernel, dim=1, inputs=[result], device=device)
    expected = 2.0 * (3.0**0.5) * 0.5
    np.testing.assert_allclose(float(result.numpy()[0]), expected, rtol=1e-2)


def test_float_annotation_is_strongly_typed_float32(test, device):
    """float annotation produces float32, which rounds 1.00000005."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_float_annotation_is_strongly_typed_float32_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(result.numpy()[0], np.float32(1.00000005))


def test_vector_dtype_float_is_float32(test, device):
    """wp.types.vector(..., dtype=float) produces float32 elements."""
    result = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(test_vector_dtype_float_is_float32_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], [1.0, 2.0, 3.0])


@wp.kernel
def test_multi_assign_weak_to_strong_kernel(result: wp.array(dtype=wp.float64)):
    """Multi-assignment: weak float literals cast to match existing strong-float symbols."""
    x = wp.float64(0.0)
    y = wp.float64(0.0)
    x, y = 3.141592653589793, 2.718281828459045
    result[0] = x
    result[1] = y


def test_multi_assign_weak_to_strong(test, device):
    result = wp.zeros(2, dtype=wp.float64, device=device)
    wp.launch(test_multi_assign_weak_to_strong_kernel, dim=1, inputs=[result], device=device)
    expected_pi = 3.141592653589793
    expected_e = 2.718281828459045
    np.testing.assert_allclose(result.numpy()[0], expected_pi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.numpy()[1], expected_e, rtol=0.0, atol=0.0)


@wp.kernel
def test_augassign_weak_to_strong_kernel(result: wp.array(dtype=wp.float64)):
    """Augmented assignment: weak float result cast to match existing strong-float symbol."""
    x = wp.float64(3.0)
    x += 0.141592653589793  # RHS is weak float; result should stay float64
    result[0] = x


def test_augassign_weak_to_strong(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_augassign_weak_to_strong_kernel, dim=1, inputs=[result], device=device)
    expected = 3.0 + 0.141592653589793
    np.testing.assert_allclose(result.numpy()[0], expected, rtol=0.0, atol=0.0)


@wp.kernel
def test_where_weak_float_kernel(result: wp.array(dtype=wp.float64)):
    """wp.where with weak float branches adapts to typed context."""
    cond = True
    val = wp.where(cond, 3.141592653589793, 2.718281828459045)
    result[0] = val


def test_where_weak_float(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_where_weak_float_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], 3.141592653589793, rtol=0.0, atol=0.0)


@wp.kernel
def test_comparison_weak_strong_kernel(result: wp.array(dtype=wp.int32)):
    """Comparison operators cast weak float to match strong float."""
    x = wp.float64(1.5)
    if x > 1.0:
        result[0] = 1
    else:
        result[0] = 0
    if 0.5 < x:
        result[1] = 1
    else:
        result[1] = 0


def test_comparison_weak_strong(test, device):
    result = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(test_comparison_weak_strong_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(result.numpy()[0], 1)
    test.assertEqual(result.numpy()[1], 1)


@wp.kernel
def test_augmented_array_store_kernel(arr: wp.array(dtype=wp.float64)):
    """Augmented array store: arr[i] += weak_float preserves array dtype precision."""
    arr[0] += 0.141592653589793


def test_augmented_array_store(test, device):
    arr = wp.array([3.0], dtype=wp.float64, device=device)
    wp.launch(test_augmented_array_store_kernel, dim=1, inputs=[arr], device=device)
    expected = 3.0 + 0.141592653589793
    np.testing.assert_allclose(arr.numpy()[0], expected, rtol=0.0, atol=0.0)


@wp.func
def _user_func_kwarg(x: wp.float32) -> wp.float32:
    return x + wp.float32(1.0)


@wp.kernel
def test_user_func_kwarg_weak_float_kernel(result: wp.array(dtype=wp.float32)):
    """User function called with a weak float keyword argument adapts to float32."""
    result[0] = _user_func_kwarg(x=2.5)


def test_user_func_kwarg_weak_float(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_user_func_kwarg_weak_float_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], 3.5, rtol=1e-6)


class TestConstantPrecision(unittest.TestCase):
    """Test suite for constant precision preservation."""

    def test_type_constructor_breaks_const_expr_chain(self):
        """wp.pow(float64_var, float32_var) errors — different strong types."""

        @wp.kernel(module="unique")
        def kernel():
            x = wp.float64(1.0)
            y = wp.float32(2.0)
            z = wp.pow(x, y)

        with self.assertRaisesRegex(RuntimeError, r"Input types must be the same"):
            wp.launch(kernel, dim=1, device="cpu")

    @unittest.expectedFailure
    def test_weak_int_float_interaction(self):
        """int + float promotion — not yet implemented."""

        @wp.kernel
        def kernel(result: wp.array(dtype=wp.float64)):
            x = 1 + 2.0
            result[0] = wp.float64(x)

        result = wp.zeros(1, dtype=wp.float64, device="cpu")
        wp.launch(kernel, dim=1, inputs=[result], device="cpu")
        self.assertEqual(float(result.numpy()[0]), 3.0)

    @unittest.expectedFailure
    def test_negative_int_literal(self):
        """int64 negative literal range — not yet implemented."""

        @wp.kernel
        def kernel(
            result_small: wp.array(dtype=wp.int64),
            result_large: wp.array(dtype=wp.int64),
        ):
            result_small[0] = wp.int64(-100)
            result_large[0] = wp.int64(-9223372036854775808)

        result_small = wp.zeros(1, dtype=wp.int64, device="cpu")
        result_large = wp.zeros(1, dtype=wp.int64, device="cpu")
        wp.launch(kernel, dim=1, inputs=[result_small, result_large], device="cpu")
        self.assertEqual(int(result_small.numpy()[0]), -100)
        self.assertEqual(int(result_large.numpy()[0]), -9223372036854775808)

    def test_typed_constructor_rejects_mismatched_variable(self):
        """Verify that passing a float64 variable to a float32 constructor errors at codegen time."""

        @wp.kernel(module="unique")
        def kernel():
            x = wp.float64(1.0)
            v = wp.vec3f(x, x, x)

        with self.assertRaisesRegex(RuntimeError, r"(multiple precisions|expected to be of the type)"):
            wp.launch(kernel, dim=1, device="cpu")

    def test_matrix_constructor_rejects_mismatched_variable(self):
        """Verify that passing a float64 variable to a float32 matrix constructor errors."""
        mat22f = wp.types.matrix((2, 2), wp.float32)

        @wp.kernel(module="unique")
        def kernel():
            x = wp.float64(1.0)
            m = mat22f(x, x, x, x)

        with self.assertRaisesRegex(RuntimeError, r"(multiple precisions|expected to be of the type)"):
            wp.launch(kernel, dim=1, device="cpu")

    def test_matrix_fill_rejects_mismatched_variable(self):
        """Verify that filling a float32 matrix with a float64 variable errors."""
        mat22f = wp.types.matrix((2, 2), wp.float32)

        @wp.kernel(module="unique")
        def kernel():
            x = wp.float64(1.0)
            m = mat22f(x)

        with self.assertRaisesRegex(RuntimeError, r"expected to be of the type"):
            wp.launch(kernel, dim=1, device="cpu")

    def test_sametypes_rejects_mismatched_variables(self):
        """Verify that sametypes functions reject strongly-typed variables with different precisions."""

        @wp.kernel(module="unique")
        def kernel():
            x = wp.float32(1.0)
            y = wp.float64(2.0)
            z = wp.pow(x, y)

        with self.assertRaisesRegex(RuntimeError, r"Input types must be the same"):
            wp.launch(kernel, dim=1, device="cpu")

    def test_scalar_infer_type_empty_raises(self):
        """scalar_infer_type raises RuntimeError (not StopIteration) when no scalar types found."""
        from warp._src.builtins import scalar_infer_type  # noqa: PLC0415

        # str is not a scalar type, not a compound type, not a float
        with self.assertRaises(RuntimeError):
            scalar_infer_type((str,))


devices = get_test_devices()

add_function_test(TestConstantPrecision, "test_float64_precision", test_float64_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_float32_backward_compat", test_float32_backward_compat, devices=devices)
add_function_test(TestConstantPrecision, "test_literal_precision", test_literal_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_large_precision_value", test_large_precision_value, devices=devices)
add_function_test(TestConstantPrecision, "test_int64_cast", test_int64_cast, devices=devices)
add_function_test(TestConstantPrecision, "test_float64_arithmetic", test_float64_arithmetic, devices=devices)
add_function_test(TestConstantPrecision, "test_float32_mul_literal", test_float32_mul_literal, devices=devices)
add_function_test(TestConstantPrecision, "test_float32_mul_variable", test_float32_mul_variable, devices=devices)
add_function_test(TestConstantPrecision, "test_float32_mul_expression", test_float32_mul_expression, devices=devices)
add_function_test(
    TestConstantPrecision, "test_chained_addition_precision", test_chained_addition_precision, devices=devices
)
add_function_test(TestConstantPrecision, "test_nested_constructor", test_nested_constructor, devices=devices)
add_function_test(TestConstantPrecision, "test_runtime_value", test_runtime_value, devices=devices)
add_function_test(TestConstantPrecision, "test_zero_literal", test_zero_literal, devices=devices)
add_function_test(TestConstantPrecision, "test_negative_literal", test_negative_literal, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3d_precision", test_vec3d_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3_backward_compat", test_vec3_backward_compat, devices=devices)
add_function_test(TestConstantPrecision, "test_mat22d_precision", test_mat22d_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_quatd_precision", test_quatd_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_transformd_precision", test_transformd_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_scalar_int_literal", test_scalar_int_literal, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3d_int_literals", test_vec3d_int_literals, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3_int_literals", test_vec3_int_literals, devices=devices)
# TODO(GH-485): Enable once weak typing of int literals is implemented.
# add_function_test(
#     TestConstantPrecision, "test_float64_arithmetic_int_literal", test_float64_arithmetic_int_literal, devices=devices
# )
# add_function_test(
#     TestConstantPrecision, "test_float32_arithmetic_int_literal", test_float32_arithmetic_int_literal, devices=devices
# )
# add_function_test(
#     TestConstantPrecision, "test_int_literal_arithmetic_float64", test_int_literal_arithmetic_float64, devices=devices
# )
add_function_test(TestConstantPrecision, "test_float16_constructor", test_float16_constructor, devices=devices)
add_function_test(TestConstantPrecision, "test_float16_int_literal", test_float16_int_literal, devices=devices)
add_function_test(
    TestConstantPrecision, "test_float16_arithmetic_precision", test_float16_arithmetic_precision, devices=devices
)
add_function_test(TestConstantPrecision, "test_vec3h_int_literals", test_vec3h_int_literals, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3d_mixed_literals", test_vec3d_mixed_literals, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3_mixed_literals", test_vec3_mixed_literals, devices=devices)
add_function_test(TestConstantPrecision, "test_mat22d_mixed_literals", test_mat22d_mixed_literals, devices=devices)
add_function_test(
    TestConstantPrecision,
    "test_literal_literal_preserves_double_precision",
    test_literal_literal_preserves_double_precision,
    devices=devices,
)
add_function_test(
    TestConstantPrecision,
    "test_literal_literal_cast_to_float32_preserves_precision",
    test_literal_literal_cast_to_float32_preserves_precision,
    devices=devices,
)
add_function_test(
    TestConstantPrecision, "test_literal_literal_int_stays_int", test_literal_literal_int_stays_int, devices=devices
)
add_function_test(
    TestConstantPrecision,
    "test_literal_literal_assignment_preserves_precision",
    test_literal_literal_assignment_preserves_precision,
    devices=devices,
)
add_function_test(
    TestConstantPrecision,
    "test_literal_literal_with_single_precision_var",
    test_literal_literal_with_single_precision_var,
    devices=devices,
)
add_function_test(
    TestConstantPrecision,
    "test_literal_literal_complex_expression",
    test_literal_literal_complex_expression,
    devices=devices,
)
# TODO(GH-485): Enable once weak typing of int literals is implemented.
# add_function_test(
#     TestConstantPrecision, "test_literal_literal_mixed_int_float", test_literal_literal_mixed_int_float, devices=devices
# )
add_function_test(
    TestConstantPrecision, "test_constant_fold_div_by_zero", test_constant_fold_div_by_zero, devices=devices
)
add_function_test(
    TestConstantPrecision, "test_constant_fold_neg_div_by_zero", test_constant_fold_neg_div_by_zero, devices=devices
)
add_function_test(
    TestConstantPrecision, "test_constant_fold_zero_div_by_zero", test_constant_fold_zero_div_by_zero, devices=devices
)
add_function_test(TestConstantPrecision, "test_constant_fold_overflow", test_constant_fold_overflow, devices=devices)
add_function_test(TestConstantPrecision, "test_constant_fold_pow", test_constant_fold_pow, devices=devices)
add_function_test(
    TestConstantPrecision, "test_constant_fold_neg_base_pow", test_constant_fold_neg_base_pow, devices=devices
)
add_function_test(
    TestConstantPrecision,
    "test_typed_constructor_accepts_literals",
    test_typed_constructor_accepts_literals,
    devices=devices,
)
add_function_test(
    TestConstantPrecision,
    "test_sametypes_accepts_literal_with_variable",
    test_sametypes_accepts_literal_with_variable,
    devices=devices,
)
add_function_test(
    TestConstantPrecision, "test_func_overload_resolution", test_func_overload_resolution, devices=devices
)
add_function_test(TestConstantPrecision, "test_func_overload_default", test_func_overload_default, devices=devices)
add_function_test(
    TestConstantPrecision, "test_builtin_default_resolution", test_builtin_default_resolution, devices=devices
)
add_function_test(TestConstantPrecision, "test_array_store_literal", test_array_store_literal, devices=devices)
add_function_test(
    TestConstantPrecision, "test_sibling_generic_retyping", test_sibling_generic_retyping, devices=devices
)
add_function_test(TestConstantPrecision, "test_warp_constant_float64", test_warp_constant_float64, devices=devices)
add_function_test(TestConstantPrecision, "test_negative_warp_constant", test_negative_warp_constant, devices=devices)
add_function_test(
    TestConstantPrecision, "test_warp_constant_constructor", test_warp_constant_constructor, devices=devices
)
add_function_test(
    TestConstantPrecision, "test_warp_constant_arithmetic", test_warp_constant_arithmetic, devices=devices
)
add_function_test(
    TestConstantPrecision, "test_int_literal_float64_constructor", test_int_literal_float64_constructor, devices=devices
)
add_function_test(TestConstantPrecision, "test_adjoint_warp_constant", test_adjoint_warp_constant, devices=devices)
add_function_test(
    TestConstantPrecision, "test_scalar_mul_literal_vec3d", test_scalar_mul_literal_vec3d, devices=devices
)
add_function_test(TestConstantPrecision, "test_sametypes_literal_first", test_sametypes_literal_first, devices=devices)
add_function_test(
    TestConstantPrecision,
    "test_builtin_result_preserves_const_expr",
    test_builtin_result_preserves_const_expr,
    devices=devices,
)
add_function_test(
    TestConstantPrecision,
    "test_float_annotation_is_strongly_typed_float32",
    test_float_annotation_is_strongly_typed_float32,
    devices=devices,
)
add_function_test(
    TestConstantPrecision, "test_vector_dtype_float_is_float32", test_vector_dtype_float_is_float32, devices=devices
)
add_function_test(
    TestConstantPrecision, "test_multi_assign_weak_to_strong", test_multi_assign_weak_to_strong, devices=devices
)
add_function_test(
    TestConstantPrecision, "test_augassign_weak_to_strong", test_augassign_weak_to_strong, devices=devices
)
add_function_test(TestConstantPrecision, "test_where_weak_float", test_where_weak_float, devices=devices)
add_function_test(TestConstantPrecision, "test_comparison_weak_strong", test_comparison_weak_strong, devices=devices)
add_function_test(TestConstantPrecision, "test_augmented_array_store", test_augmented_array_store, devices=devices)
add_function_test(
    TestConstantPrecision, "test_user_func_kwarg_weak_float", test_user_func_kwarg_weak_float, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
