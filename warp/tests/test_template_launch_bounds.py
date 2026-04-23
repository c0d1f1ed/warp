# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for templated launch_bounds_t<N>.

Verifies that kernel_dim inference from wp.tid() produces correct thread
coordinates across all supported launch configurations:

- Regular (non-tiled) launches with 1D/2D/3D/4D wp.tid()
- launch_tiled with wp.tid() ignoring the trailing block_dim dimension
- launch_tiled with wp.tid() explicitly covering all dimensions
- Manual tiled launches (wp.launch with block_dim baked into dim)
- Untiled launches that use tile functions
- Launch-dim rank validation: under both ``wp.config.optimize_tid``
  states — False (the default) pads and accepts any rank
  :math:`\\le` ``LAUNCH_MAX_DIMS``, True enforces a strict rank check
  that raises ``ValueError``
- Kernels that never call wp.tid()

Module-scope kernels compile at import time under the prevailing
``optimize_tid`` value (default False). Tests that need the opposite
mode decorate a function-local kernel after flipping the flag via the
``_optimize_tid`` context manager.
"""

import contextlib
import unittest
from types import SimpleNamespace

import numpy as np

import warp as wp

# Private helpers live under warp._src; tests import them directly.
from warp._src.context import _build_rank_error, _prepare_launch_dim, _tid_unpack
from warp._src.types import LAUNCH_MAX_DIMS
from warp.tests.unittest_utils import *

BLOCK_DIM = 64
TILE_M = wp.constant(8)
TILE_N = wp.constant(4)


@contextlib.contextmanager
def _optimize_tid(enabled):
    """Temporarily set ``wp.config.optimize_tid``.

    Must wrap the ``@wp.kernel`` decorator for any kernel whose
    ``kernel_dim`` needs to reflect the target mode (kernel_dim is locked in
    at decoration time).
    """
    old = wp.config.optimize_tid
    wp.config.optimize_tid = enabled
    try:
        yield
    finally:
        wp.config.optimize_tid = old


# ============================================================================
# Regular (non-tiled) launches
# ============================================================================


@wp.kernel
def regular_1d_kernel(out: wp.array(dtype=int)):
    i = wp.tid()
    out[i] = i


def test_regular_1d(test, device):
    N = 128
    out = wp.zeros(N, dtype=int, device=device)
    wp.launch(regular_1d_kernel, dim=N, inputs=[out], device=device)
    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(N))


@wp.kernel
def regular_2d_kernel(out: wp.array2d(dtype=int), M: int, N: int):
    i, j = wp.tid()
    out[i, j] = i * N + j


def test_regular_2d(test, device):
    M, N = 8, 16
    out = wp.zeros((M, N), dtype=int, device=device)
    wp.launch(regular_2d_kernel, dim=[M, N], inputs=[out, M, N], device=device)
    result = out.numpy()
    expected = np.arange(M * N).reshape(M, N)
    np.testing.assert_array_equal(result, expected)


@wp.kernel
def regular_3d_kernel(out: wp.array3d(dtype=int), M: int, N: int, K: int):
    i, j, k = wp.tid()
    out[i, j, k] = i * N * K + j * K + k


def test_regular_3d(test, device):
    M, N, K = 4, 8, 16
    out = wp.zeros((M, N, K), dtype=int, device=device)
    wp.launch(regular_3d_kernel, dim=[M, N, K], inputs=[out, M, N, K], device=device)
    result = out.numpy()
    expected = np.arange(M * N * K).reshape(M, N, K)
    np.testing.assert_array_equal(result, expected)


@wp.kernel
def regular_4d_kernel(out: wp.array(dtype=int), A: int, B: int, C: int, D: int):
    i, j, k, l = wp.tid()
    out[i * B * C * D + j * C * D + k * D + l] = i * B * C * D + j * C * D + k * D + l


def test_regular_4d(test, device):
    A, B, C, D = 2, 3, 4, 5
    total = A * B * C * D
    out = wp.zeros(total, dtype=int, device=device)
    wp.launch(regular_4d_kernel, dim=[A, B, C, D], inputs=[out, A, B, C, D], device=device)
    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(total))


# ============================================================================
# Launch-dim rank validation: mismatch raises ValueError with migration options
# ============================================================================


def test_launch_dim_over_rank_kernel_dim_1_error(test, device):
    """optimize_tid=True: dim rank > kernel_dim=1 raises ValueError listing all 4 intent options."""
    with _optimize_tid(True):

        @wp.kernel
        def _over_1d_kernel(out: wp.array(dtype=int)):
            i = wp.tid()
            out[i] = i

        N = 3
        out = wp.zeros(N * N, dtype=int, device=device)
        with test.assertRaises(ValueError) as cm:
            wp.launch(_over_1d_kernel, dim=(N, N), inputs=[out], device=device)
        msg = str(cm.exception)
        test.assertIn("kernel_dim=1", msg)
        test.assertIn("flat linear index over 9 threads: launch with dim=9", msg)
        test.assertIn("first-dim index (0..2, no repetition): launch with dim=3", msg)
        test.assertIn("first-dim index with repetition: unpack as `i, _ = wp.tid()`", msg)
        test.assertIn("per-dim indexing: unpack as `i, j = wp.tid()`", msg)


def test_launch_dim_over_rank_kernel_dim_2_error(test, device):
    """optimize_tid=True: dim rank > kernel_dim=2 omits launch-only options (they'd require a kernel edit too)."""
    with _optimize_tid(True):

        @wp.kernel
        def _over_2d_kernel(out: wp.array2d(dtype=int), M: int, N: int):
            i, j = wp.tid()
            out[i, j] = i * N + j

        M, N, K = 2, 3, 4
        out = wp.zeros((M, N * K), dtype=int, device=device)
        with test.assertRaises(ValueError) as cm:
            wp.launch(_over_2d_kernel, dim=(M, N, K), inputs=[out, M, N * K], device=device)
        msg = str(cm.exception)
        test.assertIn("kernel_dim=2", msg)
        # Guard: misleading launch-only fixes must NOT appear when kernel_dim >= 2.
        test.assertNotIn("flat linear", msg)
        test.assertNotIn(f"launch with dim={M * N * K}", msg)
        test.assertNotIn("no repetition", msg)
        # Kernel-side fixes still appear.
        test.assertIn("first-dim index with repetition: unpack as `i, _, _ = wp.tid()`", msg)
        test.assertIn("per-dim indexing: unpack as `i, j, k = wp.tid()`", msg)


def test_launch_dim_under_rank_error(test, device):
    """optimize_tid=True: dim rank < kernel_dim raises ValueError listing the two fix options."""
    with _optimize_tid(True):

        @wp.kernel
        def _under_2d_kernel(out: wp.array2d(dtype=int), M: int, N: int):
            i, j = wp.tid()
            out[i, j] = i * N + j

        N = 10
        out = wp.zeros((N, 1), dtype=int, device=device)
        with test.assertRaises(ValueError) as cm:
            wp.launch(_under_2d_kernel, dim=N, inputs=[out, N, 1], device=device)
        msg = str(cm.exception)
        test.assertIn("kernel_dim=2", msg)
        test.assertIn("keep 2-D kernel, launch with matching rank: dim=(10, 1)", msg)
        test.assertIn("change kernel to unpack 1 variable: `i = wp.tid()`", msg)


# ============================================================================
# No-tid kernel with multi-dimensional launch
# ============================================================================


@wp.kernel
def no_tid_kernel(out: wp.array(dtype=float), val: float):
    wp.atomic_add(out, 0, val)


def test_no_tid_kernel_multidim(test, device):
    """Kernel without wp.tid() launched with multi-dim should not crash."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(no_tid_kernel, dim=[2, 3], inputs=[out, 1.0], device=device)
    # 6 threads each atomically add 1.0
    np.testing.assert_allclose(out.numpy()[0], 6.0)


# ============================================================================
# launch_tiled: wp.tid() ignores trailing block_dim dimension
# ============================================================================


@wp.kernel
def tiled_1d_kernel(A: wp.array(dtype=float), B: wp.array(dtype=float)):
    i = wp.tid()
    a = wp.tile_load(A, shape=TILE_N, offset=i * TILE_N)
    wp.tile_store(B, a, offset=i * TILE_N)


def test_tiled_1d(test, device):
    N = TILE_N * 5
    A = wp.full(N, 42.0, dtype=float, device=device)
    B = wp.zeros(N, dtype=float, device=device)
    wp.launch_tiled(tiled_1d_kernel, dim=[int(N / TILE_N)], inputs=[A, B], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_array_equal(B.numpy(), A.numpy())


@wp.kernel
def tiled_2d_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    a = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(B, a, offset=(i * TILE_M, j * TILE_N))


def test_tiled_2d(test, device):
    M = TILE_M * 3
    N = TILE_N * 2
    rng = np.random.default_rng(42)
    A_np = rng.random((M, N)).astype(np.float32)
    A = wp.array(A_np, device=device)
    B = wp.zeros((M, N), dtype=float, device=device)
    wp.launch_tiled(
        tiled_2d_kernel,
        dim=[int(M / TILE_M), int(N / TILE_N)],
        inputs=[A, B],
        block_dim=BLOCK_DIM,
        device=device,
    )
    np.testing.assert_allclose(B.numpy(), A_np, rtol=1e-5)


# ============================================================================
# launch_tiled: kernel doesn't call wp.tid() (e.g. pure tile operations)
# ============================================================================


TILE_SIZE = wp.constant(32)


@wp.kernel
def tiled_no_tid_kernel(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float)):
    a = wp.tile_load(A, shape=(TILE_SIZE, TILE_SIZE))
    wp.tile_store(B, a)


def test_tiled_no_tid(test, device):
    """Tiled kernel with no wp.tid() should work with multi-dim launch."""
    N = 32
    rng = np.random.default_rng(123)
    A_np = rng.random((N, N)).astype(np.float32)
    A = wp.array(A_np, device=device)
    B = wp.zeros((N, N), dtype=float, device=device)
    wp.launch_tiled(tiled_no_tid_kernel, dim=[1, 1], inputs=[A, B], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_allclose(B.numpy(), A_np, rtol=1e-5)


# ============================================================================
# Manual tiled launch: wp.launch with block_dim baked into dim
# ============================================================================


@wp.kernel
def manual_tiled_kernel(out: wp.array3d(dtype=int), M: int, N: int, BD: int):
    i, j, t = wp.tid()
    out[i, j, t] = i * N * BD + j * BD + t


def test_manual_tiled(test, device):
    M, N, BD = 4, 8, 32
    out = wp.zeros((M, N, BD), dtype=int, device=device)
    wp.launch(manual_tiled_kernel, dim=[M, N, BD], inputs=[out, M, N, BD], block_dim=BD, device=device)
    result = out.numpy()
    expected = np.arange(M * N * BD).reshape(M, N, BD)
    np.testing.assert_array_equal(result, expected)


# ============================================================================
# launch_tiled dimension mismatch error
# ============================================================================


def test_tiled_dim_mismatch_error(test, device):
    """optimize_tid=True: launch_tiled raises ValueError with migration hint when kernel_dim is incompatible with dim."""
    with _optimize_tid(True):

        @wp.kernel
        def _tiled_4d_kernel(out: wp.array(dtype=float)):
            _i, _j, _k, _l = wp.tid()
            pass

        with test.assertRaises(ValueError) as cm:
            wp.launch_tiled(
                _tiled_4d_kernel,
                dim=[2, 3],
                inputs=[wp.zeros(1, dtype=float, device=device)],
                block_dim=BLOCK_DIM,
                device=device,
            )
        msg = str(cm.exception)
        test.assertIn("kernel_dim=4", msg)
        # tiled hint should appear because this is a tiled launch
        test.assertIn("For launch_tiled, dim may also match kernel_dim - 1", msg)


# ============================================================================
# Launch.set_dim rank validation
# ============================================================================


def test_set_dim_matching_rank(test, device):
    """Launch.set_dim() with a compatible rank should succeed."""
    N = 64
    out = wp.zeros(N, dtype=int, device=device)
    launch = wp.launch(regular_1d_kernel, dim=N, inputs=[out], device=device, record_cmd=True)

    # Re-set with a new 1D dim — accepted under default (padded) and optimize_tid.
    launch.set_dim(32)
    launch.launch()

    result = out.numpy()
    # Only the first 32 entries were written; rest remain 0.
    np.testing.assert_array_equal(result[:32], np.arange(32))
    np.testing.assert_array_equal(result[32:], np.zeros(N - 32))


def test_set_dim_rank_mismatch_error(test, device):
    """optimize_tid=True: Launch.set_dim() with rank != kernel_dim raises ValueError."""
    with _optimize_tid(True):

        @wp.kernel
        def _set_dim_1d_kernel(out: wp.array(dtype=int)):
            i = wp.tid()
            out[i] = i

        N = 64
        out = wp.zeros(N, dtype=int, device=device)
        launch = wp.launch(_set_dim_1d_kernel, dim=N, inputs=[out], device=device, record_cmd=True)

        with test.assertRaises(ValueError) as cm:
            launch.set_dim([8, 8])
        msg = str(cm.exception)
        test.assertIn("kernel_dim=1", msg)
        test.assertIn("flat linear index over 64 threads", msg)


def test_set_dim_preserves_tiled_flag(test, device):
    """optimize_tid=True: Launch.set_dim() on a recorded tiled launch preserves bounds.tiled=True.

    With optimize_tid off (default) ``bounds.tiled`` is always False (block_dim is folded
    into the shape and launch_coord doesn't divide), so this assertion is
    specific to the strict-mode path.
    """
    with _optimize_tid(True):

        @wp.kernel
        def _set_tiled_1d_kernel(A: wp.array(dtype=float), B: wp.array(dtype=float)):
            i = wp.tid()
            a = wp.tile_load(A, shape=TILE_N, offset=i * TILE_N)
            wp.tile_store(B, a, offset=i * TILE_N)

        N = TILE_N * 5
        A = wp.full(N, 42.0, dtype=float, device=device)
        B = wp.zeros(N, dtype=float, device=device)
        launch = wp.launch_tiled(
            _set_tiled_1d_kernel,
            dim=[int(N / TILE_N)],
            inputs=[A, B],
            block_dim=BLOCK_DIM,
            device=device,
            record_cmd=True,
        )
        # set_dim should accept user-rank dim (len == kernel_dim for this kernel)
        launch.set_dim([int(N / TILE_N)])
        test.assertTrue(launch.bounds.tiled, "tiled flag should be preserved after set_dim")
        launch.launch()
        np.testing.assert_array_equal(B.numpy(), A.numpy())


def test_set_dim_tiled_explicit_block_axis(test, device):
    """optimize_tid=True: replaying a recorded launch_tiled whose kernel unpacks the block axis.

    When ``wp.tid()`` covers the block_dim axis (kernel_dim == len(dim) + 1),
    ``_construct_tiled_bounds`` appends block_dim to the shape and leaves
    ``bounds.tiled = False``. ``set_dim`` must still route through the tiled
    path (via ``Launch.tiled``); the non-tiled path would raise because
    ``len(dim) == kernel_dim - 1``.
    """
    with _optimize_tid(True):

        @wp.kernel
        def _tiled_block_axis_kernel(out: wp.array(dtype=int)):
            _i, _t = wp.tid()

        N = 4
        out = wp.zeros(N * BLOCK_DIM, dtype=int, device=device)
        launch = wp.launch_tiled(
            _tiled_block_axis_kernel,
            dim=[N],
            inputs=[out],
            block_dim=BLOCK_DIM,
            device=device,
            record_cmd=True,
        )
        # Block axis is baked into the shape; bounds.tiled stays False.
        test.assertFalse(launch.bounds.tiled)
        test.assertTrue(launch.tiled)
        # Regression: set_dim previously consulted bounds.tiled and raised
        # on this shape. It must accept the same user-rank dim instead.
        launch.set_dim([N])
        launch.launch()


# ============================================================================
# Unit tests for launch-dim preparation helpers
# ============================================================================


def _stub_kernel(key: str = "foo", *, kernel_dim: int = 1, tid_arity: int | None = None):
    """Produce a minimal object matching the attributes _prepare_launch_dim reads.

    ``tid_arity`` defaults to ``kernel_dim`` (the common case where the kernel
    has at least one wp.tid() call). Pass ``tid_arity=0`` to simulate a kernel
    with no wp.tid() calls at all.
    """
    if tid_arity is None:
        tid_arity = kernel_dim
    adj = SimpleNamespace(kernel_dim=kernel_dim, tid_arity=tid_arity)
    return SimpleNamespace(key=key, adj=adj)


class TestTidUnpack(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(_tid_unpack(1), "i = wp.tid()")

    def test_two(self):
        self.assertEqual(_tid_unpack(2), "i, j = wp.tid()")

    def test_three(self):
        self.assertEqual(_tid_unpack(3), "i, j, k = wp.tid()")

    def test_four(self):
        self.assertEqual(_tid_unpack(4), "i, j, k, l = wp.tid()")

    def test_beyond_supported(self):
        # degrades gracefully when n > len(_TID_NAMES)
        self.assertEqual(_tid_unpack(5), "... = wp.tid()  # 5 variables")

    def test_zero(self):
        # guards against pathological input; real callers always pass n >= 1
        self.assertEqual(_tid_unpack(0), "... = wp.tid()  # 0 variables")


class TestBuildRankError(unittest.TestCase):
    # _build_rank_error is only invoked when wp.config.optimize_tid is True.
    # Pin the flag so direct-call tests reflect that.
    def setUp(self):
        self._old_optimize_tid = wp.config.optimize_tid
        wp.config.optimize_tid = True

    def tearDown(self):
        wp.config.optimize_tid = self._old_optimize_tid

    def test_over_rank_kernel_dim_1_lists_all_four_options(self):
        msg = _build_rank_error((3, 3), kernel_dim=1, kernel=_stub_kernel("foo"), tiled=False)
        self.assertIn("Launch dim (3, 3) has rank 2", msg)
        self.assertIn("kernel 'foo'", msg)
        self.assertIn("kernel_dim=1", msg)
        self.assertIn("flat linear index over 9 threads: launch with dim=9", msg)
        self.assertIn("first-dim index (0..2, no repetition): launch with dim=3", msg)
        self.assertIn("first-dim index with repetition: unpack as `i, _ = wp.tid()`", msg)
        self.assertIn("per-dim indexing: unpack as `i, j = wp.tid()`", msg)

    def test_over_rank_kernel_dim_2_omits_launch_only_options(self):
        # With kernel_dim >= 2 the launch-only options would require a second
        # kernel edit to land — don't offer them.
        msg = _build_rank_error((2, 3, 4), kernel_dim=2, kernel=_stub_kernel("baz"), tiled=False)
        self.assertIn("kernel_dim=2", msg)
        self.assertNotIn("flat linear", msg)
        self.assertNotIn("launch with dim=24", msg)
        self.assertNotIn("no repetition", msg)
        self.assertIn("first-dim index with repetition: unpack as `i, _, _ = wp.tid()`", msg)
        self.assertIn("per-dim indexing: unpack as `i, j, k = wp.tid()`", msg)

    def test_under_rank_lists_two_options(self):
        msg = _build_rank_error((10,), kernel_dim=2, kernel=_stub_kernel("bar"), tiled=False)
        self.assertIn("Launch dim (10,) has rank 1", msg)
        self.assertIn("kernel_dim=2", msg)
        self.assertIn("keep 2-D kernel, launch with matching rank: dim=(10, 1)", msg)
        self.assertIn("change kernel to unpack 1 variable: `i = wp.tid()`", msg)

    def test_under_rank_tiled_lists_kernel_dim_minus_one_option(self):
        # For tiled launches, dim rank may also match kernel_dim - 1; offer a
        # concrete option alongside the rank-matching one.
        msg = _build_rank_error((2, 3), kernel_dim=4, kernel=_stub_kernel("baz"), tiled=True)
        self.assertIn("kernel_dim=4", msg)
        # rank-kernel_dim option
        self.assertIn("keep 4-D kernel, launch with matching rank: dim=(2, 3, 1, 1)", msg)
        # rank-(kernel_dim - 1) option
        self.assertIn("keep 4-D kernel with implicit block_dim axis, launch with dim=(2, 3, 1)", msg)

    def test_under_rank_non_tiled_omits_kernel_dim_minus_one_option(self):
        # Non-tiled launches have no kernel_dim - 1 escape hatch.
        msg = _build_rank_error((2, 3), kernel_dim=4, kernel=_stub_kernel("baz"), tiled=False)
        self.assertNotIn("implicit block_dim axis", msg)

    def test_tiled_flag_appends_hint(self):
        msg = _build_rank_error((3, 3), kernel_dim=1, kernel=_stub_kernel("foo"), tiled=True)
        self.assertIn("For launch_tiled, dim may also match kernel_dim - 1", msg)

    def test_non_tiled_omits_hint(self):
        msg = _build_rank_error((3, 3), kernel_dim=1, kernel=_stub_kernel("foo"), tiled=False)
        self.assertNotIn("launch_tiled", msg)


class TestPrepareLaunchDim(unittest.TestCase):
    # Existing assertions reflect the strict-rank path; pin optimize_tid=True.
    def setUp(self):
        self._old_optimize_tid = wp.config.optimize_tid
        wp.config.optimize_tid = True

    def tearDown(self):
        wp.config.optimize_tid = self._old_optimize_tid

    def test_matching_rank_returns_canonicalized(self):
        kernel = _stub_kernel(kernel_dim=2)
        self.assertEqual(_prepare_launch_dim((3, 4), kernel), (3, 4))

    def test_scalar_int_canonicalized_for_1d(self):
        kernel = _stub_kernel(kernel_dim=1)
        self.assertEqual(_prepare_launch_dim(10, kernel), (10,))

    def test_list_canonicalized(self):
        kernel = _stub_kernel(kernel_dim=2)
        self.assertEqual(_prepare_launch_dim([3, 4], kernel), (3, 4))

    def test_zero_tid_kernel_flattens_multidim(self):
        # tid_arity=0 → any dim is accepted, flattened to total count
        kernel = _stub_kernel(kernel_dim=1, tid_arity=0)
        self.assertEqual(_prepare_launch_dim((3, 4, 5), kernel), (60,))

    def test_zero_tid_kernel_preserves_1d(self):
        kernel = _stub_kernel(kernel_dim=1, tid_arity=0)
        self.assertEqual(_prepare_launch_dim(100, kernel), (100,))

    def test_over_rank_raises_value_error(self):
        kernel = _stub_kernel(kernel_dim=1)
        with self.assertRaises(ValueError) as cm:
            _prepare_launch_dim((3, 3), kernel)
        self.assertIn("kernel_dim=1", str(cm.exception))

    def test_under_rank_raises_value_error(self):
        kernel = _stub_kernel(kernel_dim=2)
        with self.assertRaises(ValueError) as cm:
            _prepare_launch_dim(10, kernel)
        self.assertIn("kernel_dim=2", str(cm.exception))

    def test_tiled_accepts_rank_minus_one(self):
        # tiled=True allows len(dim) == kernel_dim - 1 (block_dim axis is implicit)
        kernel = _stub_kernel(kernel_dim=2)
        self.assertEqual(_prepare_launch_dim((5,), kernel, tiled=True), (5,))

    def test_tiled_still_rejects_greater_mismatch(self):
        kernel = _stub_kernel(kernel_dim=2)
        with self.assertRaises(ValueError):
            _prepare_launch_dim((3, 3, 3), kernel, tiled=True)

    def test_non_tiled_rejects_rank_minus_one(self):
        kernel = _stub_kernel(kernel_dim=2)
        with self.assertRaises(ValueError):
            _prepare_launch_dim((5,), kernel, tiled=False)


class TestPrepareLaunchDimPadding(unittest.TestCase):
    # With optimize_tid off (default), kernel_dim == LAUNCH_MAX_DIMS for any
    # tid kernel, so stub kernels match that shape.
    def setUp(self):
        self._old_optimize_tid = wp.config.optimize_tid
        wp.config.optimize_tid = False

    def tearDown(self):
        wp.config.optimize_tid = self._old_optimize_tid

    def test_pads_scalar_dim_to_kernel_dim(self):
        kernel = _stub_kernel(kernel_dim=LAUNCH_MAX_DIMS, tid_arity=1)
        self.assertEqual(_prepare_launch_dim(9, kernel), (9, 1, 1, 1))

    def test_pads_2d_dim_to_kernel_dim(self):
        kernel = _stub_kernel(kernel_dim=LAUNCH_MAX_DIMS, tid_arity=2)
        self.assertEqual(_prepare_launch_dim((3, 3), kernel), (3, 3, 1, 1))

    def test_accepts_full_rank_dim(self):
        kernel = _stub_kernel(kernel_dim=LAUNCH_MAX_DIMS, tid_arity=4)
        self.assertEqual(_prepare_launch_dim((2, 3, 4, 5), kernel), (2, 3, 4, 5))

    def test_tiled_returns_unpadded(self):
        # Tiled caller (_construct_tiled_bounds) handles block_dim/padding.
        kernel = _stub_kernel(kernel_dim=LAUNCH_MAX_DIMS, tid_arity=1)
        self.assertEqual(_prepare_launch_dim((3, 3), kernel, tiled=True), (3, 3))

    def test_over_max_dims_raises(self):
        kernel = _stub_kernel(kernel_dim=LAUNCH_MAX_DIMS, tid_arity=1)
        with self.assertRaises(ValueError) as cm:
            _prepare_launch_dim((2, 2, 2, 2, 2), kernel)
        self.assertIn("LAUNCH_MAX_DIMS", str(cm.exception))

    def test_tiled_over_max_minus_one_raises(self):
        # Tiled reserves one axis for block_dim, so cap is LAUNCH_MAX_DIMS - 1.
        kernel = _stub_kernel(kernel_dim=LAUNCH_MAX_DIMS, tid_arity=1)
        with self.assertRaises(ValueError) as cm:
            _prepare_launch_dim((2, 2, 2, 2), kernel, tiled=True)
        msg = str(cm.exception)
        self.assertIn("LAUNCH_MAX_DIMS", msg)
        self.assertIn("block_dim", msg)

    def test_zero_tid_flattens_regardless_of_flag(self):
        # No-tid kernels always hit the zero-tid bypass, not the padding branch.
        kernel = _stub_kernel(kernel_dim=1, tid_arity=0)
        self.assertEqual(_prepare_launch_dim((3, 4, 5), kernel), (60,))


# ============================================================================
# Default-mode (optimize_tid=False) end-to-end: pre-1.12 unravel semantics
# ============================================================================


def test_default_tid_unravel(test, device):
    """Default (optimize_tid=False): `i = wp.tid()` with dim=(3, 3) produces histogram {0: 3, 1: 3, 2: 3}."""
    with _optimize_tid(False):

        @wp.kernel
        def _default_tid_1d_kernel(hist: wp.array(dtype=int)):
            i = wp.tid()
            wp.atomic_add(hist, i, 1)

        hist = wp.zeros(3, dtype=int, device=device)
        wp.launch(_default_tid_1d_kernel, dim=(3, 3), inputs=[hist], device=device)
        np.testing.assert_array_equal(hist.numpy(), [3, 3, 3])


def test_default_tid_2d_kernel(test, device):
    """Default (optimize_tid=False): `i, j = wp.tid()` with dim=(3, 3) produces every (i, j) exactly once."""
    with _optimize_tid(False):

        @wp.kernel
        def _default_tid_2d_kernel(hist: wp.array2d(dtype=int)):
            i, j = wp.tid()
            wp.atomic_add(hist, i, j, 1)

        hist = wp.zeros((3, 3), dtype=int, device=device)
        wp.launch(_default_tid_2d_kernel, dim=(3, 3), inputs=[hist], device=device)
        np.testing.assert_array_equal(hist.numpy(), np.ones((3, 3), dtype=int))


def test_default_tid_tiled_1d(test, device):
    """Default (optimize_tid=False): launch_tiled with rank-1 dim, `i = wp.tid()`, histogram shows effective block_dim hits per tile.

    ``wp.launch_tiled`` forces ``block_dim=1`` on CPU, so the per-tile hit
    count is 1 there and ``BLOCK_DIM`` on GPU.
    """
    with _optimize_tid(False):

        @wp.kernel
        def _default_tid_tiled_1d_kernel(hist: wp.array(dtype=int)):
            i = wp.tid()
            wp.atomic_add(hist, i, 1)

        n_tiles = 4
        hist = wp.zeros(n_tiles, dtype=int, device=device)
        wp.launch_tiled(_default_tid_tiled_1d_kernel, dim=[n_tiles], inputs=[hist], block_dim=BLOCK_DIM, device=device)
        effective_block = 1 if wp.get_device(device).is_cpu else BLOCK_DIM
        np.testing.assert_array_equal(hist.numpy(), [effective_block] * n_tiles)


def test_default_tid_tiled_2d(test, device):
    """Default (optimize_tid=False): launch_tiled with rank-2 dim, `i, j = wp.tid()`, each (i, j) hit effective block_dim times."""
    with _optimize_tid(False):

        @wp.kernel
        def _default_tid_tiled_2d_kernel(hist: wp.array2d(dtype=int)):
            i, j = wp.tid()
            wp.atomic_add(hist, i, j, 1)

        M, N = 2, 3
        hist = wp.zeros((M, N), dtype=int, device=device)
        wp.launch_tiled(_default_tid_tiled_2d_kernel, dim=[M, N], inputs=[hist], block_dim=BLOCK_DIM, device=device)
        effective_block = 1 if wp.get_device(device).is_cpu else BLOCK_DIM
        np.testing.assert_array_equal(hist.numpy(), np.full((M, N), effective_block, dtype=int))


def test_dim_over_max_raises(test, device):
    """Default (optimize_tid=False): dim rank above LAUNCH_MAX_DIMS still raises ValueError."""
    with _optimize_tid(False):

        @wp.kernel
        def _over_max_kernel(out: wp.array(dtype=int)):
            i = wp.tid()
            out[i] = i

        out = wp.zeros(32, dtype=int, device=device)
        with test.assertRaises(ValueError) as cm:
            wp.launch(_over_max_kernel, dim=(2, 2, 2, 2, 2), inputs=[out], device=device)
        test.assertIn("LAUNCH_MAX_DIMS", str(cm.exception))


def test_optimize_tid_enforces_rank(test, device):
    """Regression: with optimize_tid on, `i = wp.tid()` with dim=(3, 3) raises ValueError."""
    with _optimize_tid(True):

        @wp.kernel
        def _strict_1d_kernel(out: wp.array(dtype=int)):
            i = wp.tid()
            out[i] = i

        out = wp.zeros(9, dtype=int, device=device)
        with test.assertRaises(ValueError) as cm:
            wp.launch(_strict_1d_kernel, dim=(3, 3), inputs=[out], device=device)
        msg = str(cm.exception)
        test.assertIn("kernel_dim=1", msg)
        test.assertIn("flat linear index", msg)


def test_default_tid_set_dim_recorded_launch(test, device):
    """Default (optimize_tid=False): recording a launch and calling set_dim with a lower-rank dim works without raising."""
    with _optimize_tid(False):

        @wp.kernel
        def _default_tid_record_kernel(out: wp.array(dtype=int)):
            i = wp.tid()
            out[i] = i

        N = 16
        out = wp.zeros(N, dtype=int, device=device)
        launch = wp.launch(_default_tid_record_kernel, dim=N, inputs=[out], device=device, record_cmd=True)
        # Re-set with a 2-D dim — with optimize_tid off this pads to 4 with 1s.
        launch.set_dim((2, 3))
        launch.launch()

        # dim=(2, 3) padded to (2, 3, 1, 1) dispatches 6 threads; coord.i =
        # linear // 3 ∈ {0, 0, 0, 1, 1, 1}, so out[0] and out[1] are written
        # (to 0 and 1 respectively) and the rest remain 0 from wp.zeros.
        expected = np.zeros(N, dtype=int)
        expected[1] = 1
        np.testing.assert_array_equal(out.numpy(), expected)


# ============================================================================
# Test class and registration
# ============================================================================


class TestTemplateLaunchBounds(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestTemplateLaunchBounds, "test_regular_1d", test_regular_1d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_2d", test_regular_2d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_3d", test_regular_3d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_4d", test_regular_4d, devices=devices)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_dim_over_rank_kernel_dim_1_error",
    test_launch_dim_over_rank_kernel_dim_1_error,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_dim_over_rank_kernel_dim_2_error",
    test_launch_dim_over_rank_kernel_dim_2_error,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_dim_under_rank_error",
    test_launch_dim_under_rank_error,
    devices=devices,
)
add_function_test(TestTemplateLaunchBounds, "test_no_tid_kernel_multidim", test_no_tid_kernel_multidim, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_1d", test_tiled_1d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_2d", test_tiled_2d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_no_tid", test_tiled_no_tid, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_manual_tiled", test_manual_tiled, devices=devices)
add_function_test(
    TestTemplateLaunchBounds, "test_tiled_dim_mismatch_error", test_tiled_dim_mismatch_error, devices=devices
)
add_function_test(TestTemplateLaunchBounds, "test_set_dim_matching_rank", test_set_dim_matching_rank, devices=devices)
add_function_test(
    TestTemplateLaunchBounds,
    "test_set_dim_rank_mismatch_error",
    test_set_dim_rank_mismatch_error,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_set_dim_preserves_tiled_flag",
    test_set_dim_preserves_tiled_flag,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_set_dim_tiled_explicit_block_axis",
    test_set_dim_tiled_explicit_block_axis,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_default_tid_unravel",
    test_default_tid_unravel,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_default_tid_2d_kernel",
    test_default_tid_2d_kernel,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_default_tid_tiled_1d",
    test_default_tid_tiled_1d,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_default_tid_tiled_2d",
    test_default_tid_tiled_2d,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_dim_over_max_raises",
    test_dim_over_max_raises,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_optimize_tid_enforces_rank",
    test_optimize_tid_enforces_rank,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_default_tid_set_dim_recorded_launch",
    test_default_tid_set_dim_recorded_launch,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
