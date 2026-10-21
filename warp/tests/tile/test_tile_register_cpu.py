# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check CPU register tile copies, including their scalar fallback paths."""

import ctypes
import functools
import math
import mmap
import sys
import unittest

import numpy as np

import warp as wp

_TILE_SHAPES = ((8,), (3, 8), (2, 3, 8), (2, 2, 3, 8))
_DTYPES = ((wp.float32, np.float32, 1), (wp.int32, np.int32, 1), (wp.float64, np.float64, 1), (wp.vec3, np.float32, 3))


@functools.cache
def _copy_kernel(rank, dtype, tile_shape, block_dim, shifted=False, one_tile=False):
    # Separate modules avoid recompiling unrelated kernels when a test changes
    # block dimensions or creates another shape/dtype specialization.
    array_type = (wp.array[dtype], wp.array2d[dtype], wp.array3d[dtype], wp.array4d[dtype])[rank - 1]
    a = wp.constant(tile_shape[0])
    b = wp.constant(tile_shape[1] if rank > 1 else 1)
    c = wp.constant(tile_shape[2] if rank > 2 else 1)
    d = wp.constant(tile_shape[3] if rank > 3 else 1)
    shift = wp.constant(int(shifted))
    single = wp.constant(one_tile)

    if rank == 1:

        @wp.kernel(module="unique", enable_backward=True)
        def copy(source: array_type, destination: array_type):
            i = wp.tid()
            if wp.static(single):
                values = wp.tile_load(source, shape=(a,), storage="register")
                wp.tile_store(destination, values, offset=(shift,))
            else:
                values = wp.tile_load(source, shape=(a,), offset=(i * a,), storage="register")
                wp.tile_store(destination, values, offset=(i * a,))

    elif rank == 2:

        @wp.kernel(module="unique", enable_backward=True)
        def copy(source: array_type, destination: array_type):
            i, j = wp.tid()
            if wp.static(single):
                values = wp.tile_load(source, shape=(a, b), storage="register")
                wp.tile_store(destination, values, offset=(0, shift))
            else:
                values = wp.tile_load(source, shape=(a, b), offset=(i * a, j * b), storage="register")
                wp.tile_store(destination, values, offset=(i * a, j * b))

    elif rank == 3:

        @wp.kernel(module="unique", enable_backward=True)
        def copy(source: array_type, destination: array_type):
            i, j, k = wp.tid()
            if wp.static(single):
                values = wp.tile_load(source, shape=(a, b, c), storage="register")
                wp.tile_store(destination, values, offset=(0, 0, shift))
            else:
                values = wp.tile_load(source, shape=(a, b, c), offset=(i * a, j * b, k * c), storage="register")
                wp.tile_store(destination, values, offset=(i * a, j * b, k * c))

    else:

        @wp.kernel(module="unique", enable_backward=True)
        def copy(source: array_type, destination: array_type):
            i, j, combined = wp.tid()
            if wp.static(single):
                values = wp.tile_load(source, shape=(a, b, c, d), storage="register")
                wp.tile_store(destination, values, offset=(0, 0, 0, shift))
            else:
                grid_last = (source.shape[3] + d - 1) // d
                k = combined // grid_last
                ell = combined % grid_last
                values = wp.tile_load(
                    source, shape=(a, b, c, d), offset=(i * a, j * b, k * c, ell * d), storage="register"
                )
                wp.tile_store(destination, values, offset=(i * a, j * b, k * c, ell * d))

    return copy


def _tile_grid(shape, tile_shape):
    grid = tuple((n + tile - 1) // tile for n, tile in zip(shape, tile_shape, strict=True))
    return grid if len(grid) < 4 else (grid[0], grid[1], grid[2] * grid[3])


def _guarded_array(shape, dtype_info, layout, source):
    """Build an offset view and an independent NumPy map of its backing cells."""
    dtype, numpy_dtype, components = dtype_info
    axes = list(range(len(shape)))
    if layout == "outer_transpose":
        axes[0], axes[1] = axes[1], axes[0]
    strides = [0] * len(shape)
    step = 2 if layout == "inner_stride" else 1
    for axis in reversed(axes):
        strides[axis] = step
        step = step * shape[axis] + (3 if layout == "outer_pad" else 0)
    count = 1 + sum((n - 1) * stride for n, stride in zip(shape, strides, strict=True)) + 20
    host_shape = (count,) if components == 1 else (count, components)
    host = np.full(host_shape, -7777 if source else -3333, dtype=numpy_dtype)
    indices = np.ones(shape, dtype=np.int64)
    for coordinates, stride in zip(np.indices(shape, dtype=np.int64), strides, strict=True):
        indices += coordinates * stride
    if source:
        values_shape = shape if components == 1 else (*shape, components)
        host[indices] = (np.arange(math.prod(values_shape)).reshape(values_shape) % 997 + 1).astype(numpy_dtype)
    backing = wp.array(host, dtype=dtype, device="cpu")
    item_bytes = np.dtype(numpy_dtype).itemsize * components
    # Start one element into the allocation: the fast path must not assume
    # 16-byte alignment, including when the row itself is contiguous.
    view = wp.array(
        ptr=backing.ptr + item_bytes,
        dtype=dtype,
        shape=shape,
        strides=tuple(stride * item_bytes for stride in strides),
        capacity=backing.capacity - item_bytes,
        device="cpu",
    )
    return backing, view, host, indices


@wp.kernel(module="unique", enable_backward=False)
def _load_high_row(source: wp.array2d[wp.float32], row: int, destination: wp.array2d[wp.float32]):
    values = wp.tile_load(source, shape=(2, 8), offset=(row, 0), storage="register", bounds_check=False, aligned=True)
    wp.tile_store(destination, values, bounds_check=False, aligned=True)


@wp.kernel(module="unique", enable_backward=False)
def _store_high_row(source: wp.array2d[wp.float32], row: int, destination: wp.array2d[wp.float32]):
    values = wp.tile_load(source, shape=(2, 8), storage="register", bounds_check=False, aligned=True)
    wp.tile_store(destination, values, offset=(row, 0), bounds_check=False, aligned=True)


class TestTileRegisterCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def setUp(self):
        previous = wp.config.enable_cpu_blocks
        self.addCleanup(setattr, wp.config, "enable_cpu_blocks", previous)
        wp.config.enable_cpu_blocks = True

    def _check_copy(
        self,
        tile_shape,
        *,
        partial=False,
        source_layout="contiguous",
        destination_layout="contiguous",
        dtype_info=_DTYPES[0],
        block_dim=1,
    ):
        rank = len(tile_shape)
        shape = tuple(2 * n - int(partial) for n in tile_shape)
        source_backing, source, source_host, source_indices = _guarded_array(shape, dtype_info, source_layout, True)
        destination_backing, destination, destination_host, destination_indices = _guarded_array(
            shape, dtype_info, destination_layout, False
        )
        expected = destination_host.copy()
        expected[destination_indices] = source_host[source_indices]
        wp.launch_tiled(
            _copy_kernel(rank, dtype_info[0], tile_shape, block_dim),
            dim=_tile_grid(shape, tile_shape),
            inputs=[source],
            outputs=[destination],
            block_dim=block_dim,
            device="cpu",
        )
        np.testing.assert_array_equal(destination_backing.numpy(), expected)
        np.testing.assert_array_equal(source_backing.numpy(), source_host)

    def test_full_tiles(self):
        for tile_shape in _TILE_SHAPES:
            with self.subTest(tile_shape=tile_shape):
                self._check_copy(tile_shape)

    def test_partial_tiles(self):
        for tile_shape in _TILE_SHAPES:
            with self.subTest(tile_shape=tile_shape):
                self._check_copy(tile_shape, partial=True)

    def test_nonmultiple_row_widths(self):
        for width in (7, 10):
            for partial in (False, True):
                with self.subTest(width=width, partial=partial):
                    self._check_copy((2, 3, width), partial=partial)

    def test_inner_strided_source_and_destination(self):
        for source_layout, destination_layout in (("inner_stride", "contiguous"), ("contiguous", "inner_stride")):
            with self.subTest(source_layout=source_layout, destination_layout=destination_layout):
                self._check_copy((2, 3, 8), source_layout=source_layout, destination_layout=destination_layout)

    def test_padded_and_transposed_outer_axes(self):
        for tile_shape in _TILE_SHAPES[2:]:
            for source_layout, destination_layout in (
                ("outer_pad", "outer_transpose"),
                ("outer_transpose", "outer_pad"),
            ):
                with self.subTest(
                    tile_shape=tile_shape, source_layout=source_layout, destination_layout=destination_layout
                ):
                    self._check_copy(tile_shape, source_layout=source_layout, destination_layout=destination_layout)

    def test_scalar_and_composite_dtypes(self):
        for dtype_info in _DTYPES[1:]:
            for partial in (False, True):
                with self.subTest(dtype=dtype_info[0], partial=partial):
                    self._check_copy((2, 3, 8), dtype_info=dtype_info, partial=partial)

    def test_visible_load_zero_padding(self):
        for tile_shape in _TILE_SHAPES:
            with self.subTest(tile_shape=tile_shape):
                source_shape = tuple(n - 1 for n in tile_shape)
                source_backing, source, source_host, source_indices = _guarded_array(
                    source_shape, _DTYPES[0], "contiguous", True
                )
                destination_backing, destination, destination_host, destination_indices = _guarded_array(
                    tile_shape, _DTYPES[0], "contiguous", False
                )
                logical_expected = np.zeros(tile_shape, dtype=np.float32)
                logical_expected[tuple(slice(n) for n in source_shape)] = source_host[source_indices]
                expected = destination_host.copy()
                expected[destination_indices] = logical_expected
                wp.launch_tiled(
                    _copy_kernel(len(tile_shape), wp.float32, tile_shape, 1),
                    dim=(1,) * min(len(tile_shape), 3),
                    inputs=[source],
                    outputs=[destination],
                    block_dim=1,
                    device="cpu",
                )
                np.testing.assert_array_equal(destination_backing.numpy(), expected)
                np.testing.assert_array_equal(source_backing.numpy(), source_host)

    def test_single_task_aliasing_preserves_load_snapshot(self):
        for tile_shape in _TILE_SHAPES:
            for shifted in (False, True):
                with self.subTest(tile_shape=tile_shape, shifted=shifted):
                    shape = (*tile_shape[:-1], tile_shape[-1] + int(shifted))
                    backing, view, host, indices = _guarded_array(shape, _DTYPES[0], "contiguous", True)
                    expected = host.copy()
                    source_slice = (*[slice(None)] * (len(shape) - 1), slice(tile_shape[-1]))
                    destination_slice = (
                        *[slice(None)] * (len(shape) - 1),
                        slice(int(shifted), tile_shape[-1] + int(shifted)),
                    )
                    expected[indices[destination_slice]] = host[indices[source_slice]]
                    # A single block-one task owns this entire register tile;
                    # no cross-task overlapping-copy contract is assumed.
                    wp.launch_tiled(
                        _copy_kernel(len(shape), wp.float32, tile_shape, 1, shifted, True),
                        dim=(1,) * min(len(shape), 3),
                        inputs=[view],
                        outputs=[view],
                        block_dim=1,
                        device="cpu",
                    )
                    np.testing.assert_array_equal(backing.numpy(), expected)

    def _check_gradient(self, tile_shape, block_dim):
        shape = tuple(2 * n - 1 for n in tile_shape)
        host = (np.arange(math.prod(shape), dtype=np.float32) % 97 + 1).reshape(shape)
        source = wp.array(host, dtype=wp.float32, device="cpu", requires_grad=True)
        destination = wp.zeros(shape, dtype=wp.float32, device="cpu", requires_grad=True)
        with wp.Tape() as tape:
            wp.launch_tiled(
                _copy_kernel(len(shape), wp.float32, tile_shape, block_dim),
                dim=_tile_grid(shape, tile_shape),
                inputs=[source],
                outputs=[destination],
                block_dim=block_dim,
                device="cpu",
            )
        tape.backward(grads={destination: wp.ones_like(destination)})
        np.testing.assert_array_equal(destination.numpy(), host)
        np.testing.assert_array_equal(source.grad.numpy(), np.ones_like(host))
        np.testing.assert_array_equal(source.numpy(), host)

    def test_partial_tile_gradients(self):
        for tile_shape in _TILE_SHAPES:
            with self.subTest(tile_shape=tile_shape):
                self._check_gradient(tile_shape, 1)

    def test_cooperative_block_control(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                self._check_copy((2, 3, 8), partial=partial, block_dim=64)
        self._check_gradient((2, 3, 8), 64)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and sys.maxsize > 2**32, "Requires a 64-bit Linux sparse mapping"
    )
    def test_row_byte_offsets_over_two_gib(self):
        # A private anonymous mapping reserves virtual address space without
        # initializing 2 GiB of physical memory. Only two small windows are
        # touched; do not convert the large Warp view to NumPy.
        row = 524289
        strides = (4096, 4)
        shape = (row + 3, 8)
        row_offset = row * strides[0]
        self.assertGreater(row_offset, 2**31)
        mapping_size = shape[0] * strides[0]
        with mmap.mmap(
            -1, mapping_size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE
        ) as mapping:
            pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            large = wp.array(
                ptr=pointer, dtype=wp.float32, shape=shape, strides=strides, capacity=mapping_size, device="cpu"
            )
            values = (np.arange(16, dtype=np.float32) + 31).reshape(2, 8)
            guard = bytes([0xA5]) * 64
            windows = []
            expected_windows = []
            for i in range(2):
                start = row_offset + i * strides[0] - len(guard)
                expected_window = guard + values[i].tobytes() + guard
                mapping[start : start + len(expected_window)] = expected_window
                windows.append((start, len(expected_window)))
                expected_windows.append(expected_window)
            small = wp.zeros((2, 8), dtype=wp.float32, device="cpu")
            # The high-row kernels also cover valid unchecked/aligned calls.
            self.assertEqual(pointer % 16, 0)
            self.assertEqual(small.ptr % 16, 0)
            wp.launch_tiled(_load_high_row, dim=1, inputs=[large, row], outputs=[small], block_dim=1, device="cpu")
            np.testing.assert_array_equal(small.numpy(), values)
            for (start, length), expected in zip(windows, expected_windows, strict=True):
                self.assertEqual(mapping[start : start + length], expected)
                mapping[start : start + length] = guard + bytes(32) + guard
            wp.launch_tiled(_store_high_row, dim=1, inputs=[small, row], outputs=[large], block_dim=1, device="cpu")
            for (start, length), expected in zip(windows, expected_windows, strict=True):
                self.assertEqual(mapping[start : start + length], expected)
            np.testing.assert_array_equal(small.numpy(), values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
