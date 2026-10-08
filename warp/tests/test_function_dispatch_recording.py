# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import sys
import unittest
import weakref
from typing import ClassVar

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices


class RecordingArray(wp.array):
    calls: ClassVar[list] = []

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        with wp.ScopedWarpFunctionDisable():
            result = func(*args, **({} if kwargs is None else kwargs))
        cls.calls.append(func)
        return result


@wp.kernel
def square(values: wp.array[float], result: wp.array[float]):
    i = wp.tid()
    result[i] = values[i] * values[i]


@wp.kernel
def increment(values: wp.array[float]):
    i = wp.tid()
    values[i] += 1.0


def test_tape_delegation(test, device):
    values = RecordingArray([2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    result = wp.zeros(2, dtype=wp.float32, device=device, requires_grad=True)
    RecordingArray.calls = []
    tape = wp.Tape()
    with tape:
        call_line = sys._getframe().f_lineno + 1
        wp.launch(square, 2, inputs=[values], outputs=[result], device=device)

    test.assertEqual(RecordingArray.calls, [wp.launch])
    test.assertEqual(len(tape.launches), 1)
    test.assertIs(tape.launches[0][3][0], values)
    test.assertEqual(
        tape.launches[0][7]["caller"],
        {"file": __file__, "lineno": call_line, "func": "test_tape_delegation"},
    )
    tape.backward(grads={result: wp.ones_like(result)})
    np.testing.assert_allclose(values.grad.numpy(), [4.0, 6.0])
    np.testing.assert_allclose(result.numpy(), [4.0, 9.0])


def test_plain_tape_callsite(test, device):
    values = wp.ones(2, dtype=wp.float32, device=device, requires_grad=True)
    result = wp.zeros_like(values)
    tape = wp.Tape()
    with tape:
        call_line = sys._getframe().f_lineno + 1
        wp.launch(square, 2, inputs=[values], outputs=[result], device=device)
    test.assertEqual(
        tape.launches[0][7]["caller"],
        {"file": __file__, "lineno": call_line, "func": "test_plain_tape_callsite"},
    )
    np.testing.assert_allclose(result.numpy(), [1.0, 1.0])


def test_capture_delegation(test, device):
    values = RecordingArray([0.0, 0.0], dtype=wp.float32, device=device)
    alias = values.view(wp.float32)
    wp.load_module(module=increment.module, device=device)
    RecordingArray.calls = []
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(increment, 2, inputs=[values], device=device)
    test.assertEqual(RecordingArray.calls, [wp.launch])

    values.zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(alias.numpy(), [1.0, 1.0])
    test.assertEqual(RecordingArray.calls, [wp.launch])


def test_disabled_tape_callsite(test, device):
    with wp.ScopedWarpFunctionDisable():
        test_plain_tape_callsite(test, device)


def test_recorded_command_delegation(test, device):
    values = RecordingArray([0.0, 0.0], dtype=wp.float32, device=device)
    owner = weakref.ref(values)
    RecordingArray.calls = []
    command = wp.launch(increment, 2, inputs=[values], device=device, record_cmd=True)
    test.assertEqual(RecordingArray.calls, [wp.launch])
    with wp.ScopedWarpFunctionDisable():
        np.testing.assert_allclose(values.numpy(), [0.0, 0.0])
    del values
    gc.collect()
    test.assertIsNotNone(owner())
    command.launch()
    command.launch()
    with wp.ScopedWarpFunctionDisable():
        np.testing.assert_allclose(owner().numpy(), [2.0, 2.0])
    test.assertEqual(RecordingArray.calls, [wp.launch])


class TestFunctionDispatchRecording(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestFunctionDispatchRecording, "test_tape_delegation", test_tape_delegation, devices=devices)
add_function_test(TestFunctionDispatchRecording, "test_plain_tape_callsite", test_plain_tape_callsite, devices=devices)
add_function_test(
    TestFunctionDispatchRecording, "test_disabled_tape_callsite", test_disabled_tape_callsite, devices=devices
)
add_function_test(TestFunctionDispatchRecording, "test_capture_delegation", test_capture_delegation, devices=devices)
add_function_test(
    TestFunctionDispatchRecording, "test_recorded_command_delegation", test_recorded_command_delegation, devices=devices
)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
