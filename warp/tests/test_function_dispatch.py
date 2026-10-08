# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import contextvars
import inspect
import sys
import threading
import unittest
from typing import ClassVar

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices

_HANDLED = object()


class _Probe:
    calls: ClassVar[list] = []
    result = _HANDLED

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        cls.calls.append((cls, func, types, args, kwargs))
        return cls.result


class _ObservedArray(wp.array):
    calls: ClassVar[list] = []

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        cls.calls.append((func, types, args, kwargs))
        with wp.ScopedWarpFunctionDisable():
            return func(*args, **kwargs)


@wp.kernel
def _increment(value: wp.array[float]):
    i = wp.tid()
    value[i] += 1.0


@wp.kernel
def _increment_tiled(value: wp.array[float]):
    i, _lane = wp.tid()
    value[i] += 1.0


class TestFunctionDispatch(unittest.TestCase):
    def setUp(self):
        _Probe.calls.clear()
        _ObservedArray.calls.clear()

    def test_original_call_shape(self):
        value = _Probe()
        arguments = [value, value]
        result = wp.launch(None, 2, arguments, device="cpu", record_cmd=True)
        self.assertIs(result, _HANDLED)
        cls, func, types, args, kwargs = _Probe.calls[0]
        self.assertIs(cls, _Probe)
        self.assertIs(func, wp.launch)
        self.assertEqual(types, (_Probe,))
        self.assertEqual(args[:2], (None, 2))
        self.assertIs(args[2], arguments)
        self.assertEqual(kwargs, {"device": "cpu", "record_cmd": True})
        self.assertNotIn("outputs", kwargs)
        self.assertEqual(len(_Probe.calls), 1)

    def test_keyword_order_follows_signature(self):
        class Destination(_Probe):
            result = NotImplemented

        class Source(_Probe):
            pass

        source, destination = Source(), Destination()
        self.assertIs(wp.copy(src=source, dest=destination), _HANDLED)
        self.assertEqual([call[0] for call in _Probe.calls], [Destination, Source])
        self.assertEqual(_Probe.calls[0][2], (Destination, Source))
        self.assertEqual(list(_Probe.calls[0][4]), ["src", "dest"])

    def test_launch_argument_groups(self):
        for function in (wp.launch, wp.launch_tiled):
            for parameter in ("inputs", "outputs", "adj_inputs", "adj_outputs"):
                with self.subTest(function=function.__name__, parameter=parameter):
                    _Probe.calls.clear()
                    value = _Probe()
                    self.assertIs(function(None, 1, **{parameter: [value]}), _HANDLED)
                    self.assertIs(_Probe.calls[0][1], function)
                    self.assertIs(_Probe.calls[0][4][parameter][0], value)

    def test_launch_positional_groups(self):
        for offset in range(4):
            with self.subTest(offset=offset):
                _Probe.calls.clear()
                arguments = [None, 1, [], [], [], []]
                arguments[offset + 2] = [_Probe()]
                self.assertIs(wp.launch(*arguments), _HANDLED)
                self.assertEqual(len(_Probe.calls), 1)

    def test_launch_group_order(self):
        class Inputs(_Probe):
            result = NotImplemented

        class Outputs(_Probe):
            result = NotImplemented

        class AdjInputs(_Probe):
            result = NotImplemented

        class AdjOutputs(_Probe):
            pass

        result = wp.launch(
            None,
            1,
            adj_outputs=[AdjOutputs()],
            outputs=[Outputs()],
            adj_inputs=[AdjInputs()],
            inputs=[Inputs()],
        )
        self.assertIs(result, _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (Inputs, Outputs, AdjInputs, AdjOutputs))
        self.assertEqual([call[0] for call in _Probe.calls], [Inputs, Outputs, AdjInputs, AdjOutputs])

    def test_nested_sequences_and_cycles(self):
        value = _Probe()
        sequence = []
        sequence.extend([sequence, ([value], (value,))])
        self.assertIs(wp.launch(None, 1, inputs=sequence), _HANDLED)
        self.assertEqual(len(_Probe.calls), 1)
        self.assertEqual(_Probe.calls[0][2], (_Probe,))
        self.assertIs(_Probe.calls[0][4]["inputs"], sequence)

    def test_discovery_does_not_consume_or_probe(self):
        consumed = []

        def generator():
            consumed.append(True)
            yield _Probe()

        class Untouched:
            @property
            def __warp_function__(self):
                raise AssertionError("Instance hooks must not be read")

            @property
            def __array_interface__(self):
                raise AssertionError("Array interfaces must not be read")

            def __dlpack__(self):
                raise AssertionError("DLPack must not be requested")

        # An instance property is a non-callable class hook, so test array
        # interface probing with a separate class that has no protocol member.
        class Foreign:
            @property
            def __array_interface__(self):
                raise AssertionError("Array interfaces must not be read")

            def __dlpack__(self):
                raise AssertionError("DLPack must not be requested")

        class CustomSequence(list):
            def __iter__(self):
                raise AssertionError("Custom containers must not be traversed")

        hidden = _Probe()
        for sequence in (generator(), {"hidden": hidden}, CustomSequence([hidden]), [Foreign()]):
            with self.subTest(sequence=type(sequence).__name__):
                _Probe.calls.clear()
                self.assertIs(wp.launch(_Probe(), 1, inputs=sequence), _HANDLED)
                self.assertEqual(len(_Probe.calls), 1)
        self.assertEqual(consumed, [])
        with self.assertRaisesRegex(TypeError, "must be callable"):
            wp.clone(Untouched())

    def test_only_designated_parameters_recurse(self):
        class Hidden(_Probe):
            pass

        hidden = Hidden()
        self.assertIs(wp.launch(_Probe(), 1, device=[hidden]), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (_Probe,))
        self.assertEqual(len(_Probe.calls), 1)
        with self.assertRaises(AttributeError):
            wp.clone([hidden])

    def test_custom_argument_container_participates(self):
        class Container(_Probe, list):
            def __iter__(self):
                raise AssertionError("Participating containers must not be traversed")

        values = Container([_Probe()])
        self.assertIs(wp.launch(None, 1, inputs=values), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (Container,))

    def test_subclass_precedence_and_repeated_types(self):
        class Base(_Probe):
            result = NotImplemented

        class Other(_Probe):
            pass

        class Child(Base):
            result = NotImplemented

        self.assertIs(wp.launch(None, 1, inputs=[Base(), Other(), Child(), Child(), Base()]), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (Child, Base, Other))
        self.assertEqual([call[0] for call in _Probe.calls], [Child, Base, Other])

    def test_multiple_inheritance_precedence(self):
        class Left(_Probe):
            result = NotImplemented

        class Right(_Probe):
            pass

        class Child(Left, Right):
            result = NotImplemented

        self.assertIs(wp.launch(None, 1, inputs=[Left(), Right(), Child()]), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (Child, Left, Right))

    def test_virtual_abc_does_not_change_precedence(self):
        class VirtualBase(_Probe, metaclass=abc.ABCMeta):
            result = NotImplemented

        class Registered(_Probe):
            pass

        VirtualBase.register(Registered)
        self.assertIs(wp.copy(VirtualBase(), Registered()), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (VirtualBase, Registered))

    def test_inherited_user_hooks_bind_concrete_types(self):
        class First(_Probe):
            result = NotImplemented

        class Second(_Probe):
            pass

        self.assertIs(wp.copy(First(), Second()), _HANDLED)
        self.assertEqual([call[0] for call in _Probe.calls], [First, Second])

    def test_false_and_empty_results_are_handled(self):
        for result in (None, False, 0, [], {}, ()):
            with self.subTest(result=result):

                class Handles(_Probe):
                    pass

                Handles.result = result
                self.assertIs(wp.clone(Handles()), result)

    def test_hook_exception_propagates(self):
        class Failure:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                raise ValueError("Hook failed")

        with self.assertRaisesRegex(ValueError, "Hook failed"):
            wp.copy(Failure(), _Probe())
        self.assertEqual(_Probe.calls, [])
        self.assertIs(wp.clone(_Probe()), _HANDLED)

    def test_instance_assignment_does_not_participate(self):
        class InstanceOnly:
            pass

        value = InstanceOnly()
        value.__warp_function__ = lambda *args: _HANDLED
        with self.assertRaisesRegex(RuntimeError, "must be arrays"):
            wp.copy(value, value)

    def test_hook_none_and_non_callable(self):
        class Disabled(_Probe):
            __warp_function__ = None

        class Invalid:
            __warp_function__ = 1

        self.assertIs(wp.copy(Disabled(), _Probe()), _HANDLED)
        self.assertEqual(_Probe.calls[0][2], (_Probe,))
        with self.assertRaisesRegex(TypeError, "must be callable or None"):
            wp.clone(Invalid())

    def test_call_shape_errors_precede_hooks(self):
        value = _Probe()
        calls = (
            lambda: wp.copy(value, value, src=value),
            lambda: wp.copy(value),
            lambda: wp.copy(value, value, unexpected=value),
            lambda: wp.copy(value, value, 0, 0, 0, None, value),
            lambda: wp.launch(None, 1, inputs=[value], unexpected=True),
            lambda: wp.launch_tiled(None, 1, inputs=[value], unexpected=True),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()
                self.assertEqual(_Probe.calls, [])

    def test_recursive_dispatch_has_delegation_guidance(self):
        class Recursive:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                return func(*args, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "ScopedWarpFunctionDisable"):
            wp.clone(Recursive())
        self.assertIs(wp.clone(_Probe()), _HANDLED)

    def test_hook_can_dispatch_a_different_function(self):
        class Nested:
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                self.assertIs(func, wp.clone)
                return wp.empty_like(_Probe())

        self.assertIs(wp.clone(Nested()), _HANDLED)
        self.assertIs(_Probe.calls[0][1], wp.empty_like)

    def test_disable_scopes_restore_after_exception(self):
        value = _Probe()
        scope = wp.ScopedWarpFunctionDisable()
        with self.assertRaisesRegex(ValueError, "Scope failure"):
            with scope:
                with scope:
                    with self.assertRaisesRegex(RuntimeError, "must be arrays"):
                        wp.copy(value, value)
                    raise ValueError("Scope failure")
        self.assertEqual(_Probe.calls, [])
        self.assertIs(wp.clone(value), _HANDLED)

    def test_disable_isolated_between_contexts_and_threads(self):
        values = []
        failures = []

        def independent_call():
            try:
                values.append(wp.clone(_Probe()))
            except Exception as error:
                failures.append(error)

        with wp.ScopedWarpFunctionDisable():
            contextvars.Context().run(independent_call)
            worker = threading.Thread(target=independent_call)
            worker.start()
            worker.join()
            with self.assertRaisesRegex(RuntimeError, "must be arrays"):
                wp.copy(_Probe(), _Probe())
        self.assertEqual(failures, [])
        self.assertEqual(values, [_HANDLED, _HANDLED])

    def test_disable_scopes_do_not_accumulate_context_variables(self):
        def check_context():
            with wp.ScopedWarpFunctionDisable():
                pass
            variables = len(contextvars.copy_context())
            for _ in range(32):
                with wp.ScopedWarpFunctionDisable():
                    pass
            self.assertEqual(len(contextvars.copy_context()), variables)

        contextvars.Context().run(check_context)

    def test_inherited_neutral_hook_and_super_deferral(self):
        class Neutral(wp.array):
            pass

        class Defers(wp.array):
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                self.assertIs(super().__warp_function__(func, types, args, kwargs), NotImplemented)
                return NotImplemented

        neutral = Neutral(dtype=wp.float32)
        defers = Defers(dtype=wp.float32)
        self.assertIs(wp.copy(neutral, _Probe()), _HANDLED)
        self.assertEqual(_Probe.calls[-1][2], (_Probe,))
        self.assertIs(wp.copy(defers, _Probe()), _HANDLED)
        self.assertEqual(_Probe.calls[-1][2], (Defers, _Probe))

    def test_native_fallback_after_all_decline(self):
        class Declining(wp.array):
            calls: ClassVar[list] = []

            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append(func)
                return super().__warp_function__(func, types, args, kwargs)

        value = Declining([1.0, 2.0], dtype=wp.float32, device="cpu")
        result = wp.clone(value)
        np.testing.assert_allclose(result.numpy(), [1.0, 2.0])
        self.assertEqual(Declining.calls, [wp.clone])
        self.assertIs(type(result), wp.array)

        class DecliningWrapper(_Probe):
            result = NotImplemented

        with self.assertRaisesRegex(RuntimeError, "must be arrays"):
            wp.copy(DecliningWrapper(), DecliningWrapper())

    def test_native_method_delegation_preserves_callsite(self):
        # Inspect the internal location explicitly for this regression: array
        # methods call covered APIs without establishing an outer dispatch.
        from warp._src.function_dispatch import get_warp_function_callsite  # noqa: PLC0415

        calls = []

        class CapturesCallsite(wp.array):
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, get_warp_function_callsite()))
                with wp.ScopedWarpFunctionDisable():
                    return func(*args, **kwargs)

        value = CapturesCallsite([0.0, 0.0], dtype=wp.float32, device="cpu")
        source = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        call_line = sys._getframe().f_lineno + 1
        value.assign(source)
        self.assertEqual(calls, [(wp.copy, (__file__, call_line, "test_native_method_delegation_preserves_callsite"))])
        np.testing.assert_allclose(value.numpy(), [1.0, 2.0])

    def test_dynamic_hook_replacement(self):
        class Dynamic(np.ndarray):
            pass

        source = np.arange(3, dtype=np.float32).view(Dynamic)
        self.assertIsInstance(wp.from_numpy(source), wp.array)
        Dynamic.__warp_function__ = classmethod(lambda cls, func, types, args=(), kwargs=None: _HANDLED)
        self.assertIs(wp.from_numpy(source), _HANDLED)
        Dynamic.__warp_function__ = None
        self.assertIsInstance(wp.from_numpy(source), wp.array)

    def test_public_function_identity_and_aliases(self):
        probe = _Probe()
        calls = (
            (wp.launch, (None, 1), {"inputs": [probe]}),
            (wp.launch_tiled, (None, 1), {"outputs": [probe]}),
            (wp.copy, (probe, probe), {}),
            (wp.clone, (probe,), {}),
            (wp.empty_like, (probe,), {}),
            (wp.zeros_like, (probe,), {}),
            (wp.ones_like, (probe,), {}),
            (wp.full_like, (probe, 2.0), {}),
            (wp.from_numpy, (probe,), {}),
            (wp.from_torch, (probe,), {}),
            (wp.to_torch, (probe,), {}),
            (wp.from_dlpack, (probe,), {}),
            (wp.to_dlpack, (probe,), {}),
        )
        for function, args, kwargs in calls:
            with self.subTest(function=function.__name__):
                _Probe.calls.clear()
                alias = function
                self.assertIs(alias(*args, **kwargs), _HANDLED)
                self.assertIs(_Probe.calls[0][1], function)
                self.assertTrue(function.__doc__)
                self.assertTrue(hasattr(function, "__wrapped__"))
                if function is not wp.launch_tiled:
                    self.assertEqual(inspect.signature(function), inspect.signature(function.__wrapped__))

    def test_conversion_hook_precedes_data_protocols(self):
        class Foreign(_Probe):
            @property
            def __array_interface__(self):
                raise AssertionError("Array interface must be queried after dispatch")

            @property
            def __cuda_array_interface__(self):
                raise AssertionError("CUDA array interface must be queried after dispatch")

            def __array__(self, *args, **kwargs):
                raise AssertionError("NumPy conversion must happen after dispatch")

            def __dlpack__(self, *args, **kwargs):
                raise AssertionError("DLPack must be requested after dispatch")

            def __dlpack_device__(self):
                raise AssertionError("DLPack device must be queried after dispatch")

        source = Foreign()
        for function in (wp.from_numpy, wp.from_torch, wp.from_dlpack):
            with self.subTest(function=function.__name__):
                self.assertIs(function(source), _HANDLED)

    def test_conversion_gradient_is_a_direct_operand(self):
        gradient = _Probe()
        source = object()
        self.assertIs(wp.from_torch(source, grad=gradient), _HANDLED)
        self.assertIs(_Probe.calls[0][1], wp.from_torch)
        self.assertEqual(_Probe.calls[0][3], (source,))
        self.assertIs(_Probe.calls[0][4]["grad"], gradient)


def test_native_delegation(test, device):
    _ObservedArray.calls.clear()
    value = _ObservedArray([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    test.assertEqual(_ObservedArray.calls, [])

    result = wp.clone(value)
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.clone])
    np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0])
    test.assertIs(type(result), wp.array)

    pointer = value.ptr
    alias = wp.array(ptr=pointer, shape=value.shape, dtype=wp.float32, device=device)
    replacement = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=device)
    test.assertIsNone(wp.copy(dest=value, src=replacement))
    test.assertEqual(value.ptr, pointer)
    np.testing.assert_allclose(alias.numpy(), [4.0, 5.0, 6.0])
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.clone, wp.copy])

    for function, expected in ((wp.zeros_like, 0.0), (wp.ones_like, 1.0)):
        _ObservedArray.calls.clear()
        result = function(value)
        test.assertEqual([call[0] for call in _ObservedArray.calls], [function])
        np.testing.assert_allclose(result.numpy(), np.full(3, expected))
    _ObservedArray.calls.clear()
    result = wp.full_like(value, 7.0)
    np.testing.assert_allclose(result.numpy(), np.full(3, 7.0))
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.full_like])


def test_launch_mutates_inputs_once(test, device):
    value = _ObservedArray([1.0, 2.0], dtype=wp.float32, device=device)
    _ObservedArray.calls.clear()
    test.assertIsNone(wp.launch(_increment, 2, inputs=[value], device=device))
    with wp.ScopedWarpFunctionDisable():
        np.testing.assert_allclose(value.numpy(), [2.0, 3.0])
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.launch])


def test_launch_tiled_composition(test, device):
    value = _ObservedArray([1.0, 2.0], dtype=wp.float32, device=device)
    _ObservedArray.calls.clear()
    test.assertIsNone(wp.launch_tiled(_increment_tiled, dim=2, inputs=[value], device=device, block_dim=1))
    with wp.ScopedWarpFunctionDisable():
        np.testing.assert_allclose(value.numpy(), [2.0, 3.0])
    test.assertEqual([call[0] for call in _ObservedArray.calls], [wp.launch_tiled])


class TestFunctionDispatchDevices(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestFunctionDispatchDevices, "test_native_delegation", test_native_delegation, devices=devices)
add_function_test(
    TestFunctionDispatchDevices, "test_launch_mutates_inputs_once", test_launch_mutates_inputs_once, devices=devices
)
add_function_test(
    TestFunctionDispatchDevices, "test_launch_tiled_composition", test_launch_tiled_composition, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
