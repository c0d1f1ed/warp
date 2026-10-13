# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for passing user-defined subclasses of the Warp array classes to Warp APIs (GH-1723).

A subclass instance is treated as its native array family (``wp.array``, ``wp.indexedarray``,
``wp.fabricarray`` or ``wp.indexedfabricarray``) wherever the parent class is accepted, while the
original object is retained at runtime.
"""

import collections
import gc
import unittest
import weakref
from collections.abc import Sequence
from typing import Any, Literal
from unittest import mock

import numpy as np

import warp as wp
import warp._src.codegen as codegen
import warp._src.types as wptypes
from warp.tests.test_fabricarray import _create_fabric_array_interface
from warp.tests.unittest_utils import (
    add_function_test,
    assert_np_equal,
    get_selected_cuda_test_devices,
    get_test_devices,
)

# ---------------------------------------------------------------------------------------------
# Application subclasses.
# The constructors require a keyword that Warp cannot know, and they record how often they run,
# so a library call that constructs a subclass instance is detected.
# ---------------------------------------------------------------------------------------------

constructor_calls = collections.Counter()


class TaggedArray(wp.array):
    def __init__(self, *args, tag, **kwargs):
        constructor_calls[type(self).__name__] += 1
        self.tag = tag
        super().__init__(*args, **kwargs)


class SecondTaggedArray(TaggedArray):
    pass


class ThirdLevelArray(SecondTaggedArray):
    pass


class Mixin:
    """An ordinary mixin without any Warp base class."""

    def hello(self):
        return "hello"


class MixinArray(Mixin, TaggedArray):
    pass


class LeftArray(TaggedArray):
    pass


class RightArray(TaggedArray):
    pass


class DiamondArray(LeftArray, RightArray):
    pass


class TaggedIndexedArray(wp.indexedarray):
    def __init__(self, *args, tag, **kwargs):
        constructor_calls[type(self).__name__] += 1
        self.tag = tag
        super().__init__(*args, **kwargs)


class IndexedChild(TaggedIndexedArray):
    pass


class TaggedFabricArray(wp.fabricarray):
    def __init__(self, *args, tag, **kwargs):
        constructor_calls[type(self).__name__] += 1
        self.tag = tag
        super().__init__(*args, **kwargs)


class FabricChild(TaggedFabricArray):
    pass


class TaggedIndexedFabricArray(wp.indexedfabricarray):
    def __init__(self, *args, tag, **kwargs):
        constructor_calls[type(self).__name__] += 1
        self.tag = tag
        super().__init__(*args, **kwargs)


class IndexedFabricChild(TaggedIndexedFabricArray):
    pass


class CrossFamilyArray(wp.array, wp.indexedarray):
    """Invalid: derives from two unrelated native array families."""


class CrossFamilyReversed(wp.indexedarray, wp.array):
    """Invalid: the same families in the opposite base order."""


class FixedChild(wp.fixedarray):
    pass


class HiddenMroMeta(type):
    def __getattribute__(cls, name):
        if name == "__mro__":
            return (cls, object)
        return super().__getattribute__(name)


class PropertyMroMeta(type):
    @property
    def __mro__(cls):
        return (cls, object)


FAMILIES = (wp.array, wp.indexedarray, wp.fabricarray, wp.indexedfabricarray)
TYPE_CODE_PREFIXES = ("a", "ia", "fa", "ifa")


# ---------------------------------------------------------------------------------------------
# Kernels and functions
# ---------------------------------------------------------------------------------------------


@wp.kernel
def read_regular(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel
def read_indexed(a: wp.indexedarray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_fabric(a: wp.fabricarray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_indexed_fabric(a: wp.indexedfabricarray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_any(a: Any, out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_generic_regular(a: wp.array[Any], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_generic_indexed(a: wp.indexedarray[Any], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_generic_fabric(a: wp.fabricarray[Any], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_generic_indexed_fabric(a: wp.indexedfabricarray[Any], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


GENERIC_KERNEL_BY_FAMILY = {
    wp.array: read_generic_regular,
    wp.indexedarray: read_generic_indexed,
    wp.fabricarray: read_generic_fabric,
    wp.indexedfabricarray: read_generic_indexed_fabric,
}


@wp.kernel
def square(values: wp.array[Any], result: wp.array[Any]):
    i = wp.tid()
    result[i] = values[i] * values[i]


@wp.kernel
def square_float(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * a[i]


@wp.kernel
def double_in_place(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] * 2.0


@wp.kernel
def sum_values(a: wp.array[float], loss: wp.array[float]):
    wp.atomic_add(loss, 0, a[wp.tid()])


@wp.kernel
def multiply_two(a: wp.array[float], out: wp.array[float]):
    out[wp.tid()] = 2.0 * a[wp.tid()]


@wp.kernel
def multiply_three(a: wp.array[float], out: wp.array[float]):
    out[wp.tid()] = 3.0 * a[wp.tid()]


@wp.struct
class Record:
    v: wp.vec3
    w: float


@wp.kernel
def read_record(a: wp.array[Record], out: wp.array[float]):
    record = a[wp.tid()]
    out[wp.tid()] = record.v[0] + record.v[1] + record.v[2] + record.w


@wp.kernel
def read_bfloat16(a: wp.array[wp.bfloat16], out: wp.array[float]):
    out[wp.tid()] = float(a[wp.tid()])


@wp.kernel
def read_or_default(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    if a.shape[0] > 0:
        out[i] = a[i]
    else:
        out[i] = -1.0


@wp.kernel
def read_vec3(a: wp.array[wp.vec3], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i][0] + a[i][1] + a[i][2]


@wp.kernel
def read_2d(a: wp.array2d[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i, 0]


@wp.kernel
def copy_generic(a: wp.array[Any], out: wp.array[Any]):
    i = wp.tid()
    out[i] = a[i]


wp.overload(copy_generic, [wp.array[wp.float32], wp.array[wp.float32]])


@wp.func
def square_element(a: float) -> float:
    return a * a


@wp.func
def split_element(a: float):
    return a + 1.0, a * 2.0


@wp.func
def add_elements(a: float, b: float):
    return a + b


@wp.func
def identify_array(a: wp.array[float]) -> wp.array[float]:
    return a


@wp.func
def identify_array(a: wp.array[wp.int32]) -> wp.array[wp.int32]:
    return a


@wp.func
def identify_generic_array(a: wp.array[Any]) -> wp.array[Any]:
    return a


@wp.func
def identify_generic_array(a: int) -> int:
    return a


@wp.func
def identify_indexed(a: wp.indexedarray[float]) -> wp.indexedarray[float]:
    return a


@wp.func
def identify_indexed(a: int) -> int:
    return a


@wp.struct
class ArrayBundle:
    a: wp.array[float]
    ia: wp.indexedarray[float]
    fa: wp.fabricarray[float]
    ifa: wp.indexedfabricarray[float]


@wp.kernel(enable_backward=False)
def read_bundle(bundle: ArrayBundle, out: wp.array[float]):
    i = wp.tid()
    out[i] = bundle.a[i] + bundle.ia[i] + bundle.fa[i] + bundle.ifa[i]


@wp.struct
class InnerStruct:
    arr: wp.array[float]


@wp.struct
class OuterStruct:
    inner: InnerStruct
    scale: float


@wp.kernel
def read_nested(o: OuterStruct, out: wp.array[float]):
    i = wp.tid()
    out[i] = o.inner.arr[i] * o.scale


@wp.func
def scaled_read(a: wp.array[float], i: int, scale: float = 3.0):
    return a[i] * scale


@wp.kernel
def read_with_default_argument(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = scaled_read(a, i)


@wp.kernel
def row_view_kernel(a: wp.array2d[Any], out: wp.array[Any]):
    i = wp.tid()
    row = a[i]  # creates a view, which reconstructs an array type from the argument type
    out[i] = row[0]


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def make_families(device, classes=(TaggedArray, TaggedIndexedArray, TaggedFabricArray, TaggedIndexedFabricArray)):
    """Return ``(array, expected values read by the kernels, typed kernel)`` for each native family."""
    cls_a, cls_ia, cls_fa, cls_ifa = classes
    values = np.arange(1, 5, dtype=np.float32)
    a = cls_a(tag="regular", data=values, dtype=float, device=device)
    indices = wp.array([3, 1, 0, 2], dtype=wp.int32, device=device)
    ia = cls_ia(tag="indexed", data=a, indices=indices)
    fa = cls_fa(tag="fabric", data=_create_fabric_array_interface(a, "a", [2, 2]), attrib="a")
    ifa = cls_ifa(tag="indexed fabric", fa=fa, indices=indices)
    permuted = values[[3, 1, 0, 2]]
    return [
        (a, values, read_regular),
        (ia, permuted, read_indexed),
        (fa, values, read_fabric),
        (ifa, permuted, read_indexed_fabric),
    ]


def make_deep_families(device):
    return make_families(device, (ThirdLevelArray, IndexedChild, FabricChild, IndexedFabricChild))


# ---------------------------------------------------------------------------------------------
# Pure Python tests (no device work)
# ---------------------------------------------------------------------------------------------


class TestArraySubclass(unittest.TestCase):
    def setUp(self):
        self.cpu = wp.get_device("cpu")

    def test_family_resolution(self):
        for (instance, _, _), family in zip(make_deep_families(self.cpu), FAMILIES, strict=True):
            with self.subTest(family=family.__name__):
                self.assertTrue(wptypes.is_array(instance))
                self.assertIs(wptypes.concrete_array_type(instance), family)

        for cls in (MixinArray, DiamondArray, SecondTaggedArray):
            instance = cls(tag="x", data=[1.0, 2.0], dtype=float, device=self.cpu)
            self.assertIs(wptypes.concrete_array_type(instance), wp.array)
        self.assertEqual(MixinArray(tag="m", shape=1, dtype=float, device=self.cpu).hello(), "hello")

        # exact classes and annotations keep reporting their own family
        self.assertIs(wptypes.concrete_array_type(wp.zeros(2, device=self.cpu)), wp.array)
        for family in FAMILIES:
            self.assertIs(wptypes.concrete_array_type(family[float]), family)
        self.assertIs(wptypes.concrete_array_type(wp.array(dtype=float)), wp.array)

    def test_non_arrays_keep_their_type(self):
        self.assertIs(wptypes.concrete_array_type(3.0), float)
        self.assertIs(wptypes.concrete_array_type(wp.float32), type)
        # class objects are not arrays, neither is a NumPy array
        self.assertIs(wptypes.concrete_array_type(TaggedArray), type)
        self.assertIs(wptypes.concrete_array_type(np.zeros(2)), np.ndarray)
        self.assertFalse(wptypes.is_array(TaggedArray))

    def test_mock_and_weak_proxies_are_not_array_subclasses(self):
        owner = TaggedArray([1.0, 2.0], tag="proxy owner", dtype=float, device=self.cpu)
        for value in (mock.Mock(spec=wp.array), weakref.proxy(owner)):
            with self.subTest(type=type(value)):
                self.assertIsInstance(value, wp.array)
                self.assertIsNone(wptypes._resolve_array_family(value))
                self.assertIs(wptypes.concrete_array_type(value), type(value))

    def test_cross_family_inheritance_is_rejected(self):
        for cls in (CrossFamilyArray, CrossFamilyReversed):
            with self.subTest(cls=cls.__name__):
                # an uninitialized instance is enough: the class decides
                instance = cls.__new__(cls)
                self.assertTrue(wptypes.is_array(instance))
                with self.assertRaisesRegex(TypeError, "native array family"):
                    wptypes.concrete_array_type(instance)
                with self.assertRaisesRegex(TypeError, "native array family"):
                    wptypes.infer_argument_types([instance], [wp.array[Any]])
                # Diagnostic rendering must remain usable even for ambiguous objects.
                instance.dtype = wp.float32
                instance.ndim = 1
                instance.shape = (1,)
                self.assertIn(cls.__name__, wptypes.type_repr(instance))

    def test_fixedarray_remains_a_distinct_compiler_type(self):
        class RaisingFixedArray(wp.fixedarray):
            def __init__(self, *args, **kwargs):
                raise AssertionError("Warp called an application fixedarray constructor")

        for shape in ((0,), (3,), (2, 3), (2, 1, 3)):
            for cls in (wp.fixedarray, FixedChild, RaisingFixedArray):
                with self.subTest(shape=shape, cls=cls.__name__):
                    value = cls.__new__(cls)
                    wp.fixedarray.__init__(value, dtype=wp.float32, shape=shape)
                    template = wp.array[wp.float32, Literal[len(shape)]]
                    self.assertIs(wptypes.concrete_array_type(value), wp.fixedarray)
                    self.assertFalse(wptypes.types_equal(value, template))
                    self.assertEqual(wptypes.get_type_code(value), f"a{len(shape)}f4")
                    self.assertTrue(wptypes.type_matches_template(value, template))
                    # These are compiler-only descriptors, including the None
                    # template path; inference must retain their fixed extents.
                    for argument in (value, None):
                        inferred = wptypes.infer_argument_types([argument], [value])[0]
                        self.assertIs(type(inferred), wp.fixedarray)
                        self.assertIs(inferred.dtype, wp.float32)
                        self.assertEqual(inferred.ndim, len(shape))
                        self.assertEqual(inferred.shape, shape)
                        self.assertEqual(inferred.size, value.size)
                        self.assertEqual(
                            codegen.Var.type_to_ctype(inferred), f"wp::fixedarray_t<{value.size}, wp::float32>"
                        )

    def test_real_mro_inference(self):
        bases = (TaggedArray, TaggedIndexedArray, TaggedFabricArray, TaggedIndexedFabricArray)
        for meta in (HiddenMroMeta, PropertyMroMeta):
            classes = tuple(meta(f"Hidden{base.__name__}", (base,), {}) for base in bases)
            for family, (value, _, _) in zip(FAMILIES, make_families(self.cpu, classes), strict=True):
                with self.subTest(metaclass=meta.__name__, family=family.__name__):
                    self.assertNotIn(family, type(value).__mro__)
                    self.assertIs(wptypes.concrete_array_type(value), family)
                    before = constructor_calls.copy()
                    self.assertEqual(wptypes.infer_argument_types([value], [Any]), [family[wp.float32]])
                    self.assertEqual(constructor_calls, before)

        class ForgedMroMeta(type):
            @property
            def __mro__(cls):
                return (cls, wp.array, object)

        provider_cls = ForgedMroMeta("UnrelatedProvider", (), {})
        provider = provider_cls()
        self.assertIs(wptypes.concrete_array_type(provider), provider_cls)
        with self.assertRaisesRegex(TypeError, "Unable to infer"):
            wptypes.infer_argument_types([provider], [Any])

    def test_subclass_annotations_remain_unsupported(self):
        for cls in (TaggedArray, TaggedIndexedArray, TaggedFabricArray, TaggedIndexedFabricArray):
            with self.subTest(cls=cls.__name__):
                self.assertFalse(wptypes.type_is_array(cls))
                with self.assertRaisesRegex(TypeError, "Subscript syntax not supported"):
                    cls[wp.float32]
                with self.assertRaisesRegex(TypeError, "Unrecognized type"):
                    wptypes.get_type_code(cls)

    def test_inference_describes_without_constructing(self):
        templates = (
            wp.array[Any, Literal[1]],
            wp.indexedarray[Any, Literal[1]],
            wp.fabricarray[Any, Literal[1]],
            wp.indexedfabricarray[Any, Literal[1]],
        )
        for (instance, _, _), template in zip(make_deep_families(self.cpu), templates, strict=True):
            with self.subTest(cls=type(instance).__name__):
                before = constructor_calls.copy()
                (inferred,) = wptypes.infer_argument_types([instance], [template])
                self.assertEqual(constructor_calls, before)
                self.assertIs(type(inferred), type(template))
                self.assertIs(inferred.dtype, wp.float32)
                self.assertEqual(inferred.ndim, 1)

        # a subclass instance used as a factory-style template must not be constructed either
        template = TaggedArray(tag="template", shape=(2, 2), dtype=float, device=self.cpu)
        before = constructor_calls.copy()
        (inferred,) = wptypes.infer_argument_types([None], [template])
        self.assertEqual(constructor_calls, before)
        self.assertEqual(inferred, wp.array2d[wp.float32])

    def test_parent_and_subclass_share_one_type(self):
        parent = wp.zeros(3, dtype=float, device=self.cpu)
        child = TaggedArray(tag="child", shape=3, dtype=float, device=self.cpu)
        sibling = SecondTaggedArray(tag="sibling", shape=3, dtype=float, device=self.cpu)
        inferred = [wptypes.infer_argument_types([a], [wp.array[Any]])[0] for a in (parent, child, sibling)]
        self.assertEqual(inferred[0], inferred[1])
        self.assertEqual(inferred[0], inferred[2])
        self.assertEqual(len({wptypes.get_signature([t]) for t in inferred}), 1)
        self.assertTrue(wptypes.types_equal(child, sibling))
        self.assertTrue(wptypes.types_equal(child, wp.array[float]))
        self.assertFalse(wptypes.types_equal(child, wp.array[wp.float64]))
        self.assertTrue(wptypes.type_matches_template(child, wp.array[float]))
        self.assertTrue(wptypes.type_matches_template(child, wp.array[Any]))
        self.assertFalse(wptypes.type_matches_template(child, wp.array[wp.float64]))
        self.assertFalse(wptypes.type_matches_template(child, wp.array2d[float]))
        self.assertFalse(wptypes.type_matches_template(child, wp.indexedarray[Any]))

    def test_families_stay_distinct(self):
        values = [instance for instance, _, _ in make_deep_families(self.cpu)]
        for i, lhs in enumerate(values):
            self.assertEqual(wptypes.get_type_code(lhs), f"{TYPE_CODE_PREFIXES[i]}1f4")
            for j, rhs in enumerate(values):
                self.assertEqual(wptypes.types_equal(lhs, rhs), i == j)

    def test_annotation_discovery_keeps_runtime_values(self):
        # General annotation discovery is deliberately unchanged. Runtime call
        # boundaries normalize values before placing types in an overload cache.
        for instance, _, _ in make_deep_families(self.cpu):
            self.assertIs(codegen.get_arg_type(instance), instance)
        annotation = wp.array[float]
        self.assertIs(codegen.get_arg_type(annotation), annotation)
        # C++ types use native families; runtime diagnostics keep Python names.
        for (instance, _, _), family in zip(make_deep_families(self.cpu), FAMILIES, strict=True):
            self.assertEqual(codegen.Var.type_to_ctype(instance), f"wp::{family.__name__}_t<wp::float32>")
            self.assertIn(type(instance).__name__, wptypes.type_repr(instance))

    def test_annotations_are_not_runtime_values(self):
        # a lightweight annotation passed as a value is still not an array that can be inferred or launched
        with self.assertRaisesRegex(TypeError, "Unable to infer"):
            wptypes.infer_argument_types([wp.array[float]], [Any])

    def test_subclass_instance_parameter_annotations_are_rejected(self):
        for annotation, _, _ in make_families(self.cpu):
            with self.subTest(family=type(annotation).__name__):

                def invalid_kernel(a: annotation):
                    pass

                def invalid_function(a: annotation):
                    return a

                class InvalidStruct:
                    arr: annotation

                for decorator, definition in (
                    (wp.kernel, invalid_kernel),
                    (wp.func, invalid_function),
                    (wp.struct, InvalidStruct),
                ):
                    with self.subTest(decorator=decorator.__name__):
                        with self.assertRaisesRegex(TypeError, "annotation"):
                            decorator(definition)

    def test_family_resolution_does_not_retain_dynamic_classes(self):
        def resolve():
            dynamic_cls = type("TemporaryArray", (TaggedArray,), {})
            array = dynamic_cls(tag="temporary", shape=1, dtype=float, device=self.cpu)
            self.assertIs(wptypes.concrete_array_type(array), wp.array)
            self.assertEqual(wptypes.infer_argument_types([array], [Any]), [wp.array[wp.float32]])
            return weakref.ref(dynamic_cls), weakref.ref(array)

        class_ref, array_ref = resolve()
        gc.collect()
        self.assertIsNone(array_ref())
        self.assertIsNone(class_ref())

    def test_custom_metaclasses_use_real_family_identity(self):
        class UnhashableMeta(type):
            __hash__ = None

        class EqualToArrayMeta(type):
            def __eq__(cls, other):
                return cls is other or other is wp.array

            def __hash__(cls):
                return hash(wp.array)

        for meta in (UnhashableMeta, EqualToArrayMeta):
            with self.subTest(metaclass=meta.__name__):
                classes = tuple(
                    meta(f"Meta{base.__name__}", (base,), {})
                    for base in (TaggedArray, TaggedIndexedArray, TaggedFabricArray, TaggedIndexedFabricArray)
                )
                for family, (array, _, _kernel), _field in zip(
                    FAMILIES, make_families(self.cpu, classes), ("a", "ia", "fa", "ifa"), strict=True
                ):
                    with self.subTest(family=family.__name__):
                        self.assertIsInstance(hash(array), int)
                        self.assertIs(wptypes.concrete_array_type(array), family)
                        before = constructor_calls.copy()
                        self.assertEqual(wptypes.infer_argument_types([array], [Any]), [family[wp.float32]])
                        self.assertEqual(constructor_calls, before)

                unrelated_cls = meta("Unrelated", (), {})
                unrelated = unrelated_cls()
                self.assertIs(wptypes.concrete_array_type(unrelated), unrelated_cls)
                self.assertFalse(wptypes.is_array(unrelated))
                with self.assertRaises(TypeError):
                    wptypes.infer_argument_types([unrelated], [Any])

    def test_scalar_function_calls_keep_fast_path(self):
        with mock.patch.object(wptypes, "_resolve_array_family", wraps=wptypes._resolve_array_family) as resolve:
            for value in (1, 2.0, True, int, float, wp.float32):
                self.assertIs(wptypes._normalize_array_argument_type(value), value)
            self.assertEqual(
                wptypes.infer_argument_types([1, 2.0, True], [Any, Any, Any]), [wp.int32, wp.float32, wp.bool]
            )
            resolve.assert_not_called()
        self.assertEqual(square_element(3.0), 9.0)
        self.assertEqual(identify_generic_array(3), 3)

    def test_regular_subclass_resolution_keeps_fast_path(self):
        mro_descriptor = wptypes._TYPE_MRO_DESCRIPTOR
        values = [
            cls(tag="fast path", shape=1, dtype=float, device=self.cpu)
            for cls in (TaggedArray, MixinArray, DiamondArray)
        ]

        class CountMroReads:
            reads = 0

            def __get__(self, cls, _owner=None):
                self.reads += 1
                return mro_descriptor.__get__(cls)

        counted_mro = CountMroReads()
        with mock.patch.object(wptypes, "_TYPE_MRO_DESCRIPTOR", counted_mro):
            for value in values:
                self.assertIs(wptypes._resolve_array_family(value), wp.array)
            self.assertIsNone(wptypes._resolve_array_family(object()))
        self.assertEqual(counted_mro.reads, 0)


# ---------------------------------------------------------------------------------------------
# Tests running on every test device
# ---------------------------------------------------------------------------------------------


def test_explicit_overload_is_shared(test, device):
    overload = wp.overload(copy_generic, [wp.array[wp.float32], wp.array[wp.float32]])
    src = TaggedArray([1.0, 2.0], tag="src", dtype=float, device=device)
    dst = TaggedArray(tag="dst", shape=2, dtype=float, device=device)
    test.assertIs(copy_generic.add_overload(copy_generic.infer_argument_types([src, dst])), overload)
    wp.launch(copy_generic, 2, inputs=[src], outputs=[dst], device=device)
    assert_np_equal(dst.numpy(), np.array([1.0, 2.0]))


def test_python_function_overloads(test, device):
    a = TaggedArray([1.0, 2.0], tag="first", dtype=float, device=device)
    b = SecondTaggedArray([3.0, 4.0], tag="second", dtype=float, device=device)
    parent = wp.clone(a)
    ints = TaggedArray([1, 2], tag="int overload", dtype=wp.int32, device=device)

    # explicit overloads select on the native type and return the original object
    for array in (a, b, parent, ints):
        test.assertIs(identify_array(array), array)

    # inferred overloads are shared between the parent and its subclasses
    test.assertIs(identify_generic_array(a), a)
    count = len(identify_generic_array.user_overloads)
    test.assertIs(identify_generic_array(b), b)
    test.assertIs(identify_generic_array(parent), parent)
    test.assertEqual(len(identify_generic_array.user_overloads), count)
    # one overload for float32 vectors, whose cached type is a lightweight annotation and not an array object
    array_overloads = [
        overload
        for overload in identify_generic_array.user_overloads.values()
        if wptypes.is_array(overload.input_types["a"]) and overload.input_types["a"] == wp.array[wp.float32]
    ]
    test.assertEqual(len(array_overloads), 1)
    test.assertIsInstance(array_overloads[0].input_types["a"], wptypes._ArrayAnnotationBase)

    indexed = make_families(device)[1][0]
    test.assertIs(identify_indexed(indexed), indexed)


def test_sequence_array_overloads(test, device):
    class SequenceArray(TaggedArray, Sequence):
        def __getitem__(self, key):
            raise AssertionError("Overload resolution must not iterate a Warp array")

    array = SequenceArray(tag="sequence", data=[1.0, 2.0], dtype=float, device=device)
    test.assertIsInstance(array, Sequence)
    test.assertIs(identify_array(array), array)
    test.assertIs(identify_generic_array(array), array)


def test_function_defaults(test, device):
    a = TaggedArray([1.0, 2.0], tag="defaults", dtype=float, device=device)
    out = wp.empty(2, dtype=float, device=device)
    wp.launch(read_with_default_argument, 2, inputs=[a], outputs=[out], device=device)
    assert_np_equal(out.numpy(), np.array([3.0, 6.0]))


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

add_function_test(TestArraySubclass, "test_shared_explicit_overload", test_explicit_overload_is_shared, devices=devices)
add_function_test(TestArraySubclass, "test_function_overloads", test_python_function_overloads, devices=devices)
add_function_test(TestArraySubclass, "test_sequence_overloads", test_sequence_array_overloads, devices=devices)
add_function_test(TestArraySubclass, "test_defaults", test_function_defaults, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
