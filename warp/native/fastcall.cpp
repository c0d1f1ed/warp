// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// METH_FASTCALL wrappers for performance-critical native functions.
// Embedded in warp.dll and loaded as a Python extension module via importlib.
#include "warp.h"

#include <Python.h>

static PyObject* fastcall_float_to_half_bits(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "float_to_half_bits() takes exactly 1 argument");
        return nullptr;
    }

    double value = PyFloat_AsDouble(args[0]);
    if (value == -1.0 && PyErr_Occurred())
        return nullptr;

    uint16_t bits = wp_float_to_half_bits(static_cast<float>(value));
    return PyLong_FromUnsignedLong(bits);
}

static PyObject* fastcall_half_bits_to_float(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "half_bits_to_float() takes exactly 1 argument");
        return nullptr;
    }

    long bits = PyLong_AsLong(args[0]);
    if (bits == -1 && PyErr_Occurred())
        return nullptr;

    float value = wp_half_bits_to_float(static_cast<uint16_t>(bits));
    return PyFloat_FromDouble(static_cast<double>(value));
}

static PyMethodDef fastcall_methods[] = {
    { "float_to_half_bits", reinterpret_cast<PyCFunction>(fastcall_float_to_half_bits), METH_FASTCALL,
      "Convert a float to float16 bit pattern" },
    { "half_bits_to_float", reinterpret_cast<PyCFunction>(fastcall_half_bits_to_float), METH_FASTCALL,
      "Convert a float16 bit pattern to float" },
    { nullptr, nullptr, 0, nullptr },
};

static PyModuleDef fastcall_module = {
    PyModuleDef_HEAD_INIT, "_warp_fastcall", "Warp METH_FASTCALL native bindings", -1, fastcall_methods,
};

extern "C" WP_API PyObject* PyInit__warp_fastcall() { return PyModule_Create(&fastcall_module); }
