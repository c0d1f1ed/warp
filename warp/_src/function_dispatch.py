# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type-based dispatch for supported public Warp host functions.

This module deliberately depends only on the standard library so that array
types can use its neutral hook and initialization scope during import.
"""

import functools
import inspect
import sys
from contextvars import ContextVar

_suppression_depth = ContextVar("warp_function_suppression_depth", default=0)
_disable_scopes = ContextVar("warp_function_disable_scopes", default=())
_active_functions = ContextVar("warp_function_active_functions", default=())
_callsite = ContextVar("warp_function_callsite", default=None)

# A single registration table also makes the documented API coverage inspectable.
WARP_FUNCTIONS = {}


def default_warp_function(cls, func, types, args=(), kwargs=None):
    """Defer to the next override without executing the Warp implementation."""
    return NotImplemented


def get_warp_function_callsite():
    """Return the originating ``(filename, lineno, function)`` or ``None``.

    Store locations rather than frames so that dispatch does not retain operands
    or application stack frames. A disabled delegation retains its hook's origin.
    """
    return _callsite.get()


def _capture_warp_function_callsite():
    """Capture the application caller of a dispatch wrapper without retaining frames."""
    frame = sys._getframe(2)
    # Uncovered methods can call a covered function; retain their original
    # application caller just as allocation and copy diagnostics do.
    while frame is not None:
        package = frame.f_globals.get("__package__") or ""
        if package != "warp._src" and not package.startswith("warp._src."):
            break
        frame = frame.f_back
    location = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name) if frame is not None else None
    del frame
    return location


class ScopedWarpFunctionDisable:
    """Temporarily disable ``__warp_function__`` dispatch in the current context.

    Use this scope inside an override to execute a public Warp function with its
    native behavior. Nested scopes restore the preceding state even when an
    exception is raised. Other threads and independently entered contexts retain
    their own dispatch state.

    Example:
        An override can delegate and then process the native result::

            with wp.ScopedWarpFunctionDisable():
                result = func(*args, **kwargs)
    """

    def __enter__(self):
        token = _suppression_depth.set(_suppression_depth.get() + 1)
        _disable_scopes.set((*_disable_scopes.get(), token))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        scopes = _disable_scopes.get()
        _disable_scopes.set(scopes[:-1])
        _suppression_depth.reset(scopes[-1])


def suppress_warp_function(func):
    """Suppress dispatch throughout an internal implementation, such as initialization."""

    @functools.wraps(func)
    def suppressed(*args, **kwargs):
        token = _suppression_depth.set(_suppression_depth.get() + 1)
        try:
            return func(*args, **kwargs)
        finally:
            _suppression_depth.reset(token)

    return suppressed


class _ArgumentExtractor:
    """Precompute the call-shape validation and ordered operand locations."""

    def __init__(self, signature, sequence_parameters):
        self.signature = signature
        parameters = tuple(signature.parameters.values())
        sequence_parameters = frozenset(sequence_parameters)
        unknown = sequence_parameters.difference(signature.parameters)
        if unknown:
            raise ValueError(f"Unknown Warp function sequence parameters: {', '.join(sorted(unknown))}")

        positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        self.positional_names = tuple(p.name for p in parameters if p.kind in positional)
        self.keyword_names = frozenset(p.name for p in parameters if p.kind in keyword)
        self.has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
        self.has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
        self.required = tuple(
            (p.name, self.positional_names.index(p.name) if p.kind in positional else -1)
            for p in parameters
            if p.kind in (*positional, inspect.Parameter.KEYWORD_ONLY) and p.default is inspect.Parameter.empty
        )
        self.locations = tuple(
            (
                p.name,
                self.positional_names.index(p.name) if p.kind in positional else -1,
                p.kind in keyword,
                p.name in sequence_parameters,
            )
            for p in parameters
            if p.kind in (*positional, inspect.Parameter.KEYWORD_ONLY)
        )

    def candidates(self, args, kwargs):
        # Avoid binding a Signature on the ordinary path; use it only to report
        # an invalid call with Python's familiar argument-binding diagnostics.
        nargs = len(args)
        invalid = not self.has_varargs and nargs > len(self.positional_names)
        if not invalid:
            invalid = any(name in kwargs and name in self.keyword_names for name in self.positional_names[:nargs])
        if not invalid and not self.has_varkw:
            invalid = any(name not in self.keyword_names for name in kwargs)
        if not invalid:
            invalid = any(
                (index < 0 or index >= nargs) and (name not in kwargs or name not in self.keyword_names)
                for name, index in self.required
            )
        if invalid:
            self.signature.bind(*args, **kwargs)

        candidates = []
        seen_types = set()
        seen_containers = set()

        def visit(value, recurse=False):
            cls = type(value)
            hook = getattr(cls, "__warp_function__", None)
            if hook is not None and getattr(hook, "__func__", hook) is not default_warp_function:
                if not callable(hook):
                    raise TypeError(f"{cls.__name__}.__warp_function__ must be callable or None")
                if cls not in seen_types:
                    seen_types.add(cls)
                    for index, (previous, _) in enumerate(candidates):
                        # Actual MRO membership intentionally ignores virtual ABCs.
                        if previous in cls.__mro__[1:]:
                            candidates.insert(index, (cls, hook))
                            break
                    else:
                        candidates.append((cls, hook))
                return
            if recurse and cls in (list, tuple) and id(value) not in seen_containers:
                seen_containers.add(id(value))
                for item in value:
                    visit(item, recurse=True)

        for name, index, accepts_keyword, recurse in self.locations:
            if 0 <= index < nargs:
                visit(args[index], recurse)
            elif accepts_keyword and name in kwargs:
                visit(kwargs[name], recurse)
        if self.has_varargs:
            for value in args[len(self.positional_names) :]:
                visit(value)
        if self.has_varkw:
            for name, value in kwargs.items():
                if name not in self.keyword_names:
                    visit(value)
        return candidates


def warp_function(func=None, *, sequence_parameters=(), signature=None):
    """Register a public function for operand-based ``__warp_function__`` dispatch.

    Args:
        func: Function whose default body is executed under dispatch suppression.
        sequence_parameters: Parameters whose exact list and tuple values are
            recursively inspected in addition to direct argument values.
        signature: An explicit signature or callable supplying argument metadata
            for a forwarding function such as ``launch_tiled``.
    """
    if func is None:
        return functools.partial(warp_function, sequence_parameters=sequence_parameters, signature=signature)

    dispatch_signature = signature if isinstance(signature, inspect.Signature) else inspect.signature(signature or func)
    extractor = _ArgumentExtractor(dispatch_signature, sequence_parameters)

    @functools.wraps(func)
    def dispatch(*args, **kwargs):
        if _suppression_depth.get():
            if _callsite.get() is not None:
                return func(*args, **kwargs)
            origin_token = _callsite.set(_capture_warp_function_callsite())
            try:
                return func(*args, **kwargs)
            finally:
                _callsite.reset(origin_token)

        active = _active_functions.get()
        if dispatch in active:
            raise RuntimeError(
                f"Recursive Warp function dispatch for {func.__name__}; "
                "use wp.ScopedWarpFunctionDisable() to delegate to the default implementation"
            )

        candidates = extractor.candidates(args, kwargs)
        origin_token = _callsite.set(_capture_warp_function_callsite())
        try:
            if candidates:
                types = tuple(cls for cls, _ in candidates)
                active_token = _active_functions.set((*active, dispatch))
                try:
                    for _, hook in candidates:
                        result = hook(dispatch, types, args, kwargs)
                        if result is not NotImplemented:
                            return result
                finally:
                    _active_functions.reset(active_token)

            token = _suppression_depth.set(_suppression_depth.get() + 1)
            try:
                return func(*args, **kwargs)
            finally:
                _suppression_depth.reset(token)
        finally:
            _callsite.reset(origin_token)

    WARP_FUNCTIONS[dispatch] = extractor
    return dispatch
