Function Overrides
==================

.. currentmodule:: warp

Array subclasses and wrappers can implement ``__warp_function__`` to intercept
supported host Python operations, such as :func:`launch` or :func:`clone`.
The hook runs before Warp specializes a kernel, submits work, or converts a
source array. It receives the original operands and can return an alternative
result, observe the operation by delegating to Warp, or decline it.

Defining a Hook
---------------

Define the hook as a synchronous class method::

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        ...

``cls`` is the concrete participating type, including when it inherits the hook.
``func`` is the public callable: an override can compare it with ``wp.launch`` or
use it as a registry key. Imported aliases of a covered function also dispatch.
``types`` is the tuple of unique types with active overrides, in dispatch order.
It excludes ordinary arrays and other objects without an override.

``args`` and ``kwargs`` preserve the caller's positional tuple and supplied
keyword arguments. Warp does not insert omitted defaults, unwrap operands, or
replace them with copies. Any result other than the singleton ``NotImplemented``
finishes the operation and is returned unchanged, including ``None`` and
``False``. Exceptions propagate immediately.

Return ``NotImplemented`` before making changes if the hook cannot handle the
function or participating types. Warp tries the next candidate, then executes
its native implementation once if every candidate declines. Subclasses without
an override retain native behavior. The neutral hook inherited from Warp's array
base also returns ``NotImplemented``; ``super().__warp_function__(...)`` defers to
the superclass hook rather than executing Warp's implementation.

A declining hook must leave operands, argument containers, and tracing state
unchanged. Warp cannot roll back mutations, submitted work, or capsules consumed
by an override.

Native fallback uses the original operands and applies the usual validation.
A wrapper that Warp cannot otherwise accept must adapt its operands while
handling the call. Returning ``NotImplemented`` does not unwrap it.

Delegating to Warp
------------------

Use :class:`ScopedWarpFunctionDisable` to invoke Warp's implementation without
calling the same hook again. For example, this subclass records successful
operations::

    import warp as wp


    class ObservedArray(wp.array):
        calls = []

        @classmethod
        def __warp_function__(cls, func, types, args=(), kwargs=None):
            if not all(issubclass(t, ObservedArray) for t in types):
                return NotImplemented

            with wp.ScopedWarpFunctionDisable():
                result = func(*args, **({} if kwargs is None else kwargs))

            cls.calls.append(func)
            return result


    # Attach subclass metadata before enabling its hooks.
    with wp.ScopedWarpFunctionDisable():
        x = ObservedArray([1.0, 2.0], dtype=wp.float32, device="cpu")

    y = wp.clone(x)
    wp.copy(dest=x, src=y)
    assert ObservedArray.calls == [wp.clone, wp.copy]

Hooks remain enabled while an override runs, so it can call a different
supported function and receive that function's hooks. Re-entering the same
function without a disable scope raises ``RuntimeError`` with delegation
guidance. A hook handling mixed types should coordinate their behavior before
disabling their overrides.

Disable scopes nest, restore their state after exceptions, and affect the
current execution context rather than a process-wide flag. They follow Python's
normal context inheritance rules. This isolation does not change Warp's runtime,
tape, or capture concurrency guarantees.

Warp suppresses hooks throughout every selected native implementation, including
when no candidate participates. As a result, a call to ``clone`` produces one
event; its internal allocation and copy do not produce more events. Likewise,
``launch_tiled`` is observed under its own identity without an additional
``launch`` event. Native array initialization suppresses hooks while it prepares
the array; keep an explicit disable scope active until subclass metadata is ready.

Operand Discovery and Ordering
------------------------------

Warp resolves ``__warp_function__`` on the operand's type. Instance attributes
are not consulted. An absent hook or one set to ``None`` does not participate;
a non-callable hook raises ``TypeError``.

Direct supplied arguments are visited in the function's declared parameter
order, regardless of keyword insertion order. For ``launch`` and
``launch_tiled``, Warp also recursively inspects exact ``list`` and ``tuple``
containers in ``inputs``, ``outputs``, ``adj_inputs``, and ``adj_outputs``.
Discovery detects cycles and does not descend into an object with an active
hook. It does not iterate generators or custom sequences, traverse dictionaries
or struct fields, or inspect arbitrary object attributes. A custom argument
container can implement its own hook. Discovery does not make a nested container
a valid native kernel argument.

Each concrete participating type is called at most once. Two subclasses that
inherit the same user hook participate separately. On encountering a new type,
Warp inserts it before the first existing type that is its strict superclass,
or appends it if there is no such superclass. Actual Python inheritance controls
this priority; virtual ABC registration does not. Encountering ``Base``,
``Other``, then ``Child`` where ``Child`` inherits ``Base`` produces
``(Child, Base, Other)``.

For ``copy``, the destination precedes the source. For launches, forward inputs
precede forward outputs, then adjoint inputs and adjoint outputs. Normal Python
call errors such as duplicated parameters are raised before any hook runs.

Covered APIs
------------

The following public functions support the protocol:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Functions
     - Operands and behavior
   * - :func:`launch`, :func:`launch_tiled`
     - Direct arguments and the four launch argument sequences.
   * - :func:`copy`, :func:`clone`
     - Source and destination operands; delegated ``copy`` still returns ``None``.
   * - :func:`empty_like`, :func:`zeros_like`, :func:`ones_like`, :func:`full_like`
     - The template array and other direct arguments.
   * - :func:`from_numpy`
     - Direct arguments, including a NumPy subclass with a hook.
   * - :func:`from_torch`, :func:`to_torch`
     - Direct source and explicit gradient arguments; hooks run before an
       optional PyTorch import or conversion.
   * - :func:`from_dlpack`, :func:`to_dlpack`
     - Direct arguments; hooks run before requesting or consuming a capsule.

Plain NumPy arrays and PyTorch tensors do not acquire this hook automatically.
Discovery never reads array contents or invokes ``__array__``, ``__dlpack__``,
or an array-interface property. A delegated conversion retains Warp's existing
device, stream, gradient, lifetime, and ``return_ctype`` behavior. Decide whether
to handle a DLPack call before requesting its data; consume a capsule only once.

Storage, Mutation, and Results
------------------------------

The hook receives the actual objects, preserving repeated-object identity and
storage aliases. Delegation must preserve destination storage and ownership,
especially for ``copy`` and kernel launches. Replacing a destination with a
new allocation changes what the caller observes. A kernel can mutate an array
passed through ``inputs``; the launch input and output groups are not complete
read and write declarations.

Warp does not preserve subclass types or copy arbitrary metadata into results.
An override owns that policy. Fresh allocations can receive new metadata;
conversions sharing storage must retain the source owner and alias relationship.
Generic reconstruction or reassignment of ``__class__`` is not a supported
wrapping recipe. Subclass inputs use their registered Warp array family for
native validation and generic specialization while retaining their original
Python object and gradient descriptor. Kernel annotations continue to use
Warp's base array types, such as ``wp.array[float]``.

Coverage Limits
---------------

The protocol applies to covered host functions. Array constructors, methods such
as ``.numpy()``, ``.to()``, ``.zero_()``, and ``.reshape()``, Python operators,
``wp.map``, reductions, scans, sorting, Paddle and JAX integrations, and mesh and
volume APIs are outside this set. Internal calls to covered functions outside a
suppressed native implementation may dispatch, but they do not provide a stable
hook for the outer operation. Factories such as ``wp.zeros(shape=...)`` have no
array operand from which to select a hook and are not registered.

Compiled kernel operations and calls to ``@wp.func`` inside kernels do not
dispatch. With ``record_cmd=True``, a ``launch`` hook observes construction of a
:class:`Launch` object rather than its later execution. ``Launch.launch()``,
CUDA graph replay, and API capture replay do not invoke these host hooks.
Tracing systems that need array-free operations, uncovered methods, command
execution, or replay must provide their own instrumentation.
