# Warp Function Override Protocol

**Status**: Implemented

**Issue**: [GH-1834](https://github.com/NVIDIA/warp/issues/1834)

**Dependency**: [GH-1723](https://github.com/NVIDIA/warp/issues/1723), array subclass input compatibility

## Motivation

Add a `__warp_function__` protocol that lets array subclasses and wrappers intercept
supported public Warp functions. Users keep calling APIs such as `wp.launch`,
`wp.clone`, and `wp.copy`; downstream libraries can observe the operation, execute
an alternative implementation, or delegate to Warp and process the result.

LEAPP motivates this extension. Its proposed Warp integration associates arrays
with tracing context and FX graph proxies through a `TracedWpArray` subclass.
The background notes in `LEAPP--FC.md` describe a global patcher that wraps Warp
functions to propagate tracing metadata and discover regions for API capture.
An operand dispatch protocol gives those responsibilities a supported entry point
and works when callers retain imported references to a supported function.

The hook addresses interception of host Python calls involving tracked operands.
LEAPP still owns region discovery, framework transitions, mutation tracking, and
export. CUPTI monitoring, `.wrp` serialization, PTX portability, ONNX custom
operators, and loading captures from bytes are separate concerns.

Subclass compatibility and function dispatch solve different problems. GH-1723
allows a properly initialized subclass to pass the same validation as its base
array family. This proposal adds control over an operation and its results;
it does not require Warp to preserve subclass types or metadata automatically.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Intercept `wp.launch` before kernel specialization, argument packing, or submission | Must | Include positional and keyword calls and all four argument sequences |
| R2 | Provide the `func`, `types`, `args`, and `kwargs` protocol arguments | Must | Preserve the original call and omitted defaults |
| R3 | Support array subclasses and wrappers that implement the protocol | Must | Wrappers must adapt their operands before default execution when necessary |
| R4 | Specify deterministic precedence and `NotImplemented` fallback | Must | One invocation per participating concrete type |
| R5 | Provide safe delegation to the default implementation | Must | No global patching or process-wide dispatch switch |
| R6 | Preserve default validation, return values, ownership, streams, and side effects | Must | Include mutation, gradients, tape recording, and capture |
| R7 | Document an explicit set of participating APIs and intentional exclusions | Must | Coverage is part of the public contract |
| R8 | Keep overhead small for ordinary arrays | Should | Avoid data access and runtime signature inspection on the common path |

**Non-goals:**

- Dispatch from compiled kernel code, `@wp.func` calls inside kernels, or native
  runtime operations.
- Automatic subclass preservation in allocations, views, gradients, or conversions.
- Implicit wrapper unwrapping, a new array storage ABI, or subclass-specific kernel
  annotations such as `CustomArray[float]`.
- Observing every allocation, CUDA operation, command execution, or graph replay.
- Implementing LEAPP tracing or changing capture formats and deployment runtimes.

## Design

### Protocol contract

A participating type defines a synchronous class method:

```python
@classmethod
def __warp_function__(cls, func, types, args=(), kwargs=None): ...
```

The arguments have the following meaning:

| Argument | Meaning |
| --- | --- |
| `cls` | The concrete participating class, including when it inherits a user override |
| `func` | The public dispatching callable, identical to the corresponding `wp` export |
| `types` | A tuple of unique participating concrete classes, in dispatch order |
| `args` | The original positional argument tuple |
| `kwargs` | A dictionary containing only the original keyword arguments |

Warp does not normalize the hook's arguments, insert defaults, replace array
objects, or rewrite positional arguments as keywords. A hook can inspect all
operands through `args` and `kwargs`; `types` is not a complete list of operand
types because it excludes objects without an active override.

Any result other than the singleton `NotImplemented` completes dispatch. This
includes `None`, `False`, and empty containers. Warp returns that result unchanged
and does not then run its own implementation. Exceptions from a hook propagate;
they do not select the next candidate.

A hook must decide whether it can handle the function and participating types
before making execution or tracing changes. Returning `NotImplemented` must leave
operands, argument containers, and tracing state unchanged. Warp cannot roll back
an allocation, mutation, capsule consumption, kernel submission, or graph update
performed by a declining hook.

### Candidate discovery

Resolve `__warp_function__` on `type(value)`, rather than on the instance. This
avoids evaluating arbitrary instance properties and makes dispatch a type-level
contract. An absent hook or a hook set to `None` does not participate. A non-callable
hook produces a `TypeError`. Implementations must use the class method convention
above; assigning a hook to an individual instance is unsupported.

Each registered function supplies a small argument extractor. The extractor
visits supplied arguments in declared parameter order, regardless of keyword
insertion order. Extra positional and keyword arguments, where supported, follow
their original order. It uses parameter metadata prepared at registration time;
it does not call `inspect.signature()` on each invocation. Normal Python call-shape
errors, such as duplicate arguments, remain errors before hook invocation.

All direct argument values are checked. For `launch`, additionally inspect the
elements of `inputs`, `outputs`, `adj_inputs`, and `adj_outputs`, whether those
parameters were supplied positionally or by keyword. `launch_tiled` uses the same
sequence extraction rules. Traverse only objects whose exact type is `list` or
`tuple`, recursively, with identity-based cycle protection. An object that has an
active hook is treated as a participant rather than traversed.

Do not iterate arbitrary generators or custom sequence implementations, traverse
dictionary contents, inspect object attributes, or descend into Warp struct
fields. A custom argument container can implement its own hook. Additional
structured discovery can be added through explicit extractors later. Discovering
a hook in a nested list does not make that list a valid native kernel argument;
default execution still applies the existing argument rules.

Collect each concrete participating type once, even when several operands have
that type or one object occurs in several parameters. Two concrete subclasses
that inherit the same user hook still participate separately, with their own
`cls` binding. Do not deduplicate by hook function identity.

### Dispatch order

Use actual Python inheritance, as represented by the class MRO, for precedence.
For each newly encountered type, insert it immediately before the first existing
type that is its strict superclass. If there is no such superclass, append it.
The resulting list becomes `types` and determines hook invocation order.

For example, encountering `[Base, Other, Child]`, where `Child` inherits from
`Base` and `Other` is unrelated, produces `[Child, Base, Other]`. This defines the
interaction between subclass priority and encounter order without relying on an
ambiguous comparison sort. Virtual ABC registration does not change precedence.

For `copy`, the destination is encountered before the source. For `launch`, the
forward inputs precede forward outputs, then adjoint inputs and adjoint outputs,
according to its declared parameter order. Reordering keyword arguments does
not change that order.

### Default behavior and deferral

Add a neutral `__warp_function__` class method to the common `Array` base in
[`types.py`](../warp/_src/types.py). It returns `NotImplemented`. Warp excludes
that exact inherited implementation from candidate discovery, so ordinary arrays
and subclasses without a user override do not become dispatch candidates.

The neutral base supports `super().__warp_function__(func, types, args, kwargs)`
as a deferral mechanism. It does not execute the operation. An application
superclass may implement its own behavior, in which case normal Python `super()`
resolution applies. Calling `super()` does not itself restart Warp's candidate
selection.

Try each candidate until one returns a result. If every candidate declines, run
Warp's default implementation once with the original operands. This intentionally
differs from NumPy's all-decline `TypeError`: a Warp subclass can decline a call
and retain native behavior. NumPy's protocol and argument extraction model provide
the comparison point in [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html).

Fallback does not strip metadata or extract storage from wrappers. A wrapper
that the native API cannot already accept must handle the call and supply valid
operands itself. Otherwise the usual native validation error is raised. A wrapper
already accepted through an existing array interface retains that behavior.

### Delegation and reentrancy

Expose `wp.ScopedWarpFunctionDisable()` to disable this protocol in the current
execution context. An override uses it to call `func` with the original subclass
operands, or with explicitly adapted wrapper operands, and then processes the
result after leaving the scope.

This example observes successful operations without preserving the result's
subclass type:

```python
import warp as wp


class ObservedArray(wp.array):
    calls = []

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, ObservedArray) for t in types):
            return NotImplemented

        kwargs = {} if kwargs is None else kwargs
        with wp.ScopedWarpFunctionDisable():
            result = func(*args, **kwargs)

        cls.calls.append(func)
        return result


with wp.ScopedWarpFunctionDisable():
    x = ObservedArray([1.0, 2.0], dtype=wp.float32, device="cpu")

y = wp.clone(x)
wp.copy(dest=x, src=y)
assert ObservedArray.calls == [wp.clone, wp.copy]
```

This example uses the subclass compatibility delivered with the protocol. `super()` is appropriate
for deferral; the disable scope is appropriate for executing Warp and examining
or wrapping its result. A handler that takes ownership of a mixed-type operation
must coordinate with those types before disabling their hooks.

Keep dispatch enabled while a hook runs. This allows a hook to intentionally call
a different supported API and receive that API's hooks. Track active dispatching
functions; re-entering the same function from its hook without a disable scope
raises a `RuntimeError` with delegation guidance instead of overflowing the stack.

Execute every default function body under dispatch suppression, including when
no custom candidate was found. Consequently, a public `clone` call is one observed
operation; its internal `empty_like` and `copy` calls do not generate additional
hook events. Similarly, `launch_tiled` is observed under its own identity and its
internal `launch` is suppressed. This boundary prevents duplicate tracing and
makes observation independent of helper composition.

Native array initialization also runs under internal suppression. For example,
constructing an array from host data currently calls `warp.copy` internally;
that copy must not invoke a subclass hook before its initialization is complete.
Subclass authors should keep suppression active until their own tracing metadata
is attached, as shown by the explicit construction scope in the example.

Use `ContextVar` state for suppression depth and active functions. Nested disable
scopes are supported, and tokens are restored in `finally` blocks after success
or failure. The protocol does not share a mutable enable flag between threads or
execution contexts. This isolation does not change Warp's runtime, tape, or
capture concurrency guarantees. Hooks are synchronous; scopes cover synchronous
execution and follow Python's normal context inheritance rules.

### Initial API coverage

The following APIs participate as one documented set.

| APIs | Discovery and behavior |
| --- | --- |
| `wp.launch`, `wp.launch_tiled` | Direct arguments and the four launch argument sequences |
| `wp.copy` | Include both destination and source; preserve the `None` return value |
| `wp.clone` | Observe the outer clone operation and allow explicit result wrapping |
| `wp.empty_like`, `wp.zeros_like`, `wp.ones_like`, `wp.full_like` | Include the template array and other direct operands |
| `wp.from_numpy` | Allow a NumPy subclass or wrapper with this hook to handle import |
| `wp.from_torch`, `wp.to_torch` | Include direct source and gradient operands; retain existing return variants |
| `wp.from_dlpack`, `wp.to_dlpack` | Discover before requesting or consuming a capsule |

These are host function hooks. Plain NumPy arrays and PyTorch tensors do not
implement this new protocol automatically. LEAPP can add it to its traced foreign
types to observe calls such as `wp.from_torch(traced_tensor)`.

Array constructors, array methods such as `.numpy()`, `.to()`, `.zero_()`, and
`.reshape()`, Python operators, `wp.map`, reductions, scans, sorting, Paddle and
JAX integrations, mesh and volume APIs, and other functions are outside the
initial set. Their internal calls to a covered function can still be intercepted
when they occur outside a suppressed default implementation, but this is not a
stable hook for the outer operation.

Factories such as `wp.zeros(shape=...)` have no array operand from which to select
a hook and are not registered in this phase. A library that needs to observe
array-free operations or uncovered methods must use explicit tracing regions or
additional instrumentation. The first release can replace LEAPP patches for the
covered operand calls; removing its entire patcher requires broader coverage.

For `wp.launch(record_cmd=True)`, the hook observes command construction, which
returns a `Launch`, rather than kernel execution. Subsequent `Launch.launch()`
calls do not re-enter this protocol. CUDA graph replay and API capture replay
also do not invoke these host hooks. Command execution and replay observation
would require separately designed hooks.

### Mutation and framework interoperability

The protocol passes the actual operands, retaining repeated-object identity and
aliasing. Warp does not clone an array to adapt its type. An override that delegates
must preserve destination storage and ownership, especially for `copy` and
`launch`. Replacing a mutated destination with a newly allocated wrapper changes
the caller's observable behavior and is not transparent delegation.

The `inputs` and `outputs` launch groups are not complete read and write
declarations. A kernel can mutate an array supplied through `inputs`. LEAPP must
account for kernel mutation and shared storage when updating proxies, rather than
infer all writes from `outputs` or a returned value.

Result propagation belongs to the override. A fresh allocation can receive new
metadata, while a shared-storage conversion must preserve the relationship to the
source and retain its storage owner. A generic constructor call or reassignment
of `__class__` is not a supported wrapping recipe. Warp does not copy arbitrary
subclass attributes or attach FX proxies.

Conversion hooks run before importing an optional framework, requesting DLPack
data, inspecting an array interface, or synchronizing a stream. Discovery never
reads array contents or calls `.numpy()`, `__array__`, `__dlpack__`, or
`__cuda_array_interface__`. The delegated implementation retains existing device,
stream, gradient, lifetime, and `return_ctype` behavior. A DLPack capsule must be
consumed once, after the handling decision.

### Subclass compatibility prerequisite

This protocol builds on the separate subclass compatibility implementation for
GH-1723 in the base branch.
Forward argument packing in [`context.py`](../warp/_src/context.py) accepts
subclasses through their registered array family. Generic type inference and
type-code generation in [`types.py`](../warp/_src/types.py) use the same family
normalization, and `map` uses it in generated annotations and cache keys.

Normalize subclass instances to their registered array family for kernel type
matching, while retaining the original Python object for ownership, gradients,
and tracing. Apply the ordinary dtype, dimensionality, layout, and configured
device-access checks. Do not route a Warp subclass through foreign array-interface
conversion as a substitute: that path can bypass native checks and does not carry
the same gradient descriptor semantics.

This prerequisite does not change annotation support or require output types to
match input subclasses. The dispatch layer likewise leaves annotations and the
native array descriptor ABI unchanged.

### Implementation boundaries

Add an internal [`warp/_src`](../warp/_src) dispatch module containing the decorator,
argument extractor metadata, neutral hook, context state, and disable scope. Keep
its dependencies limited to the standard library so that early-loaded array
types can use it without importing `context` or `utils` and creating a cycle.

Decorate functions where they are defined in `context.py`, `torch.py`, and
`dlpack.py`, before their public re-exports in `warp/__init__.py`. Use
`functools.wraps` to preserve names, docstrings, annotations, and introspected
signatures. The hook receives the dispatch wrapper, so comparisons such as
`func is wp.launch` and registries keyed by public callables are reliable. Do not
install wrappers dynamically at tracing start or only patch the `wp` namespace.

The decorator dispatches before the original function body initializes devices,
validates array metadata, specializes kernels, records work, or performs
conversion. Once default execution is selected, existing validation and native
execution remain responsible for those operations. No C++ or CUDA changes are
needed for the hook itself.

Maintain a single internal registry for the participating functions and their
extractors. Use it to check documentation coverage and function identity. Avoid
caching negative hook lookups on user types without an invalidation strategy;
replacing a class hook must not leave stale dispatch behavior.

Capture the originating application call site in the dispatch context, without
retaining Python frame objects. Use it for tape records and allocation/autograd
diagnostics during default delegation. Merely skipping wrapper frames is
insufficient: a user override would otherwise become the first external caller.
Nested independent dispatches get their own originating location. Preserve the
existing default execution exactly once, including tape records, retained command
operands, and capture recording.

The ordinary-array path performs only type-level eligibility checks, declared
container traversal, and context bookkeeping. It performs no data movement or
optional framework import. Measure host overhead for repeated ordinary launches,
copies, and allocation helpers before merging; optimize extraction if needed
without weakening discovery or introducing timing assertions in unit tests.

### Alternatives considered

**Continue global monkey-patching.** It can cover more operations immediately,
but interacts with imported aliases, patch installation order, other integrations,
and restoration. It also makes LEAPP depend on Warp implementation details.

**Add a global observer callback.** This can observe operations without tracked
operands, but does not provide type-specific override selection or result
propagation. It would be a separate instrumentation API.

**Wrap native launches only.** This sees submissions but misses allocation helpers,
conversions, and the original Python operands needed for metadata propagation.

**Automatically preserve every subclass.** Warp cannot infer constructor
requirements, metadata policies, or wrapper ownership rules. Explicit overrides
let each library choose its result behavior.

**Raise an error whenever all hooks decline.** This follows the usual NumPy
override rule but prevents an observing subclass from selectively deferring to
native Warp. Native fallback better matches the requested behavior; invalid
operands still fail native validation.

## Testing Strategy

Add `warp/tests/test_function_dispatch.py` to `default_suite` in
`warp/tests/unittest_suites.py`. Use `unittest.TestCase` for device-independent
dispatch tests and `add_function_test()` with `get_test_devices()` where actual
kernel execution must cover CPU and CUDA.

| Area | Required checks |
| --- | --- |
| Discovery | Positional and keyword calls; keyword reordering; hooks found only in outputs or adjoint sequences; nested list and tuple traversal; cycles; no generator consumption or attribute probing |
| Precedence | Child before parent; unrelated encounter order; multiple inheritance; repeated types; distinct types inheriting one hook; neutral base arrays do not preempt wrappers |
| Results | Deferral to another hook and then native fallback; `None`, `False`, and empty results; no extra default execution after a handled call; exceptions propagate |
| Delegation | Explicit disable scope; nested scopes; restoration after exceptions; recursive same-function error; different-function calls from hooks; isolation across execution contexts |
| API identity | Every registered function passes its public callable; imported aliases work; signatures and documentation remain discoverable |
| Composition | One `clone` event with no inner allocation or copy events; one `launch_tiled` event; no hooks during native construction; `super()` defers rather than executes |
| Native validation | Ordinary arrays and subclasses retain dtype, dimensionality, layout, generic specialization, and configured device-access behavior |
| Mutation | Destination identity and storage aliases remain intact; kernels can mutate input operands; copies execute once |
| Autodiff and recording | Correct gradients; one tape and capture record; original user call site; recorded commands retain storage owners; hook timing matches command construction |
| Interoperability | Foreign traced types can intercept import; shared-storage conversion retains owners; existing gradient and descriptor return variants; DLPack consumption occurs once |

Keep kernel fixtures in real `.py` files and use public `warp` imports. Verify
numerical results with `np.testing.assert_allclose()` and synchronize the tested
device when an asynchronous test has no implicit synchronization. Do not clear
kernel or LTO caches from the tests and do not add timing assertions.

Run the new dispatch tests together with the affected array, launch, tape,
interop, and API capture tests. Import and documentation checks should also
exercise installations without optional PyTorch support. A small LEAPP-style
test should delegate a launch, convert a tracked operand to PyTorch when available,
and preserve tracing metadata through explicit wrapping without patching Warp.

## Delivery

The implementation builds on GH-1723 subclass compatibility and includes the full
initial coverage table, protocol and device regression tests, and recording and
interoperability tests. The public Python guide documents deferral versus
execution, method and replay exclusions, and the supported API list. The protocol
has its own Towncrier fragment; GH-1723 records the prerequisite separately.

Array methods and additional helpers are candidates for a follow-up. Extending
coverage changes what a downstream tracer observes, so additions should be
documented and tested as public behavior.
