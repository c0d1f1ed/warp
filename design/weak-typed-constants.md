# Weakly-Typed Constants

**Status**: In progress

**Issue**: [GH-1297](https://github.com/NVIDIA/warp/issues/1297)

## Motivation

Warp historically mapped Python's `float` to `float32` at annotation time. This
caused silent precision loss when users wrote literal constants in contexts that
expected higher (or lower) precision:

```python
# Before: the literal 3.141592653589793 was truncated to float32
x = wp.float64(3.141592653589793)   # x ≈ 3.1415927 (7 digits)

# After: the literal propagates at double precision through the compiler
x = wp.float64(3.141592653589793)   # x ≈ 3.141592653589793 (16 digits)
```

The same issue affected compound constructors (`wp.vec3d(1.1, 2.2, 3.3)`),
binary operators mixing literals with typed variables, and assignment to
higher-precision variables.

## Requirements

### Precision Adaptation

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R1  | Float literals adapt precision to surrounding typed context | Must | Core feature |
| R2  | Literal-only expressions compute in float64 for accuracy | Must | Emitted as C++ `double`; compiler constant-folds |
| R3  | Scalar constructors (`float32()`, `float64()`) produce strongly-typed results | Must | Prevents weak typing from leaking out |
| R4  | Compound constructors (`vec3d()`, `mat22f()`) pass literal precision to components | Should | Enables `wp.vec3d(1.1, 2.2, 3.3)` with full float64 precision |
| R5  | Builtin function results with weak float input preserve weak typing | Must | `wp.sqrt(3.0)` computes in float64 and adapts to context |
| R6  | Module constants (`wp.PI`, `wp.E`, `wp.TAU`) adapt precision in typed constructors | Must | `wp.float64(wp.PI)` preserves full double precision |
| R10 | Int literals adapt precision to surrounding typed context | Must | Mirrors R1 for int |
| R11 | Weak int compatible with any strong int type (int8..uint64) | Must | Int-to-int adaptation |
| R12 | Weak int promotes to weak float in mixed expressions | Must | `2 * 3.0` → weak float; matches Python/C++/CUDA |

R1 includes these concrete guarantees:
- `wp.float64(3.141592653589793)` produces the exact double-precision value
- Negative literals (`-3.14...`) preserve full precision in float64 contexts
- `float32_var * 2.0` continues to produce `float32` results (literal adapts down)

R5 details:
- Builtins with a `float64` overload select it when the input is a weakly-typed
  float, compute in float64, and return a weakly-typed float result.
- The same builtin with a strongly-typed `float64` input returns a strongly-typed
  `float64` result (the weak-to-weak propagation only applies when the input is
  itself weakly typed).
- The CUDA compiler is expected to constant-fold these calls when all inputs are
  compile-time constants. Verify by examining PTX output for the absence of
  runtime calls. Builtins confirmed to be constant-folded should be annotated
  accordingly.

R12 details:
- When a weak int and a weak float appear in the same binary expression (e.g.,
  `2 * 3.0`), the weak int is promoted to a weak float constant. The resulting
  float preserves full double precision, matching Python and C++/CUDA semantics.
- Weak int also promotes to strong float when mixed with a strongly-typed float
  operand (e.g., `2 * float32_var` → weak int cast to float32).
- Weak int promotes to strong float in compound constructors (e.g.,
  `wp.vec3d(1, 2, 3)` → int literals promoted to float64 components).
- If a weak int constant value exceeds the representable range of the target
  float type, raise a `WarpCodegenOverflowError` (derived from Python's
  `OverflowError`).

### Type Safety

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R7  | Two different strong types remain incompatible | Must | e.g., `float32` vs `float64` is always an error |
| R8  | Annotated `float` parameters remain float32 | Must | Warp kernel syntax: `float` is shorthand for `float32` |

### Overload Preference

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R13 | Weak float prefers float64 overloads for builtins | Must | Preserves precision; CUDA compiler constant-folds |
| R14 | Weak float prefers float64 overloads for user functions | Must | Consistency; avoids unexpected precision loss |
| R15 | Config option to restore legacy float32-default behavior | Must | Opt-in for users who prefer float32 performance |

### Backward Compatibility

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R9  | No impact on existing user code that only uses float32 | Must | All existing tests must pass |

**Non-goals**: None.

## Design

### Approach

Introduce **weakly-typed constants** where Python's built-in `float` and `int`
types are treated as distinct types during code generation, separate from Warp's
`float32` and `int32`. A weakly-typed constant:

- Emits as C++ `double` (float) or `int64` (int) for maximum range/precision
- Adapts to the first strongly-typed value found in the same expression
- Defaults to `float32` / `int32` when no typed context is available (backward compat)
The key insight is distinguishing two uses of `float`/`int` in user code:

1. **Annotated `float`/`int`** (function parameter types) — In Warp's kernel
   syntax, `float` is shorthand for CUDA's 32-bit `float` and `int` is
   shorthand for 32-bit `int`. These are converted to `float32`/`int32` early
   in `Adjoint.__init__` via `canonicalize_dtype()`. This is strongly typed.
   The same applies to builtin `input_types` — bare `int` means `int32` and
   bare `float` means `float32`, distinct from the capitalized generic type
   placeholders `Int`, `Float`, and `Scalar` (defined in `types.py`).
2. **Literal `float`/`int`** (from `add_constant` for numeric literals like
   `3.14` or `42`) — stays as Python `float`/`int` internally, emits as C++
   `double`/`int64`. This is weakly typed and adapts to context. Python has no
   single-precision literal syntax, so weak typing bridges the gap between
   Python's float64 literals and Warp's float32 default.

**Implementation invariant**: Function signatures (both builtins and user
functions) must contain only canonicalized types — never weak `float` or `int`.
Weak types exist exclusively as argument types during overload resolution.
Builtin registrations that currently use bare `int`/`float` in `input_types`
(e.g., `scalar_types_all`, `range`, `randu`) must be canonicalized to
`int32`/`float32`.

### Alternatives Considered

**Always use float64 for literals** — Partially adopted. Weak floats now prefer
float64 overloads (R13, R14), but they still adapt *down* to float32 when a
strongly-typed float32 context is present (e.g., `float32_var + 1.0` stays
float32). The key distinction from "always float64" is that weak floats remain
adaptable rather than being locked to float64.

**Add a new `WeakFloat` sentinel type** — Rejected as unnecessarily complex.
Reusing Python's `float` as the weak type is natural since it already exists in
the type system and the codegen already handles it.

**Const-expression tracking (`is_const_expr` / `preserves_const_expr`)** — An
earlier prototype tracked which expressions were "constant" to decide when to
preserve precision. This was removed because it coupled precision semantics to
constness, creating confusing edge cases. The current approach is simpler: the
*type* (`float` vs `float32`) carries the precision information directly.

### Key Implementation Details

#### Weak-Type Invariant

Weak `float` and `int` types exist only for **constant literals** — variables
created by `add_constant()` with a known value. When a `Var` is created without
a constant (i.e., a computed value), the type is canonicalized:

- `Var(type=float, constant=3.14)` → stays `float` (weakly typed)
- `Var(type=float, constant=None)` → canonicalized to `float32`
- `Var(type=int, constant=42)` → stays `int` (weakly typed)
- `Var(type=int, constant=None)` → canonicalized to `int32`

This ensures weak types never appear for computed intermediate values —
only for literals and module constants with known values.

#### Negative Literal Folding

Negative literals like `-3.14` are represented in Python's AST as
`UnaryOp(USub, Constant(3.14))`. To preserve precision, `emit_UnaryOp` folds
the negation *before* evaluating the operand, creating a single
`add_constant(-3.14)` call. Without this folding, the compiler would emit a
positive constant (`Var(type=float, constant=3.14)`) followed by a negation
operation, potentially losing the constant's value for downstream optimizations.

#### Type Flow Through the Compiler

```
Source: x = 3.14
  → emit_Constant → add_constant(3.14)
    → Var(type=float, constant=3.14)   # weakly typed

Source: y: float = 3.14
  → Adjoint.__init__ converts annotation float → float32
    → y is float32 (strongly typed)

Source: z = wp.float64(3.14)
  → _get_constructor_target_type → float64
    → add_constant(3.14, target_type=float64)
    → Var(type=float64, constant=3.14)  # strongly typed

Source: z = wp.float64(wp.PI)
  → _get_constructor_target_type → float64
    → emit_Attribute evaluates wp.PI → Var(type=float, constant=π)
    → emit_Call casts weak float arg to float64 via _cast_to
    → Var(type=float64, constant=π)    # strongly typed, full precision

Source: n = 42
  → emit_Constant → add_constant(42)
    → Var(type=int, constant=42)       # weakly typed

Source: n = wp.int64(42)
  → _get_constructor_target_type → int64
    → add_constant(42, target_type=int64)
    → Var(type=int64, constant=42)     # strongly typed
```

#### Overload Resolution

Generic overloads (those using `Float`/`Scalar` generic types) match `float` via
`scalar_types_match` — the function treats `float` as compatible with `Float`
and `Scalar` generic type placeholders.

For concrete overloads, weak float prefers `float64` when available (R13, R14).
This applies to both builtins and user-defined functions:

- **Builtins**: `wp.sin(1.0)` selects `sin(float64)` and returns a weakly-typed
  float result that adapts to context. The CUDA compiler constant-folds the
  float64 computation when all inputs are compile-time constants, so there is no
  runtime cost for literal-only expressions.
- **User functions**: When a user-defined function has both `float32` and
  `float64` overloads, weak float selects the `float64` overload. Unlike
  builtins, user functions are not guaranteed to be constant-folded, so this may
  incur a runtime cost for double-precision computation. Users who prefer float32
  performance can explicitly downcast: `func(wp.float32(1.0))`.
- **Fallback**: If no `float64` overload exists, weak float is cast to `float32`
  and resolution is retried. This handles builtins like `expect_eq` that only
  have concrete overloads, and user functions with a single `float` (= float32)
  parameter.

A configuration option (R15) controls this behavior. When disabled, the legacy
behavior is restored: weak float resolves to `float32` overloads by default.

##### Breaking Change

Previously, `array_f32[i] = func_x(1.0)` where `func_x` has both float32 and
float64 overloads would select the float32 overload. With the new behavior, the
float64 overload is selected, and the result must be explicitly downcast:

```python
# Before (implicit float32):
array_f32[i] = func_x(1.0)

# After (explicit downcast required):
array_f32[i] = wp.float32(func_x(1.0))
# Or direct the overload via the argument:
array_f32[i] = func_x(wp.float32(1.0))
```

This is expected to be rare in practice — user-defined functions seldom have
both float32 and float64 overloads, and are rarely called with only constant
arguments.

#### Casting Rules (in `add_call`)

After overload resolution, before calling `value_func`:

1. Scan all bound arguments for strongly-typed floats (scalar, compound scalar
   type, or array dtype).
2. If a strongly-typed float is found, cast all weakly-typed `float` arguments
   to that precision.
3. If **all** arguments are weakly-typed `float`, do nothing — they stay as
   `float` (C++ `double`) for maximum precision.

The `_cast_to` helper handles the actual conversion:
- For constants with known values → creates a new constant at the target type
  (e.g., `add_constant(3.14, target_type=float32)`)
- For computed values → first aliases the variable as `float64` (since weakly-
  typed `float` emits as C++ `double`, which is 64-bit), then applies a scalar
  cast builtin (e.g., `float32(alias)`)

#### Special Cases

- **User functions**: Annotated `float` parameters are always float32 (R8).
  When a user function has both float32 and float64 overloads, weak float
  arguments prefer the float64 overload (R14). When only a single `float`
  (= float32) parameter exists, the weak float is implicitly cast to float32.
  Resolution uses a two-phase approach: first, cast weak float/int arguments
  to float32/int32 and attempt resolution. If that fails (e.g., no matching
  overload), retry with float64. This pre-cast-before-resolution strategy
  avoids "no overload found" errors for the common case of single-precision
  user functions. Keyword arguments follow the same casting logic.
- **Scalar constructors** (`float32()`, `float64()`, etc.): Always produce
  strongly-typed results, preventing weak typing from propagating out.
  `_get_constructor_target_type` detects scalar constructors by `func.key`
  (not by checking for literal args), so module constants like `wp.PI` also
  get cast to the target precision rather than falling through to the
  `float32` retry path.
- **Compound constructors** (`vector`, `matrix`, `quaternion`, `transformation`):
  When called with a typed variant (e.g., `vec3d`), the target scalar type is
  detected via `_get_constructor_target_type` and literal arguments are created
  at that precision. Generic constructors without a `dtype` keyword default to
  `float32` for all-float-literal arguments.
- **Binary operations with compound types**: `add_builtin_call` detects when a
  scalar operand is weakly-typed `float` and the other operand is a compound
  type (vector, matrix, tile), then casts the scalar to match the compound's
  scalar precision.
- **Store operations**: `array_store`, `store`, and atomic operations cast
  weakly-typed `float` values to match the target array/address element type.
- **Assignment**: When assigning to a variable, three cases arise:
  (a) **Weak → strong**: a weak value is cast to match the variable's existing
  strong type (e.g., `x: float64 = 0.0; x = 3.14` casts the literal to
  float64). (b) **Strong replaces weak**: the variable was previously weakly
  typed and now receives a strongly-typed value — the variable's type is
  updated to the strong type. (c) **Incompatible strong types**: two different
  strong types raise a type error (R7). These rules apply uniformly to regular
  assignment, multi-assignment (tuple unpacking), and augmented assignment
  (`+=`, `-=`, etc.).
- **Return statements**: Weakly-typed `float` returns are cast to match the
  annotated return type, or to `float32` when no annotation exists.
- **Chained comparisons**: `emit_Compare` casts `float` operands to match
  strongly-typed comparands.

#### `scalar_infer_type` Changes

The `scalar_infer_type` function in `builtins.py` was updated to:
- Accept `float` as a valid scalar type in the input set
- Discard `float` when a strongly-typed float is also present (the strongly-typed
  float wins)
- Return `float` (weak) when only weakly-typed `float` remains, preserving
  adaptability for downstream context

#### Helper Predicates

The `is_weak_float(t)` and `is_strong_float(t)` predicates in `types.py` make
the weak-typing intent self-documenting throughout the codebase. Use these
instead of raw `t is float` / `t in float_types` checks in weak-typing logic.
Related helpers in `builtins.py`: `_resolve_dtype_default()` (defaults weakly-
typed float to `float32`) and `_check_dtype_mismatch()` (validates dtype
compatibility while allowing weakly-typed float).

#### Error Message Canonicalization

Weak types are an internal implementation detail and must not be exposed to
users. The `type_str()` function in `context.py` maps Python's `float` to
`"float32"` and Python's `int` to `"int32"` so that error messages always show
the canonical Warp type name. Without this, users would see confusing messages
like `"expected float32, got float"` instead of
`"expected float32, got float32"`.

#### Dispatch and Default Value Handling

Raw Python values that enter the compiler through dispatch arguments or
parameter defaults (e.g., `def foo(x: float, scale=0.5)`) must be explicitly
cast to non-weak types via `add_constant(value, target_type=...)`. Without
this, a default value like `0.5` would create a weakly-typed `Var`, leaking
weak typing into user code paths where it is not expected.

## Testing Strategy

- **`test_constant_precision.py`** — numerical accuracy: literal propagation,
  constructor precision, binary operator casting, compound type interactions,
  chained comparisons, module constants (`wp.PI`), adjoint precision, and
  edge cases (division by zero, overflow, constant folding).
- **`test_constant_precision.py`** — behavioral correctness: typed constructors
  accepting bare literals, sametypes functions with literals, scalar-vector
  multiplication, `@wp.func` overload resolution, builtin overload resolution
  (verifying float64 preference for weak float), and error rejection for
  strongly-typed variable mismatches.
- **Overload preference tests**: Verify that `wp.sin(1.0)` computes in float64
  precision and returns a weakly-typed result. Verify that user functions with
  both float32/float64 overloads select the float64 overload for weak float
  arguments.
- **Config option tests**: Verify the legacy config restores float32-default
  behavior for both builtins and user functions. Test both config states.
- **Constexpr verification**: For builtins expected to be constant-folded by
  the CUDA compiler, examine PTX output to confirm the absence of runtime
  calls when passing float literal arguments.
- **Int-to-float promotion tests**: Verify `2 * 3.0` produces a weak float
  at double precision. Verify weak int promotes to match strong float in binary
  ops and compound constructors (`wp.vec3d(1, 2, 3)`). Verify
  `WarpCodegenOverflowError` is raised when an int literal exceeds the target
  float type's representable range.
- **Existing test updates**: Six test files were updated to adjust expectations
  for the new float literal behavior (e.g., constructor type inference now
  produces `float32` from literals instead of raising errors).
- **Device coverage**: Tests use `get_test_devices()` via `add_function_test()`
  to run on all available devices (CPU + CUDA).
