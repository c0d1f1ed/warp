# Weakly-Typed Constants

**Status**: Implemented (float); in progress (int)

**Issue**: [GH-485](https://github.com/NVIDIA/warp/issues/485)

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
| R5  | Builtin function results with weak float input preserve weak typing | Must | `wp.sqrt(3.0)` adapts to context, not locked to float32 |
| R6  | Module constants (`wp.PI`, `wp.E`, `wp.TAU`) adapt precision in typed constructors | Must | `wp.float64(wp.PI)` preserves full double precision |
| R10 | Int literals adapt precision to surrounding typed context | Must | Mirrors R1 for int |
| R11 | Weak int compatible with any strong int type (int8..uint64) | Must | Int-to-int adaptation |
| R12 | Weak int compatible with any strong float type (int-to-float promotion) | Must | e.g., `wp.vec3d(1, 2, 3)` |

R1 includes these concrete guarantees:
- `wp.float64(3.141592653589793)` produces the exact double-precision value
- Negative literals (`-3.14...`) preserve full precision in float64 contexts
- `float32_var * 2.0` continues to produce `float32` results (literal adapts down)

### Type Safety

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R7  | Two different strong types remain incompatible | Must | e.g., `float32` vs `float64` is always an error |
| R8  | Annotated `float` parameters remain float32 (backward compat) | Must | GH-485 backward compatibility |

### Backward Compatibility

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R9  | No impact on existing user code that only uses float32 | Must | All existing tests must pass |

**Non-goals**: None — weak typing now covers both float and int literals.

## Design

### Approach

Introduce **weakly-typed constants** where Python's built-in `float` and `int`
types are treated as distinct types during code generation, separate from Warp's
`float32` and `int32`. A weakly-typed constant:

- Emits as C++ `double` (float) or `int64` (int) for maximum range/precision
- Adapts to the first strongly-typed value found in the same expression
- Defaults to `float32` / `int32` when no typed context is available (backward compat)
- Weak int is also compatible with strong float types (int-to-float promotion)

The key insight is distinguishing two uses of `float`/`int` in user code:

1. **Annotated `float`/`int`** (function parameter types) — converted to
   `float32`/`int32` early in `Adjoint.__init__` via `canonicalize_dtype()`.
   This is strongly typed.
2. **Literal `float`/`int`** (from `add_constant` for numeric literals like
   `3.14` or `42`) — stays as Python `float`/`int` internally, emits as C++
   `double`/`int64`. This is weakly typed and adapts to context.

### Alternatives Considered

**Always use float64 for literals** — Rejected because it would break backward
compatibility. Existing code like `float32_var + 1.0` would promote to float64,
changing behavior silently.

**Add a new `WeakFloat` sentinel type** — Rejected as unnecessarily complex.
Reusing Python's `float` as the weak type is natural since it already exists in
the type system and the codegen already handles it.

**Const-expression tracking (`is_const_expr` / `preserves_const_expr`)** — An
earlier prototype tracked which expressions were "constant" to decide when to
preserve precision. This was removed because it coupled precision semantics to
constness, creating confusing edge cases. The current approach is simpler: the
*type* (`float` vs `float32`) carries the precision information directly.

### Key Implementation Details

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

Generic overloads (those using `Float`/`Scalar` `TypeVar`s) match `float` via
`scalars_equal_generic` — the function was updated to treat `float` as
compatible with `Float` and `Scalar` type variables.

Concrete overloads do **not** match `float` directly. If resolution fails with
`float` arguments, a retry mechanism casts them to `float32` and resolves again.
This handles builtins like `expect_eq` that only have concrete overloads.

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

- **User functions**: `float` arguments are implicitly cast to `float32` before
  resolution for backward compatibility (annotated `float` params are `float32`).
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
- **Assignment**: `emit_Assign` and `emit_AugAssign` allow weakly-typed `float`
  to be assigned to/from strongly-typed float variables without type errors.
- **Return statements**: Weakly-typed `float` returns are cast to match the
  annotated return type, or to `float32` when no annotation exists.
- **Chained comparisons**: `emit_Compare` casts `float` operands to match
  strongly-typed comparands.

#### `scalar_infer_type` Changes

The `scalar_infer_type` function in `builtins.py` was updated to:
- Accept `float` as a valid scalar type in the input set
- Discard `float` when a strongly-typed float is also present (the strongly-typed
  float wins)
- Return `float32` when only weakly-typed `float` remains (fallback)

#### Helper Predicates

The `is_weak_float(t)` and `is_strong_float(t)` predicates in `types.py` make
the weak-typing intent self-documenting throughout the codebase. Use these
instead of raw `t is float` / `t in float_types` checks in weak-typing logic.
Related helpers in `builtins.py`: `_resolve_dtype_default()` (defaults weakly-
typed float to `float32`) and `_check_dtype_mismatch()` (validates dtype
compatibility while allowing weakly-typed float). Note that `type_str()` in
`context.py` maps Python's `float` to `"float32"` so that error messages use
the canonical Warp type name rather than exposing the internal representation.

## Testing Strategy

- **`test_constant_precision.py`** — numerical accuracy: literal propagation,
  constructor precision, binary operator casting, compound type interactions,
  chained comparisons, module constants (`wp.PI`), adjoint precision, and
  edge cases (division by zero, overflow, constant folding).
- **`test_constant_precision.py`** — behavioral correctness: typed constructors
  accepting bare literals, sametypes functions with literals, scalar-vector
  multiplication, `@wp.func` overload resolution, builtin default resolution,
  and error rejection for strongly-typed variable mismatches.
- **Existing test updates**: Six test files were updated to adjust expectations
  for the new float literal behavior (e.g., constructor type inference now
  produces `float32` from literals instead of raising errors).
- **Device coverage**: Tests use `get_test_devices()` via `add_function_test()`
  to run on all available devices (CPU + CUDA).
