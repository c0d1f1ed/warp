# Refactor: Type-Matching Rename and Clarification

## Motivation

The type-matching functions in `warp/_src/types.py` use names and parameters
that obscure their intent. The `match_generic` boolean does double duty
(controlling generic type matching AND weak-float equivalence), and function names
like `scalars_equal_generic` are ambiguous — "generic" in the name means
something different from the `match_generic` parameter. This plan proposes
renames and minor restructuring to make the logic self-documenting.

## Changes

### 1. Rename `match_generic` parameter to `match_generic_type`

**Files:** `warp/_src/types.py`, `warp/_src/codegen.py`

The parameter controls whether generic type placeholders (`Any`, `Int`, `Float`,
`Scalar`) match concrete types. Renaming to `match_generic_type` makes this
explicit and improves readability of the tricky weak-float guard:

```python
# Before (what does "generic" mean here?):
if not match_generic and (is_weak_float(a) or is_weak_float(b)):

# After (clear: we skip this during generic type resolution):
if not match_generic_type and (is_weak_float(a) or is_weak_float(b)):
```

### 2. Rename functions

**Files:** `warp/_src/types.py`, `warp/_src/codegen.py`, and all call sites

| Current                  | Proposed                  | Rationale                                      |
|--------------------------|---------------------------|-------------------------------------------------|
| `scalars_equal_generic`  | `scalar_types_match`      | Answers "does scalar A match scalar B?"         |
| `types_equal_generic`    | `types_match`             | Matches any types (scalars, arrays, tuples…)    |
| `scalars_equal`          | `scalar_types_equal`      | Symmetry with `types_equal`; thin wrapper       |

`types_equal` keeps its current name (it's already clear).

### 3. Add docstring to `scalar_types_match`

**File:** `warp/_src/types.py`

The function currently has no docstring despite being the key function for
overload resolution semantics. Add:

```python
def scalar_types_match(a, b, match_generic_type=True):
    """Check whether scalar types *a* and *b* are compatible.

    When *match_generic_type* is True (overload resolution), generic type
    placeholders (Any, Int, Float, Scalar) match any compatible concrete
    type, but weak types (Python ``int`` and ``float``) remain distinct
    from their Warp counterparts so that the correct overload is selected.

    When *match_generic_type* is False (structural equality), only concrete
    types are compared, and weak types are treated as equivalent to their
    Warp counterparts (``int32`` and ``float32``).
    """
```

### 4. Restructure weak-type equivalence block

**File:** `warp/_src/types.py`, bottom of `scalar_types_match`

Replace the dense `if a is not b` block with separate, self-explanatory
branches. Keep the `not match_generic_type` guard on weak float.

The guard is necessary because scalar type constructors register concrete
overloads for every pair of scalar types (builtins.py ~1050):

```python
scalar_types_all = [*scalar_types, bool, int, float]
for t in scalar_types_all:
    for u in scalar_types_all:
        add_builtin(t.__name__, input_types={"a": u}, ...)
```

This creates overloads like `float64(float32)` AND `float64(float)`. Since
`float32` appears before `float` in the list, and overload resolution uses
first-match (codegen.py `resolve_func`), removing the guard would make
`wp.float64(3.14)` match `float64(float32)` — truncating the literal to
float32 precision before widening to float64.

Weak int does NOT need the guard because int32 → int64 widening is exact,
so matching `int64(int32)` instead of `int64(int)` is harmless. However,
if Warp adds large-literal support in the future (values exceeding int32
range), the same guard would become necessary for weak int. Add a
`not match_generic_type` guard to weak int as well, for symmetry and
future-safety.

```python
    # --- Weak-type equivalence rules ---
    # During overload resolution (match_generic_type=True), weak types must
    # remain distinct from their Warp counterparts. This prevents e.g.
    # wp.float64(3.14) from matching the float64(float32) overload (which
    # would truncate the literal to float32 precision) instead of the
    # float64(float) overload (which preserves full precision).
    #
    # During structural comparison (match_generic_type=False), weak types
    # are treated as equivalent to their Warp counterparts.
    if not match_generic_type:
        if is_weak_int(a) and canonicalize_dtype(b) is int32:
            return True
        if is_weak_int(b) and canonicalize_dtype(a) is int32:
            return True
        if is_weak_float(a) and canonicalize_dtype(b) is float32:
            return True
        if is_weak_float(b) and canonicalize_dtype(a) is float32:
            return True

    return a is b
```

### 5. Improve canonicalization comment in `context.py`

**File:** `warp/_src/context.py`, around line 352

Link the two canonicalization sites so future readers know they must stay in
sync:

```python
# Before:
# Convert Python float/int arg types for overload matching

# After:
# Canonicalize Python float/int to Warp types (float32/int32) so that
# literal args match overloads whose annotations were also canonicalized
# (see Adjoint.__init__ _convert_annotation).
```

## Call sites to update

A non-exhaustive list of references that need renaming:

- `warp/_src/types.py`: definitions and internal calls
  - `scalars_equal_generic` (definition + calls from `types_equal_generic`,
    `scalars_equal`)
  - `types_equal_generic` (definition + recursive calls, `seq_match_ellipsis`)
  - All `match_generic=` keyword arguments
- `warp/_src/codegen.py`: `func_match_args` calls `types_equal_generic`
- Any other files referencing these functions (search for all call sites before
  starting)

## Validation

Run the existing type-matching and overload tests to confirm no behavioral
change:

```
uv run build_lib.py --quick
uv run --extra dev -m warp.tests -s autodetect -k TestCodeGen -k TestFunc
```

Also run the int literal tests on the current branch:

```
uv run warp/tests/test_int_literals.py
```
