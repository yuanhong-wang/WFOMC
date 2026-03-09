# Cython WFOMC Algorithms — Design Document

**Date:** 2026-02-25
**Status:** Approved
**Goal:** Speed up WFOMC algorithms using Cython (Option 1: typed loop variables, Python `object` for ring elements). All existing tests in `tests/wfomc_test.py` must pass.

---

## 1. Approach

Use Cython with typed integer loop variables and Cython compiler directives. All `fmpq` and `fmpq_mpoly` (from python-flint) remain as Python `object` — their arithmetic already executes at C level via FLINT, and python-flint 0.8.0 ships no `.pxd` files for direct `cimport`.

**Speedup sources:**
- Integer loop counters (`cdef int i, j, k, n_cells, domain_size`) — eliminates boxing/unboxing in every iteration
- `multinomial`/`multinomial_less_than` partition generators — pure-C integer recursion (heaviest Python overhead in practice)
- Replace `reduce(lambda ...)` patterns with explicit `cdef int` loops
- Cython directives: `boundscheck=False`, `wraparound=False`, `nonecheck=False`

**Not in scope:** C-level FLINT arithmetic (no `extern from`, no `fmpq_set_any_ref` capsule, no `fmpq_mpoly_t` wrapper). This keeps the build portable and maintenance-free.

**Expected speedup:** 2–8× depending on domain size.

---

## 2. File Layout

```
src/wfomc/
├── _compat.py                          ← new: fallback helper + WFOMC_CYTHON_ONLY
├── algo/
│   ├── __init__.py                     ← updated: try-import Cython versions
│   ├── StandardWFOMC.py                ← unchanged (Python fallback)
│   ├── FastWFOMC.py                    ← unchanged (Python fallback)
│   ├── IncrementalWFOMC.py             ← unchanged (Python fallback)
│   ├── RecursiveWFOMC.py               ← unchanged (Python fallback)
│   ├── StandardWFOMC.pyx               ← new
│   ├── FastWFOMC.pyx                   ← new
│   ├── IncrementalWFOMC.pyx            ← new
│   └── RecursiveWFOMC.pyx              ← new
├── utils/
│   ├── __init__.py                     ← updated: try-import Cython multinomial
│   ├── multinomial.py                  ← unchanged (Python fallback)
│   └── multinomial.pyx                 ← new
└── cell_graph/
    ├── __init__.py                     ← updated: try-import Cython cell_graph
    ├── cell_graph.py                   ← unchanged (Python fallback)
    └── cell_graph.pyx                  ← new
```

---

## 3. Cython Strategy per Module

### `multinomial.pyx` (highest priority)
Called by every algorithm on every partition iteration. Pure Python integer recursion → pure C integer loops.

```cython
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
def multinomial(int length, int total_sum):
    cdef int value
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in multinomial(length - 1, total_sum - value):
                yield (value,) + permutation

def multinomial_less_than(int length, int total_sum):
    cdef int i, value
    ...

cdef class MultinomialCoefficients:
    # typed Pascal triangle operations
    @staticmethod
    def coef(tuple lst) -> int:
        cdef int s, ret = 1
        ...
    @staticmethod
    def comb(int a, int b) -> int:
        cdef int ret
        ...
```

### `cell_graph.pyx`
Type integer loop indices in `get_term`, `get_d_term`, `get_d_term`, `get_J_term`. Leave `RingElement` as `object`.

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
def get_term(self, int iv, int bign, tuple partition) -> object:
    cdef int nval, s
    cdef object accum, smul

def get_d_term(self, int l, int n, int cur=0) -> object:
    cdef int ni, clique_size
    cdef object r, s, mult, ret
```

### `IncrementalWFOMC.pyx`
The dominant hot loop. Replace `reduce(lambda x, y: x * y, (...))` with explicit `cdef int k` loop. Type all table-index operations.

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
def incremental_wfomc(context, circle_len=None):
    cdef int cur_idx, j, k, n_cells, domain_size, pred_idx
    cdef list old_ivec
    cdef object w, w_old, w_new, accum
    # replace reduce(lambda) with:
    accum = Rational(1, 1)
    for k in range(n_cells):
        accum = accum * cell_graph.get_two_table_weight(
            (cell, cells[k])
        ) ** old_ivec[k]
    w_new = w_new * accum
```

### `FastWFOMC.pyx`
Type partition loop variables; `fmpq`/`fmpq_mpoly` body stays as `object`.

```cython
cdef int i, j, l, domain_size = len(domain)
cdef object body, coef, res_, mul
```

### `StandardWFOMC.pyx`
Type partition enumeration indices.

```cython
cdef int i, j, n_i, n_j, n_cells
cdef object res, coef, cell_config_weight
```

### `RecursiveWFOMC.pyx`
Type DFS loop variables.

```cython
cdef int l, cell_num = len(cell_weights)
cdef object res, w_l, value
```

---

## 4. Fallback & Test Control

`src/wfomc/_compat.py`:
```python
import os

CYTHON_ONLY = os.environ.get("WFOMC_CYTHON_ONLY", "0") == "1"

def try_import_cython(cython_mod: str, python_fallback: str):
    """Import Cython module if available; fall back to Python version."""
    try:
        return __import__(cython_mod, fromlist=["*"])
    except ImportError:
        if CYTHON_ONLY:
            raise ImportError(
                f"Cython module '{cython_mod}' not available and "
                "WFOMC_CYTHON_ONLY=1. Run: python setup.py build_ext --inplace"
            )
        return __import__(python_fallback, fromlist=["*"])
```

Each `__init__.py` uses this to re-export functions:
```python
# algo/__init__.py
from wfomc._compat import try_import_cython as _try
_m = _try("wfomc.algo.IncrementalWFOMC", "wfomc.algo.IncrementalWFOMC")
incremental_wfomc = _m.incremental_wfomc
```

---

## 5. Build System

### `pyproject.toml` changes
```toml
[build-system]
requires = ["setuptools>=68", "cython>=3.0"]
build-backend = "setuptools.backends.legacy:build"

[dependency-groups]
dev = [
    "cython>=3.0",
    "ipython>=9.6.0",
    "pytest>=8.4.1",
]
```

### `setup.py` (new file)
```python
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension("wfomc.utils.multinomial",     ["src/wfomc/utils/multinomial.pyx"]),
    Extension("wfomc.cell_graph.cell_graph", ["src/wfomc/cell_graph/cell_graph.pyx"]),
    Extension("wfomc.algo.IncrementalWFOMC", ["src/wfomc/algo/IncrementalWFOMC.pyx"]),
    Extension("wfomc.algo.RecursiveWFOMC",   ["src/wfomc/algo/RecursiveWFOMC.pyx"]),
    Extension("wfomc.algo.FastWFOMC",        ["src/wfomc/algo/FastWFOMC.pyx"]),
    Extension("wfomc.algo.StandardWFOMC",    ["src/wfomc/algo/StandardWFOMC.pyx"]),
]
setup(ext_modules=cythonize(extensions, language_level=3, annotate=True))
```

Build command: `uv run python setup.py build_ext --inplace`

---

## 6. Testing

All existing tests in `tests/wfomc_test.py` serve as the correctness validation. No new tests needed.

```bash
# Build and test with Cython enforced
uv run python setup.py build_ext --inplace
WFOMC_CYTHON_ONLY=1 uv run pytest tests/ -v

# Test Python fallback still works
uv run pytest tests/ -v   # (without building, falls back to Python)
```

---

## 7. Implementation Order

1. Build system: `pyproject.toml` + `setup.py` + install `cython` dep
2. `src/wfomc/_compat.py`
3. `utils/multinomial.pyx` + update `utils/__init__.py`
4. `cell_graph/cell_graph.pyx` + update `cell_graph/__init__.py`
5. `algo/IncrementalWFOMC.pyx`
6. `algo/FastWFOMC.pyx`
7. `algo/StandardWFOMC.pyx`
8. `algo/RecursiveWFOMC.pyx`
9. Update `algo/__init__.py`
10. Build + run full test suite with `WFOMC_CYTHON_ONLY=1`
