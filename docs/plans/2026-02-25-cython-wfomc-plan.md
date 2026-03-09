# Cython WFOMC Algorithms Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compile the WFOMC algorithm modules with Cython (typed integer loop variables, Python `object` for ring elements) to reduce Python interpreter overhead, while keeping all existing tests passing.

**Architecture:** Each `.py` file in `algo/`, `cell_graph/`, and `utils/` gets a parallel `.pyx` file with `cdef int` annotations on loop variables and Cython directives. A `_compat.py` helper provides try-import with `WFOMC_CYTHON_ONLY` fallback control. The build backend switches from hatchling to setuptools to support Cython `ext_modules`.

**Tech Stack:** Cython ≥ 3.0, setuptools ≥ 68, python-flint (unchanged), existing test suite (`pytest`).

---

## Task 1: Add Cython to dependencies and switch build backend

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Replace the `[build-system]` section and add `cython` to dev deps:

```toml
[build-system]
requires = ["setuptools>=68", "cython>=3.0"]
build-backend = "setuptools.backends.legacy:build"
```

In `[dependency-groups]`:
```toml
[dependency-groups]
dev = [
    "cython>=3.0",
    "ipython>=9.6.0",
    "pytest>=8.4.1",
]
```

Also add `[tool.setuptools.packages.find]` so setuptools discovers the src layout:
```toml
[tool.setuptools.packages.find]
where = ["src"]
```

**Step 2: Sync and verify cython is importable**

```bash
uv sync
uv run python -c "import cython; print(cython.__version__)"
```
Expected: prints a version ≥ 3.0.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: switch to setuptools backend, add cython dev dependency"
```

---

## Task 2: Create setup.py for Cython extensions

**Files:**
- Create: `setup.py`

**Step 1: Write setup.py**

```python
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        "wfomc.utils.multinomial",
        ["src/wfomc/utils/multinomial.pyx"],
    ),
    Extension(
        "wfomc.cell_graph.cell_graph",
        ["src/wfomc/cell_graph/cell_graph.pyx"],
    ),
    Extension(
        "wfomc.algo.IncrementalWFOMC",
        ["src/wfomc/algo/IncrementalWFOMC.pyx"],
    ),
    Extension(
        "wfomc.algo.RecursiveWFOMC",
        ["src/wfomc/algo/RecursiveWFOMC.pyx"],
    ),
    Extension(
        "wfomc.algo.FastWFOMC",
        ["src/wfomc/algo/FastWFOMC.pyx"],
    ),
    Extension(
        "wfomc.algo.StandardWFOMC",
        ["src/wfomc/algo/StandardWFOMC.pyx"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        annotate=True,          # generates .html annotation files
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )
)
```

**Step 2: Verify setup.py is valid (no .pyx files yet — expect file-not-found, not syntax error)**

```bash
uv run python setup.py --version
```
Expected: prints the package version (not a Cython error).

**Step 3: Commit**

```bash
git add setup.py
git commit -m "build: add setup.py for Cython ext_modules"
```

---

## Task 3: Create `_compat.py` fallback helper

**Files:**
- Create: `src/wfomc/_compat.py`

**Step 1: Write `_compat.py`**

```python
import os

# Set WFOMC_CYTHON_ONLY=1 to raise loudly when Cython .so is missing.
# Default: silently fall back to pure Python.
CYTHON_ONLY = os.environ.get("WFOMC_CYTHON_ONLY", "0") == "1"


def try_import_cython(cython_mod: str, python_fallback: str):
    """Import Cython extension if compiled; fall back to Python module.

    Args:
        cython_mod: Fully-qualified module name for the Cython .so.
        python_fallback: Fully-qualified module name for the Python .py.

    Returns:
        The imported module (Cython if available, else Python).

    Raises:
        ImportError: If Cython .so is missing and WFOMC_CYTHON_ONLY=1.
    """
    try:
        return __import__(cython_mod, fromlist=["*"])
    except ImportError:
        if CYTHON_ONLY:
            raise ImportError(
                f"Cython module '{cython_mod}' is not compiled. "
                "Run: python setup.py build_ext --inplace\n"
                "Or unset WFOMC_CYTHON_ONLY to use the Python fallback."
            )
        return __import__(python_fallback, fromlist=["*"])
```

**Step 2: Verify it imports cleanly**

```bash
uv run python -c "from wfomc._compat import try_import_cython, CYTHON_ONLY; print('ok', CYTHON_ONLY)"
```
Expected: `ok False`

**Step 3: Commit**

```bash
git add src/wfomc/_compat.py
git commit -m "feat: add _compat.py for Cython/Python fallback dispatch"
```

---

## Task 4: Cythonize `multinomial.pyx`

**Files:**
- Create: `src/wfomc/utils/multinomial.pyx`
- Modify: `src/wfomc/utils/__init__.py`

**Step 1: Create `multinomial.pyx`**

This is a line-for-line copy of `multinomial.py` with `cdef int` annotations added. The key changes are: typed function arguments, typed local variables, typed `MultinomialCoefficients` method locals.

```cython
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from __future__ import annotations

import functools


def multinomial(int length, int total_sum):
    """
    Generate tuples of `length` non-negative integers summing to `total_sum`.
    """
    cdef int value
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in multinomial(length - 1, total_sum - value):
                yield (value,) + permutation


def multinomial_less_than(int length, int total_sum):
    """
    Generate tuples of `length` non-negative integers summing to at most `total_sum`.
    """
    cdef int i, value
    if length == 0:
        yield ()
        return
    if length == 1:
        for i in range(total_sum + 1):
            yield (i,)
    else:
        for value in range(total_sum + 1):
            for permutation in multinomial_less_than(length - 1, total_sum - value):
                yield (value,) + permutation


class MultinomialCoefficients(object):
    """
    Multinomial coefficients backed by a precomputed Pascal triangle.
    """
    pt: list = None
    n: int = 0

    @staticmethod
    def setup(int n):
        cdef int i
        cdef list lst, newlist
        if n <= MultinomialCoefficients.n:
            return
        pt = []
        lst = [1]
        for i in range(n + 1):
            pt.append(lst)
            newlist = []
            newlist.append(lst[0])
            for j in range(len(lst) - 1):
                newlist.append(lst[j] + lst[j + 1])
            newlist.append(lst[-1])
            lst = newlist
        MultinomialCoefficients.pt = pt
        MultinomialCoefficients.n = n

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def coef(tuple lst) -> int:
        cdef int ret = 1
        cdef tuple tmplist
        if MultinomialCoefficients.pt is None:
            raise RuntimeError(
                'Please initialize MultinomialCoefficients first by '
                '`MultinomialCoefficients.setup(n)`'
            )
        if sum(lst) > MultinomialCoefficients.n:
            raise RuntimeError(
                f'The sum {sum(lst)} of input is larger than precomputed '
                f'maximal sum {MultinomialCoefficients.n}, '
                'please re-initialized MultinomialCoefficients using bigger n'
            )
        tmplist = lst
        while len(tmplist) > 1:
            ret *= MultinomialCoefficients.comb(sum(tmplist), tmplist[-1])
            tmplist = tmplist[:-1]
        return ret

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def comb(int a, int b) -> int:
        if a < b:
            return 0
        elif b == 0:
            return 1
        else:
            return MultinomialCoefficients.pt[a][b]
```

**Step 2: Update `src/wfomc/utils/__init__.py`**

Replace the `from .multinomial import ...` line with a try-import:

```python
import numpy as np

from wfomc._compat import try_import_cython as _try_cy

_multinomial_mod = _try_cy("wfomc.utils.multinomial", "wfomc.utils.multinomial")
MultinomialCoefficients = _multinomial_mod.MultinomialCoefficients
multinomial = _multinomial_mod.multinomial
multinomial_less_than = _multinomial_mod.multinomial_less_than

from .polynomial_flint import (
    Rational, Poly, RingElement, expand,
    coeff_monomial, create_vars, coeff_dict, round_rational,
)


def format_np_complex(num: np.ndarray) -> str:
    return '{num.real:+0.04f}+{num.imag:+0.04f}j'.format(num=num)


__all__ = [
    "MultinomialCoefficients",
    "multinomial",
    "multinomial_less_than",
    "Poly",
    "Rational",
    "RingElement",
    "round_rational",
    "expand",
    "coeff_monomial",
    "coeff_dict",
    "create_vars",
    "format_np_complex",
]
```

**Step 3: Verify Python fallback still works (before building)**

```bash
uv run python -c "
from wfomc.utils import multinomial, MultinomialCoefficients
MultinomialCoefficients.setup(5)
print(list(multinomial(2, 3)))
print(MultinomialCoefficients.coef((1, 2)))
"
```
Expected: `[(0, 3), (1, 2), (2, 1), (3, 0)]` and `3`.

**Step 4: Build `multinomial.pyx`**

```bash
uv run python setup.py build_ext --inplace 2>&1 | grep -E "multinomial|error|warning"
```
Expected: compiles without error, creates `src/wfomc/utils/multinomial.cpython-*.so`.

**Step 5: Verify Cython version loads**

```bash
WFOMC_CYTHON_ONLY=1 uv run python -c "
from wfomc.utils import multinomial, MultinomialCoefficients
MultinomialCoefficients.setup(5)
print(list(multinomial(2, 3)))
print(MultinomialCoefficients.coef((1, 2)))
"
```
Expected: same output as Step 3, using the `.so`.

**Step 6: Run existing tests (Python fallback)**

```bash
uv run pytest tests/wfomc_test.py -x -q
```
Expected: all pass.

**Step 7: Run existing tests (Cython enforced)**

```bash
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -x -q
```
Expected: all pass.

**Step 8: Commit**

```bash
git add src/wfomc/utils/multinomial.pyx src/wfomc/utils/__init__.py
git commit -m "feat(cython): add multinomial.pyx with typed int loop variables"
```

---

## Task 5: Cythonize `cell_graph.pyx`

**Files:**
- Create: `src/wfomc/cell_graph/cell_graph.pyx`
- Modify: `src/wfomc/cell_graph/__init__.py`

**Step 1: Create `src/wfomc/cell_graph/cell_graph.pyx`**

Copy the full content of `cell_graph.py` verbatim, then make these targeted changes:

1. Add the directive header at the top (before all imports):
```cython
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
```

2. In `OptimizedCellGraph.get_term`, add typed locals after the method signature:
```cython
    def get_term(self, iv: int, bign: int, partition: tuple) -> object:
        cdef int iv_int = iv, bign_int = bign
        cdef int nval, s
        cdef object accum, smul
        # rest of body unchanged
```

3. In `OptimizedCellGraph.get_d_term`, add typed locals:
```cython
    def get_d_term(self, l: int, n: int, cur: int = 0) -> object:
        cdef int clique_size = len(self.cliques[l])
        cdef int ni
        cdef object r, s, mult, ret
        # rest of body unchanged
```

4. In `OptimizedCellGraph.get_J_term`, add typed locals:
```cython
    def get_J_term(self, l: int, nhat: int) -> object:
        cdef object thesum
        # rest of body unchanged
```

5. In `OptimizedCellGraphWithPC.get_J_term`, add:
```cython
    def get_J_term(self, l: int, clique_config: tuple) -> object:
        cdef int i, j
        cdef object ret, r
        cdef int sumn = 0
        # rest of body unchanged
```

6. In `OptimizedCellGraphWithPC.get_d_term`, add:
```cython
    def get_d_term(self, l: int, n: int, par_idx: int, cur: int = 0) -> object:
        cdef int cells_num = len(self.clique_partitions[l][par_idx])
        cdef int ni
        cdef object r, s, mult, ret, w
        # rest of body unchanged
```

7. In `CellGraph.find_independent_sets` and `find_independent_cliques` (in `OptimizedCellGraph`):
```cython
        cdef int i, j
```
at the top of each method body.

**Step 2: Update `src/wfomc/cell_graph/__init__.py`**

```python
from wfomc._compat import try_import_cython as _try_cy

_cg_mod = _try_cy("wfomc.cell_graph.cell_graph", "wfomc.cell_graph.cell_graph")
CellGraph = _cg_mod.CellGraph
OptimizedCellGraph = _cg_mod.OptimizedCellGraph
OptimizedCellGraphWithPC = _cg_mod.OptimizedCellGraphWithPC
build_cell_graphs = _cg_mod.build_cell_graphs

from .components import Cell, TwoTable

__all__ = [
    'CellGraph',
    'OptimizedCellGraph',
    'OptimizedCellGraphWithPC',
    'build_cell_graphs',
    'Cell',
    'TwoTable',
]
```

**Step 3: Build cell_graph.pyx**

```bash
uv run python setup.py build_ext --inplace 2>&1 | grep -E "cell_graph|error|Error"
```
Expected: compiles without error.

**Step 4: Run tests with Cython enforced**

```bash
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -x -q
```
Expected: all pass.

**Step 5: Commit**

```bash
git add src/wfomc/cell_graph/cell_graph.pyx src/wfomc/cell_graph/__init__.py
git commit -m "feat(cython): add cell_graph.pyx with typed int loop variables"
```

---

## Task 6: Cythonize `IncrementalWFOMC.pyx`

**Files:**
- Create: `src/wfomc/algo/IncrementalWFOMC.pyx`

This is the most important hot loop. The key change beyond typed loop variables is replacing the `reduce(lambda x, y: x * y, (...))` call with an explicit `for k` loop, which Cython can lower to a tight C loop.

**Step 1: Create `src/wfomc/algo/IncrementalWFOMC.pyx`**

```cython
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from wfomc.cell_graph import build_cell_graphs
from wfomc.context import WFOMCContext
from wfomc.network import UnaryEvidenceEncoding
from wfomc.utils import RingElement, Rational, MultinomialCoefficients


def incremental_wfomc(context: WFOMCContext, circle_len=None) -> object:
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred
    res = Rational(0, 1)
    predecessor_preds = context.predecessor_preds
    pred_orders = None
    cdef int pred_max_order = 0
    if predecessor_preds is not None:
        pred_orders = list(predecessor_preds.keys())
        pred_max_order = max(pred_orders)
    circular_predecessor_pred = context.circular_predecessor_pred
    cdef int domain_size = len(domain)
    cdef int circle_len_int
    if circle_len is None:
        circle_len_int = domain_size
    else:
        circle_len_int = circle_len

    for cell_graph, weight in build_cell_graphs(
        formula, get_weight,
        leq_pred=leq_pred,
        predecessor_preds=predecessor_preds
    ):
        cells = cell_graph.get_cells()
        cdef int n_cells = len(cells)

        def helper(cell, pc_pred, pc_ccs):
            cdef int i
            for i, p in enumerate(pc_pred):
                if cell.is_positive(p) and pc_ccs[i] > 0:
                    return i
            return None

        if context.unary_evidence_encoding == UnaryEvidenceEncoding.PC:
            pc_pred, pc_ccs = zip(*context.partition_constraint.partition)
            table = dict()
            for i, cell in enumerate(cells):
                j = helper(cell, pc_pred, pc_ccs)
                if j is None:
                    continue
                table[
                    (
                        tuple(int(k == i) for k in range(n_cells)),
                        None if pred_orders is None else tuple(
                            cell for _ in range(pred_max_order)
                        ),
                        None if circular_predecessor_pred is None else cell
                    )
                ] = (
                    cell_graph.get_cell_weight(cell),
                    tuple(cc - 1 if k == j else cc
                          for k, cc in enumerate(pc_ccs))
                )
        else:
            table = dict(
                (
                    (
                        tuple(int(k == i) for k in range(n_cells)),
                        None if pred_orders is None else tuple(
                            cell for _ in range(pred_max_order)
                        ),
                        None if circular_predecessor_pred is None else cell
                    ),
                    (
                        cell_graph.get_cell_weight(cell),
                        None
                    )
                )
                for i, cell in enumerate(cells)
            )

        cdef int cur_idx, j_idx, k, pred_idx
        for cur_idx in range(domain_size - 1):
            old_table = table
            table = dict()
            for j_idx, cell in enumerate(cells):
                w = cell_graph.get_cell_weight(cell)
                for (ivec, last_cells, first_cell), (w_old, old_ccs) in old_table.items():
                    old_ivec = list(ivec)
                    if old_ccs is not None:
                        idx = helper(cell, pc_pred, old_ccs)
                        if idx is None:
                            continue
                        new_ccs = tuple(
                            cc - 1 if k == idx else cc
                            for k, cc in enumerate(old_ccs)
                        )
                    else:
                        new_ccs = None

                    w_new = w_old * w
                    if cur_idx == circle_len_int - 2 and first_cell is not None:
                        w_new = w_new * cell_graph.get_two_table_with_pred_weight(
                            (first_cell, cell), 1
                        )
                        old_ivec[cells.index(first_cell)] -= 1
                    if last_cells is not None:
                        for pred_idx in pred_orders:
                            if cur_idx >= pred_idx - 1:
                                pred_cell = last_cells[-pred_idx]
                                w_new = w_new * cell_graph.get_two_table_with_pred_weight(
                                    (cell, pred_cell), pred_idx
                                )
                                old_ivec[cells.index(pred_cell)] -= 1
                        new_last_cells = last_cells[1:] + (cell,)
                    else:
                        new_last_cells = None

                    # CHANGED: replace reduce(lambda) with explicit typed loop
                    cdef object accum = Rational(1, 1)
                    for k in range(n_cells):
                        accum = accum * cell_graph.get_two_table_weight(
                            (cell, cells[k])
                        ) ** old_ivec[k]
                    w_new = w_new * accum

                    ivec = tuple(
                        (num if k != j_idx else num + 1)
                        for k, num in enumerate(ivec)
                    )
                    new_last_cells = (
                        tuple(new_last_cells)
                        if new_last_cells is not None else None
                    )
                    w_new = w_new + table.get(
                        (ivec, new_last_cells, first_cell),
                        (Rational(0, 1), ())
                    )[0]
                    table[(tuple(ivec), new_last_cells, first_cell)] = (
                        w_new, new_ccs
                    )
        res = res + weight * sum(w for w, _ in table.values())

    if context.unary_evidence_encoding == UnaryEvidenceEncoding.PC:
        res = res / MultinomialCoefficients.coef(
            tuple(
                i for _, i in context.partition_constraint.partition
            )
        )
    return res
```

**Step 2: Build**

```bash
uv run python setup.py build_ext --inplace 2>&1 | grep -E "IncrementalWFOMC|error|Error"
```
Expected: compiles without error.

**Step 3: Run incremental-specific tests**

```bash
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -x -q -k "incremental or MATH"
```
Expected: all pass.

**Step 4: Commit**

```bash
git add src/wfomc/algo/IncrementalWFOMC.pyx
git commit -m "feat(cython): add IncrementalWFOMC.pyx with typed loop variables"
```

---

## Task 7: Cythonize `FastWFOMC.pyx`

**Files:**
- Create: `src/wfomc/algo/FastWFOMC.pyx`

**Step 1: Create `src/wfomc/algo/FastWFOMC.pyx`**

```cython
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from collections import defaultdict
from itertools import product
from logzero import logger
from contexttimer import Timer

from wfomc.cell_graph import build_cell_graphs
from wfomc.context import WFOMCContext
from wfomc.network import PartitionConstraint
from wfomc.utils import MultinomialCoefficients, multinomial_less_than, RingElement, Rational
from wfomc.fol import Const, Pred, QFFormula


def fast_wfomc(context: WFOMCContext, modified_cell_symmetry: bool = False) -> object:
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    partition_constraint = context.partition_constraint
    if partition_constraint is None:
        return _fast_wfomc(formula, domain, get_weight, modified_cell_symmetry)
    else:
        return _fast_wfomc_with_pc(formula, domain, get_weight, partition_constraint)


def _fast_wfomc(formula, domain, get_weight, modified_cell_symmetry: bool = False) -> object:
    cdef int domain_size = len(domain)
    cdef int i, j, l
    cdef object res = Rational(0, 1)
    cdef object res_, coef, body, mul, weight

    for opt_cell_graph, weight in build_cell_graphs(
        formula, get_weight, optimized=True,
        domain_size=domain_size,
        modified_cell_symmetry=modified_cell_symmetry
    ):
        cliques = opt_cell_graph.cliques
        nonind = opt_cell_graph.nonind
        i2_ind = opt_cell_graph.i2_ind
        nonind_map = opt_cell_graph.nonind_map

        res_ = Rational(0, 1)
        with Timer() as t:
            for partition in multinomial_less_than(len(nonind), domain_size):
                mu = tuple(partition)
                if sum(partition) < domain_size:
                    mu = mu + (domain_size - sum(partition),)
                coef = MultinomialCoefficients.coef(mu)
                body = Rational(1, 1)

                for i, clique1 in enumerate(cliques):
                    for j, clique2 in enumerate(cliques):
                        if i in nonind and j in nonind:
                            if i < j:
                                body = body * opt_cell_graph.get_two_table_weight(
                                    (clique1[0], clique2[0])
                                ) ** (partition[nonind_map[i]] *
                                      partition[nonind_map[j]])

                for l in nonind:
                    body = body * opt_cell_graph.get_J_term(
                        l, partition[nonind_map[l]]
                    )
                    if not modified_cell_symmetry:
                        body = body * opt_cell_graph.get_cell_weight(
                            cliques[l][0]
                        ) ** partition[nonind_map[l]]

                opt_cell_graph.setup_term_cache()
                mul = opt_cell_graph.get_term(len(i2_ind), 0, partition)
                res_ = res_ + coef * mul * body
        res = res + weight * res_
    logger.info('WFOMC time: %s', t.elapsed)
    return res


def _fast_wfomc_with_pc(formula, domain, get_weight,
                         partition_constraint: PartitionConstraint) -> object:
    logger.info('Invoke faster WFOMC with partition constraint')
    logger.info('Partition constraint: %s', partition_constraint)
    cdef int domain_size = len(domain)
    cdef int i, j, l
    cdef object res = Rational(0, 1)
    cdef object res_, coef, body, weight

    for opt_cell_graph, weight in build_cell_graphs(
        formula, get_weight,
        optimized=True, domain_size=domain_size,
        modified_cell_symmetry=True,
        partition_constraint=partition_constraint
    ):
        cliques = opt_cell_graph.cliques
        nonind = opt_cell_graph.nonind
        nonind_map = opt_cell_graph.nonind_map

        pred_partitions = list(num for _, num in partition_constraint.partition)
        partition_cliques = opt_cell_graph.partition_cliques

        res_ = Rational(0, 1)
        with Timer() as t:
            for configs in product(
                *(list(multinomial_less_than(len(partition_cliques[idx]), constrained_num))
                  for idx, constrained_num in enumerate(pred_partitions))
            ):
                coef = Rational(1, 1)
                remainings = list()
                overall_config = list(0 for _ in range(len(cliques)))
                clique_configs = defaultdict(list)
                for idx, (constrained_num, config) in enumerate(zip(pred_partitions, configs)):
                    remainings.append(constrained_num - sum(config))
                    mu = tuple(config) + (constrained_num - sum(config),)
                    coef = coef * MultinomialCoefficients.coef(mu)
                    for num, clique_idx in zip(config, partition_cliques[idx]):
                        overall_config[clique_idx] = overall_config[clique_idx] + num
                        clique_configs[clique_idx].append(num)

                body = opt_cell_graph.get_i1_weight(remainings, overall_config)

                for i, clique1 in enumerate(cliques):
                    for j, clique2 in enumerate(cliques):
                        if i in nonind and j in nonind:
                            if i < j:
                                body = body * opt_cell_graph.get_two_table_weight(
                                    (clique1[0], clique2[0])
                                ) ** (overall_config[nonind_map[i]] *
                                      overall_config[nonind_map[j]])

                for l in nonind:
                    body = body * opt_cell_graph.get_J_term(
                        l, tuple(clique_configs[nonind_map[l]])
                    )
                res_ = res_ + coef * body
        res = res + weight * res_
    logger.info('WFOMC time: %s', t.elapsed)
    return res
```

**Step 2: Build**

```bash
uv run python setup.py build_ext --inplace 2>&1 | grep -E "FastWFOMC|error|Error"
```

**Step 3: Run fast-specific tests**

```bash
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -x -q
```
Expected: all pass.

**Step 4: Commit**

```bash
git add src/wfomc/algo/FastWFOMC.pyx
git commit -m "feat(cython): add FastWFOMC.pyx with typed loop variables"
```

---

## Task 8: Cythonize `StandardWFOMC.pyx`

**Files:**
- Create: `src/wfomc/algo/StandardWFOMC.pyx`

**Step 1: Create `src/wfomc/algo/StandardWFOMC.pyx`**

```cython
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from wfomc.cell_graph import CellGraph, Cell, build_cell_graphs
from wfomc.context import WFOMCContext
from wfomc.utils import MultinomialCoefficients, multinomial, RingElement, Rational


def get_config_weight_standard(cell_graph: CellGraph,
                               cell_config: dict) -> object:
    cdef int i, j
    cdef object res = Rational(1, 1)
    cdef object n_i, n_j
    cells_items = list(cell_config.items())
    for i, (cell_i, n_i) in enumerate(cells_items):
        if n_i == 0:
            continue
        res = res * cell_graph.get_nullary_weight(cell_i)
        break
    for i, (cell_i, n_i) in enumerate(cells_items):
        if n_i == 0:
            continue
        res = res * cell_graph.get_cell_weight(cell_i) ** n_i
        res = res * cell_graph.get_two_table_weight(
            (cell_i, cell_i)
        ) ** (n_i * (n_i - 1) // 2)
        for j, (cell_j, n_j) in enumerate(cells_items):
            if j <= i:
                continue
            if n_j == 0:
                continue
            res = res * cell_graph.get_two_table_weight(
                (cell_i, cell_j)
            ) ** (n_i * n_j)
    return res


def standard_wfomc(context: WFOMCContext) -> object:
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    cdef object res = Rational(0, 1)
    cdef int domain_size = len(domain)
    cdef int n_cells
    cdef object coef, weight

    for cell_graph, weight in build_cell_graphs(formula, get_weight):
        res_ = Rational(0, 1)
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        for partition in multinomial(n_cells, domain_size):
            coef = MultinomialCoefficients.coef(partition)
            cell_config = dict(zip(cells, partition))
            res_ = res_ + coef * get_config_weight_standard(cell_graph, cell_config)
        res = res + weight * res_
    return res
```

**Step 2: Build and test**

```bash
uv run python setup.py build_ext --inplace 2>&1 | grep -E "StandardWFOMC|error|Error"
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -x -q
```
Expected: all pass.

**Step 3: Commit**

```bash
git add src/wfomc/algo/StandardWFOMC.pyx
git commit -m "feat(cython): add StandardWFOMC.pyx with typed loop variables"
```

---

## Task 9: Cythonize `RecursiveWFOMC.pyx`

**Files:**
- Create: `src/wfomc/algo/RecursiveWFOMC.pyx`

**Step 1: Create `src/wfomc/algo/RecursiveWFOMC.pyx`**

```cython
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import math
import functools
from collections import Counter
import pynauty
import networkx as nx

from wfomc.cell_graph import CellGraph, build_cell_graphs
from wfomc.utils import RingElement, Rational
from wfomc.utils.polynomial_flint import expand
from wfomc.fol.syntax import Const, Pred, QFFormula
from wfomc.context import WFOMCContext


class NautyContext(object):
    def __init__(self, domain_size, cell_weights, edge_weights):
        self.graph = None
        self.layer_num_for_convert = 0
        self.node_num = 0
        self.edge_color_num = 0
        self.edge_weight_to_color = {1: 0}
        self.edge_color_mat = []
        self.vertex_color_no = 0
        self.vertex_weight_to_color = {}
        self.adjacency_dict = {}
        self.cache_for_nauty = {}
        self.ig_cache = IsomorphicGraphCache(domain_size)

        self.edgeWeight_to_edgeColor(edge_weights)
        self.calculate_adjacency_dict(len(cell_weights))
        self.create_graph()

    def edgeWeight_to_edgeColor(self, edge_weights):
        for lst in edge_weights:
            tmp_list = []
            for w in lst:
                if str(w) not in self.edge_weight_to_color:
                    self.edge_color_num += 1
                    self.edge_weight_to_color[str(w)] = self.edge_color_num
                tmp_list.append(self.edge_weight_to_color[str(w)])
            self.edge_color_mat.append(tmp_list)

    def calculate_adjacency_dict(self, int cell_num):
        cdef int i, j, l, k
        self.layer_num_for_convert = math.ceil(math.log2(self.edge_color_num + 1))
        self.node_num = cell_num * self.layer_num_for_convert

        adjacency_dict = {}
        for i in range(self.node_num):
            adjacency_dict[i] = []

        c2layers = {}
        for k in range(self.edge_color_num + 1):
            bi = bin(k)[2:][::-1]
            layers = [i for i in range(len(bi)) if bi[i] == '1']
            c2layers[k] = layers

        for i in range(cell_num):
            for j in range(cell_num):
                layers = c2layers[self.edge_color_mat[i][j]]
                for l in layers:
                    adjacency_dict[l * cell_num + i].append(l * cell_num + j)

        for i in range(cell_num):
            clique = [i + j * cell_num for j in range(self.layer_num_for_convert)]
            for ii in clique:
                for jj in clique:
                    if ii != jj:
                        adjacency_dict[ii].append(jj)
        self.adjacency_dict = adjacency_dict

    def create_graph(self):
        self.graph = pynauty.Graph(
            self.node_num, directed=False,
            adjacency_dict=self.adjacency_dict
        )

    def update_graph(self, colored_vertices):
        self.graph.set_vertex_coloring(colored_vertices)
        return self.graph

    def get_vertex_color(self, weight):
        if str(weight) not in self.vertex_weight_to_color:
            self.vertex_weight_to_color[str(weight)] = self.vertex_color_no
            self.vertex_color_no += 1
        return self.vertex_weight_to_color[str(weight)]

    def cellWeight_To_vertexColor(self, cell_weights):
        vertex_colors = [self.get_vertex_color(w) for w in cell_weights]
        color_dict = Counter(vertex_colors)
        color_kind = tuple(sorted(color_dict))
        color_count = tuple(color_dict[num] for num in color_kind)
        return vertex_colors, color_kind, color_count

    def extend_vertex_coloring(self, colored_vertices, int no_color):
        cdef int i
        ext_colored_vertices = []
        for i in range(self.layer_num_for_convert):
            ext_colored_vertices += [x + no_color * i for x in colored_vertices]
        no_color *= self.layer_num_for_convert
        vertex_coloring = [set() for _ in range(no_color)]
        for i in range(len(ext_colored_vertices)):
            vertex_coloring[ext_colored_vertices[i]].add(i)
        return vertex_coloring


class TreeNode(object):
    def __init__(self, cell_weights, depth):
        self.cell_weights = cell_weights
        self.depth = depth
        self.cell_to_children = dict()


class IsomorphicGraphCache(object):
    def __init__(self, int domain_size):
        self.cache = [{} for _ in range(domain_size)]
        self.cache_hit_count = [0] * domain_size

    def get(self, int level, color_kind, color_count, can_label):
        if color_kind not in self.cache[level]:
            self.cache[level][color_kind] = {}
            return None
        if color_count not in self.cache[level][color_kind]:
            self.cache[level][color_kind][color_count] = {}
            return None
        if can_label not in self.cache[level][color_kind][color_count]:
            return None
        self.cache_hit_count[level] += 1
        return self.cache[level][color_kind][color_count][can_label]

    def set(self, int level, color_kind, color_count, can_label, value):
        if color_kind not in self.cache[level]:
            self.cache[level][color_kind] = {}
        if color_count not in self.cache[level][color_kind]:
            self.cache[level][color_kind][color_count] = {}
        self.cache[level][color_kind][color_count][can_label] = value


def adjust_vertex_coloring(colored_vertices):
    sorted_colors = sorted(set(colored_vertices))
    rank = {v: i for i, v in enumerate(sorted_colors)}
    return [rank[x] for x in colored_vertices], len(sorted_colors)


ENABLE_ISOMORPHISM = True

def dfs_wfomc_real(cell_weights, edge_weights, int domain_size,
                   nauty_ctx: NautyContext, node=None) -> object:
    cdef int l, cell_num = len(cell_weights)
    cdef object res = 0
    cdef object w_l, value

    for l in range(cell_num):
        w_l = cell_weights[l]
        new_cell_weights = [expand(cell_weights[i] * edge_weights[l][i])
                            for i in range(cell_num)]
        if domain_size - 1 == 1:
            value = sum(new_cell_weights)
        else:
            original_vertex_colors, vertex_color_kind, vertex_color_count = \
                nauty_ctx.cellWeight_To_vertexColor(new_cell_weights)
            if ENABLE_ISOMORPHISM:
                adjust_vertex_colors, no_color = adjust_vertex_coloring(
                    original_vertex_colors
                )
                key = tuple(adjust_vertex_colors)
                if key not in nauty_ctx.cache_for_nauty:
                    can_label = pynauty.certificate(
                        nauty_ctx.update_graph(
                            nauty_ctx.extend_vertex_coloring(
                                adjust_vertex_colors, no_color
                            )
                        )
                    )
                    nauty_ctx.cache_for_nauty[key] = can_label
                else:
                    can_label = nauty_ctx.cache_for_nauty[key]
            else:
                can_label = tuple(original_vertex_colors)

            value = nauty_ctx.ig_cache.get(
                domain_size - 1, vertex_color_kind, vertex_color_count, can_label
            )
            if value is None:
                value = dfs_wfomc_real(
                    new_cell_weights, edge_weights, domain_size - 1, nauty_ctx, None
                )
                value = expand(value)
                nauty_ctx.ig_cache.set(
                    domain_size - 1, vertex_color_kind, vertex_color_count,
                    can_label, value
                )
        res = res + w_l * value
    return res


def find_independent_sets(cell_graph: CellGraph):
    cdef int i, j
    g = nx.Graph()
    cells = cell_graph.cells
    g.add_nodes_from(range(len(cells)))
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if cell_graph.get_two_table_weight(
                    (cells[i], cells[j])) != Rational(1, 1):
                g.add_edge(i, j)

    self_loop = set()
    for i in range(len(cells)):
        if cell_graph.get_two_table_weight(
                (cells[i], cells[i])) != Rational(1, 1):
            self_loop.add(i)

    non_self_loop = g.nodes - self_loop
    if len(non_self_loop) == 0:
        i1_ind = set()
    else:
        i1_ind = set(nx.maximal_independent_set(g.subgraph(non_self_loop)))
    g_ind = set(nx.maximal_independent_set(g, nodes=i1_ind))
    i2_ind = g_ind.difference(i1_ind)
    non_ind = g.nodes - i1_ind - i2_ind
    return list(i1_ind), list(i2_ind), list(non_ind)


ROOT = TreeNode([], 0)


def recursive_wfomc(context: WFOMCContext) -> object:
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred

    cdef int domain_size = len(domain)
    cdef object res = Rational(0, 1)
    cdef object weight, res_

    for cell_graph, weight in build_cell_graphs(
            formula, get_weight, leq_pred=leq_pred):
        cell_weights = cell_graph.get_all_weights()[0]
        edge_weights = cell_graph.get_all_weights()[1]

        nauty_ctx = NautyContext(domain_size, cell_weights, edge_weights)
        global ROOT
        ROOT.cell_weights = cell_weights
        res_ = dfs_wfomc_real(cell_weights, edge_weights, domain_size, nauty_ctx, ROOT)
        res = res + weight * res_
    return res
```

**Step 2: Build and test**

```bash
uv run python setup.py build_ext --inplace 2>&1 | grep -E "RecursiveWFOMC|error|Error"
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -x -q
```
Expected: all pass.

**Step 3: Commit**

```bash
git add src/wfomc/algo/RecursiveWFOMC.pyx
git commit -m "feat(cython): add RecursiveWFOMC.pyx with typed loop variables"
```

---

## Task 10: Update `algo/__init__.py` with try-import

**Files:**
- Modify: `src/wfomc/algo/__init__.py`

**Step 1: Update `algo/__init__.py`**

The `Algo` enum must stay in this file. Only the function imports become try-imports:

```python
from enum import Enum

from wfomc._compat import try_import_cython as _try_cy

_std = _try_cy("wfomc.algo.StandardWFOMC", "wfomc.algo.StandardWFOMC")
_fast = _try_cy("wfomc.algo.FastWFOMC", "wfomc.algo.FastWFOMC")
_incr = _try_cy("wfomc.algo.IncrementalWFOMC", "wfomc.algo.IncrementalWFOMC")
_rec = _try_cy("wfomc.algo.RecursiveWFOMC", "wfomc.algo.RecursiveWFOMC")

standard_wfomc = _std.standard_wfomc
fast_wfomc = _fast.fast_wfomc
incremental_wfomc = _incr.incremental_wfomc
recursive_wfomc = _rec.recursive_wfomc

__all__ = [
    "standard_wfomc",
    "fast_wfomc",
    "incremental_wfomc",
    "recursive_wfomc",
]


class Algo(Enum):
    STANDARD = 'standard'
    FAST = 'fast'
    FASTv2 = 'fastv2'
    INCREMENTAL = 'incremental'
    RECURSIVE = 'recursive'

    def __str__(self):
        return self.value
```

**Step 2: Final full test run with WFOMC_CYTHON_ONLY=1**

```bash
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -v
```
Expected: all tests pass.

**Step 3: Verify Python fallback works (simulate no Cython .so)**

```bash
# Test that Python fallback path still works without Cython
uv run pytest tests/wfomc_test.py -v
```
Expected: all tests pass (uses Python modules).

**Step 4: Commit**

```bash
git add src/wfomc/algo/__init__.py
git commit -m "feat(cython): update algo/__init__.py with try-import for Cython extensions"
```

---

## Task 11: Update CLAUDE.md and verify build documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add build commands to CLAUDE.md**

Add to the Commands section:
```markdown
# Build Cython extensions (required before WFOMC_CYTHON_ONLY=1)
uv run python setup.py build_ext --inplace

# Run tests with Cython extensions enforced
WFOMC_CYTHON_ONLY=1 uv run pytest tests/

# Run tests with Python fallback (no Cython required)
uv run pytest tests/
```

**Step 2: Final verification — clean build from scratch**

```bash
# Remove all .so files
find src -name "*.so" -delete
find src -name "*.c" -delete
# Rebuild
uv run python setup.py build_ext --inplace
# Run full suite
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -v --tb=short
```
Expected: clean compilation, all tests green.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add Cython build commands to CLAUDE.md"
```

---

## Quick Reference

```bash
# Build all Cython extensions
uv run python setup.py build_ext --inplace

# Run with Cython enforced (fails loudly if .so missing)
WFOMC_CYTHON_ONLY=1 uv run pytest tests/wfomc_test.py -v

# Run with Python fallback (always works)
uv run pytest tests/wfomc_test.py -v

# Rebuild single module (e.g. after editing IncrementalWFOMC.pyx)
uv run python setup.py build_ext --inplace 2>&1 | grep IncrementalWFOMC

# Check which .so files exist
find src -name "*.so" | sort
```
