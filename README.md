# Exact Lifted Counter for Two-Variable Logic and Extensions

This tool is for counting the models (or combinatorical structures) from the two-variable fragment of first-order logic and extensions.

### Installation

Install UV via:
[github](https://github.com/astral-sh/uv) or
```
pip install uv
```

Sync the dependencies:
```
uv sync
```

### How to use
```
$ uv run wfomc -i [input] -a [algo] -e [unary_evidence_strategy] [-l [linear_order_encoding]]
```
where
- `input` is the input file with the suffix `.wfomcs` or `.mln`
- `algo` is the algorithm to use, including:
  - `standard`: the standard WFOMC algorithm in Beame et al. (2015)
  - `fast`: the fast WFOMC algorithm in Timothy van Bremen and Ondrej Kuzelka (2021)
  - `fastv2` (default): the optimized fast WFOMC algorithm
  - `incremental`: the incremental WFOMC algorithm for linear order axiom in Toth and Kuzelka (2022)
  - `incremental3`: the incremental WFOMC algorithm with factorized counting-quantifier
    and unary-evidence support (also handles modulo counting quantifiers)
  - `recursive`: the recursive WFOMC algorithm for linear order axiom in Meng et al. (2024)
  - `propositional`: ground the (Skolemized) sentence over the domain and count with the
    external [ganak](https://github.com/meelgroup/ganak) propositional model counter.
    Intended as a ground-truth baseline. Requires a ganak binary; see *Propositional counter* below.
- `unary_evidence_strategy` is the unary evidence strategy, including:
  - `auto` (default): selects the best supported implementation for each
    algorithm
  - `ccs`: forces the modular auxiliary-predicate and cardinality-constraint
    encoding
- `linear_order_encoding` (only used by `-a propositional`) controls how the order axioms
  (`LEQ` / `PRED` / `CIRCULAR_PRED`) are encoded:
  - `pin` (default): pin every ground order atom to a canonical sorted order/cycle;
    `decode_result` applies the `n!` multiplier. Cheap and fast.
  - `axioms`: emit the FO³ axioms / definitions explicitly. Pin-free but materially slower.

Unary evidence is represented once as a `UnaryEvidencePartition` and compiled
into a `CellEvidenceAllocation` for each cell graph. With `auto`, the current
implementation mapping is:

| Algorithm | Unary evidence strategy |
| --- | --- |
| `standard` | evidence-profile configuration coefficients |
| `fast` | classic CCS fallback |
| `fastv2` | evidence-expanded cell graph view |
| `incremental` | threaded evidence-profile capacities |
| `incremental3` | evidence-profile configuration coefficients |
| `recursive` | classic CCS fallback |

The explicit `ccs` strategy is available for every algorithm as an independent,
modular correctness reference.

Unary evidence grouping assumes the sentence does not distinguish named domain
constants. Such inputs fail fast rather than silently overcounting.

### Python API

You can also call the solver directly from a Python script:

```python
from sympy import symbols
from wfomc import Algo, Const, Pred, Rational, WFOMCProblem, fol_parse, to_sc2, wfomc

x = symbols("x")
domain = {Const("a"), Const("b"), Const("c")}
sentence = to_sc2(fol_parse(r"\forall X: (P(X))"))
weights = {
    Pred("P", 1): (x, Rational(1, 1)),
}

problem = WFOMCProblem(sentence, domain, weights)
result = wfomc(problem, algo=Algo.FASTv2)

print(result)                  # exact result
print(result.constant_value())  # SymPy Rational, or None for polynomial results
for degrees, coeff in result.terms([x]):
    print(degrees, coeff)
```

`wfomc(...)` returns a `WFOMCResult`, not a raw FLINT polynomial. Use:

- `result.is_zero()`
- `result.is_constant()`
- `result.constant_value()`
- `result.is_polynomial()`
- `result.variable_names()`
- `result.terms([...])`

The underlying solver still uses FLINT internally for exact polynomial arithmetic,
but callers should treat that as an implementation detail.

`WFOMCProblem(..., weights=...)` accepts a dictionary from `Pred` to
`(positive_weight, negative_weight)`. Each weight may be:

- a Python `int`;
- a Python `float`, converted exactly through `fractions.Fraction(float_value)`;
- a SymPy expression (`sympy.Expr`), including `sympy.Rational`, symbols, and
  polynomial expressions such as `x`, `2*x + 1`, or `x*y`.

Avoid passing raw FLINT values or `sympy.Poly` objects as weights. If you need a
polynomial weight, pass the corresponding SymPy expression instead, e.g.
`x**2 + 3*x + 1`.

## Input format

The input file with the suffix `.wfomcs` contains the following information **in order**:
1. First-order sentence with at most two logic variables (must in capital letters, e.g., `X`, `Y`, `Z`, etc.), see [fol_grammar.py](sampling_fo2/parser/fol_grammar.py) for details, e.g.,
  * `\forall X: (\forall Y: (R(X, Y) <-> Z(X, Y)))`
  * `\forall X: (\exists Y: (R(X, Y)))`
  * `\exists X: (F(X) -> \forall Y: (R(X, Y)))`
  * ..., even more complex sentence...
2. Domain: 
  * `domain=3` or
  * `domain={p1, p2, p3}`, where `p1`, `p2`, `p3` are the constants in the domain (must start with a lowercase letter).
3. Weighting (optional): `positve_weight negative_weight predicate`
4. Cardinality constraint (optional): 
  * `|P| = k`
  * `|P| > k`
  * `|P| >= k`
  * `|P| < k`
  * `|P| <= k`
5. Unary evidence (optional): 
  * `P(p1), ~P(p3)`

### Use linear order constraint

To use linear order constraint (or linear order axiom), just use the predefined predicate `LEQ` in the input file. 
For the `head-tail` example in [Lifted Inference with Linear Order Axiom.](https://doi.org/10.1609/aaai.v37i10.26449), you can write the sentence as:
```
\forall X: (\forall Y: (~H(X) | ~T(X))) &
\forall X: (\forall Y: (H(Y) & LEQ(X, Y) -> H(X))) &
\forall X: (\forall Y: (T(X) & LEQ(X, Y) -> T(Y))) &
```

The $k$-th predecessors are predifined as `PREk`, e.g., `PRE2(X, Y)` means `Y` is the 2nd predecessor of `X` in the linear order.
See [predk](models/predk/) for more examples.
The circular predecessor `CIRCULAR_PRED` is also predefined with `CIRCULAR_PRED(X, Y)` means `Y` is the predecessor of `X` in a circular order.
The output count of circular order is always divided by the domain size to avoid overcounting.

> **Note: To use linear order constraint, you must use the `incremental`, `incremental3`, `recursive`, or `propositional` algorithm. To use $k$-th predecessor or circular predecessor, you must use the `incremental` or `propositional` algorithm.**


### Example input file

- 2 colored graphs:
```
\forall X: (\forall Y: ((E(X,Y) -> E(Y,X)) &
                        (R(X) | B(X)) &
                        (~R(X) | ~B(X)) &
                        (E(X,Y) -> ~(R(X) & R(Y)) & ~(B(X) & B(Y)))))

V = 10
```

- 2 regular graphs:
```
\forall X: (~E(X,X)) &
\forall X: (\forall Y: ((E(X,Y) -> E(Y,X)) &
                        (E(X,Y) <-> (F1(X,Y) | F2(X,Y))) &
                        (~F1(X, Y) | ~F2(X,Y)))) &
\forall X: (\exists Y: (F1(X,Y))) & 
\forall X: (\exists Y: (F2(X,Y)))

V = 6
|E| = 12
```

- 2 regular graphs where `\exists_{=2} Y: (E(X,Y))` means there are exactly 2 edges from each node (please refer to [Weighted First-Order Model Counting in the Two-Variable Fragment With Counting Quantifiers](https://jair.org/index.php/jair/article/view/12320/26673):
```
\forall X: (~E(X,X)) &
\forall X: (\forall Y: (E(X,Y) -> E(Y,X))) &
\forall X: (\exists_{=2} Y: (E(X,Y)))

V = 6
```

- Transformed from `friends-smokes` MLN:
```
\forall X: (~fr(X,X)) &
\forall X: (\forall Y: (fr(X,Y) -> fr(Y,X))) &
\forall X: (\forall Y: (aux(X,Y) <-> (fr(X,Y) & sm(X) -> sm(Y)))) &
\forall X: (\exists Y: (fr(X,Y)))

person = 10
2.7 1 aux
```

> **Note: Now you can also directly input the MLN in the form defined in [mln_grammar.py](sampling_fo2/parser/mln_grammar.py)**
```
~friends(X,X).
friends(X,Y) -> friends(Y,X).
2.7 friends(X,Y) & smokes(X) -> smokes(Y)
\forall X: (\exists Y: (fr(X,Y))).
# or 
\exists Y: (fr(X,Y)).

person = 10
```

> Add unary evidence:
```
~friends(X,X).
friends(X,Y) -> friends(Y,X).
2.7 friends(X,Y) & smokes(X) -> smokes(Y)
\forall X: (\exists Y: (fr(X,Y))).

person = {alice, bob, charlie, david, eve}

smokes(alice), ~smokes(bob)
``` 

More examples are in [models](models/)

## Propositional counter

The `propositional` algorithm grounds the universally quantified (Skolemized) sentence over every pair of domain elements and hands the resulting weighted CNF to [ganak](https://github.com/meelgroup/ganak). It is intended as a textbook-definition ground-truth baseline against which the lifted algorithms can be checked.

It supports plain FO² with rational weights, cardinality constraints and counting quantifiers (`∃_{=k}`), unary evidence, and three order axioms: `LEQ` (linear order), `PRED` (= `PRED1`, the immediate linear predecessor), and `CIRCULAR_PRED` (the immediate circular predecessor). The `PREDk` family for `k > 1` raises an error.

The encoding of the order axioms is selected per call via the `--linear-order-encoding` (short `-l`) CLI flag, the `linear_order_encoding=` keyword argument on `wfomc()` / `propositional_wfomc()`, or the module-level constant `LINEAR_ORDER_ENCODING` in [src/wfomc/algo/PropositionalWFOMC.py](src/wfomc/algo/PropositionalWFOMC.py):

| Setting | Mechanism | Multiplier | When to use |
|---|---|---|---|
| `"pin"` (default) | Each ground `LEQ` / `PRED1` / `CIRCULAR_PRED` atom is pinned to its value under a canonical sorted order / cycle on the domain. | `× n!` via `decode_result`. | Default; the cheap textbook trick. |
| `"axioms"` | The order axioms are emitted explicitly as FO³ definitions (with Tseitin auxiliaries). | None. | A pin-free, FO-axiomatic baseline (materially slower). |

**Unary evidence handling.** Direct unit-clause evidence (`UnaryEvidenceStrategy.AUTO`) is element-specific and therefore breaks pin-and-multiply's symmetry argument when an order axiom is present. The solver's `resolve_unary_evidence_strategy` automatically picks the right strategy for the propositional algorithm:

| `linear_order_encoding` | Has order axiom? | Effective strategy | ganak mode |
|---|---|---|---|
| `"pin"` | yes | `ccs` (symmetric fingerprint counts) | `--mode 3` (polynomial) |
| `"pin"` | no | `auto` (direct unit clauses) | `--mode 1` |
| `"axioms"` | any | `auto` (direct unit clauses) | `--mode 1` |

An explicit `-e ccs` always forces the cardinality-constraint encoding for every algorithm.

ganak is invoked in two modes: exact rational weighted counting (`--mode 1`) when no symbolic/polynomial weights are involved, and multivariate-polynomial weighted counting (`--mode 3`) when cardinality constraints or counting quantifiers introduce symbolic weights. Install the pinned binary into the active uv environment with:

```
uv run wfomc-install-ganak
```

The runtime lookup order is the `--ganak-path` argument to `propositional_wfomc()`, the `GANAK` environment variable, then `ganak` on `PATH`.

## References

Please refer to [reference.bib](reference.bib) for the references of the algorithms.
