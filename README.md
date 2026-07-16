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
$ uv run wfomc --input [input] --algo [algo]
```
where
- `input` is the input file with the suffix `.wfomcs` or `.mln`
- `algo` is the algorithm to use, including:
  - `standard`: the standard WFOMC algorithm in Beame et al. (2015)
  - `fast`: the fast WFOMC algorithm in Timothy van Bremen and Ondrej Kuzelka (2021)
  - `fastv2`: the optimized fast WFOMC algorithm
  - `incremental`: the incremental WFOMC algorithm for linear order axiom in Toth and Kuzelka (2022)
  - `incremental3`: the incremental WFOMC algorithm with factorized counting-quantifier
    and unary-evidence support (also handles modulo counting quantifiers)
  - `recursive`: the recursive WFOMC algorithm for linear order axiom in Meng et al. (2024)
  - `propositional`: directly ground the source sentence, without normalization or
    logical reductions, and count it with the external
    [ganak](https://github.com/meelgroup/ganak) propositional model counter.
  - `propositional-reduced`: normalize and reduce the problem first, then ground the
    resulting quantifier-free sentence and count it with Ganak. This preserves the
    former propositional implementation for performance and regression comparisons.
`tail-signature` is an experimental Python-API adapter requiring an external
engine factory, so it is intentionally hidden from CLI choices.

The CLI defaults to `standard`. Advanced evidence, order-encoding, arithmetic,
and external-engine options are currently available through the Python API.
Use `-v` for phase summaries and timings, or `-vv` for bounded DEBUG details.
The Python library emits standard `wfomc.*` logging records without configuring
the application's handlers.

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
from wfomc import AlgoName, parse_problem, solve

problem = parse_problem(r"""
\forall X: (P(X))
domain = {a, b, c}
2 1 P
""")
result = solve(problem, algo=AlgoName.FASTV2)

print(result)
print(result.constant_value())
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

Programmatic callers may also construct a typed `Problem` with the builders in
`wfomc.fol`. Exact FLINT values are an internal representation; public callers
should prefer integers, fractions, and parsed model weights.

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
See [predk](models/linear_order/predk/) for more examples.
The circular predecessor `CIRCULAR_PRED` is also predefined with `CIRCULAR_PRED(X, Y)` means `Y` is the predecessor of `X` in a circular order.
The output count of circular order is always divided by the domain size to avoid overcounting.

> **Note: To use linear order constraint, you must use the `incremental`, `incremental3`, `recursive`, `propositional`, or `propositional-reduced` algorithm. To use $k$-th predecessor or circular predecessor, use an algorithm whose feature validation accepts that input.**


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

Both propositional modes hand an explicit weighted CNF to [ganak](https://github.com/meelgroup/ganak), but they reach that CNF through different paths:

| Algorithm | Preparation path | Intended use |
|---|---|---|
| `propositional` | Source `Problem` → finite-domain expansion of ordinary and counting quantifiers → Tseitin CNF | Reduction-independent correctness baseline |
| `propositional-reduced` | normalization → unary-evidence/counting/existential/cardinality reductions → quantifier-free grounding | Compatibility, performance, and reduction-regression comparison |

The direct path supports arbitrary Boolean placement of ordinary and counting quantifiers, ground unary and binary evidence, and simple global constraints of the form `|P| <= k`, `|P| = k`, or `|P| >= k`. The reduced path inherits the reduction pipeline's narrower feature limits: for example, binary evidence and counting sections that cannot be lowered to UFO² plus cardinality constraints are rejected.

The order encoding can be selected through `AlgoOptions(linear_order_encoding=...)` in the Python API:

| Setting | Mechanism | Multiplier | When to use |
|---|---|---|---|
| `"pin"` (default) | Pin each order atom to a canonical sorted order/cycle. | `× n!` during result decoding | Cheap symmetry-based baseline |
| `"axioms"` | Emit explicit FO³ order axioms. | None | Pin-free baseline; materially slower |

Direct grounding uses ground evidence clauses. If named constants or ground evidence make an ordered problem asymmetric, it defaults to the explicit axioms encoding and rejects an explicitly requested unsound pin encoding. The reduced path preserves the former behavior: with order pinning, unary evidence is first converted to symmetric cardinality constraints.

ganak is invoked in two modes: exact rational weighted counting (`--mode 1`) when no symbolic/polynomial weights are involved, and multivariate-polynomial weighted counting (`--mode 3`) when cardinality constraints or counting quantifiers introduce symbolic weights. Install the pinned binary into the active uv environment with:

```
uv run wfomc-install-ganak
```

The runtime lookup order is `RuntimeOptions.propositional_ganak_path`, the `GANAK` environment variable, then `ganak` on `PATH`. Lifted algorithms also reuse this Ganak installation after their bounded PySAT pair-factor fast path; if Ganak is unavailable or times out, they automatically use the installed PySDD backend instead.

## References

Please refer to [reference.bib](reference.bib) for the references of the algorithms.
