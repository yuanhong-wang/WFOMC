# Exact Lifted Counter for Two-Variable Logic and Extensions

This tool counts models and combinatorial structures in the two-variable
fragment of first-order logic and its extensions.

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
  - `boundary-profile`: the Boundary-Profile decomposition-tree DP with
    automatic plan search
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
`boundary-profile` is a native beta algorithm available from both the CLI and
the `AlgoName` Python API.

Boundary-Profile builds one decomposition tree per structural cell-graph
variant and reuses it across domain sizes. Its default tree search is domain
independent. To select that one reusable tree with the cost model for a chosen
reference size, pass `--bp-tree-reference-domain-size N`, or use
`AlgoOptions(boundary_profile_options=BoundaryProfileOptions(
tree_reference_domain_size=N))` in the Python API. The actual domain still
controls local weight tables, state bounds, and execution statistics; it does
not trigger another tree search.

The CLI defaults to `standard`. Use `-e/--evidence-strategy` to override the
selected algorithm's unary-evidence preparation, and
`--exact-symbolic-backend` to choose the exact polynomial backend. Use `-v` for
phase summaries and timings, or `-vv` for bounded DEBUG details. The Python
library emits standard `wfomc.*` logging records without configuring the
application's handlers.

`--existential-strategy {counting,skolem}` is an `incremental3`-only option.
Other lifted algorithms always apply their fixed weighted-Skolem reduction,
while `propositional` directly expands source quantifiers over the finite domain.

Unary evidence is represented once as `UnaryEvidence` and compiled
into a `CellEvidenceAllocation` for each cell graph. When the CLI/API override
is omitted, the current algorithm defaults are:

| Algorithm | Unary evidence strategy |
| --- | --- |
| `standard` | evidence-profile configuration coefficients |
| `fast` | classic CCS fallback |
| `fastv2` | evidence-expanded cell graph view |
| `incremental` | threaded evidence-profile capacities |
| `incremental3` | evidence-profile configuration coefficients |
| `recursive` | classic CCS fallback |
| `boundary-profile` | classic CCS fallback |

The explicit `ccs` strategy is available for the lifted algorithms that support
unary evidence. Direct propositional grounding uses ground unit clauses.

Unary evidence grouping assumes the sentence does not distinguish named domain
constants. Such inputs fail fast rather than silently overcounting.

### Python API

You can also call the solver directly from a Python script:

```python
from wfomc import (
    AlgoName,
    Domain,
    compile_problem,
    parse_problem,
    solve,
)

instance = parse_problem(r"""
\forall X: (P(X))
domain = {a, b, c}
2 1 P
""")
result = solve(instance, algo=AlgoName.FASTV2)

print(result)
print(result.constant_value())

# Compilation is domain-free and can be reused across domain sizes.
compiled = compile_problem(instance.problem, algo=AlgoName.FASTV2)
for size in (5, 10, 20):
    print(size, solve(compiled, Domain.of_size(size)))
```

`solve(...)` returns a `WFOMCResult`, not a raw FLINT polynomial. Use:

- `result.is_zero()`
- `result.is_constant()`
- `result.constant_value()`
- `result.is_polynomial()`
- `result.variable_names()`
- `result.terms([...])`

The underlying solver still uses FLINT internally for exact polynomial arithmetic,
but callers should treat that as an implementation detail.

`parse_problem(...)` parses text, while `parse_problem_file(...)` parses a
`.wfomcs` or `.mln` path. Both return an explicit `ProblemInstance` with
separate `.problem` and `.domain` fields; file parsing also records
`.source_path` as non-semantic provenance. Programmatic callers may construct a
typed `Problem` with the builders in `wfomc.fol`, then pass a separate `Domain`
to `solve`. Exact FLINT values are an internal representation; public callers
should prefer integers, fractions, and parsed model weights.

The top-level package also exports the configuration types `AlgoOptions`,
`BoundaryProfileOptions`, `EvidenceStrategy`, `ExistentialStrategy`,
`LinearOrderEncoding`, `WeightOptions`, `RuntimeOptions`, and `RuntimeContext`.

## Input format

The input file with the suffix `.wfomcs` contains the following information **in order**:
1. First-order sentence with at most two logical variables (written with
   capital letters such as `X` and `Y`), see
   [wfomcs.py](src/wfomc/parser/grammar/wfomcs.py) for details, e.g.,
  * `\forall X: (\forall Y: (R(X, Y) <-> Z(X, Y)))`
  * `\forall X: (\exists Y: (R(X, Y)))`
  * `\exists X: (F(X) -> \forall Y: (R(X, Y)))`
  * ..., even more complex sentence...
2. Domain: 
  * `domain=3` or
  * `domain={p1, p2, p3}`, where `p1`, `p2`, `p3` are the constants in the domain (must start with a lowercase letter).
3. Weighting (optional): `positive_weight negative_weight predicate`
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

The $k$-th predecessor predicates are predefined as `PREDk`, e.g.,
`PRED2(X, Y)` means `Y` is the second predecessor of `X` in the linear order.
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

> **Note: You can also directly input an MLN in the form defined in
> [mln.py](src/wfomc/parser/grammar/mln.py).**
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

The order encoding can be selected with
`--linear-order-encoding {pin,axioms}` (`-l`) in the CLI or
`AlgoOptions(linear_order_encoding=LinearOrderEncoding.AXIOMS)` in the Python
API. Both types are importable directly from `wfomc`. The option is valid only
for `propositional` and `propositional-reduced`:

| Setting | Mechanism | Multiplier | When to use |
|---|---|---|---|
| `"pin"` (default) | Pin each order atom to a canonical sorted order/cycle. | `× n!` during result decoding | Cheap symmetry-based baseline |
| `"axioms"` | Emit explicit FO³ order axioms. | None | Pin-free baseline; materially slower |

Direct grounding uses ground evidence clauses. If named constants or ground evidence make an ordered problem asymmetric, it defaults to the explicit axioms encoding and rejects an explicitly requested unsound pin encoding. The reduced path preserves the former behavior: with order pinning, unary evidence is first converted to symmetric cardinality constraints.

ganak is invoked in two modes: exact rational weighted counting (`--mode 1`) when no symbolic/polynomial weights are involved, and multivariate-polynomial weighted counting (`--mode 3`) when cardinality constraints or counting quantifiers introduce symbolic weights. Install the pinned binary into the active uv environment with:

```
uv run wfomc-install-ganak
```

The installer checks out Ganak
`82a1d1fb6f0d6fb4a46b825f84b29567728ae483` together with its compatible
Arjun revision `1553e6b3ebdd76ba3b66d3fece4cf8de4e2743ce`. Both revisions are
fixed because Ganak's source build otherwise fetches Arjun from its moving
`master` branch.

For the two propositional modes, the runtime lookup order is CLI `--ganak-path`
/ `RuntimeOptions.propositional_ganak_path`, the `GANAK` environment variable,
then `ganak` on `PATH`. Lifted algorithms discover Ganak through `GANAK` or
`PATH` for their pair-factor backend; if Ganak is unavailable or times out,
they automatically use the installed PySDD backend instead.

## Development

To implement and register another solver, see
[Adding a New Algorithm](docs/adding-an-algorithm.md). The guide covers the
reduced and direct-source integration paths, input-template caching, feature
declarations, registration, and required tests.

## References

Please refer to [reference.bib](reference.bib) for the references of the algorithms.
