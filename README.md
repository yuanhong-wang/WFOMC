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
$ uv run wfomc -i [input] -a [algo] -e [unary_evidence_encoding] [-l [linear_order_encoding]]
```
where
- `input` is the input file with the suffix `.wfomcs` or `.mln`
- `algo` is the algorithm to use, including:
  - `standard`: the standard WFOMC algorithm in Beame et al. (2015)
  - `fast`: the fast WFOMC algorithm in Timothy van Bremen and Ondrej Kuzelka (2021)
  - `fastv2` (default): the optimized fast WFOMC algorithm
  - `incremental`: the incremental WFOMC algorithm for linear order axiom in Toth and Kuzelka (2022)
  - `recursive`: the recursive WFOMC algorithm for linear order axiom in Meng et al. (2024)
  - `propositional`: ground the (Skolemized) sentence over the domain and count with the external [ganak](https://github.com/meelgroup/ganak) propositional model counter. Intended as a ground-truth baseline. Requires a ganak binary; see *Propositional counter* below.
- `unary_evidence_encoding` is the encoding for unary evidence, including:
  - `ccs` (default): using cardinality constraints to encode unary evidence, see Wang et al. (2024)
  - `pc`: **only work for the algorithms `fast`, `fastv2` and `incremental`**
- `linear_order_encoding` (only used by `-a propositional`) controls how the order axioms (`LEQ` / `PRED` / `CIRCULAR_PRED`) are encoded:
  - `pin` (default): pin every ground order atom to a canonical sorted order/cycle; `decode_result` applies the `n!` multiplier. Cheap, fast.
  - `axioms`: emit the FO³ axioms / definitions explicitly. Pin-free but materially slower. See *Propositional counter* below for the trade-off.

### Propositional counter

The `propositional` algorithm grounds the universally quantified (Skolemized) sentence over every pair of domain elements and hands the resulting weighted CNF to [ganak](https://github.com/meelgroup/ganak). It is intended as a textbook-definition ground-truth baseline against which the lifted algorithms can be checked.

It supports plain FO² with rational weights, cardinality constraints and counting quantifiers (`∃_{=k}`), unary evidence with the default `ccs` encoding, and three order axioms: `LEQ` (linear order), `PRED` (= `PRED1`, the immediate linear predecessor), and `CIRCULAR_PRED` (the immediate circular predecessor). The `PREDk` family for `k > 1` and the partition-constraint (`pc`) unary-evidence encoding raise an error.

The encoding of the order axioms is selected per call via the `--linear-order-encoding` (short `-l`) CLI flag, the `linear_order_encoding=` keyword argument on `wfomc()` / `propositional_wfomc()`, or, as a fallback for both, the module-level constant `LINEAR_ORDER_ENCODING` in [src/wfomc/algo/PropositionalWFOMC.py](src/wfomc/algo/PropositionalWFOMC.py):

| Setting | Mechanism | Multiplier | Cost | When to use |
|---|---|---|---|---|
| `"pin"` (default) | Each ground `LEQ` / `PRED1` / `CIRCULAR_PRED` atom is pinned to its value under a canonical sorted order / cycle on the domain. | `× n!` via `decode_result`. | `O(n²)` unit clauses, no aux. Fast. | Default; the cheap textbook trick. |
| `"axioms"` | `LEQ` is axiomatized as a total order (FO³ transitivity); `PRED1` is defined via the FO³ "immediately below" pattern with Tseitin auxiliaries; `CIRCULAR_PRED` is defined as `PRED1 ∨ (LEQ-min ∧ LEQ-max)`. | None — axioms already range `LEQ` over all `n!` orders. | `O(n³)` clauses + Tseitin aux. Materially slower under ganak `--mode 3` (polynomial weighting). | When you want a pin-free, FO-axiomatic baseline. |

Both options give the same answers (the pin trick exploits domain symmetry: model count is invariant under permutations of the domain, so any fixed pinning multiplied by `n!` recovers the total). Slow cases for the `"axioms"` setting — `predecessor.wfomcs` and the `MATH/*` `CIRCULAR_PRED` problems — are gated behind `WFOMC_RUN_SLOW=1` in the test suite; under the default `"pin"` they run in seconds and are tested unconditionally.

**Unary evidence handling.** Direct unit-clause evidence (`UnaryEvidenceEncoding.NONE`) is element-specific and therefore breaks pin-and-multiply's symmetry argument when an order axiom is present. The solver automatically picks the right encoding for the propositional algorithm:

| `LINEAR_ORDER_ENCODING` | Has order axiom? | Evidence encoding | ganak mode |
|---|---|---|---|
| `"pin"` | yes | `CCS` (symmetric fingerprint counts) | `--mode 3` (polynomial) |
| `"pin"` | no | `NONE` (direct unit clauses) | `--mode 1` |
| `"axioms"` | any | `NONE` (direct unit clauses) | `--mode 1` |

Skipping CCS in the no-order and axioms cases avoids a needless trip through ganak's polynomial mode; on the test suite this gives a 2–4× speed-up on `unary_evidence/*` (e.g. `unary_evidence/employment.mln`: 204 ms → 47 ms).

ganak is invoked in two modes:

- exact rational weighted counting (`--mode 1`) when no symbolic/polynomial weights are involved,
- multivariate-polynomial weighted counting (`--mode 3`) when cardinality constraints or counting quantifiers introduce symbolic weights.

The polynomial mode only emits its result on the **`devel` branch** of ganak — the released v2.6.1 binary does not print mode-3 output. Build ganak from `devel` and point the integration at the binary in one of the following ways (search order):

- `--ganak-path` argument when calling `propositional_wfomc()` programmatically;
- the `GANAK` environment variable;
- `ganak` on `PATH`.

Build instructions (macOS / Linux, requires `cmake`, `gmp`, `mpfr`, `flint`):

```
git clone --recurse-submodules -b devel https://github.com/meelgroup/ganak
cd ganak && mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON .. && make -j ganak-bin
export GANAK=$(pwd)/ganak
```

Then run:

```
uv run wfomc -i models/2-regular-graph.wfomcs -a propositional
```

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

> **Note: To use linear order constraint, you must use the `incremental` or `recursive` algorithm. To use $k$-th predecessor or circular predecessor, you must use the `incremental` algorithm.**


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

## References

Please refer to [reference.bib](reference.bib) for the references of the algorithms.
