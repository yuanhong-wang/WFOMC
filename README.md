# Exact Lifted Counter for Two-Variable Logic

This tool is for counting the models (or combinatorical structures) from the two-variable fragment of first-order logic.


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
$ uv run wfomc -i [input] -a [algo] -e [unary_evidence_encoding]
```
where
- `input` is the input file with the suffix `.wfomcs` or `.mln`
- `algo` is the algorithm to use, including:
  - `standard`: the standard WFOMC algorithm in Beame et al. (2015)
  - `fast`: the fast WFOMC algorithm in Timothy van Bremen and Ondrej Kuzelka (2021)
  - `fastv2` (default): the optimized fast WFOMC algorithm
  - `incremental`: the incremental WFOMC algorithm for linear order axiom in Toth and Kuzelka (2022)
  - `recursive`: the recursive WFOMC algorithm for linear order axiom in Meng et al. (2024)
- `unary_evidence_encoding` is the encoding for unary evidence, including:
  - `ccs` (default): using cardinality constraints to encode unary evidence, see Wang et al. (2024)
  - `pc`: **only work for `fast` and `fastv2`**

## References

Please refer to [reference.bib](reference.bib) for the references of the algorithms.