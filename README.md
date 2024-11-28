# Exact Lifted Counter for Two-Variable Logic

This tool is for counting the models (or combinatorical structures) from the two-variable fragment of first-order logic.


## Input format

The input file with the suffix `.wfomcs` contains the following information **in order**:
1. First-order sentence with at most two logic variables, see [fol_grammar.py](sampling_fo2/parser/fol_grammar.py) for details, e.g.,
  * `\forall X: (\forall Y: (R(X, Y) <-> Z(X, Y)))`
  * `\forall X: (\exists Y: (R(X, Y)))`
  * `\exists X: (F(X) -> \forall Y: (R(X, Y)))`
  * ..., even more complex sentence...
2. Domain: 
  * `domain=3` or
  * `domain={p1, p2, p3}`
3. Weighting (optional): `positve_weight negative_weight predicate`
4. Cardinality constraint (optional): 
  * `|P| = k`
  * `|P| > k`
  * `|P| >= k`
  * `|P| < k`
  * `|P| <= k`

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
\forall X: (\existes Y: (fr(X,Y))).
# or 
\exists Y: (fr(X,Y)).

person = 10
```

More examples are in [models](models/)


### Installation
Install the package:
```
$ pip install -e .
```


### How to use
```
$ python sampling_fo2/wfomc.py -i [input] -a [algo]
```

## References


```
@inproceedings{DBLP:conf/uai/BremenK21,
  author       = {Timothy van Bremen and
                  Ondrej Kuzelka},
  editor       = {Cassio P. de Campos and
                  Marloes H. Maathuis and
                  Erik Quaeghebeur},
  title        = {Faster lifting for two-variable logic using cell graphs},
  booktitle    = {Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial
                  Intelligence, {UAI} 2021, Virtual Event, 27-30 July 2021},
  series       = {Proceedings of Machine Learning Research},
  volume       = {161},
  pages        = {1393--1402},
  publisher    = {{AUAI} Press},
  year         = {2021},
  url          = {https://proceedings.mlr.press/v161/bremen21a.html},
  timestamp    = {Fri, 17 Dec 2021 17:06:27 +0100},
  biburl       = {https://dblp.org/rec/conf/uai/BremenK21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```
@article{kuzelka_weighted_2021,
  title = {Weighted First-Order Model Counting in the Two-Variable Fragment with Counting Quantifiers},
  author = {Kuzelka, Ondrej},
  year = {2021},
  month = mar,
  journal = {Journal of Artificial Intelligence Research},
  volume = {70},
  eprint = {2007.05619},
  pages = {1281--1307},
  issn = {1076-9757},
  doi = {10.1613/jair.1.12320}
}
```
```
@article{toth_lifted_2022-1,
  title = {Lifted Inference with Linear Order Axiom},
  author = {T{\'o}th, Jan and Ku{\v z}elka, Ond{\v r}ej},
  year = {2022},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {37},
  number = {10},
  pages = {12295--12304},
  doi = {10.1609/aaai.v37i10.26449}
}
```
```
@incollection{endriss_more_2024,
  title = {A More Practical Algorithm for Weighted First-Order Model Counting with Linear Order Axiom},
  booktitle = {Frontiers in {{Artificial Intelligence}} and {{Applications}}},
  author = {Meng, Qiaolan and T{\'o}th, Jan and Wang, Yuanhong and Wang, Yuyi and Ku{\v z}elka, Ond{\v r}ej},
  editor = {Endriss, Ulle and Melo, Francisco S. and Bach, Kerstin and {Bugar{\'i}n-Diz}, Alberto and {Alonso-Moral}, Jos{\'e} M. and Barro, Sen{\'e}n and Heintz, Fredrik},
  year = {2024},
  month = oct,
  publisher = {IOS Press},
  isbn = {978-1-64368-548-9},
  keywords = {linter/error}
}
```
