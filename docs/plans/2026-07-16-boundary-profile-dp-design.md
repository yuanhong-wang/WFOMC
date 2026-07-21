# Boundary-Profile DP Design

**Goal:** Implement the Boundary-Profile decomposition dynamic program from
`reorder_wfomc/draft.tex` as a native WFOMC algorithm, including decomposition
tree search and the structural/hot-loop optimizations used by
`tail_signature_opt27.py`.

## Requirements

### Functional

- Materialize the cell-level master sum as local tables `W[i][n]`, a symmetric
  interaction matrix `R[i][j]`, and a nullary-branch graph weight.
- Build and search binary Boundary-Profile decomposition trees rather than
  requiring a user-supplied tree.
- Evaluate every selected tree exactly for scalar and symbolic arithmetic.
- Preserve cardinality-constraint and CCS-evidence semantics through the
  framework's existing reductions and result decoders.
- Reuse structural cases from FastWFOMC and opt27: independent cells,
  symmetric cliques, disconnected interaction components, tail-signature
  caterpillars, balanced repeated composition, and cached interaction powers.
- Remain a separate algorithm so its correctness and performance can be
  compared with `fastv2`, `incremental3`, and `tail-signature`.

### Non-functional

- All numeric construction and operations go through `ArithmeticContext`.
- Planning is deterministic and exposes width/cost diagnostics.
- The solver frees child tables after their parent is evaluated and avoids
  unbounded caches of polynomial values.
- A planner estimate must reject or avoid candidates whose predicted state or
  join space is materially worse than another available candidate.
- Correctness is established against direct master-sum enumeration and the
  existing algorithms on scalar, cardinality, evidence, and backend fixtures.

## Normalized Recurrence

The paper stores

```text
F_t(c) = sum_n prod_i W_i(n_i) / n_i! * internal_interactions(n).
```

The implementation stores the factorial-scaled message

```text
G_t(c) = |c|! F_t(c).
```

For a leaf `i`, `G_i[(n,)] = W_i[n]`. For children `a` and `b`,

```text
G_t[c] += choose(|a| + |b|, |a|)
          * G_a[a] * G_b[b] * cross_kernel(a, b)
```

whenever the two projected child states add to `c`. The root answer is
`G_root[(N,)]`. This form uses only semiring addition, multiplication, powers,
and integer coefficients; it avoids introducing factorial denominators into
FLINT values.

## Architecture

```text
Problem -> reductions/compiler -> CellGraphData -> MasterSumComponent
                                                |-> BP planner -> BPPlan
                                                `-> BP kernel -> WFOMCResult
```

- `wfomc.algo.master_sum` owns the representation shared by native
  Boundary-Profile and the external tail-signature adapter.
- `wfomc.algo.boundary_profile.input` materializes algorithm input and plans.
- `plan.py` interns unhashable arithmetic interaction values into integer
  labels, constructs boundary classes/projections, generates candidate trees,
  estimates their work, and selects the best candidate.
- `kernel.py` evaluates a plan in postorder with total-cardinality buckets.
- `solve.py` combines nullary branches in the same way as existing cell-graph
  algorithms.

## Decomposition Search

The planner creates several candidates and compares a domain-size-aware cost:

1. a tail-signature caterpillar using signature-class lookahead ordering;
2. a greedy agglomerative tree minimizing estimated child-state pairs and
   parent boundary width;
3. a structure-aware tree that first exposes non-neutral connected components,
   symmetric cliques, and an independent-cell remainder;
4. balanced joins for homogeneous/repeated blocks when their recursive
   signatures agree.

For a node with `d` boundary classes, the state upper bound is
`choose(N + d, d)`. A join is scored from the two child bounds, the number of
non-unit cross-class interactions, parent width, and cumulative subtree cost.
The selected plan records its source, BP width, join width, estimated state
count, and estimated pair count.

Independent cells use an opt27/FastWFOMC-style weight-aware closing rule when
their internal interactions are one and every local table is exponential.
Otherwise they remain ordinary BP leaves, preserving exactness for symbolic
or evidence-derived local tables. Disconnected joins specialize only the
cross kernel to one; they still apply the general child-to-parent projection,
because boundary classes may merge at the parent.

## Kernel Optimizations

- Group states by total cardinality.
- Use a dense `N+1` vector for one-dimensional tables and sparse maps for wider
  tables initially.
- Compile child-state projections once per node.
- Store only non-unit cross-class interactions.
- For each fixed left state, compute one interaction step per right boundary
  class and build transient powers `step^0..step^N`; discard them before the
  next left state.
- Skip zero child entries and zero cross kernels.
- Use `ArithmeticContext.add_product`, truncation, zero/one fast paths, and
  solve-scoped structural caches.
- Release child tables after their unique parent has been computed.
- Reuse recursively identical subtree tables and join skeletons only after the
  base tree implementation is verified.

## Paper Corrections Enforced by the Implementation

- The exponentially-neutral condition is `W_i(n) = theta_i^n`; writing
  `theta_i^n / n!` double-counts the factorial in the paper's `F_t` definition.
- A disconnected join has cross kernel one, but its child boundary classes may
  still merge at the parent, so projection is not generally the identity.
- Articulation blocks overlap at the separator while BP-tree children are
  disjoint. Articulation acceleration therefore requires a separately proved
  separator state or the disjoint `(A - {v}), (B - {v}), {v}` construction; it
  is not assumed by the base join recurrence.

## Failure Modes and Mitigations

- **Planner explosion:** cache profiles by cell bit mask, bound beam/exact
  searches, and always retain deterministic greedy/caterpillar candidates.
- **Unhashable polynomial/ball weights:** use interned integer interaction
  labels; arithmetic equality is only used while interning.
- **Large tables:** estimate before evaluation, group by total, use the
  one-dimensional dense specialization, and release children eagerly.
- **Approximate equality misses a merge:** this only widens the table; the
  planner never merges values unless equality is positively established.
- **Global constraints couple blocks:** cardinality and CCS markers remain in
  arithmetic values, so joins do not factor them out or inspect their syntax.

## Verification

- Unit-test boundary classes, no-split projection, cross-label uniformity,
  candidate determinism, and cost selection.
- Compare the kernel with direct master-sum enumeration for random small
  symmetric matrices and local tables.
- Compare API results with `standard`, `fastv2`, and `incremental3` on existing
  scalar, cardinality, and CCS evidence cases.
- Exercise exact scalar, `fmpq_poly`, default `fmpq_mpoly`, float, and Arb
  backends where supported.
- Run the repository suite and a 30-second/4-GiB benchmark over representative
  slow families, recording selected plan, widths, states, joins, time, and RSS.

## Decision Record

**Decision:** Add a separate native `boundary-profile` algorithm, share only
the master-sum materialization with tail-signature, and use factorial-scaled
messages.

**Consequences:** The implementation fits existing algorithm ownership and
decoder boundaries, can consume symbolic markers without new constraint state,
and can be benchmarked independently. It adds a nontrivial planner and a new
table kernel; automatic selection between lifted algorithms remains out of
scope until comparative data exists.

**Alternatives considered:** Replacing fastv2 would discard its mature closed
forms; embedding BP in incremental3 would mix incompatible state semantics;
turning the external tail-signature adapter into a multi-mode engine would keep
core correctness and arithmetic outside the framework.
