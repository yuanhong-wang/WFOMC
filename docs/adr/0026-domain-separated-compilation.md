# ADR-0026: Domain-Separated Compilation

## Status

Accepted

## Context

`Problem` currently owns both the logical query and a concrete domain. As a
result, `compile_problem` normalizes, reduces, compiles weights, builds cell
graphs, and allocates solver caches again for every domain size. The resulting
`CompiledProblem` also contains a domain and concrete arithmetic degree limits,
so its name does not describe a reusable compiled artifact.

This prevents the intended workload of compiling one logical problem and
evaluating it for several domain sizes. It also leaves ownership unclear:
reduction, numeric compilation, algorithm input construction, and per-domain
state are mixed between algorithm specs and engine orchestration.

## Decision

The public source model is split into two immutable values:

- `Problem` contains the formula, weights, constraints, and evidence. It never
  contains a domain size, domain elements, file provenance, or algorithm
  options.
- `Domain` contains the concrete domain elements and optional circular-order
  size.
- `ProblemInstance` is a parser/convenience value with explicit `.problem`,
  `.domain`, and optional `.source_path` fields. Source path is diagnostic
  provenance and does not participate in `Problem` equality or engine cache
  keys. The instance is not a problem stage used by the engine.

The engine exposes three separate operations:

- `compile_problem(problem, algo, options)` returns a reusable, domain-free
  `CompiledProblem`;
- `instantiate_problem(compiled, domain)` creates a `ProblemExecution` for one
  concrete domain;
- `solve(compiled, domain)` evaluates that execution. The convenience form
  `solve(problem, domain, algo, options)` compiles and instantiates internally.

`CompiledProblem` records the selected algorithm, resolved options, analyzed
features, and domain-free compiled branches. `ProblemExecution` owns a tuple of
engine-level `ExecutionBranch` values. Each execution branch pairs the logical
problem stage with one concrete algorithm input and its decoder.

The engine owns lifecycle and caching:

1. analyze the domain-free source problem;
2. resolve the algorithm spec and options;
3. apply declared reductions;
4. compile branch weights and formulas;
5. ask the algorithm to build a domain-free input template;
6. instantiate reductions, arithmetic bounds, decoders, and algorithm input
   for a concrete domain;
7. run the algorithm, decode branches, and aggregate the result.

Every registered algorithm implements the staged interface. `AlgoSpec` only
declares option resolution, input-template construction, an optional structural
template key, solving, and the two reduction-policy flags. The engine, rather
than each algorithm spec, applies reductions, compiles numeric branches, checks
domain applicability, instantiates decoders, and assembles execution branches.
Input-template dataclasses own their concrete `.instantiate(...)` operation;
there is no separate `instantiate_input` callback in `AlgoSpec`.
The method is enforced through two nominal algorithm contracts:
`ReducedInputTemplate` accepts a `CompiledBranchInstance`, while
`GroundingInputTemplate` accepts a `Domain`. `AlgoSpec.build_input_template`
returns their explicit union; engine orchestration rejects a template whose
nominal kind does not match its compiled branch. Runtime `getattr`/`callable`
probing is not part of this boundary.

Standard, Fast,
FastV2, Incremental, Incremental3, and Recursive cache structural cell-graph
input templates. Domain size is introduced when arithmetic bounds, evidence
capacities, order sizes, and solver-local caches are instantiated.
Independent-clique classification is part of Fast's static layout and does not
inspect a range of sizes.

Incremental3 keeps native counting sections in its reduced problem. Its
domain-simplified counting automaton is part of the input-template key:
domains that select the same automaton reuse a graph, while a genuine automaton
shape change creates another template.

The two propositional algorithms also use staged compilation. Direct grounding
caches the source formula and compiled weights; reduction-first grounding
caches its reduced numeric branches. Both necessarily construct a fresh ground
CNF for each concrete domain.

Logical reductions used by staged algorithms operate on
`ReducedProblem`. Domain-dependent integers are represented by small
`DomainExpr` values and evaluated only during instantiation. Reduction result
corrections are stored as `DecoderSpec` data rather than closures over a
concrete domain.

Unary-evidence reduction produces one reduced profile constraint with a
parametric empty profile of size `n - observed_count`. A zero-sized lifted
profile is omitted when the reduced problem is instantiated. CCS leaves
unmarked elements as the implicit empty profile. Fast and FastV2 choose the
corresponding closed or open cell-graph shape while instantiating their input,
so the input-template cache may hold both shapes without turning this
Fast-specific choice into logical reduction branches.

The runtime cache is split by lifetime:

- feature and compiled-problem caches are keyed only by source problem,
  algorithm, and resolved options;
- algorithm input templates are keyed by compiled branch and static variant;
- executions and results are keyed additionally by the full `Domain`;
- input-template, concrete-execution, and result buckets have independent
  configurable bounds. Evicted operation objects own their recursive caches
  and are therefore reclaimable.

Compiler utilities and execution artifacts live in `wfomc.engine`. Algorithm
packages own only option/capability declarations, input templates, concrete
algorithm inputs, and solvers. Reduction packages own only pure logical
transformations.

Cross-layer data contracts are lower than all three packages:

- `wfomc.options` owns `EvidenceStrategy` and `ExistentialStrategy`;
- `wfomc.problem` owns only source/domain values: `Problem`, `Domain`, and
  `ProblemInstance`;
- `wfomc.stages` owns neutral derived-stage data: `FeatureSet`,
  `ReducedProblem`, decoder specifications, `GroundingProblem`, compiled
  reduced branches, and concrete branch instances;
- engine-owned runtime types may refer to algorithm inputs, but neutral problem
  data never refers to engine, algorithms, or reductions.

The enforced dependency direction is:

```text
algo      -> problem/stages/options
reduction -> problem/stages/options
engine    -> algo + reduction + problem/stages/options
```

Imports used only under `TYPE_CHECKING` follow the same rule. A static AST test
rejects reverse imports instead of allowing type hints to create hidden cycles.
The engine converts `RuntimeContext` into the smaller algorithm-owned
`SolveContext`, so solver signatures do not import engine runtime types.

## Consequences

- One `CompiledProblem` can be evaluated for multiple domain sizes.
- Cache keys no longer have an ambiguous `include_domain_size` switch.
- A compiled artifact cannot accidentally retain one concrete domain.
- Lifted algorithms reuse normalization, reduction, weight compilation, and
  compatible cell-graph structures across domain sizes.
- Each Fast/FastV2 execution receives fresh mutable term caches, so reuse does
  not leak results between sizes.
- Propositional algorithms reuse logical and numeric compilation while keeping
  grounding explicitly domain-sized.
- Algorithm and reduction packages can be imported and type-checked without
  importing the engine or each other.
- Parser callers must explicitly distinguish the parsed logical problem from
  its domain.
- `parse_problem_file` is the unambiguous file entry point; the legacy
  `parse_input` alias is not part of the current API.
