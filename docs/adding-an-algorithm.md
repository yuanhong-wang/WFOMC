# Adding a New Algorithm

This guide explains how to add a solver to the current domain-separated WFOMC
engine. Read [ADR-0015](adr/0015-algorithm-package-convention.md) for the
package convention and [ADR-0026](adr/0026-domain-separated-compilation.md)
for the compilation/cache design.

## 1. Understand the boundary

The engine owns the common pipeline:

```text
Problem + AlgoName/options
  -> feature analysis and option validation
  -> optional logical reductions
  -> numeric/weight compilation
  -> algorithm-owned InputTemplate (cached)
  + Domain
  -> algorithm-owned AlgoInput
  -> algorithm solve
  -> engine-owned decoding, branch aggregation, and result caching
```

An algorithm owns only:

- its supported features and option policy;
- construction of its reusable input template;
- instantiation of that template for one domain;
- its solver kernel.

It must not import `wfomc.engine` or `wfomc.reduction`. The engine is the only
layer that depends on both. This direction is enforced by
`tests/unit/test_dependency_boundaries.py`.

## 2. Choose an integration path

Choose the reduced path unless the algorithm fundamentally needs the original
source formula.

| Path | `AlgoSpec` setting | Compiled branch | Template base class | Use when |
| --- | --- | --- | --- | --- |
| Reduced/lifted | `uses_reduction=True` (default) | `CompiledReducedBranch` | `ReducedInputTemplate` | The algorithm consumes the normalized and reduced quantifier-free problem. |
| Direct source | `uses_reduction=False` | `GroundingProblem` | `GroundingInputTemplate` | The algorithm must ground or otherwise process the unreduced source formula itself. |
| Configuration variant | Reuse another spec's builder and solver | Same as reused algorithm | Same as reused algorithm | Only a small configuration flag changes, as in `fastv2`. |

For the normal reduced path, the engine instantiates domain-dependent
cardinality bounds, evidence capacities, arithmetic degree limits, and result
decoders before it calls the input template. A solver must not repeat those
steps.

## 3. Create the package

Create `src/wfomc/algo/<name>/` with this layout:

```text
<name>/
  __init__.py   # package docstring only
  input.py      # AlgoInput, InputTemplate, and input construction
  solve.py      # solve(input, context) -> WFOMCResult
  spec.py       # option policy, cache key, and SPEC
```

Add files such as `kernel.py`, `graph.py`, or `operations.py` only when they
own a distinct implementation responsibility. Do not re-export implementation
types from the package `__init__.py`; internal callers should import from their
defining module.

The `standard` package is the smallest complete reduced-path example. The
`propositional` package demonstrates the direct-source path, and `fastv2`
demonstrates a configuration-only variant.

## 4. Define the input and reusable template

Every concrete input extends `AlgoInput` and therefore carries the branch's
`ArithmeticContext`. A reduced-path template extends `ReducedInputTemplate`:

```python
# src/wfomc/algo/example/input.py
from dataclasses import dataclass

from wfomc.algo.core import AlgoInput, ReducedInputTemplate
from wfomc.arithmetic import ArithmeticValue
from wfomc.stages import CompiledBranchInstance, CompiledReducedBranch


@dataclass(frozen=True)
class ExampleInput(AlgoInput):
    values: tuple[ArithmeticValue, ...]
    domain_size: int


@dataclass(frozen=True)
class ExampleInputTemplate(ReducedInputTemplate):
    values: tuple[ArithmeticValue, ...]

    def instantiate(self, concrete: CompiledBranchInstance) -> ExampleInput:
        return ExampleInput(
            arithmetic=concrete.arithmetic,
            values=tuple(
                concrete.arithmetic.coerce(value) for value in self.values
            ),
            domain_size=len(concrete.domain),
        )


def build_input_template(
    compiled: CompiledReducedBranch,
) -> ExampleInputTemplate:
    # Replace this with the expensive, domain-reusable preparation.
    return ExampleInputTemplate(
        tuple(value for pair in compiled.weights.values() for value in pair)
    )
```

Keep expensive domain-independent work in `build_input_template`. Keep
domain-dependent work in `instantiate`. In particular:

- do not store a concrete `Domain` or domain size in a reusable reduced
  template;
- build expensive structural objects such as decomposition trees in the
  template, then reuse their object identity across concrete domains;
- rebind cached numeric values through `concrete.arithmetic`, because degree
  limits can differ by domain size;
- materialize domain-sized tables, bounds, diagnostics, and mutable execution
  state only in `instantiate`;
- create fresh mutable solver caches during instantiation rather than sharing
  them between domain sizes;
- use `ArithmeticContext.zero()`, `one()`, `from_int()`, `coerce()`, and its
  arithmetic methods instead of constructing FLINT values directly.

## 5. Implement the solver

The solver accepts only the algorithm-owned concrete input and an optional
`SolveContext`. It returns the undecoded branch value wrapped in
`WFOMCResult`:

```python
# src/wfomc/algo/example/solve.py
from wfomc.algo.core import SolveContext
from wfomc.result import WFOMCResult

from .input import ExampleInput


def solve(
    algo_input: ExampleInput,
    context: SolveContext | None = None,
) -> WFOMCResult:
    if not isinstance(algo_input, ExampleInput):
        raise TypeError("example algorithm expects an ExampleInput")

    arithmetic = algo_input.arithmetic
    result = arithmetic.zero()
    for value in algo_input.values:
        result = arithmetic.add(result, value)
    return WFOMCResult(result)
```

`SolveContext` is for engine-supplied external dependencies, currently the
explicit Ganak path. Do not pass the engine runtime or caches into an
algorithm. The engine applies reduction decoders, output-symbol projection,
linear-order correction, and branch aggregation after `solve` returns.

Override `AlgoInput.include_order_factorial()` only if the algorithm explicitly
models the order axioms and must not receive the engine's usual order
multiplier. See `GroundCNFInput` for the existing example.

## 6. Declare `SPEC`

`spec.py` adapts the algorithm's strongly typed builder to the small engine
contract:

```python
# src/wfomc/algo/example/spec.py
from collections.abc import Hashable

from wfomc.algo.core import (
    AlgoBranch,
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    option_resolver,
)
from wfomc.stages import CompiledReducedBranch

from .input import ExampleInputTemplate, build_input_template
from .solve import solve


def build_example_input_template(
    branch: AlgoBranch,
    input_key: Hashable,
    _options: AlgoOptions,
) -> ExampleInputTemplate:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Example requires a compiled reduced branch")
    if input_key is not None:
        raise TypeError("Example input does not use structural keys")
    return build_input_template(branch)


SPEC = AlgoSpec(
    name=AlgoName.EXAMPLE,
    resolve_options=option_resolver(
        algo=AlgoName.EXAMPLE,
        default_unary_evidence=EvidenceStrategy.CCS,
        supported_unary_evidence=(EvidenceStrategy.CCS,),
    ),
    solve=solve,
    build_input_template=build_example_input_template,
    maturity=AlgoMaturity.BETA,
)


__all__ = ["SPEC"]
```

Use `option_resolver` to declare only features the implementation actually
handles:

- `supported_unary_evidence` lists the accepted evidence preparations;
- `supports_linear_order` enables `LEQ`;
- `supports_predk_or_circular` enables predecessor/circular predicates;
- `supports_mod_counting` enables modulo counting;
- `supports_binary_evidence` enables ground binary evidence;
- `supported_existential_strategies` is needed only for algorithms that expose
  a configurable existential strategy.

Set `reduce_counting_quantifiers=False` only when the input and solver consume
counting sections natively, as `incremental3` does. Set
`uses_reduction=False` only for a direct-source implementation.

Start a runnable but not yet broadly validated implementation at
`AlgoMaturity.BETA`. `STABLE` and `BETA` algorithms automatically appear in
the CLI; `UNAVAILABLE` algorithms remain registered but hidden.

## 7. Direct-source variation

A direct-source algorithm keeps the same package layout but changes three
parts:

1. Its template extends `GroundingInputTemplate` and implements
   `instantiate(domain: Domain)`.
2. Its spec builder requires `GroundingProblem` instead of
   `CompiledReducedBranch`.
3. Its `AlgoSpec` sets `uses_reduction=False`.

The `GroundingProblem` contains the original `Problem`, source feature set,
compiled weights, and arithmetic context. The algorithm owns all per-domain
grounding or source transformation. The engine uses an identity decoder because
no logical reduction was applied. See `wfomc.algo.propositional.input` and
`wfomc.algo.propositional.spec` for the complete implementation.

Do not select this path merely to bypass an inconvenient reduction. Doing so
makes the algorithm responsible for source quantifiers, evidence, cardinality
constraints, order predicates, and their correctness.

## 8. Add an input-template key only when necessary

If the same template works for every domain size, leave
`AlgoSpec.input_template_key` unset. The engine passes `None` to the builder
and reuses one template.

If the template's structure changes with the domain, define:

```python
def input_template_key(
    branch: AlgoBranch,
    domain: Domain,
) -> Hashable:
    ...
```

Return the smallest immutable structural discriminator. Do not return the
whole `Domain` or `domain.size` merely because arithmetic values change with
`n`; those belong in `InputTemplate.instantiate`. Existing legitimate keys
include the set of active evidence profiles and incremental3's counting-state
shape.

The engine caches templates by compiled problem, branch index, and this key.
An unnecessarily broad key silently defeats cross-domain reuse.

An algorithm may expose a reference domain size for selecting one reusable
structure. Treat that reference as an `AlgoOptions` compilation setting, not as
the actual `input_template_key`: the selected structure still belongs to the
domain-free template and must be reused by all concrete domains mapped to the
same structural variant. Boundary-Profile's cached tree is the reference
implementation of this pattern.

## 9. Register the algorithm

Registration has two required edits in `src/wfomc/algo/core.py`:

1. Add a documented member to `AlgoName`.
2. Map it to `wfomc.algo.<name>.spec` in `_SPEC_MODULES`.

The spec module must expose exactly one `SPEC`, and `SPEC.name` must match the
registry key. No engine edit or package-level forwarding function is needed.

Also add the user-facing name and support description to the algorithm list in
the root `README.md`. Add CLI-specific options only if existing `AlgoOptions`
cannot express the algorithm. If a new option affects compilation or template
selection, include it in the engine cache key and test that different option
values do not reuse incompatible artifacts.

## 10. Test the integration

At minimum, add tests for:

1. `algo_spec(AlgoName.EXAMPLE)` returns the registered spec and expected
   maturity.
2. Unsupported features fail with `UnsupportedFeatureError`; they must never
   be silently ignored.
3. The input builder returns the correct nominal template type.
4. One compiled problem can instantiate multiple domain sizes, and the
   `algo_input_templates` cache is hit when the structural key is unchanged.
5. The solver matches a trusted algorithm on small representative models.
6. Rounded and symbolic arithmetic either work end to end or fail early with
   `ArithmeticBackendError`.
7. A `STABLE` or `BETA` algorithm appears in `wfomc --help`.

Add the algorithm to the appropriate model families in
`tests/integration/test_algorithm_consistency.py` only after its supported
feature set is correct.

Run:

```bash
uv run pytest -q tests/unit/test_engine_compile.py \
  tests/unit/test_dependency_boundaries.py tests/unit/test_cli.py
uv run pytest -q tests/integration/test_algorithm_consistency.py
uv run pytest -q
```

## Final checklist

- [ ] Package follows `input.py` / `solve.py` / `spec.py` ownership.
- [ ] Algorithm imports neither `wfomc.engine` nor `wfomc.reduction`.
- [ ] `SPEC` declares an honest feature and maturity policy.
- [ ] Reusable templates contain no accidental concrete-domain state.
- [ ] `input_template_key` contains only structural domain dependence.
- [ ] Numeric construction goes through `ArithmeticContext`.
- [ ] Solver returns `WFOMCResult` and leaves decoding to the engine.
- [ ] `AlgoName`, `_SPEC_MODULES`, README, and tests are updated.
- [ ] Cross-domain cache reuse and full regression tests pass.
