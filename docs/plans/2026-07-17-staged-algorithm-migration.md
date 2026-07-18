# Staged Algorithm Migration Plan

**Goal:** Move every registered algorithm onto the domain-separated compile,
input-template, and instantiation contract.

**Architecture:** Algorithms compile logical and numeric data once. Cell-graph
algorithms cache structural graphs and rebind arithmetic/profile data per
domain. Propositional algorithms cache their compiled source or reduced branch
but perform grounding per domain. Incremental3 preserves its native counting
quantifiers and selects a counting-state-shaped input template at
instantiation, since that state can genuinely simplify with the domain size.

---

### Task 1: Share staged cell-graph structure selection

- Extract the lifted-profile and CCS open/closed structural variant logic from
  Fast into a small cell-graph input helper.
- Add reusable numeric/component rebinding helpers.
- Keep concrete evidence allocation in input instantiation.

### Task 2: Migrate Standard, Incremental, and Recursive

- Compile domain-free reduced branches and feature sets.
- Build static input templates from structural cell graphs.
- Instantiate arithmetic, profile allocation, order sizes, and decoders per
  domain.
- Remove their compatibility `prepare` paths.

### Task 3: Migrate Incremental3

- Preserve counting sections during staged reductions.
- Build domain-sensitive native counting state before selecting the template.
- Cache templates by immutable counting-state shape plus evidence structure.
- Instantiate concrete numeric weights and profile allocations per domain.

### Task 4: Migrate both Propositional algorithms

- Cache source arithmetic for direct propositional grounding.
- Cache reduced logical/numeric branches for reduction-first grounding.
- Keep CNF grounding and order/evidence clause materialization per domain.
- Preserve direct-grounding and reduction-first semantics.

### Task 5: Migrate the unavailable treewidth extension point

- Use staged hooks while preserving the current instantiation-time
  `UnsupportedFeatureError`.

### Task 6: Verify

- Add compile/instantiate cache tests across multiple domain sizes for every
  migrated algorithm.
- Run focused algorithm, reduction, engine, and propositional tests.
- Run the complete test suite, Ruff, package build, CLI smoke tests, and
  `git diff --check`.
