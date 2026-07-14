# 当前 C2 Normal Form 的形态与边界

## 当前形态

`C2NormalForm` 是算法无关、sectioned、typed 的 C2 中间表示：

```text
C2NormalForm
├── qf_formula: Formula | None
├── forall_exists: tuple[Formula, ...]
├── exists: tuple[Formula, ...]
├── forall_counts: tuple[ForallCountSection, ...]
├── counts: tuple[CountSection, ...]
├── count_definitions: tuple[CountDefinition, ...]
└── requires_nonempty_domain: bool
```

各字段的语义是：

- `qf_formula` 是采用隐式全称闭包的无量词公式；
- `forall_exists` 保存直接的 `∀x∃y body`；
- `exists` 保存直接的 `∃x body`；
- `forall_counts` 保存顶层、无条件的 row count；
- `counts` 保存顶层、无条件的 global count；
- `count_definitions` 保存嵌入布尔上下文的 count marker 及其唯一 section；
- `requires_nonempty_domain` 标记 Scott abstraction 对非空域的要求。

例如：

```text
∀X: P(X) ∨ (∃_=2 Y: R(X,Y))
```

会表示为：

```text
qf_formula = P(X) ∨ @count(X)
count_definitions = (
    @count(X) ↔ ForallCountSection(=, 2, R(X,Y), X, Y),
)
forall_counts = ()
```

这里的 count 是有条件的定义，不会同时出现在 `forall_counts` 中。这样可以避免同一个 section 既被当成直接约束、又被当成 marker 定义。

## 已保证的不变量

- source 的每个 subformula 最多使用两个变量；
- solver vocabulary 只接受 unary/binary predicate；
- `qf_formula` 不包含 quantifier，并采用 implicit universal closure；
- `forall_exists`、`exists` 的 quantifier shape 固定；
- count body 必须是 typed `Atom`；
- `CountSection.counted_var` 必填；
- `ForallCountSection.outer_var` 和 `counted_var` 必填；
- count body 的 term 顺序必须与 section variables 一致；
- global definition 使用 nullary marker，row definition 使用 outer-variable marker；
- fresh predicate/variable 避开 source 和 `Problem` 已声明名称；
- normalize 输入必须是 closed sentence；
- `Not`、`Implies`、`Iff` 等 Boolean structure 不会为进入 NNF 而展开；
- modulo direct global/row count 可以进入 typed section；
- unsupported downstream capability 会 fail fast。

## 已移除的第二事实来源

- `universal` 已替换为真正的 `qf_formula`，不再二次解析 universal prefix；
- `_extract_universal_sections()` 已删除；
- `PredicateDefinition` 已删除，fresh predicate equivalence 只写入 `qf_formula`；
- `CountSection` / `ForallCountSection` 不再保存 `source` 或 `marker`；
- `CountDefinition` 是嵌入式 count section 的唯一 owner；
- `to_sentence()` 和无损 round-trip 的假象已删除；
- trivial accessor wrappers 已删除，算法直接读取显式字段。

## 仍然存在的能力边界

### Valid 不等于 runnable

IR 可以保留 `= != < <= > >= mod` 和 embedded count definitions，具体算法只消费其支持的子集：

- UFO2/cardinality reduction 不支持 embedded count definition；
- 当前 row-count reduction 主要支持 exact count；
- incremental3 支持直接的 `=`, `<=`, `mod`，但暂不支持 embedded count definition；
- negated modulo count 表示为 `¬marker` 加保留正向 modulo section 的 `CountDefinition`。

这些限制属于算法契约，不应通过丢失 normal form 信息来掩盖。不同算法可以在后续 reduction 中选择不同的 lower 方式。

### 仍是混合 IR

ordinary existential sections 仍保存完整 `Formula`，count 使用 dataclass section。这是有意保留的折中：当前字段边界清楚，继续为 `forall_exists` / `exists` 新建类型的收益不足以抵消额外封装。

### 不是完整、canonical 的一般 C2

- predicate arity ≤ 2 是当前 WFOMC cell-graph solver fragment 的约束；
- 等价公式可能因 conjunction 顺序和 fresh-definition traversal 顺序得到不同 IR；
- Scott abstraction 仍要求非空域。

因此它是 validated、algorithm-independent 的 solver C2 IR，不是数学意义上的唯一 canonical form。

## 下一步

最重要的后续工作是为需要支持嵌入式 count 的算法实现明确的 `CountDefinition` lowering。comparator 是否展开为 exact count，也应由目标算法的 reduction 决定，而不是继续扩大 `C2NormalForm` 或在 normalize 阶段丢失原始结构。
