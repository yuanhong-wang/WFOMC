# C2 Normal Form 与 Kuželka 2021 Sections 6.2/7 的对齐分析

参考论文：Kuželka, *Weighted First-Order Model Counting in the Two-Variable Fragment With Counting Quantifiers*，Section 6.2 与 Section 7（PDF pages 14-17）。

## 论文实际建立的 pipeline

论文没有定义一个长期保留所有 counting syntax 的“C2 normal-form object”。它证明的是一条 reduction pipeline。

### Section 6.2：任意位置的 exact count

对于任意子公式：

```text
∃=k y ψ(x,y)
```

重复执行：

```text
∃=k y ψ(x,y)  ↦  Aψ(x)
```

并加入 definition：

```text
∀x∀y: Bψ(x,y) ↔ ψ(x,y)
Aψ(x) ↔ ∃=k y Bψ(x,y)
```

其中 `Aψ` 是 fresh unary predicate，`Bψ` 是 fresh binary predicate。

关键点是第二条 equivalence 不是最终 artifact。Section 6.2 继续用 Lemma 4 和 Proposition 3 把它降低成已经可处理的形式：

- 普通 FO² clauses；
- `∀x∃=k y` constraints；
- cardinality constraints；
- fresh predicates；
- extension-count correction factor；
- 必要时使用负 weight 的辅助 predicate 消除 negated exact count。

因此论文中的“definition”是 reduction obligation，不是留给 solver 猜测的 metadata。

### Section 7：一般 counting comparator 先化为 exact count

论文把一般 counting 归约到 `∃=k`：

```text
∃≤k y ψ(x,y)
```

改写为：

```text
∀y ¬ψ(x,y)
∨ ∃=1 y ψ(x,y)
∨ ...
∨ ∃=k y ψ(x,y)
```

而：

```text
∃≥k y ψ(x,y)
```

改写为：

```text
¬(∃≤k-1 y ψ(x,y))
```

然后交给 Section 6.2 的 arbitrary-context exact-count reduction处理。

论文核心 IR 因而只需要 exact count。`≤`、`≥` 以及由否定产生的其他 comparator 是 normalization 输入语法，不需要长期存在于核心 count section。

## 与当前实现的对应关系

### 已对齐

- `PredicateDefinition(atom=Bψ, body=ψ)` 对应 `Bψ(x,y) ↔ ψ(x,y)`；
- `CountDefinition(marker=Aψ, section=...)` 对应 `Aψ(x) ↔ ∃=k y Bψ(x,y)`；
- global exact count 转 cardinality constraint 与论文思路一致；
- `reduce_exact_row_count()` 用多个互斥辅助 predicates、cardinality 和 factorial correction 处理 `∀x∃=k y`，与 Section 6.1/6.2 所依赖的基础 reduction 同方向；
- fresh predicate 必须与原 vocabulary 分离，当前 used-name allocator 已满足这一前提；
- ArithmeticContext 能表示负 rational weight，具备实现 Proposition 3 negative-weight gadget 的数值基础。

### 未对齐

#### 1. CountDefinition 目前是终止点，不是可执行 reduction obligation

normalizer 能生成 embedded count 的 `CountDefinition`，但 standard/UFO2 reduction 拒绝它，incremental3 也 fail fast。

这意味着当前实现只完成了 Section 6.2 最后一个自然段的“replace subformula + add definition”，没有实现前面 Lemma 4/Proposition 3 所需的 definition lowering。因此目前不能声称支持 arbitrary-context C2 counting。

#### 2. 核心 CountSection 保留了过多 comparator

当前 IR 接受：

```text
= != < <= > >= mod
```

但 Section 7 的理论 normalizer 应先把 ordinary inequalities 化为 exact-count boolean combinations。保留所有 comparator 把 source syntax、算法优化和理论核心 IR 混在了一层。

#### 3. Negated exact/mod count 的处理不完整

论文使用 Proposition 3 的 fresh `C,D` predicates 和负 weight gadget 处理 exact-count subformula 的否定。

算法并不都支持 `!=` 等 comparator。Modulo 不属于论文 Sections 6.2/7，应作为单独扩展设计，不能用论文结论为其完整性背书。当前 IR 对 negated modulo 不再构造 comparator dual，而是保留正向 modulo `CountDefinition`，并在 QF body 中否定 marker；是否执行该 definition 仍由算法 reduction 决定。

#### 4. PredicateDefinition 同时是 metadata 和 universal clause

当前 normalizer 已把 `Bψ ↔ ψ` 写进 universal，同时保留 `PredicateDefinition`。论文只需要实际 definition sentence；双重表示不是理论要求。

#### 5. 当前 IR 混入了算法专用扩展

`<=` mask、mod section、`requires_nonempty_domain` 等是实现策略或扩展能力，不是论文 exact-count reduction 的核心 normal form。

## 更合适的核心边界

结合论文，建议将 pipeline 明确成：

```text
Raw C2 formula with counting
    │
    ├─ comparator normalization (Section 7)
    │      ≤ / ≥ / < / > / != → boolean combinations of =k
    │
    ├─ exact-count flattening (Section 6.2)
    │      ψ → Bψ, count-subformula → Aψ
    │
    ├─ count-definition lowering (Lemma 4 + Proposition 3)
    │      eliminate Aψ ↔ ∃=k Bψ
    │
    ▼
Reduced C2 counting theory
    ├── qf_universal_body
    ├── forall_exists
    ├── exact_global_counts
    ├── exact_row_counts
    ├── cardinality_constraints
    ├── auxiliary_weights
    └── correction_factor
```

这里 `CountDefinition` 只存在于 exact-count flattening 与 definition lowering 之间，不应进入最终 algorithm input。

## 对当前 C2NormalForm 的实现结论

### 已保留

- QF universal body；
- ordinary `∀x∃y` sections；
- exact global count；
- exact row count；
- `CountDefinition`，用于保留 embedded count 的 marker 语义。

### 已改变

- `PredicateDefinition` 已删除，definition clause 只写入 QF body；
- `universal` 已改为直接保存 QF body 的 `qf_formula`；
- `_extract_universal_sections()` 已删除；
- embedded count section 只由 `CountDefinition` 持有，不再混入 direct count fields。

### 算法特定 reduction

- modulo counting；
- incremental3 的直接 `<=` mask 优化。

`CountSection` 暂时保留 comparator，而不在通用 normalize 阶段统一展开。不同算法可以按自身能力选择 exact-count lowering、直接 mask 或拒绝。这保留了算法无关 IR 的信息，也避免为所有算法强制同一条 reduction 路径。

## 最重要的结论

论文支持的是“任意 C2 counting formula 可以通过一系列 WFOMC-preserving reductions 降到可 lifted 计算的形式”，不是“把任意 comparator 和 unresolved marker definition 装进一个 dataclass 就得到了最终 C2 normal form”。

因此当前实现下一步最关键的工作不是继续增加 section 字段，而是为需要执行 embedded count 的算法实现明确的 `CountDefinition` lowering。普通 comparator 是否收敛到 exact count，由目标算法的 reduction 决定。
