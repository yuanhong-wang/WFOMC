# 代码复用审查（2026-07-11）

## 结论

当前仓库不存在需要建立“大一统工具层”的问题。值得复用的代码主要是重复表达同一业务规则，而不是普通的循环、排序或 dataclass 字段。建议优先处理三个点：

1. `CellGraphData` 统一提供 weight matrix 投影；
2. FOL traversal 统一使用 `formula_children()`；
3. cardinality constraint 自己负责判断一个计数是否满足约束。

算法 `spec.prepare()` 模板和 fast evidence/non-evidence graph 也有明显重复，但抽取时必须保留算法入口的可读性，不应引入通用 hook 框架。

## 判断标准

只有同时满足以下条件才建议复用：

- 重复代码表达的是同一个业务规则，而不仅是形状相似；
- 修改该规则时，当前实现确实存在多处改漏风险；
- 复用后的调用点比原代码更容易理解；
- 不引入 `object`、动态属性探测或多层 callback。

## P1：建议优先复用

状态：以下三项已完成。

### 1. CellGraphData 的权重投影

同一个转换在多个算法重复出现：

- `incremental/input.py::_ordered_component()`；
- `incremental3/input.py::_counting_component()`；
- `tail_signature/input.py::_component()`；
- `fast/graph.py::_GraphView.get_all_weights()`。

它们都把 `TwoTable` matrix 转成 `ArithmeticValue` matrix：

```python
tuple(tuple(table.get_weight() for table in row) for row in data.two_tables)
```

建议由 `CellGraphData` 提供：

```python
def pair_weights(self) -> tuple[tuple[ArithmeticValue, ...], ...]: ...

def conditioned_pair_weights(
    self, evidence: frozenset[Literal]
) -> tuple[tuple[ArithmeticValue, ...], ...]: ...
```

这是数据自身的稳定投影，放在 class method 比放入 `algo/utils.py` 更明确。算法仍负责决定使用普通、predecessor 或 evidence-conditioned table。

### 2. FOL children traversal

`fol/syntax.py` 已提供 `formula_children()`，但 `fol/analysis.py` 又维护了 `_formula_children()`，逐类列举 `Not`、`And`、`Or`、quantifier 等节点。新增语法节点时，两处必须同步，否则 predicates/constants/free-vars/height 可能漏遍历。

建议删除 analysis 内部版本，直接复用 `syntax.formula_children()`。`rewrite.py` 中的 `_map_children()` 不是同一职责：它负责按节点类型重建公式，应继续独立存在。

### 3. Cardinality constraint 的满足关系

`reduce_cardinality_constraints.py::_compare()` 实际是 `LinearCardinalityConstraint` 的领域规则。当前 constraint 负责校验 modulus，reduction 却负责解释 comparator，职责被拆开。

建议增加：

```python
class LinearCardinalityConstraint:
    def accepts(self, value: int) -> bool: ...
```

decoder 只计算线性项总和，然后调用 `constraint.accepts(total)`。这同时避免未来 parser、测试或其他算法再实现一套 EQ/NE/LT/LE/GT/GE/MOD 判断。

## P2：可以复用，但应小范围实施

### 4. 常规算法 reduction sequence

standard、fast、incremental、recursive、propositional、tail-signature 重复声明：

```python
(
    reduce_unary_evidence_for_options,
    reduce_counting_quantifiers,
    reduce_existential_quantifiers,
    reduce_cardinality_constraints,
)
```

可以在 `algo/core.py` 或 `reduction/__init__.py` 暴露一个命名明确的 `GENERAL_REDUCTIONS` tuple。incremental3 继续声明自己的 counting-DP sequence。

不建议把整个 `prepare()` 抽成接收任意 callback/kwargs 的通用 builder。各算法传给 `build_input()` 的数据不同，保留十几行显式循环更容易理解。

### 5. spec.prepare 的 branch compile 循环

standard、incremental、recursive、propositional 的 `prepare()` 几乎相同；fast、incremental3 和 tail-signature 在同一骨架上增加算法数据。

若后续该循环继续增加统一步骤，可以提供一个非常窄的 helper：输入 reduced branches 和一个类型明确的 `build_input(compiled, features)` 函数，输出 `PreparedBranch`。目前收益有限，优先复用 reduction tuple 即可。

不要做 `BaseAlgoSpec`、继承层级或注册 hook；这些会重新引入之前已经清理掉的 planning 封装。

### 6. fast graph 的基础关系判断

`_OptimizedAnalysis` 与 `_EvidenceOptimizedAnalysis` 都实现了：

- cell index 到 weight 的视图；
- interaction graph 构造；
- self-loop 判断；
- symmetric clique 的贪心分组；
- clique 间 relation consistency 判断。

但 evidence 版本还需要 profile-aware seed、expanded cell identity 和 profile partition，不能直接合并成一个大类。

建议只抽取两个纯函数：

- 从 cell indices 和 pair-weight accessor 构建 interaction graph；
- 按显式 `matches(left_clique, candidate)` 分组 indices。

不要让 evidence analysis 继承 non-evidence analysis，也不要恢复通用 graph facade。两条算法路径的业务差异应继续显式保留。

## 不建议抽取的相似代码

### Problem / ReducedProblem / CompiledProblem 属性

三个阶段都有 `has_unary_evidence`、`has_binary_evidence` 等属性，但含义和可用字段并不完全一致。通过基类或 mixin 复用几行属性会模糊阶段边界。保持重复更好；feature analyzer 如需统一类型，应使用窄 Protocol，而不是继承。

### UnaryEvidence / BinaryEvidence

两者都有 `is_empty` 和 `cache_key_parts()`，但 literal shape 不同。建立泛型 EvidenceCollection 只能省少量代码，却增加公开模型数量，没有必要。

### 各算法的 solve 循环

fast、incremental、incremental3 都会遍历 components 并累加 `graph_weight * component_result`，但 component result 的计算、evidence coefficient 和状态空间完全不同。只复用 ArithmeticContext，不复用控制流。

### 排序和 tuple 化

`dict(sorted(..., key=str))`、`tuple(tuple(...))` 在多处出现，但大多是各阶段建立确定性输出，不是独立业务能力。建立 generic collection utils 会降低局部可读性。

### FLINT context 转换

cardinality decoder、Ganak adapter 和 weight compilation 都会投影 polynomial context，但方向和约束不同：cardinality 删除内部 symbols，Ganak 重命名 generators，weight compilation 对齐 branch ArithmeticContext。不能仅因都调用 FLINT context API 就合并。

## 次要清理机会

- `incremental/input.py::_same_predicate()` 可直接使用 `Predicate.identity()`；
- evidence model 与 `Literal.cache_key_parts()` 的 string-predicate fallback 相似，可在决定是否继续支持 string predicate 后统一；如果 public evidence 仍允许 string，优先让 evidence constructor 尽早转成 typed `Predicate`；
- `TailSignaturePolynomialContext` 已确认不可达：tail-signature 的 reduction sequence 先执行 cardinality reduction并清空 constraints，随后 `prepare()` 才把 constraints 传给 input builder，因此该字段永远是 `None`，可以删除；
- `MultinomialCoefficients.setup(domain_size)` 是全局状态式 API，多个算法使用时存在隐式共享；更适合改成按 domain size 缓存的 class method，而不是新增 utils wrapper。

## 推荐实施顺序

1. `LinearCardinalityConstraint.accepts()`，删除 reduction 私有 comparator；
2. `CellGraphData.pair_weights()`，替换三个算法的 table 投影；
3. analysis 复用 `formula_children()`；
4. 统一 `GENERAL_REDUCTIONS`；
5. 单独重构 fast graph 的两个纯算法片段；
6. 最后判断 prepare helper 是否仍有足够收益。

前三项是低风险、边界清晰的复用；后面三项应分别提交和验证，避免一次性把算法结构重新抽象化。
