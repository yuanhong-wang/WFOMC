# `fol.normal_form.c2` 正确性审查（2026-07-11）

> 这是修复过程中的历史审计记录。2026-07-12 起 normalizer 已要求 closed sentence、保留 Boolean structure，并用 negated marker 表示 negated modulo；当前结构以 `c2-normal-form-current-shape-2026-07-11.md` 为准。

## 结论

当前实现不能视为一个正确、封闭的 C2 normalizer。正常的直接 `∀∃`、全局精确计数、row exact count、NNF 和简单 Scott abstraction 路径有测试覆盖，但 normalizer 与 validator 的契约存在多处漏洞，其中有几处会直接改变公式语义或让算法静默忽略约束。

## 修复状态

本报告列出的问题现已全部处理：

- predicate/variable fresh names 使用 used-name set 避碰；Problem 会把 weights、evidence、cardinality 中的声明一并保留给 normalizer；
- modulo count 不再走 fallback，direct global/row 和 embedded count 会产生 typed section，negated modulo 使用 negated marker；
- source normalization 检查 closed sentence、两变量 subformula 与 solver 的 unary/binary vocabulary；
- validator 检查 universal、`forall_exists`、`exists`、count body、marker、definition scope 和 QF contract；
- incremental3 对 unsupported comparator 和 embedded count definition fail fast；
- Scott abstraction 显式标记 non-empty-domain requirement，reduction 在空域拒绝；
- `quantifier_free_body()` 对非 QF 结果报错；`normalize(C2NormalForm)` 也会执行 validation。

以下章节保留原始发现和反例，作为回归测试依据。

本次审查运行了现有 normal-form/parser/feature 测试，49 项全部通过；下面的问题均不在现有测试覆盖内，并已用最小 typed formula 复现。

## Critical

### 1. Fresh predicate 会与用户 predicate 冲突并改变语义

位置：`normalize.py::_fresh_atom()`。

fresh name 直接使用 `@c2_quant_0`、`@c2_count_0`、`@c2_rel_0`，没有收集或避开 source predicate names。

反例：

```text
@c2_quant_0(X) ∨ ∃Y E(X,Y)
```

normalizer 把 `∃Y E(X,Y)` 的 marker 也命名为 `@c2_quant_0(X)`。生成的定义会把用户 predicate 强制解释成 `∃Y E(X,Y)`，原来的析取退化成 marker 本身，语义发生变化。

修复要求：normalizer 初始化时收集全部 `(name, arity)`，所有内部 predicate 通过唯一 fresh-name allocator 创建；不要依赖“用户通常不会使用 @ 前缀”。

### 2. Alpha-renaming 的 fresh variable 也会冲突并捕获变量

位置：`normalize.py::_AlphaRenamer._binder()`。

shadowed `X` 固定改名成 `X_c2_0`，但没有避开公式中已经存在的 `X_c2_0`。

反例：

```text
∀X ∀X_c2_0 ∃X P(X, X_c2_0)
```

inner `X` 被改成已绑定的 `X_c2_0`，body 从 `P(innerX, outerX_c2_0)` 变成 `P(X_c2_0, X_c2_0)`。随后 marker 甚至被错误判定为 nullary。该转换不再与 source 等价或等可满足。

修复要求：alpha-renamer 预先收集所有 variable names，并从全局 used-name set 分配 fresh variable；scope environment 仍按 binder identity 管理。

### 3. Incremental3 会静默忽略部分合法 comparator 和 count-definition 语义

位置：`algo/incremental3/counting_state.py::build_counting_state_for_normal_form()`。

normal-form validator 接受 `= != < <= > >= mod`，但 incremental3 只处理 `mod`、`=`、`<=`，其余 comparator 没有报错，直接不加入 state/mask。例如：

```text
∃_{>2} X U(X)
∀X ∃_{!=2} Y R(X,Y)
```

都会生成 normal-form section，但 incremental3 不施加该约束。

embedded boolean count 生成 `count_definitions` 和 marker；incremental3 同样不消费 marker 与 section 之间的条件关系，却会把 section 当成无条件计数约束。这也会改变语义。

修复要求：在算法 planning/input boundary 明确验证算法支持的 normal-form subset。未实现的 comparator 和非空 `count_definitions` 必须拒绝，不能由 state builder 跳过。

## High

### 4. Modulo fallback 返回的不是 normalized C2NormalForm

位置：`normalize.py::normalize()`。

只要 normalization 抛出包含 `Modulo counting` 的 `NormalizeError`，入口就返回：

```python
C2NormalForm(universal=sentence)
```

这绕过 alpha-renaming、NNF、section extraction、predicate definition 和 scope validation。

反例：

```text
P(X) ∨ ∃_{0 mod 2} Y R(X,Y)
```

结果的 `universal` 就是原句；`quantifier_free_body()` 仍返回含 counting quantifier 的公式，`is_quantifier_free()` 为 false，但 `validate_normal_form()` 通过。

修复要求：删除 exception-message fallback。Modulo count 应由 normalizer 正式产生 `CountSection` / `ForallCountSection`，或者在明确不支持的 boolean context 抛出 `NormalizeError`。

### 5. 实现没有验证输入属于 C2

normalizer 会接受并 validate：

```text
∀X ∀Y ∀Z R(X,Y,Z)
```

输出仍保留三个不同变量，`quantifier_free_body()` 得到 `R(X,Y,Z)`。当前 validation 不检查：

- 整个 sentence 是否 closed；
- 任一 subformula 是否最多使用两个 variable symbols；
- normal-form universal body 是否符合算法实际支持的 unary/binary vocabulary；
- `∀∃` 与 count body 是否有 scope 外的 free variables。

修复要求：在 normalization 前执行 source-fragment validation，在 normalization 后执行 IR invariant validation。算法只支持 unary/binary predicate 时，应单独明确限制，不要把它与逻辑学定义中的 FO2/C2 限制混为一条含糊检查。

### 6. Scott abstraction 隐含非空 domain 假设，但系统允许空 domain

exists/forall marker 的反向定义通过 witness existential 实现。例如 exists marker 包含：

```text
∃Y (¬marker ∨ body)
```

在空 domain 上，即使正确解释应是 `marker = false`，该 witness 仍为 false，导致转换不可满足。forall marker 有相同问题。

`Problem.domain` 默认可以为空，仓库也有空 domain 测试，没有统一的 non-empty domain invariant，因此该假设目前不成立。

修复方案二选一：

- 明确禁止空 domain，并在 Problem/reduction 入口验证；
- 修改 definitional normalization，使其在空 domain 上仍保持语义。

## Medium

### 7. `validate_normal_form()` 没有验证主要 IR 不变量

validator 当前只遍历显式 `forall_counts`、`counts`、definitions 的少数字段。它不验证：

- `universal` 是否为合法 Formula、是否只包含 universal/QF 内容；
- `forall_exists` 和 `exists` 的 quantifier shape；
- 通过 `universal` fallback 隐藏的 count sections；
- count body 是否为对应 arity 的 typed Atom；
- `count_definition.marker == count_definition.section.marker`；
- definition marker、body、variables 的 arity/free-variable 一致性；
- `predicate_definitions` 是否已经由 universal equivalence 表达；
- fresh symbols 是否互不冲突。

现有测试甚至把 string body/marker 当作 valid normal form，这与 typed pipeline 的目标冲突。

### 8. `quantifier_free_body()` 的返回契约没有被保证

该函数名称和调用者都假设结果 QF，但 `_extract_universal_sections()` 遇到无法识别的 quantified/embedded count structure 时可能直接返回含 quantifier 的 body。函数本身不做 `is_quantifier_free()` 检查。

建议在 C2NormalForm validation 保证结构后，再让该 accessor 使用直接字段；不应继续作为对任意 `universal` 公式进行容错解析的第二套 normalizer。

### 9. `normalize(C2NormalForm)` 直接原样返回

传入现成 C2NormalForm 时不会 validate，也不会 canonicalize。虽然 `begin_reduction()` 随后会调用 validator，但公开的 `normalize()` 单独使用时不能保证输出满足其名称承诺。

## 当前正确或基本合理的部分

- implication elimination 后进入 NNF 的 comparator dual 映射正确；
- `=0` / `<=0` 转 universal negation，以及 `>0` / `>=1` 转 existential 的规则正确；
- non-atomic count body 通过 fresh relation + universal iff 变成 atomic body的方向合理；
- 在 non-empty domain 且 fresh names 不冲突的前提下，exists/forall Scott marker 的两个方向基本正确；
- direct global exact count 和 direct row exact count 的 section 数据完整；
- shadowing 不发生 fresh-name collision 时，scope environment 的递归方式基本正确。

## 建议修复顺序

1. 建立统一 fresh-name allocator，同时修 predicate 与 variable collision；
2. 删除 modulo fallback，正式定义 mod 支持边界；
3. 在 normalizer 前后分别增加 source C2 validation 和 IR validation；
4. incremental3 对不支持 comparator/count definitions fail fast；
5. 决定 empty-domain 语义并修复 Scott abstraction 或禁止空域；
6. 收紧 C2NormalForm 字段类型和 validator，删除 string/object valid cases；
7. 简化 `norm_form.py` 的 universal section 容错提取逻辑，使 normalized IR 成为唯一事实来源。

不建议在修复这些 correctness 问题之前继续给 normalize 增加更多 comparator 或算法分支。
