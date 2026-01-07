from .fol_grammar import function_free_logic_grammar
from .cardinality_constraints_grammar import cc_grammar

"""
这个文件的核心作用是定义 WFOMC 问题的文本语法规则。它使用 Lark 解析器库的语法格式，将一个完整的 WFOMC 问题描述分解成不同的组成部分，如逻辑公式、论域、权重和约束。

最终，这些分散的语法规则被拼接成一个名为 grammar 的完整字符串，这个字符串将作为“语法蓝图”被 Lark 解析器使用。
"""
domain_grammar = r"""
    domain: domain_name "=" domain_spec
    domain_name: CNAME
    ?domain_spec: INT               -> int_domain
        | ("{" domain_elements "}") -> set_domain
    domain_elements: element ("," element)*
    element: CNAME
"""

"""
`domain` 规则：定义一个论域的完整结构。
它由一个 `domain_name`、一个等号 `=` 和一个 `domain_spec` 组成。
例如: "D = 5" 或 "D = {a, b, c}"
`domain_name` 规则：论域的名称是一个 CNAME（通用名称，通常是字母、数字和下划线的组合）。
`domain_spec` 规则：定义论域的具体内容。
    `?` 表示这个规则在解析树中会被“内联”，即它的子节点会直接挂在父节点上。
    
    可能性1：一个整数（INT）。`-> int_domain` 是一个别名，
    在转换器（Transformer）中，处理这个分支的方法将被命名为 `int_domain`。
    例如: "5"
    
    可能性2 (`|`)：一个用花括号 `{}` 包围的元素列表。别名为 `set_domain`。
    例如: "{a, b, c}"

`domain_elements` 规则：定义花括号内的元素列表。
它由一个 `element`，后面跟着零个或多个由逗号分隔的 `element` 组成。
`*` 表示 "零个或多个"。
`element` 规则：列表中的单个元素是一个 CNAME。
"""

# `grammar` 变量将所有部分拼接成一个完整的语法定义。
grammar = r"""
    ?wfomcs: ffl domain weightings cardinality_constraints
    weightings: weighting*
    weighting: weight weight predicate

    weight: SIGNED_FLOAT | SIGNED_INT
""" + domain_grammar + cc_grammar + function_free_logic_grammar
"""
`wfomcs` 规则：这是整个语法的起始规则（start rule）。
一个完整的 WFOMC 问题由四个部分组成：`ffl` (无函数逻辑公式), `domain` (论域), `weightings` (权重), `cardinality_constraints` (基数约束)。
`weightings` 规则：权重部分由零个或多个 `weighting` 定义组成。

`weighting` 规则：单个权重定义。
它由两个 `weight` 和一个 `predicate` 组成。
这通常对应于谓词为 True 时的权重和为 False 时的权重。
例如: "1.0 0.0 P(x)"

`weight` 规则：权重值可以是一个有符号浮点数或有符号整数。
`SIGNED_FLOAT` 和 `SIGNED_INT` 是 Lark 内置或在其他地方定义的终端符号。


最后，通过字符串拼接，将主语法、论域语法、基数约束语法和一阶逻辑语法
合并成一个单一、完整的语法字符串，供 Lark 解析器使用。
"""