# Graph Model Counter for Odd-Degree Vertices

This script is a specialized tool for a graph theory combinatorial problem. It counts the number of simple undirected graphs on `n` vertices that satisfy two specific conditions:
1.  The graph has exactly `k` edges.
2.  The graph has exactly `m` vertices of odd degree.

The script works by translating these graph properties into a propositional logic formula in Conjunctive Normal Form (CNF) and then using an external model counter (`ganak`) to find the number of satisfying assignments, which corresponds to the number of such graphs.

## Features

-   **First-Order Logic to CNF Conversion**: Encodes graph properties into a CNF formula.
-   **Parameterization**: Allows specifying the number of vertices (`n`), the number of odd-degree vertices (`m`), and the number of edges (`k`).
-   **XOR Chain Encoding**: Uses an efficient XOR chain to encode the odd-degree constraint for each vertex.
-   **Cardinality Constraints**: Leverages the `pysat` library to enforce "exactly-k" and "exactly-m" constraints.
-   **Automated Counting**: Integrates with the `ganak` model counter to perform the final count.
-   **Data Export**: The main execution block iterates through various `n`, `m`, and `k` values and saves the results to a `odd_degree.csv` file.

## Prerequisites

1.  **Python 3**: The script is written in Python.
2.  **Python Libraries**:
    -   `pysat`
    -   `sympy`
    -   `logzero`
    You can install them using pip:
    ```bash
    pip install python-sat sympy logzero
    ```
3.  **Ganak Model Counter**: The script requires the `ganak` executable. You need to download and build it, then update the `ganak_path` variable in the script to point to its location.
    -   Ganak Repository: [https://github.com/arminbiere/ganak](https://github.com/arminbiere/ganak)
    -   In the script, modify this line:
        ```python
        ganak_path = "/path/to/your/ganak/executable"
        ```

## How It Works

The script defines a set of first-order logic constraints for a simple undirected graph:

1.  **Irreflexivity**: `∀X: ¬E(X,X)` (No self-loops).
2.  **Symmetry**: `∀X,Y: E(X,Y) → E(Y,X)` (Edges are undirected).
3.  **Odd Degree Definition**: `∀X: (Odd(X) ↔ (∃_{1 mod 2} Y: E(X,Y)))` (A vertex `X` is marked as `Odd` if and only if its degree is odd). This is implemented using an XOR chain over all potential edges connected to `X`.
4.  **Odd Vertex Count**: `∃_{=m} X: Odd(X)` (There are exactly `m` vertices with an odd degree). This is encoded using a cardinality constraint.
5.  **Edge Count**: `|E| = k` (The total number of edges is exactly `k`). This is also encoded using a cardinality constraint.

The `CNFContext` class manages the conversion process. It grounds these formulas for a given domain size `n`, generating a set of propositional clauses. These clauses are then written to a `.cnf` file in the standard DIMACS format. Finally, the script calls `ganak` as a subprocess to count the models of this CNF file and prints the result.

## Usage

The script is designed to be run directly from the command line.

```bash
python ganak_odd_degree.py
```

Upon execution, the script will:
1.  Iterate through a range of values for `n` (vertices), `m` (odd-degree vertices), and `k` (edges).
2.  For each `(n, m, k)` combination, it generates a corresponding CNF file (e.g., `m-odd-degree-graph-origin_n_3_m_2_k_1.cnf`).
3.  It calls `ganak` to count the models for each CNF file.
4.  The progress is printed to the console (e.g., `n=3, m=2, k=1 -> 有效模型数量: 3`).
5.  All results are compiled and saved into a CSV file named `odd_degree.csv` in the same directory.

---

# 奇数度顶点图模型计数器

本脚本是一个用于解决图论组合问题的专用工具。它旨在计算满足以下两个特定条件的 `n` 个顶点的简单无向图的数量：
1.  图恰好有 `k` 条边。
2.  图恰好有 `m` 个奇数度的顶点。

该脚本的工作原理是将这些图属性转换为合取范式（CNF）的命题逻辑公式，然后使用外部模型计数器（`ganak`）来计算满足条件的赋值数量，这个数量即对应满足条件的图的数量。

## 功能特性

-   **一阶逻辑到CNF转换**：将图的属性编码为CNF公式。
-   **参数化**：允许用户指定顶点数（`n`）、奇数度顶点数（`m`）和边数（`k`）。
-   **XOR链编码**：使用高效的XOR链来编码每个顶点的奇偶度约束。
-   **基数约束**：利用 `pysat` 库来强制实现“恰好等于k”和“恰好等于m”的约束。
-   **自动计数**：集成了 `ganak` 模型计数器以执行最终的计数任务。
-   **数据导出**：主程序会遍历不同的 `n`、`m` 和 `k` 值，并将结果保存到 `odd_degree.csv` 文件中。

## 环境要求

1.  **Python 3**: 脚本使用Python编写。
2.  **Python库**:
    -   `pysat`
    -   `sympy`
    -   `logzero`
    您可以使用pip安装它们：
    ```bash
    pip install python-sat sympy logzero
    ```
3.  **Ganak模型计数器**: 脚本需要 `ganak` 可执行文件。您需要下载并编译它，然后在脚本中更新 `ganak_path` 变量以指向其位置。
    -   Ganak仓库: [https://github.com/arminbiere/ganak](https://github.com/arminbiere/ganak)
    -   在脚本中，修改这一行：
        ```python
        ganak_path = "/path/to/your/ganak/executable"
        ```

## 工作原理

该脚本为简单无向图定义了一组一阶逻辑约束：

1.  **无自环性**: `∀X: ¬E(X,X)` (没有自环)。
2.  **对称性**: `∀X,Y: E(X,Y) → E(Y,X)` (边是无向的)。
3.  **奇数度定义**: `∀X: (Odd(X) ↔ (∃_{1 mod 2} Y: E(X,Y)))` (一个顶点 `X` 被标记为 `Odd` 当且仅当它的度为奇数)。此约束通过对连接到 `X` 的所有潜在边进行XOR链编码来实现。
4.  **奇数顶点数量**: `∃_{=m} X: Odd(X)` (恰好有 `m` 个顶点的度为奇数)。此约束使用基数约束进行编码。
5.  **边数量**: `|E| = k` (图中边的总数恰好为 `k`)。此约束同样使用基数约束进行编码。

`CNFContext` 类负责管理整个转换过程。它将这些公式在给定的域大小 `n` 上进行“基境化”（Grounding），生成一组命题逻辑子句。这些子句随后被写入一个标准DIMACS格式的 `.cnf` 文件。最后，脚本通过子进程调用 `ganak` 来计算此CNF文件的模型数量，并输出结果。

## 如何使用

该脚本可以直接从命令行运行。

```bash
python ganak_odd_degree.py
```

执行后，脚本将：
1.  遍历一系列 `n`（顶点数）、`m`（奇数度顶点数）和 `k`（边数）的值。
2.  对于每个 `(n, m, k)` 组合，生成一个对应的CNF文件（例如 `m-odd-degree-graph-origin_n_3_m_2_k_1.cnf`）。
3.  调用 `ganak` 对每个CNF文件进行模型计数。
4.  将进度打印到控制台（例如 `n=3, m=2, k=1 -> 有效模型数量: 3`）。
5.  所有结果将被汇总并保存到同一目录下的 `odd_degree.csv` 文件中。