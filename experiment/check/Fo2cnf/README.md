# WFOMC to CNF Converter and Model Counter

This document is available in English and Chinese.

- [English Readme](#english-readme)
- [中文 Readme](#中文-readme)

---

## English Readme

### Introduction

This is a command-line tool designed to convert logic problems defined in the Weighted First-Order Model Counting (WFOMC) format (`.wfomc`) into the standard Conjunctive Normal Form (CNF), and then invoke various back-end solvers to count the number of models.

The core functionality of this tool is to "ground" first-order logic formulas, which contain quantifiers, over a given finite domain. This process generates an equivalent propositional logic formula without variables, which is then converted into the CNF format suitable for SAT solvers.

### Key Features

*   **Rich Logic Support**:
    *   Supports universal (`∀`) and existential (`∃`) quantifiers.
    *   Supports Counting quantifiers (e.g., `#x.phi(x) = k`).
    *   Supports Modulo quantifiers (e.g., `#x.phi(x) mod k = r`).
    *   Supports Cardinality Constraints (e.g., `|E| <= k`).
*   **Linear Order Axioms**: Automatically identifies the `LEQ` predicate and adds axioms for linear order (reflexivity, anti-symmetry, transitivity, totality), allowing solvers to explore all possible orderings.
*   **Multiple Counter Backends**:
    *   **Ganak**: For exact model counting.
    *   **ApproxMC**: For approximate model counting.
    *   **PySAT**: A built-in, pure Python exact model counter, useful when external tools are not available.
*   **Standardized Output**: Generates files in the standard DIMACS CNF format, which can be used by almost all SAT-related tools.
*   **Caching Mechanism**: If a CNF file for a given input and domain size already exists, the script will skip the conversion process and proceed directly to counting, saving time.

### Installation Guide

#### 1. Prerequisites

*   **Python**: Python 3.8 or newer is recommended.
*   **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

#### 2. Install Python Dependencies

This project depends on `pysat`, `sympy`, and `logzero`. You can install them all at once using `pip` and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
*(You need to create the `requirements.txt` file first, as previously discussed.)*

#### 3. Install External Counters (Optional but Recommended)

To use `ganak` or `approxmc` for efficient counting, you need to download and compile them separately.

*   **Ganak**:
    1.  Visit the [Ganak GitHub repository](https://github.com/meelgroup/ganak).
    2.  Follow the instructions in their documentation to compile it.
    3.  Note the path to the generated `ganak` executable.

*   **ApproxMC**:
    1.  Visit the [ApproxMC GitHub repository](https://github.com/meelgroup/approxmc).
    2.  Follow the instructions in their documentation to compile it.
    3.  Note the path to the generated `approxmc` executable.

### Usage

The tool is operated via the command line.

#### Basic Command Format

```bash
python fo2cnf.py --input <input_file> --domain <domain_size> --counter <counter_name> [options]
```

#### Command-Line Arguments

*   `--input, -i`: **(Required)** Path to the input `.wfomc` file.
*   `--domain, -d`: The size of the domain. Defaults to `5`.
*   `--counter, -c`: The model counter to use. Choices are `ganak`, `pysat`, `approxmc`. Defaults to `ganak`.
*   `--ganak_path`: Path to the `ganak` executable. Not needed if `ganak` is in the system's `PATH`.
*   `--approxmc_path`: Path to the `approxmc` executable.
*   `--epsilon`: The tolerance parameter `epsilon` for `approxmc`. Defaults to `0.01`.
*   `--delta`: The confidence parameter `delta` for `approxmc`. Defaults to `0.01`.

#### Examples

Assuming we have an input file named `graph_coloring.wfomcs`.

1.  **Exact counting with Ganak (domain size 4)**
    ```bash
    python fo2cnf.py -i models/graph_coloring.wfomcs -d 4 -c ganak --ganak_path /path/to/your/ganak
    ```

2.  **Approximate counting with ApproxMC (domain size 10)**
    ```bash
    python fo2cnf.py -i models/graph_coloring.wfomcs -d 10 -c approxmc --approxmc_path /path/to/your/approxmc
    ```

3.  **Exact counting with PySAT (domain size 3)**
    ```bash
    python fo2cnf.py -i models/graph_coloring.wfomcs -d 3 -c pysat
    ```

### Output Description

*   **CNF File**: The script generates a `.cnf` file in the same directory as the input file, e.g., `graph_coloring_domain_size_4.cnf`.
*   **Log File**: A `performance.log` file is created in the script's directory, recording detailed runtime information and errors.
*   **Model Count Result**: The final model count is printed directly to the console.

```
INFO:__main__:Result:
 InputFile: models/graph_coloring.wfomcs
 Domain Size: 4
 Counter: ganak
 Model Count: 96
```

---

## 中文 Readme

### 简介

这是一个命令行工具，旨在将以权重一阶模型计数（WFOMC）格式（`.wfomc`）定义的逻辑问题，转换为标准的合取范式（CNF）文件，并调用多种后端求解器来计算其模型的数量。

该工具的核心功能是将带有量词的一阶逻辑公式，在一个给定的有限论域（domain）上进行“基化”（grounding），从而生成一个等价的、不含变量的命题逻辑公式，并最终将其转换为可供SAT求解器处理的CNF格式。

### 主要功能

*   **丰富的逻辑支持**:
    *   支持全称量词 (`∀`) 和存在量词 (`∃`)。
    *   支持计数（Counting）量词（例如，`#x.phi(x) = k`）。
    *   支持模（Modulo）量词（例如，`#x.phi(x) mod k = r`）。
    *   支持基数约束（Cardinality Constraints）（例如，`|E| <= k`）。
*   **线性序公理**: 自动识别 `LEQ` 谓词，并为其添加线性序的公理（自反性、反对称性、传递性、完全性），允许求解器探索所有可能的排序。
*   **多种计数后端**:
    *   **Ganak**: 用于精确模型计数。
    *   **ApproxMC**: 用于近似模型计数。
    *   **PySAT**: 内置的纯Python精确模型计数器，方便在没有外部工具时使用。
*   **标准化输出**: 生成标准的 DIMACS CNF 格式文件，可被几乎所有的SAT相关工具使用。
*   **缓存机制**: 如果一个输入的CNF文件已经存在，脚本会跳过转换过程，直接进行计数，以节省时间。

### 安装指南

#### 1. 环境准备

*   **Python**: 建议使用 Python 3.8 或更高版本。
*   **克隆仓库**:
    ```bash
    git clone <你的仓库URL>
    cd <仓库目录>
    ```

#### 2. 安装 Python 依赖

本项目依赖于 `pysat`, `sympy` 和 `logzero`。您可以通过 `pip` 和提供的 `requirements.txt` 文件一键安装：

```bash
pip install -r requirements.txt
```
*(您需要先按照之前的建议创建 `requirements.txt` 文件)*

#### 3. 安装外部计数器 (可选，但推荐)

为了使用 `ganak` 或 `approxmc` 进行高效计数，您需要单独下载并编译它们。

*   **Ganak**:
    1.  访问 [Ganak GitHub 仓库](https://github.com/meelgroup/ganak)。
    2.  按照其文档指引进行编译。
    3.  记录下生成的可执行文件 `ganak` 的路径。

*   **ApproxMC**:
    1.  访问 [ApproxMC GitHub 仓库](https://github.com/meelgroup/approxmc)。
    2.  按照其文档指引进行编译。
    3.  记录下生成的可执行文件 `approxmc` 的路径。

### 使用方法

本工具通过命令行进行操作。

#### 基本命令格式

```bash
python fo2cnf.py --input <输入文件> --domain <论域大小> --counter <计数器> [其他选项]
```

#### 命令行参数

*   `--input, -i`: **(必需)** 输入的 `.wfomc` 文件路径。
*   `--domain, -d`: 论域的大小。默认为 `5`。
*   `--counter, -c`: 使用的模型计数器。可选值为 `ganak`, `pysat`, `approxmc`。默认为 `ganak`。
*   `--ganak_path`: `ganak` 可执行文件的路径。如果 `ganak` 在系统 `PATH` 中，则无需指定。
*   `--approxmc_path`: `approxmc` 可执行文件的路径。
*   `--epsilon`: `approxmc` 的容忍度参数 `epsilon`。默认为 `0.01`。
*   `--delta`: `approxmc` 的置信度参数 `delta`。默认为 `0.01`。

#### 使用示例

假设我们有一个名为 `graph_coloring.wfomcs` 的输入文件。

1.  **使用 Ganak 进行精确计数 (论域大小为 4)**
    ```bash
    python fo2cnf.py -i models/graph_coloring.wfomcs -d 4 -c ganak --ganak_path /path/to/your/ganak
    ```

2.  **使用 ApproxMC 进行近似计数 (论域大小为 10)**
    ```bash
    python fo2cnf.py -i models/graph_coloring.wfomcs -d 10 -c approxmc --approxmc_path /path/to/your/approxmc
    ```

3.  **使用 PySAT 进行精确计数 (论域大小为 3)**
    ```bash
    python fo2cnf.py -i models/graph_coloring.wfomcs -d 3 -c pysat
    ```

### 输出说明

*   **CNF 文件**: 脚本会在输入文件所在的目录下生成一个 `.cnf` 文件，例如 `graph_coloring_domain_size_4.cnf`。
*   **日志文件**: 在脚本同级目录下会生成 `performance.log` 文件，记录详细的运行信息和错误。
*   **模型计数结果**: 最终的模型数量会直接打印在控制台。

```
INFO:__main__:Result:
 InputFile: models/graph_coloring.wfomcs
 Domain Size: 4
 Counter: ganak
 Model Count: 96
```