import numpy as np

from collections import Counter, defaultdict
from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.context.dr_context import DRWFOMCContext, ConfigUpdater, HashableArrayWrapper
from wfomc.fol.syntax import AUXILIARY_PRED_NAME, X, Y, AtomicFormula, Const, Pred, QFFormula, top, a, b
from wfomc.utils import multinomial, MultinomialCoefficients
from wfomc.utils.polynomial import Rational, coeff_dict, expand
from wfomc.utils.third_typing import RingElement


def domain_recursive_wfomc(context: DRWFOMCContext) -> RingElement:
    # # 从 context 中取出域、权重函数、线性序谓词和公式
    domain: set[Const] = context.domain
    get_weight = context.get_weight
    leq_pred: Pred = context.leq_pred
    formula = context.uni_formula
    c_type_shape = context.c_type_shape
    result = Rational(0, 1)
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size) # 预计算多项式系数
    ## 遍历所有 cell graph
    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight, leq_pred):
        cells = cell_graph.get_cells()
        n_cells = len(cells) # # 1-type cell 数
        w2t, w, r = context.build_weight(cells, cell_graph)  # 构建 cell 的权重、类型映射及递归转移参数
        ## 处理一元 counting mod 约束（∃_{r mod k}）
        unary_masks = []  # # 每个约束对应一个 mask 和 (r,k) [(np.int8[n_cells], r, k), …]
        for pred, r_mod, k_mod in context.unary_mod_constraints:
            mask = np.fromiter(
                (1 if cell.is_positive(pred) else 0 for cell in cells), # cell 是否满足该一元谓词
                dtype=np.int8, count=n_cells
            )
            unary_masks.append((mask, r_mod, k_mod))
            # 一阶 1-type cell 是否把某个一元谓词 pred 标成 True”转换成一个长度为 n_cells 的 0‒1 向量mask。比如，cells = [B(X)^LEQ(X,X)^~@aux0(X,X)^~A(X), @aux0(X,X)^A(X)^LEQ(X,X)^~B(X)], 那么unary_masks = [([0 1], 0, 2)]

        t_updates = context.build_t_updates(r, n_cells, domain_size)  # 构建 t_updates（pairwise 组合时状态更新表）
        shape = (n_cells,) + tuple(c_type_shape) # 配置数组的维度: (cell 数) × (扩展谓词维度) × (计数约束维度)
        Cache_F = dict()  # 全局缓存，用于 update_config 的结果
        config_updater = ConfigUpdater(t_updates, shape, Cache_F)  # update_config
        update_config = config_updater.update_config  # this is a function
        Cache_T = dict()  # 缓存递归计算结果，避免重复子问题

        ## ========== 定义核心递归函数 ==========
        def domain_recursion(config):
            if config in Cache_T: # 缓存命中，直接返回
                return Cache_T[config]

            if config.array.sum() == 0:  # 如果当前配置已分配完所有元素
                return Rational(1, 1) # 基础情况，返回 1

            T = defaultdict(lambda: Rational(0, 1))
            new_config = HashableArrayWrapper(  # 拷贝当前配置，避免原地修改
                np.array(config.array, copy=True, dtype=np.uint8)
            )
            ## 选择最后一个非零 cell（固定策略）作为 target_c
            target_c = tuple(np.argwhere(new_config.array > 0)[-1])  # Select last non-zero cell as target (processing order strategy)
            new_config.array[target_c] -= 1  # 从配置中移除一个元素
            F = dict() # 初始化 F（配对状态集）：target cell + 辅助配置
            u_config = np.zeros(shape, dtype=np.uint8)
            u_config = HashableArrayWrapper(u_config)
            F[(target_c, u_config)] = Rational(1, 1)

            ## 依次将 target 与其余元素配对（外层循环：other_c）
            for other_c in np.argwhere(new_config.array > 0):
                other_c = tuple(other_c.flatten())  # Get cell coordinates
                F_new = defaultdict(lambda: Rational(0, 1))
                l = new_config.array[other_c]  # other_c 的剩余个数

                ## 内层循环：更新所有可能的配对状态
                for (target_c, u_config), W in F.items():
                    F_update = update_config(target_c, other_c, l)  # 调用 update_config 计算 target_c 与 other_c 交互 l 次后的所有可能结果
                    for target_c_new, u_config_update in F_update.keys():  # 合并更新：累加新状态的权重
                        F_config_new = HashableArrayWrapper(
                            u_config.array + u_config_update.array
                        )
                        F_new[(target_c_new, F_config_new)] += W * F_update[(target_c_new, u_config_update)]
                F = F_new # 更新 F

            ## 过滤掉未满足 stop_condition 的状态，得到递归下一层的入口状态集 T
            for (last_target_c,
                 last_F_config), W in F.items():  # Among alAl the pairing methods, only retain the portion where the target elements have fully met the constraints, and merge their weights into the configuration T of the next layer of recursion.
                if context.stop_condition(last_target_c): # Check the predicate status. Filter out valid results: The predicate status of target_c must be all zeros.
                    T[last_F_config] += W

            ## 递归累加子问题结果
            ret = Rational(0, 1)
            for recursive_config, weight in T.items():
                W = domain_recursion(recursive_config)  # Recursive call to subconfiguration
                ret = ret + (weight * W)
            Cache_T[config] = ret  # 缓存当前配置的结果
            return ret

        ## ========== 主循环：遍历所有多项式配置 ==========
        for config in multinomial(n_cells, domain_size):
            ## --- 检查一元 mod 约束 ---
            skip = False
            for mask, r_mod, k_mod in unary_masks:  # 遍历每个约束
                config_total_unary_constraint = (mask @ np.fromiter(config, dtype=np.int32))  # config 是当前 1-type 配置，元素是“第 i 个 cell 放了多少个元素”。mask @ config 就是向量点积 —— 自动算出 整个结构里满足 pred 的元素总数
                if config_total_unary_constraint % k_mod != r_mod:
                    skip = True
                    break
            if skip:  # 有一个约束不满足，就跳过这个配置
                continue
            ## 初始化配置数组并赋值
            init_config = np.zeros(shape, dtype=np.uint8)
            W = Rational(1, 1)
            for i, n in enumerate(config):
                init_config[(i,) + w2t[i]] = n  # i is cell index, n is element num of cell i
                W = W * (w[i] ** n)
            init_config = HashableArrayWrapper(init_config)
            ## # 调用递归计算该配置的加权结果
            dr_res = domain_recursion(init_config)  # call recursison
            # 累加：多项式系数 × 权重 × 递归结果 × cell graph 权重
            result += (MultinomialCoefficients.coef(config) * W * dr_res * graph_weight)
    return result
