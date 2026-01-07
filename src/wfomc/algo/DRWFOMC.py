import numpy as np
from collections import Counter, defaultdict
from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.context.dr_context import DRWFOMCContext, ConfigUpdater, HashableArrayWrapper
from wfomc.fol import Const, Pred
from wfomc.utils import multinomial, MultinomialCoefficients, Rational, coeff_dict, expand, RingElement
from logzero import logger


def domain_recursive_wfomc(context: DRWFOMCContext) -> RingElement:
    # 从 context 中取出域、权重函数、线性序谓词和公式
    domain: set[Const] = context.domain # 获取问题的域
    get_weight = context.get_weight # 获取谓词权重函数
    leq_pred: Pred = context.leq_pred # 获取线性序谓词（如果存在）
    formula = context.uni_formula # 获取一阶逻辑公式
    c_type_shape = context.c_type_shape # 获取 c type 的形状信息
    WFOMC_result = Rational(0, 1)  # 初始化最终结果 WFOMC_result 为0。
    domain_size = len(domain)  # 获取域的大小
    # --- 遍历所有 cell graph
    MultinomialCoefficients.setup(domain_size) # 初始化多项式系数计算器，预先计算阶乘等，用于后续快速计算组合数。
    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight, leq_pred):
        cells = cell_graph.get_cells()  # 获取当前图中的所有单元格（Cell）及其数量
        n_cells = len(cells)  # 1-type cell 数
        w2t, w, r = context.build_weight(
            cells, cell_graph
        )  # w2t: 从单元格索引到谓词状态字典的字典，值代表需要满足的数量。w: 每个单元格类型的权重字典。r: 单元格对之间的关系字典。
        logger.debug("Weight mapping w2t: %s", w2t)
        logger.debug("Weight w: %s", w)
        # logger.debug("Weight r: %s", r)
        # 处理一元约束
        # 为一元约束（如 ∃=k X: P(X)）构建掩码，用于后续快速检查。
        unary_mask = context.build_unary_mask(cells)
        t_update_dict = context.build_t_update_dict(
            r, n_cells
        )  # 构建 t_update_dict 预计算当两个元素配对时，它们的状态会如何更新。这相当于一个巨大的状态转移查找表。
        c1_type_shape = (n_cells,) + tuple(
            c_type_shape
        )  # c1 type 的维度: (cell 数) × (扩展谓词维度) × (计数约束维度)
        Cache_H = dict()   # 缓存 ConfigUpdater 的内部计算结果。
        config_updater = ConfigUpdater(
            t_update_dict, c1_type_shape, Cache_H)  # f
        f = config_updater.f  # this is a function
        Cache_T = dict()  # 缓存递归计算结果，避免重复子问题

        # ========== 定义核心递归函数 ==========
        def domain_recursion(config):
            if config in Cache_T:  # 如果当前配置的结果已经计算过，直接从缓存返回。
                return Cache_T[config]

            # --- 递归出口 ---
            if config.array.sum() == 0: # 如果config中所有元素都已被处理 (sum=0)， 基础情况，返回 1
                return Rational(1, 1)

            # --- 递归步骤 ---
            result_of_target_c_list = Rational(0, 1)
            # --- 选择一个目标元素target_c，也就是config字典的索引，它将被用来和所有其他剩余元素连接。
            if context.contain_linear_order_axiom(): # 如果问题包含线性序，则需要遍历所有可能的元素作为target，因为它们不再对称。
                target_c_list = list(tuple(i) for i in np.argwhere(config.array > 0))
            else:  # 否则，根据对称性，我们只需选择最后一个非零类型的元素作为target即可。
                target_c_list = list((tuple(np.argwhere(config.array > 0)[-1]),))

            # --- 遍历所有选出的目标 ---
            for target_c in target_c_list:  # 遍历备选c1_type 索引列表
                T = defaultdict(lambda: Rational(0, 1)) # 创建字典，存储进入下一层递归的入口状态及其权重。
                config_new = HashableArrayWrapper(  # 拷贝当前配置，避免原地修改,创建一个新的配置
                    np.array(config.array, copy=True, dtype=np.uint8)
                )
                config_new.array[target_c] -= 1  # 从中移除我们刚刚选出的一个 target_c 对应索引里面对应的元素。
                #
                G = dict()  # 中间状态字典G, 跟踪 target_c 与其他元素配对过程中的中间状态。
                G_config = HashableArrayWrapper(np.zeros(c1_type_shape, dtype=np.uint8)) # 初始化辅助配置 G_config，全为0，表示当前没有任何状态更新。
                G[(target_c, G_config)] = Rational(1, 1) # 初始状态字典，权重为1。
                #
                # --- 连接过程 ---
                for other_c in np.argwhere(config_new.array > 0): # 依次将 target 与其余元素other_c连接, other_c 是一个c1type config里面的索引
                    other_c = tuple(other_c.flatten())  # Get cell coordinates
                    G_new = defaultdict(lambda: Rational(0, 1)) # 新的中间状态字典 G_new，用于存储更新后的配对状态。
                    l = config_new.array[other_c]  # 获取other_c 索引对应字典里面的剩余元素个数

                    # --- 内层循环：更新所有可能的配对状态
                    for (target_c, G_config), W in G.items(): # 遍历之前所有的中间状态。
                        H = f(
                            target_c, other_c, l
                        ) # 调用 `f` 函数（即 ConfigUpdater），计算 `target_c`  与 `l` 个 `other_c` 配对的所有可能结果，返回的结果是 H 字典。
                        for (target_c_new, H_config_new) in H.keys(): # 遍历 H 中的每一种可能结果。
                            G_config_new = HashableArrayWrapper(
                                G_config.array + H_config_new.array
                            ) # 将 `other_c` 元素的状态变化 (`H_config_new`) 累加到之前的辅助配置 (`G_config`) 中。
                            G_new[(target_c_new, G_config_new)] += (
                                W * H[(target_c_new, H_config_new)]
                            ) # 将新状态的权重累加到 `G_new` 中。新权重 = 上一步权重 W * 本次配对权重 H[...]
                    G = G_new # 用 `G_new` 更新 `G`，准备与下一种 `other_c` 进行配对。
                # 
                # --- 过滤与递归 ---
                for (target_c, G_config), W in G.items():  # 当 `target_c` 与池中所有元素连接完毕后，`G` 中存储了所有最终状态。
                    if context.stop_condition(
                        target_c
                    ):  # stop_condition 检查 target_c 的状态是否已“满足”（例如，计数量词的要求已达到）。如果满足，说明这次对 target_c 的连接是正确的。
                        T[G_config] += W

                # 递归累加子问题结果
                result_of_target_c = Rational(0, 1)
                for T_config, weight in T.items(): # 对所有有效的下一层入口配置，进行递归调用。
                    W = domain_recursion(T_config) # 递归调用子问题。
                    result_of_target_c += (weight * W) # 将子问题的结果与当前路径的权重相乘，并累加。
                result_of_target_c_list += result_of_target_c
                
            # --- 缓存并返回 ---
            Cache_T[config] = result_of_target_c_list  # 将当前配置 `config` 的最终计算结果存入缓存。
            return result_of_target_c_list

        # ========== 主循环：遍历所有多项式config ==========
        for config in multinomial(n_cells, domain_size): # multinomial生成所有将domain_size个元素分配到 n_cells个类型中的可能方式，即config。
            logger.debug("Config: %s", config)
            # --- 检查一元约束 ---
            if any(
                context.check_unary_constraints(config, unary_mask)
            ): # 返回结果为布尔值，表示是否违反约束。如果当前分区方式不满足任何一个一元约束，则直接跳过，因为它不可能是一个有效的模型。这里相当于预剪枝，是一个优化。满足了一元约束的话，才继续往下计算。
                continue 
            # --- 初始化递归的起始配置 ---
            init_config = np.zeros(
                c1_type_shape, dtype=np.uint8)  # 初始化一个全为0的数组，维度是c1type。即对应所有可能的格子空间，值为0
            W = Rational(1, 1) # 初始化该config的基础权重 W 为1。
            for i, n in enumerate(config):  # 把1type config中的所有元素，根据状态w2t，放到c1 type config的init_config中
                init_config[(i,) + w2t[i]] = n
                W = W * (w[i] ** n)  # 计算这个config的基础权重（不含组合数）。
            init_config = HashableArrayWrapper(init_config)  # 将 NumPy 数组包装成一个可哈希的对象，这样它才能被用作字典的键（用于缓存）。
            # --- 调用递归计算该配置的加权结果 ---
            result_config = domain_recursion(init_config)  # call recursison
            # --- 累加结果 ---
            if context.contain_linear_order_axiom():  # 线性序公理下，元素是可区分的，不需要乘以多项式系数
                WFOMC_result += W * result_config * graph_weight
            else:  # 在非线性序下，元素是不可区分的，需要乘以多项式系数（组合数）来计算排列方式。
                WFOMC_result += MultinomialCoefficients.coef(
                    config) * W * result_config * graph_weight # 累加：多项式系数 × 权重 × 递归结果 × cell graph 权重
    return expand(WFOMC_result)
