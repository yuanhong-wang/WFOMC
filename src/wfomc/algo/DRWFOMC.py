import numpy as np
from collections import Counter, defaultdict
from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.context.dr_context import DRWFOMCContext, ConfigUpdater, HashableArrayWrapper
from wfomc.fol import Const, Pred
from wfomc.utils import multinomial, MultinomialCoefficients
from wfomc.utils.polynomial_flint import Rational, coeff_dict, expand, RingElement


def domain_recursive_wfomc(context: DRWFOMCContext) -> RingElement:
    # Extract the domain, weight function, linear order predicate and formula from the context.
    domain: set[Const] = context.domain
    get_weight = context.get_weight
    leq_pred: Pred = context.leq_pred
    formula = context.uni_formula
    c_type_shape = context.c_type_shape
    result = Rational(0, 1)
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)
    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight, leq_pred):
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        w2t, w, r = context.build_weight(
            cells, cell_graph
        )
        unary_mask = context.build_unary_mask(cells)
        t_updates = context.build_t_updates(
            r, n_cells, domain_size
        )
        shape = (n_cells,) + tuple(
            c_type_shape
        )
        Cache_F = dict()
        config_updater = ConfigUpdater(
            t_updates, shape, Cache_F)  # update_config
        update_config = config_updater.update_config  # this is a function
        Cache_T = dict()

        def domain_recursion(config):
            if config in Cache_T:
                return Cache_T[config]

            if config.array.sum() == 0:
                return Rational(1, 1)

            final_res = Rational(0, 1)
            if (
                context.contain_linear_order_axiom()
            ):
                target_cs = list(tuple(i)
                                 for i in np.argwhere(config.array > 0))
            else:
                target_cs = list((tuple(np.argwhere(config.array > 0)[-1]),))

            for target_c in target_cs:
                T = defaultdict(lambda: Rational(0, 1))
                new_config = HashableArrayWrapper(
                    np.array(config.array, copy=True, dtype=np.uint8)
                )
                new_config.array[target_c] -= 1
                F = dict()
                u_config = np.zeros(shape, dtype=np.uint8)
                u_config = HashableArrayWrapper(u_config)
                F[(target_c, u_config)] = Rational(1, 1)

                for other_c in np.argwhere(new_config.array > 0):
                    other_c = tuple(other_c.flatten())  # Get cell coordinates
                    F_new = defaultdict(lambda: Rational(0, 1))
                    l = new_config.array[other_c]

                    for (target_c, u_config), W in F.items():
                        F_update = update_config(
                            target_c, other_c, l
                        )
                        for (
                            target_c_new,
                            u_config_update,
                        ) in F_update.keys():
                            F_config_new = HashableArrayWrapper(
                                u_config.array + u_config_update.array
                            )
                            F_new[(target_c_new, F_config_new)] += (
                                W * F_update[(target_c_new, u_config_update)]
                            )
                    F = F_new

                for (
                    last_target_c,
                    last_F_config,
                ), W in (
                    F.items()
                ):  # Among alAl the pairing methods, only retain the portion where the target elements have fully met the constraints, and merge their weights into the configuration T of the next layer of recursion.
                    if context.stop_condition(
                        last_target_c
                    ):  # Check the predicate status. Filter out valid results: The predicate status of target_c must be all zeros.
                        T[last_F_config] += W

                res = Rational(0, 1)
                for recursive_config, weight in T.items():
                    W = domain_recursion(
                        recursive_config
                    )  # Recursive call to subconfiguration
                    res = res + (weight * W)
                final_res += res
            Cache_T[config] = final_res
            return final_res

        for config in multinomial(n_cells, domain_size):
            if any(
                context.check_unary_constraints(config, unary_mask)
            ):
                continue

            init_config = np.zeros(shape, dtype=np.uint8)
            W = Rational(1, 1)
            for i, n in enumerate(config):
                init_config[(i,) + w2t[i]] = (
                    n  # i is cell index, n is element num of cell i
                )
                W = W * (w[i] ** n)
            init_config = HashableArrayWrapper(init_config)
            dr_res = domain_recursion(init_config)  # call recursison

            if (
                context.contain_linear_order_axiom()
            ):
                result += W * dr_res * graph_weight
            else:
                result += (
                    MultinomialCoefficients.coef(
                        config) * W * dr_res * graph_weight
                )
    return result
