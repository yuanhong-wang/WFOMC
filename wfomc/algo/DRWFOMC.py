import numpy as np

from collections import Counter, defaultdict
from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.context.dr_context import DRWFOMCContext, ConfigUpdater, HashableArrayWrapper
from wfomc.fol.syntax import AUXILIARY_PRED_NAME, X, Y, AtomicFormula, Const, Pred, QFFormula, top, a, b
from wfomc.utils import multinomial, MultinomialCoefficients
from wfomc.utils.polynomial import Rational, coeff_dict, expand
from wfomc.utils.third_typing import RingElement


def domain_recursive_wfomc(context: DRWFOMCContext) -> RingElement:
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
        w2t, w, r = context.build_weight(cells, cell_graph)  # build_weight
        t_updates = context.bulid_wij(r, n_cells, domain_size)  # t_updates
        # shape = (n_cells,) + tuple(2 for _ in ext_preds) + tuple(k + 1 for k in cnt_params)
        shape = (n_cells,) + tuple(c_type_shape)

        Cache_F = dict()  # Global Cache_F
        config_updater = ConfigUpdater(t_updates, shape, Cache_F)  # update_config
        update_config = config_updater.update_config  # this is a function

        Cache_T = dict()  # key = (config)

        # core of recursion
        def domain_recursion(config):
            if config in Cache_T:
                return Cache_T[config]

            if config.array.sum() == 0:  # All elements have been consumed
                return Rational(1, 1)

            T = defaultdict(lambda: Rational(0, 1))
            new_config = HashableArrayWrapper(  # Create a copy of current configuration (to avoid modifying original)
                np.array(config.array, copy=True, dtype=np.uint8)
            )
            # select element
            target_c = tuple(np.argwhere(new_config.array > 0)[-1])  # Select last non-zero cell as target (processing order strategy)
            new_config.array[target_c] -= 1  # Remove this element
            F = dict()
            u_config = np.zeros(shape, dtype=np.uint8)
            u_config = HashableArrayWrapper(u_config)
            F[(target_c, u_config)] = Rational(1, 1)

            # Pair with "other elements" in sequence (outer loop)
            for other_c in np.argwhere(new_config.array > 0):
                other_c = tuple(other_c.flatten())  # Get cell coordinates
                F_new = defaultdict(lambda: Rational(0, 1))
                l = new_config.array[other_c]  # How many elements of this type remain
                # Expand F by calling update_config (inner core)
                for (target_c, u_config), W in F.items():
                    F_update = update_config(target_c, other_c, l)  # Calculate all possible results after target_c interacts with other_c for num times
                    for target_c_new, u_config_update in F_update.keys():  # Merge transition results
                        F_config_new = HashableArrayWrapper(
                            u_config.array + u_config_update.array
                        )
                        F_new[(target_c_new, F_config_new)] += W * F_update[(target_c_new, u_config_update)]
                F = F_new  # Update current transition path set

            # # Filter "target satisfied" terminal states → T
            for (last_target_c,
                 last_F_config), W in F.items():  # Among all the pairing methods, only retain the portion where the target elements have fully met the constraints, and merge their weights into the configuration T of the next layer of recursion.
                if all(i == 0 for i in last_target_c[1:]):  # # Check the predicate status. Filter out valid results: The predicate status of target_c must be all zeros.
                    T[last_F_config] += W

            # Recursively calculate weighted sum of all subproblems
            ret = Rational(0, 1)
            for recursive_config, weight in T.items():
                W = domain_recursion(recursive_config)  # Recursive call to subconfiguration
                ret = ret + (weight * W)
            Cache_T[config] = ret  # Cache current configuration result
            return ret

        # main code
        for config in multinomial(n_cells, domain_size):
            init_config = np.zeros(shape, dtype=np.uint8)
            W = Rational(1, 1)
            for i, n in enumerate(config):
                init_config[(i,) + w2t[i]] = n  # i is cell index, n is element num of cell i
                W = W * (w[i] ** n)
            init_config = HashableArrayWrapper(init_config)
            dr_res = domain_recursion(init_config)  # call recursison
            result += (MultinomialCoefficients.coef(config) * W * dr_res * graph_weight)
    return result
