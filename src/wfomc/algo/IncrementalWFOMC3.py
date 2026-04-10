import numpy as np
from collections import defaultdict
from loguru import logger

from wfomc.cell_graph import build_cell_graphs
from wfomc.context import IncrementalWFOMC3Context, ConfigUpdater, HashableArrayWrapper
from wfomc.fol import Const, Pred
from wfomc.utils import multinomial, MultinomialCoefficients, Rational, expand, RingElement


def incremental_wfomc3(context: IncrementalWFOMC3Context) -> RingElement:
    # Extract the domain, weight function, linear order predicate and formula from the context.
    domain: set[Const] = context.domain # Obtain the domain of the problem
    get_weight = context.get_weight # Obtain the predicate weight function
    leq_pred: Pred = context.leq_pred # Obtain the linear order predicate (if any)
    formula = context.uni_formula # Obtain the first-order logic formula
    c_type_shape = context.c_type_shape # Obtain the shape information of c type
    WFOMC_result = Rational(0, 1)  # Initialize the final result WFOMC_result to 0.
    domain_size = len(domain)  # Obtain the size of the domain
    # --- Iterate over all cell graphs
    MultinomialCoefficients.setup(domain_size) # Initialize the multinomial coefficients calculator, precompute factorials, etc., for fast combination calculations later.
    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight, leq_pred):
        cells = cell_graph.get_cells()  # Obtain all cells in the current graph and their count
        n_cells = len(cells)  # Number of 1-type cells
        w2t, w, r = context.build_weight(
            cells, cell_graph
        )  # w2t: dictionary from cell index to predicate state dictionary, values represent required counts. w: weight dictionary for each cell type. r: relationship dictionary between cell pairs.
        logger.debug("Weight mapping w2t: %s", w2t)
        logger.debug("Weight w: %s", w)
        # logger.debug("Weight r: %s", r)
        # Handle unary constraints
        # Build masks for unary constraints (e.g., ∃=k X: P(X)) for fast checking later.
        unary_mask = context.build_unary_mask(cells)
        t_update_dict = context.build_t_update_dict(
            r, n_cells
        )  # Build t_update_dict to precompute how states update when two elements are paired. This acts as a large state transition lookup table.
        c1_type_shape = (n_cells,) + tuple(
            c_type_shape
        )  # Dimension of c1 type: (number of cells) × (extended predicate dimension) × (count constraint dimension)
        Cache_H = dict()   # Cache internal computation results of ConfigUpdater.
        config_updater = ConfigUpdater(
            t_update_dict, c1_type_shape, Cache_H)  # f
        f = config_updater.f  # this is a function
        Cache_T = dict()  # Cache recursive computation results to avoid redundant subproblems
        # ========== Define the core recursive function ==========
        def domain_recursion(config):
            if config in Cache_T:  # If the result for the current configuration has already been computed, return it directly from the cache.
                return Cache_T[config]

            # --- Base case ---
            if config.array.sum() == 0: # If all elements in config have been processed (sum=0), base case, return 1
                return Rational(1, 1)

            # --- Recursive step ---
            result_of_target_c_list = Rational(0, 1)
            # --- Select a target element target_c, which is an index in the config dictionary, to be paired with all other remaining elements.
            if context.contain_linear_order_axiom(): # If the problem contains a linear order, we need to iterate over all possible elements as targets because they are no longer symmetric.
                target_c_list = list(tuple(i) for i in np.argwhere(config.array > 0))
            else:  # Otherwise, due to symmetry, we only need to select the last non-zero type element as the target.
                target_c_list = list((tuple(np.argwhere(config.array > 0)[-1]),))

            # --- Iterate over all selected targets ---
            for target_c in target_c_list:  # Iterate over candidate c1_type index list
                T = defaultdict(lambda: Rational(0, 1)) # Create a dictionary to store entry states and their weights for the next recursion level.
                config_new = HashableArrayWrapper(  # Copy the current configuration to avoid in-place modification, creating a new configuration
                    np.array(config.array, copy=True, dtype=np.uint8)
                )
                config_new.array[target_c] -= 1  # Remove the element corresponding to the selected target_c index from the configuration.
                #
                G = dict()  # Intermediate state dictionary G, tracking intermediate states during pairing of target_c with other elements.
                G_config = HashableArrayWrapper(np.zeros(c1_type_shape, dtype=np.uint8)) # Initialize auxiliary configuration G_config, all zeros, indicating no state updates currently.
                G[(target_c, G_config)] = Rational(1, 1) # Initial state dictionary, weight is 1.
                #
                # --- Connection process ---
                for other_c in np.argwhere(config_new.array > 0): # Sequentially connect target with other elements other_c, other_c is an index in c1type config
                    other_c = tuple(other_c.flatten())  # Get cell coordinates
                    G_new = defaultdict(lambda: Rational(0, 1)) # New intermediate state dictionary G_new, used to store updated pairing states.
                    l = config_new.array[other_c]  # Get the number of remaining elements corresponding to the other_c index in the dictionary.

                    # --- Inner loop: update all possible pairing states
                    for (target_c, G_config), W in G.items(): # Iterate over all previous intermediate states.
                        H = f(
                            target_c, other_c, l
                        ) # Call the `f` function (i.e., ConfigUpdater) to compute all possible results of pairing `target_c` with `l` `other_c` elements, returning a dictionary H.
                        for (target_c_new, H_config_new) in H.keys(): # Iterate over each possible result in H.
                            G_config_new = HashableArrayWrapper(
                                G_config.array + H_config_new.array
                            ) # Accumulate the state changes of `other_c` elements (`H_config_new`) to the previous auxiliary configuration (`G_config`).
                            G_new[(target_c_new, G_config_new)] += (
                                W * H[(target_c_new, H_config_new)]
                            ) # Accumulate the weight of the new state to `G_new`. New weight = previous weight W * current pairing weight H[...]
                    G = G_new # Update `G` with `G_new`, preparing to pair with the next `other_c`.
                #
                # --- Filtering and recursion ---
                for (target_c, G_config), W in G.items():  # After `target_c` has been connected with all elements in the pool, `G` stores all final states.
                    if context.stop_condition(
                        target_c
                    ):  # stop_condition checks whether the state of target_c has been "satisfied" (e.g., the requirements of counting quantifiers have been met). If satisfied, it means this connection of target_c is correct.
                        T[G_config] += W

                # Recursively accumulate subproblem results
                result_of_target_c = Rational(0, 1)
                for T_config, weight in T.items(): # Recursively call for all valid next-level entry configurations.
                    W = domain_recursion(T_config) # Recursively call subproblems.
                    result_of_target_c += (weight * W) # Multiply the subproblem result by the current path weight and accumulate.
                result_of_target_c_list += result_of_target_c

            # --- Cache and return ---
            Cache_T[config] = result_of_target_c_list  # Store the final computed result of the current configuration `config` in the cache.
            return result_of_target_c_list

        # ========== Main loop: iterate over all polynomial configs ==========
        for config in multinomial(n_cells, domain_size): # multinomial generates all possible ways to distribute domain_size elements into n_cells types, i.e., config.
            logger.debug("Config: %s", config)
            # --- Check the unary constraint ---
            if any(
                context.check_unary_constraints(config, unary_mask)
            ): # Returns a boolean indicating whether any unary constraint is violated. If the current partition does not satisfy any unary constraint, it is skipped as it cannot be a valid model. This acts as a form of pruning optimization. Only if unary constraints are satisfied does the computation proceed.
                continue
            # --- Initialize the starting configuration for recursion ---
            init_config = np.zeros(
                c1_type_shape, dtype=np.uint8)  # Initialize an array of zeros with the shape of c1_type. This corresponds to all possible grid spaces, with values set to 0.
            W = Rational(1, 1) # Initialize the base weight W for this config as 1.
            for i, n in enumerate(config):  # Place all elements from the 1-type config into the c1-type config's init_config according to the state w2t.
                init_config[(i,) + w2t[i]] = n
                W = W * (w[i] ** n)  # Calculate the base weight for this config (excluding combinatorial factors).
            init_config = HashableArrayWrapper(init_config)  # Wrap the NumPy array into a hashable object so it can be used as a dictionary key (for caching).
            # --- Recursively compute the weighted result for this configuration ---
            result_config = domain_recursion(init_config)  # call recursion
            # --- Accumulate results ---
            if context.contain_linear_order_axiom():  # Under the linear order axiom, elements are distinguishable, so there is no need to multiply by polynomial coefficients
                WFOMC_result += W * result_config * graph_weight
            else:  # Under non-linear order, elements are indistinguishable, so polynomial coefficients (combinatorial numbers) are needed to calculate permutations.
                WFOMC_result += MultinomialCoefficients.coef(
                    config) * W * result_config * graph_weight # Accumulate: polynomial coefficient × weight × recursive result × cell graph weight
    return expand(WFOMC_result)
