#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pulp
import bitarray
import numpy as np
from typing import List, Set


# Function to solve a hitting set problem instance
def find_hitting_set(x: List[np.ndarray], y: List[float]) -> List[int]:
    """
    Finds a minimal hitting set for the given sets using PuLP.

    Args:
        x: A list of numpy arrays, where each array represents a row of binary values.
        y: A list of floats representing target values.

    Returns:
        A list representing the minimal hitting set.
    """
    # Build constraints from the input data based on the target values
    sets_to_hit = [set(np.where(row == 1)[0]) for row, target in zip(x, y) if target > 0.1]

    # Create the universe of elements
    universe = set.union(*sets_to_hit)

    # Create the LP problem
    prob = pulp.LpProblem("HittingSet", pulp.LpMinimize)

    # Create binary decision variables for each element in the universe
    variables = pulp.LpVariable.dicts("x", universe, cat=pulp.LpBinary)

    # Objective function: Minimize the number of elements in the hitting set
    prob += pulp.lpSum(variables[element] for element in universe)

    # Constraint: Each set must be covered by at least one element in the hitting set
    for i, set_i in enumerate(sets_to_hit):
        prob += pulp.lpSum(variables[element] for element in set_i) >= 1, f"Cover_{i}"

    # Solve the LP problem
    prob.solve()

    # Extract the minimal hitting set
    result = [element for element, var in variables.items() if var.value() > 0.5]

    return result


def find_greedy_hitting_set(bit_vectors):
    universe_size = len(bit_vectors[0])
    hitting_set = bitarray(universe_size)
    hitting_set.setall(0)

    covered_sets = bitarray(len(bit_vectors))
    covered_sets.setall(0)

    while covered_sets.count(0) > 0:
        # Select the element that appears in the maximum number of uncovered sets
        element_coverage = [0] * universe_size
        for i, bv in enumerate(bit_vectors):
            if not covered_sets[i]:
                for j in range(universe_size):
                    if bv[j]:
                        element_coverage[j] += 1
        
        max_element = element_coverage.index(max(element_coverage))
        
        # Add this element to the hitting set
        hitting_set[max_element] = 1
        
        # Update covered sets
        for i, bv in enumerate(bit_vectors):
            if bv[max_element]:
                covered_sets[i] = 1

    return hitting_set