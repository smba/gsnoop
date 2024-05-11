#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pulp
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

import numpy as np
from typing import List, Set

def find_greedy_hitting_set(x: List[np.ndarray], y: List[float]) -> List[int]:
    """
    Finds a hitting set for the given sets using a greedy algorithm.

    Args:
        x: A list of numpy arrays, each representing a row of binary values indicating set membership.
        y: A list of floats representing target values which dictate if the set should be considered.

    Returns:
        A set representing the hitting set.
    """
    # Initialize the sets to hit based on y values greater than a threshold (e.g., 0.1)
    sets_to_hit = {i: set(np.where(row == 1)[0]) for i, (row, target) in enumerate(zip(x, y)) if target > 0.1}

    # Create the universe of all elements and initialize the covered sets tracker
    universe = set().union(*sets_to_hit.values())
    covered_sets = set()
    
    # Initialize the hitting set
    hitting_set = set()

    # While there are uncovered sets
    while covered_sets != sets_to_hit.keys():
        # Find the element that is in the maximum number of uncovered sets
        coverage_count = {element: 0 for element in universe}
        for set_id, elements in sets_to_hit.items():
            if set_id not in covered_sets:
                for element in elements:
                    coverage_count[element] += 1

        # Select the element with the maximum coverage
        max_element = max(coverage_count, key=coverage_count.get)
        hitting_set.add(max_element)

        # Update the set of covered sets
        for set_id, elements in sets_to_hit.items():
            if max_element in elements:
                covered_sets.add(set_id)

    return list(hitting_set)
