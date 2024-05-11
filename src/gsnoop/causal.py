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
