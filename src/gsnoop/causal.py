#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pulp
import numpy as np
from typing import List, Set, Dict

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



def find_greedy_hitting_set(x: List[np.ndarray], y: List[float], threshold: float) -> Set[int]:
    """
    Implements Hochbaum's greedy approximation algorithm for the hitting set problem.

    Parameters:
        Set (List[Set[int]]): A list of sets containing integers, representing the collection
                              of subsets from which we are trying to find a minimal hitting set.

    Returns:
        Set[int]: A set of integers representing the minimal hitting set, i.e., a set of elements
                  such that each subset in the original list has at least one element in common
                  with this hitting set.
    """

    Set = [set(np.where(row == 1)[0]) for row, target in zip(x, y) if target > threshold]

    # Initialize the hitting set as empty
    hitting_set: Set[int] = set()

    # While there are still sets left
    while Set:
        # Count the frequency of each element in the sets
        frequency: Dict[int, int] = {}
        for subset in Set:
            for element in subset:
                if element not in frequency:
                    frequency[element] = 0
                frequency[element] += 1

        # Find the element with the highest frequency
        max_freq_element: int = max(frequency, key=frequency.get)

        # Add the element with the highest frequency to the hitting set
        hitting_set.add(max_freq_element)

        # Remove all sets hit by the element with the highest frequency
        Set = [subset for subset in Set if max_freq_element not in subset]

    return list(hitting_set)

if __name__ == '__main__':
    pass

