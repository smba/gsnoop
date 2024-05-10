#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
from pysat.examples.hitman import Hitman
from sklearn.cluster import KMeans

# Function to check if all constraints are covered by the given hitting set
def constraints_covered(hitting_set, constraints):
    """
    Determine if all constraints are covered by the hitting set.
    
    Args:
        hitting_set (set): Set containing the hitting elements.
        constraints (list of arrays): List of constraint clauses.

    Returns:
        np.ndarray: Boolean array indicating if each clause is covered.
    """
    hitting_set = set(hitting_set)
    covered = np.array([bool(set(clause) & hitting_set) for clause in constraints])
    return covered

# Function to check if a particular clause is covered by the hitting set
def is_clause_covered(clause, hitting_set):
    """
    Determine if a specific clause is covered by the hitting set.
    
    Args:
        clause (array): A specific constraint clause.
        hitting_set (set): Set containing the hitting elements.

    Returns:
        bool: True if the clause is covered, otherwise False.
    """
    return bool(set(clause) & set(hitting_set))

# Function to check all constraints in parallel using multiprocessing
def parallel_constraints_covered(hitting_set, constraints):
    """
    Check if all constraints are covered using multiprocessing.
    
    Args:
        hitting_set (set): Set containing the hitting elements.
        constraints (list of arrays): List of constraint clauses.

    Returns:
        np.ndarray: Boolean array indicating if each clause is covered.
    """
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(is_clause_covered, [(clause, hitting_set) for clause in constraints])
    return np.array(results)

# Function to solve causal constraints using clustering and hitting set solver
def find_hitting_set(x, y):
    """
    Find the minimal hitting set that covers the causal constraints.
    
    Args:
        x (np.ndarray): Binary feature matrix.
        y (np.ndarray): Target values.

    Returns:
        list: Sorted list of elements forming the minimal hitting set.
    """
    # Build constraints from the input data based on the target values
    constraints = [np.where(row == 1)[0] for row, target in zip(x, y) if target > 0.1]
    x_constraints = np.vstack([x[i, :] for i, target in enumerate(y) if target > 0.1])

    # Apply KMeans clustering to identify representative clusters
    kmeans = KMeans(n_clusters=x.shape[1], n_init='auto')
    kmeans.fit(x_constraints)
    centers = (kmeans.cluster_centers_ > 0.5).astype(int)

    # Find indices of centroids that represent the cluster centers
    centroid_indices = [np.argmax([np.sum(np.bitwise_and(center, row)) for row in x_constraints]) for center in centers]

    # Initialize the Hitman hitting set solver
    hittingset_solver = Hitman(solver="g42", htype="rc2", mxs_minz=True)

    # Bootstrap initial hitting set with the selected centroid constraints
    for idx in centroid_indices:
        hittingset_solver.hit(constraints[idx])

    # Iteratively identify and add constraints to achieve a valid hitting set
    while True:
        # Get the current minimal hitting set
        hitset = hittingset_solver.get()
        
        # Check if all constraints are covered by the current hitting set
        covered = parallel_constraints_covered(hitset, constraints)

        if not np.all(covered):
            # Identify constraints not covered by the current hitting set
            uncovered_indices = np.where(covered == 0)[0]
            dist = np.array([sum(i not in clause for i in hitset) for clause in constraints])
            max_dist = np.max(dist[uncovered_indices])
            max_false_indices = np.where(dist[uncovered_indices] == max_dist)[0]
            select = uncovered_indices[max_false_indices]

            # Randomly select an uncovered constraint to add to the hitting set
            selected_index = np.random.choice(select, size=1)
            for i in selected_index:
                hittingset_solver.hit(constraints[i])

        else:
            # Return the sorted hitting set when all constraints are covered
            return sorted(hitset)

