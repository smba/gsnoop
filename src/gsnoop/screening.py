#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from joblib import Parallel, delayed
from typing import List, Set, Dict
from sklearn.linear_model import SGDRegressor

import pulp
import itertools

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


def fit_lasso_model(alpha: float, x: np.ndarray, y: np.ndarray) -> SGDRegressor:
    """
    Fits an L1-regularized (SGDRegressor) model to the data and returns the trained model.

    Parameters:
    - alpha (float): Regularization parameter.
    - x (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.

    Returns:
    - SGDRegressor: The fitted model.
    """
    model = SGDRegressor(penalty="l1", alpha=alpha, random_state=1, max_iter=5000)
    model.fit(x, y)
    return model


def find_alpha_limit(
    x: np.ndarray, y: np.ndarray, screening_split: int = 5, tolerance: float = 1e-5
) -> float:
    """
    Identifies the smallest alpha value that results in zero features being selected.

    Parameters:
    - x (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - screening_split (int): Number of alpha values to test in the initial split.

    Returns:
    - float: The smallest alpha value that results in zero features being selected.
    """
    # Initial range of alpha values
    lower_alpha = 0
    upper_alpha = 10

    while True:
        # Generate alpha values within the current range
        screening_alphas = np.linspace(lower_alpha, upper_alpha, screening_split)

        # Fit models for each alpha value
        models = Parallel(n_jobs=-1)(
            delayed(fit_lasso_model)(a, x, y) for a in screening_alphas
        )

        # Count non-zero coefficients for each model
        counts = [np.sum(m.coef_ != 0) for m in models]

        stepsize = np.abs(screening_alphas[0] - screening_alphas[1])

        if all(counts):
            upper_alpha *= 1 + 0.1

        # sweet spot
        elif any(counts) and not all(counts):
            # Find the index of the first model with zero non-zero coefficients
            zero_count_idx = counts.index(0)

            # Update the alpha range based on the index of zero features
            lower_alpha = screening_alphas[zero_count_idx] - stepsize
            upper_alpha = screening_alphas[zero_count_idx] + stepsize

            if abs(lower_alpha - upper_alpha) < tolerance:
                return screening_alphas[zero_count_idx]

        # only zeros, decrease lower_alpha
        else:
            lower_alpha = max(lower_alpha * (1 - 0.1), 0)


def baseline_screening(
    x: np.ndarray, y: np.ndarray, n_simulations: int = 100
) -> List[int]:
    """
    Performs feature screening using Lasso Regression (using SGDRegressor).

    Parameters:
    - x (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - n_simulations (int): Number of simulations for hyperparameter optimization.

    Returns:
    - List[int]: Ranked list of most important feature indices.
    """
    params = {
        "alpha": np.linspace(0, 10, 1000),
    }
    search = HalvingGridSearchCV(SGDRegressor(penalty="l1", max_iter=5000), params)
    search.fit(x, y)

    model = search.best_estimator_
    return list(sorted(np.where(model.coef_ != 0)[0]))


def stable_screening(
    x: np.ndarray, y: np.ndarray, n_simulations: int = 100
) -> List[int]:
    """

    Parameters:
    - x (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - n_simulations (int): Number of simulations for hyperparameter optimization.

    Returns:
    - List[int]: Ranked list of most important feature indices.
    """

    # Determine the smallest alpha value that results in zero features being selected
    alpha_limit = find_alpha_limit(x, y)

    # Generate random alphas for hyperparameter optimization
    alphas = alpha_limit * np.random.random(size=n_simulations)

    # Train models using these alphas and count non-zero coefficients
    models = Parallel(n_jobs=-1)(delayed(fit_lasso_model)(a, x, y) for a in alphas)
    counts = np.array([np.sum(m.coef_ != 0) for m in models])

    unique, counts = np.unique(counts, return_counts=True)
    frequencies = dict(zip(unique, counts))

    most_stable_size = max(frequencies, key=frequencies.get)
    # get one of those models

    idx = np.where(unique == most_stable_size)[0][0]
    model = models[idx]
    options = np.where(model.coef_ != 0)[0]

    return list(sorted(options))


def stepwise_screening(
    x: np.ndarray,
    y: np.ndarray,
    std_tolerances: List[float] = [5e-2, 2.5e-2, 1e-2, 5e-3],
) -> List[List[int]]:
    """
    Identifies important features stepwise until the R² score drops below the threshold.

    Parameters:
    - x (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - std_tolerances (float): Minimum threshold.

    Returns:
    - List[List[int]]: Indices of the most important features in descending order.
    """

    x = np.copy(x)
    y = np.copy(y)

    options = [
        [] for tol in std_tolerances
    ]  # Stores indices of the most important features
    std = np.std(y)  # = 1.0 since we receive standardized data
    params = {
        "alpha": np.linspace(0, 1, 100),
    }

    for _ in range(x.shape[1]):

        # Train linear model using stochastic gradient descent, hyperparameter optimization for R2
        search = HalvingGridSearchCV(SGDRegressor(penalty="l2", max_iter=5000), params)
        search.fit(x, y)
        model = search.best_estimator_

        # Get feature importance coefficients and rank features
        coefs = model.coef_

        importances = np.argsort(np.abs(coefs))[::-1]

        # Most important feature
        opt = None
        seen = []
        for o in importances:
            if o not in seen:
                opt = o
                break

        # Adjust the target values to remove the influence of the most important feature
        mask_negative = np.where(x[:, opt] == -1)[0]
        mask_positive = np.where(x[:, opt] == 1)[0]
        y[mask_negative] += coefs[opt]
        y[mask_positive] -= coefs[opt]

        # Remove the most important feature from consideration in subsequent iterations
        x[:, opt] = 0

        new_std = np.std(y)

        loss = std - new_std
        seen.append(opt)
        for k, tol in enumerate(std_tolerances):
            if loss >= tol:
                options[k].append(opt)

        if loss >= np.min(std_tolerances):
            std = new_std
        else:
            break

    return options


# Function to solve a hitting set problem instance
def find_hitting_set(x: List[np.ndarray]) -> List[int]:
    """
    Finds a minimal hitting set for the given sets using PuLP.

    Args:
        x: A list of numpy arrays, where each array represents a row of binary values.
        y: A list of floats representing target values.

    Returns:
        A list representing the minimal hitting set.
    """
    # Build constraints from the input data based on the target values
    sets_to_hit = [set(np.where(row == 1)[0]) for row in x]

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


def find_greedy_hitting_set(x: List[np.ndarray]) -> Set[int]:
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

    Set = [set(np.where(row == 1)[0]) for row in x]

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


if __name__ == "__main__":
    pass
