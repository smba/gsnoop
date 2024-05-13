#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from joblib import Parallel, delayed
from typing import List
from sklearn.linear_model import SGDRegressor

def fit_lasso_model(alpha: float, x: np.ndarray, y: np.ndarray) -> SGDRegressor:
    """
    Fits an L1-regularized (Lasso) model to the data and returns the trained model.

    Parameters:
    - alpha (float): Regularization parameter.
    - x (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.

    Returns:
    - SGDRegressor: The fitted model.
    """
    model = SGDRegressor(penalty="l1", alpha=alpha, random_state=42, max_iter=5000)
    model.fit(x, y)
    return model

def find_alpha_limit(
    x: np.ndarray,
    y: np.ndarray,
    screening_split: int = 5,
    tolerance: float = 1e-5
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
    upper_alpha = 20

    while True:
        # Generate alpha values within the current range
        screening_alphas = np.linspace(lower_alpha, upper_alpha, screening_split)

        # Fit models for each alpha value
        models = Parallel(n_jobs=-1)(delayed(fit_lasso_model)(a, x, y) for a in screening_alphas)

        # Count non-zero coefficients for each model
        counts = [np.sum(m.coef_ != 0) for m in models]

        stepsize = np.abs(screening_alphas[0] - screening_alphas[1])
        
        if all(counts):
            upper_alpha *= (1 + 0.1)

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


def lasso_screening(
    x: np.ndarray,
    y: np.ndarray,
    n_simulations: int = 100
) -> List[int]:
    """
    Performs feature screening using a stepwise Lasso approach with SGDRegressor.

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

    # Stack coefficients and calculate feature rankings
    coefs = np.vstack([m.coef_ for m in models])
    coef_sums = np.sum(np.abs(coefs), axis=0)
    ranking = np.argsort(coef_sums)[::-1]

    # Find the "best" alpha: one with the least sensitive count
    mean_count = np.mean(counts)
    count_diff = np.abs(counts - mean_count)
    best_alpha_idx = np.argmin(count_diff)

    # Compute used features for the "best" alpha
    best_features = np.where(models[best_alpha_idx].coef_ != 0)[0]
    best_features_sorted = sorted(best_features)

    return list(ranking[:int(mean_count)])

def group_screening(
    x: np.ndarray,
    y: np.ndarray,
    r2_threshold: float = 0.1
) -> List[int]:
    """
    Identifies important features stepwise until the R² score drops below the threshold.

    Parameters:
    - x (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - r2_threshold (float): Minimum R² score threshold to stop the feature selection.

    Returns:
    - List[int]: Indices of the most important features in descending order.
    """
    options = []  # Stores indices of the most important features
    score = 1.0  # Initialize with a high score for the first iteration

    while score >= r2_threshold:
        # Train linear model using stochastic gradient descent
        model = SGDRegressor(penalty=None, random_state=42)
        model.fit(x, y)

        # Get feature importance coefficients and rank features
        coefs = model.coef_
        importances = np.argsort(np.abs(coefs))[::-1]
        opt = importances[0]  # Most important feature

        # Calculate the R² score of the model
        score = model.score(x, y)

        # Adjust the target values to remove the influence of the most important feature
        mask_negative = np.where(x[:, opt] == -1)[0]
        mask_positive = np.where(x[:, opt] == 1)[0]
        y[mask_negative] += coefs[opt]
        y[mask_positive] -= coefs[opt]

        # Remove the most important feature from consideration in subsequent iterations
        x[:, opt] = 0
        options.append(opt)

    return list(options)

if __name__ == '__main__':
    pass
