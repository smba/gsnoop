#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

def diff_transform_x(x: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise differences between rows of the input array x.

    Args:
        x (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Array containing pairwise differences.
    """
    return np.vstack([x[i, :] - x[j, :] for i, j in itertools.combinations(range(x.shape[0]), 2)])


def diff_transform_y(y: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise differences between elements of the input array y.

    Args:
        y (np.ndarray): Input 1D array.

    Returns:
        np.ndarray: Array containing pairwise differences.
    """
    return np.array([y[i] - y[j] for i, j in itertools.combinations(range(y.shape[0]), 2)])


def diff_transform(x: np.ndarray, y: np.ndarray, scaler: StandardScaler = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform both feature and target arrays using pairwise differences.

    Args:
        x (np.ndarray): Input 2D feature array.
        y (np.ndarray): Input 1D target array.
        scaler (StandardScaler): Optional standard scaler instance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed feature and target arrays.
    """
    x_ = diff_transform_x(x)
    y_ = diff_transform_y(y)
    if scaler is None:
        scaler = StandardScaler()
    y_ = scaler.fit_transform(y_.reshape(-1, 1)).ravel()
    return x_, y_

def xor_transform_x(x: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise differences between rows of the input array x.

    Args:
        x (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Array containing pairwise differences.
    """
    np.vstack([
        np.bitwise_xor(x[i, :], x[j, :])
        for i, j in itertools.combinations(range(x.shape[0]), 2)
    ])

def xor_transform_y(y: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise differences between elements of the input array y.

    Args:
        y (np.ndarray): Input 1D array.

    Returns:
        np.ndarray: Array containing pairwise differences.
    """
    return np.abs(diff_transform_y(y))

def xor_transform(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise XOR operations between rows of the input array x.

    Args:
        x (np.ndarray): Input 2D array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed feature and target arrays.
    """
    return xor_transform_x(x), xor_transform_y(y)


def precision(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the precision metric.

    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[int]): List of predicted labels.

    Returns:
        float: Precision score.
    """
    return len(set(y_true).intersection(set(y_pred))) / max(1, len(set(y_pred)))


def recall(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the recall metric.

    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[int]): List of predicted labels.

    Returns:
        float: Recall score.
    """
    return len(set(y_true).intersection(set(y_pred))) / max(1, len(set(y_true)))


def jaccard(y1: List[int], y2: List[int]) -> float:
    """
    Calculate the Jaccard similarity coefficient.

    Args:
        y1 (List[int]): First list of labels.
        y2 (List[int]): Second list of labels.

    Returns:
        float: Jaccard similarity score.
    """
    return len(set(y1).intersection(set(y2))) / max(1, len(set(y1).union(set(y2))))


def f1(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the F1 score.

    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[int]): List of predicted labels.

    Returns:
        float: F1 score.
    """
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return (2 * p * r) / max(1, (p + r))
