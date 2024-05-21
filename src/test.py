import warnings

warnings.filterwarnings("ignore")

# Import required libraries and modules
import numpy as np
from gsnoop.util import diff_transform, xor_transform
from gsnoop.screening import (
    baseline_screening,
    stepwise_screening,
    stable_screening,
    find_hitting_set,
    find_greedy_hitting_set,
)
from sklearn.preprocessing import StandardScaler

# Set the random seed for reproducibility
np.random.seed(14)

# Define the size of the problem space
n_features = 30
n_configs = 90


# Define a simple performance function to simulate system behavior
# The function models the performance of a system given a configuration of features
def performance_oracle(x):
    return (
        x[0] * x[1] * 123 + x[2] * 45 + x[3] * x[4] * 67 + 0.0002
    )  # + np.random.normal(0, 0.5)


# Generate random configurations and evaluate performance using the defined oracle
x = np.random.choice(2, size=(n_configs, n_features))
y = np.array(list(map(performance_oracle, x)))
scaler = StandardScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

x_diff_transformed, y_diff_transformed = diff_transform(x, y)

# Conduct baseline screening using LASSO
print("baseline")
print(baseline_screening(x, y))
print(baseline_screening(x_diff_transformed, y_diff_transformed))

print("stable")
print(stable_screening(x, y))
print(stable_screening(x_diff_transformed, y_diff_transformed))

print("stepwise")
print(y_diff_transformed)
print(stepwise_screening(x, y))
print(stepwise_screening(x_diff_transformed, y_diff_transformed))

x_xor_transformed2, y_xor_transformed2 = xor_transform(x, y)
x_xor_transformed2 = np.vstack(
    [
        x_xor_transformed2[i, :]
        for i in range(x_xor_transformed2.shape[0])
        if y_xor_transformed2[i] != 0
    ]
)

print("MHS solving")
causal_options = find_greedy_hitting_set(x_xor_transformed2)
print(sorted(causal_options))

