import warnings
warnings.filterwarnings('ignore')

# Import required libraries and modules
import numpy as np
from gsnoop.util import diff_transform, xor_transform
from gsnoop.screening import baseline_screening, stepwise_screening, stable_screening
from gsnoop.causal import find_hitting_set, find_greedy_hitting_set
import time

# Set the random seed for reproducibility
np.random.seed(14)

# Define the size of the problem space
n_features = 100
n_configs = 50


# Define a simple performance function to simulate system behavior
# The function models the performance of a system given a configuration of features
def performance_oracle(x):
    return (
        x[0] * x[1] * 123 + x[2] * 45 + x[3] * x[4] *67 + 0.0002
    )  # + np.random.normal(0, 0.5)


# Generate random configurations and evaluate performance using the defined oracle
x = np.random.choice(2, size=(n_configs, n_features))
y = np.array(list(map(performance_oracle, x)))
x_diff_transformed, y_diff_transformed = diff_transform(x, y)

# Conduct baseline screening using LASSO
print('baseline')
print(baseline_screening(x, y))
print(baseline_screening(x_diff_transformed, y_diff_transformed))

print('stable')
print(stable_screening(x, y))
print(stable_screening(x_diff_transformed, y_diff_transformed))

print('stepwise')
threshold = 0.5
print(stepwise_screening(x, y, threshold))
print(stepwise_screening(x_diff_transformed, y_diff_transformed, threshold))

x_xor_transformed, y_xor_transformed = xor_transform(x, y)
x_xor_transformed = np.vstack(
    [
        x_xor_transformed[i, :]
        for i in range(x_xor_transformed.shape[0])
        if y_xor_transformed[i] != 0
    ]
)

print('MHS solving')
causal_options = find_greedy_hitting_set(x_xor_transformed)
print(sorted(causal_options))

