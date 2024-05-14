# Import required libraries and modules
import numpy as np
from gsnoop.util import diff_transform, xor_transform
from gsnoop.screening import group_screening, lasso_screening
from gsnoop.causal import find_hitting_set, find_greedy_hitting_set
import time

# Set the random seed for reproducibility
np.random.seed(1)

# Define the size of the problem space
n_features = 150
n_configs = 150

# Define a simple performance function to simulate system behavior
# The function models the performance of a system given a configuration of features
def performance_oracle(x):
    return x[0] * x[1] * 123 + x[3] * 45 + x[4] * x[5] * x[6] * 67 + 0.0001#+ np.random.normal(0, 0.5)

# Generate random configurations and evaluate performance using the defined oracle
x = np.random.choice(2, size=(n_configs, n_features))
y = np.array(list(map(performance_oracle, x)))

# Conduct baseline screening using LASSO
lasso_options = lasso_screening(x, y)
print(lasso_options)
print("Lasso Screening Results:", lasso_options)

# Perform group screening using the difference transformation
x_diff_transformed, y_diff_transformed = diff_transform(x, y)
group_options = group_screening(x_diff_transformed, y_diff_transformed)
print("Group Screening Results:", group_options)
print(group_options)
# Perform causal analysis using the XOR transformation
#x_xor_transformed = xor_transform(x, y, k = 1e-13)
#causal_options = find_hitting_set(x_xor_transformed)
#print("MHS solving  using PuLP:", causal_options)

x_xor_transformed = xor_transform(x, y, k = 1e-13)
causal_options = find_greedy_hitting_set(x_xor_transformed)
print("MHS solving using Hochbaum's Approximation", causal_options)
print(causal_options)

x_xor_transformed = xor_transform(x, y, k = 0.05)
causal_options = find_greedy_hitting_set(x_xor_transformed)
print("MHS solving using Hochbaum's Approximation with NOISE", causal_options)
print(causal_options)
