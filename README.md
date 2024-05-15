# A Framework for Group-based Screening
![Run Tests](https://github.com/smba/gsnoop/actions/workflows/test.yml/badge.svg) [![codecov](https://codecov.io/gh/smba/gsnoop/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/smba/gsnoop)


A framework for testing group-based screening strategies for _feature selection_ or _dimensionality reduction_ of large parameter spaces. The methods provided aim at identifying input features (e.g., software configuration options) associated with variation in a dependent variable (e.g., software performance). To elicit variation due to higher-order interactions (effects due to combinations of parameters), we analyze the difference of each pair of observations instead of the training set of obeservations directly. We provide two realizations of group-based screening, a step-wise _variance decomposition_, and a _solving-approach_, conceiving feature selection as the minimal hitting set problem. 

## Description
## Installation
```bash
# install
pip install git+https://github.com/smba/gsnoop.git@main # install 
```
```bash
# upgrade existing installation
pip install --upgrade git+https://github.com/smba/gsnoop.git@main
```
```bash
# last resort
pip install --upgrade --force-reinstall --ignore-installed git+https://github.com/smba/gsnoop.git@main 
```

## Example Usage
```python
# Import required libraries and modules
import numpy as np
from gsnoop.util import diff_transform, xor_transform
from gsnoop.screening import group_screening, lasso_screening
from gsnoop.causal import find_hitting_set

# Set the random seed for reproducibility
np.random.seed(1)

# Define the size of the problem space
n_features = 50
n_configs = 50

# Define a simple performance function to simulate system behavior
# The function models the performance of a system given a configuration of features
def performance_oracle(x):
    return x[0] * x[1] * 123 + x[3] * 45 + x[4] * x[5] * 67 + 0.01

# Generate random configurations and evaluate performance using the defined oracle
x = np.random.choice(2, size=(n_configs, n_features))
y = np.array(list(map(performance_oracle, x)))

# Conduct baseline screening using LASSO
lasso_options = lasso_screening(x, y)
print("Lasso Screening Results:", lasso_options)

# Perform group screening using the difference transformation
x_diff_transformed, y_diff_transformed = diff_transform(x, y)
group_options = group_screening(x_diff_transformed, y_diff_transformed)
print("Group Screening Results:", group_options)

# Perform causal analysis using the XOR transformation
x_xor_transformed, y_xor_transformed = xor_transform(x, y)
causal_options = find_hitting_set(x_xor_transformed, y_xor_transformed)
print("Causal Screening Results:", causal_options)

```
