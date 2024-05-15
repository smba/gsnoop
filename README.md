# A Framework for Group-based Screening
![Run Tests](https://github.com/smba/gsnoop/actions/workflows/test.yml/badge.svg) [![codecov](https://codecov.io/gh/smba/gsnoop/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/smba/gsnoop)
 ![with-coffee](https://img.shields.io/badge/made%20with-%E2%98%95%EF%B8%8F%20coffee-yellow.svg)

A framework for testing group-based screening strategies for _feature selection_ or _dimensionality reduction_ of large parameter spaces. The methods provided aim at identifying input features (e.g., software configuration options) associated with variation in a dependent variable (e.g., software performance). To elicit variation due to higher-order interactions (effects due to combinations of parameters), we analyze the difference of each pair of observations instead of the training set of obeservations directly. We provide two realizations of group-based screening, a step-wise _variance decomposition_, and a _solving-approach_, conceiving feature selection as the minimal hitting set problem. 

## Description

### Pairwise Feature Groups
```python
>>> import numpy as np
>>> X = np.random.choice([0, 1], size=(5, 10)) # 5 configurations over 10 features
>>> y = np.random.exponential(10, size=5) # generate some useless data
>>> X
array([[0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
       [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
       [1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
       [1, 1, 0, 1, 0, 1, 1, 0, 0, 0]])
>>> y
array([4.42994617, 2.26268041, 0.42066676, 0.76405986, 4.94017651])
```
#### Group Construction via Configuration Differences
```python
>>> import gsnoop.util as util
>>> x_diff = util.diff_transform_x(X)
>>> y_diff = util.diff_transform_y(y)
>>> x_diff
array([[-1,  1, -1,  0,  0, -1,  1, -1,  0,  0],
       [ 0,  0,  0, -1,  0, -1,  0, -1,  0, -1],
       [-1,  0,  0,  0,  0,  0,  1,  0, -1,  0],
       [-1,  0,  0, -1,  1, -1,  0,  0,  0,  0],
       [ 1, -1,  1, -1,  0,  0, -1,  0,  0, -1],
       [ 0, -1,  1,  0,  0,  1,  0,  1, -1,  0],
       [ 0, -1,  1, -1,  1,  0, -1,  1,  0,  0],
       [-1,  0,  0,  1,  0,  1,  1,  1, -1,  1],
       [-1,  0,  0,  0,  1,  0,  0,  1,  0,  1],
       [ 0,  0,  0, -1,  1, -1, -1,  0,  1,  0]])
>>> y_diff
array([ 2.16726577,  4.00927942,  3.66588631, -0.51023034,  1.84201365,
        1.49862054, -2.6774961 , -0.34339311, -4.51950975, -4.17611665])
```
#### Group Construction via Bit-wise XOR
```python
>>> x_xor = util.xor_transform_x(X)
>>> y_xor = util.xor_transform_y(y)
>>> x_xor
array([[1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
       [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
       [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
       [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
       [0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
       [1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
       [0, 0, 0, 1, 1, 1, 1, 0, 1, 0]])
>>> y_xor
array([2.16726577, 4.00927942, 3.66588631, 0.51023034, 1.84201365,
       1.49862054, 2.6774961 , 0.34339311, 4.51950975, 4.17611665])
```

#### Shortcuts
```python
>>> x_diff, y_diff = util.diff_transform(X, y)
>>> x_xor, y_xor = util.xor_transform(X, y)
```

### Stepwise Variance Decomposition
### Minimal Hitting Set 
[hitting set problem](https://en.wikipedia.org/wiki/Set_cover_problem#Hitting_set_formulation)

```python
>>> sets_to_hit = [set(np.where(x_ == 1)[0]) for x_ in x_xor]
>>> for s in sets_to_hit:
...     print(s)
... 
{0, 1, 2, 5, 6, 7}
{9, 3, 5, 7}
{0, 8, 6}
{0, 3, 4, 5}
{0, 1, 2, 3, 6, 9}
{1, 2, 5, 7, 8}
{1, 2, 3, 4, 6, 7}
{0, 3, 5, 6, 7, 8, 9}
{0, 9, 4, 7}
{3, 4, 5, 6, 8}
```

## How to..

### Installation
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

### Example Usage
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
