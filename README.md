# Group-based Feature Selection ![Run Tests](https://github.com/smba/gsnoop/actions/workflows/test.yml/badge.svg)

## Description

### Group Construction

### Stepwise Decomposition
The `gsnoop.screening.group_screening` function is a stepwise feature selection method that identifies the most important features in a dataset until a certain performance threshold is reached.  The function takes in a group feature matrix `x`, a target vector of response differences `y`, and a minimum R² score threshold `r2_threshold`. It then trains a linear model using Stochastic Gradient Descent (SGD) and calculates the importance of each feature. The feature with the highest importance is considered the most important.  The function then adjusts the target values to remove the influence of the most important feature and removes this feature from consideration in the next iteration. This process continues until the R² score of the model drops below the specified threshold. The function returns a list of indices of the most important features in descending order of importance. This list can be used to select the most relevant features for further analysis or modeling.

### Minimal Hitting Set (MHS)

## How to..

### Install
```bash
pip install git+https://github.com/smba/gsnoop.git@main # install 
pip install --upgrade git+https://github.com/smba/gsnoop.git@main # upgrade
pip install --upgrade --force-reinstall --ignore-installed git+https://github.com/smba/gsnoop.git@main # alles neu
```

### Documentation
<details>
  <summary>#### Example usage</summary>
  
  ```python
import numpy as np

from gsnoop.util import diff_transform, xor_transform
from gsnoop.screening import group_screening
from gsnoop.causal import find_hitting_set

np.random.seed(1)

# Specify problem space
n_features = 50
n_configs = 100

# Specify simple performance oracle
func = lambda x: x[0] * x[1] * 123 + x[3] * 45 + x[4] * x[5] * 67 + 0.01

# Draw random sample, compute performance
x = np.random.choice(2, size=(n_configs, n_features))
y = np.array(list(map(func, x)))

# Perform stepwise 'group screening'
x_, y_ = diff_transform(x, y)
group_options = group_screening(x_, y_)

# print(group_options)
# > [0, 1, 2, 3, 4, 5]

# Perform causal group screening
x_, y_ = xor_transform(x, y)
causal_options = find_hitting_set(x_, y_)

# print(causal_options)
# > [0, 1, 2, 3, 4, 5]
```

</details>


