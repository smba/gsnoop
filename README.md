# Group-based Feature Selection
## How to..

### Install
```bash
pip install git+https://github.com/smba/gsnoop.git@main # install 
pip install --upgrade git+https://github.com/smba/gsnoop.git@main # upgrade
pip install --upgrade --force-reinstall git+https://github.com/smba/gsnoop.git@main # yay
```

### Basic Usage
```python
import numpy as np

from gsnoop.util import diff_transform, xor_transformation
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
x_, y_ = xor_transform(x)
y_ = np.abs(y_)
causal_options = causal_screening(x_, y_)

# print(causal_options)
# > [0, 1, 2, 3, 4, 5]
```
