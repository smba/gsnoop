# Group-based Feature Selection ![Run Tests](https://github.com/smba/gsnoop/actions/workflows/test.yml/badge.svg)

## Description

### Group Construction


### Stepwise Decomposition
The `gsnoop.screening.group_screening` function is a stepwise feature selection method that identifies the most important features in a dataset until a certain performance threshold is reached.  The function takes in a group feature matrix `x`, a target vector of response differences `y`, and a minimum R² score threshold `r2_threshold`. It then trains a linear model using Stochastic Gradient Descent (SGD) and calculates the importance of each feature. The feature with the highest importance is considered the most important.  The function then adjusts the target values to remove the influence of the most important feature and removes this feature from consideration in the next iteration. This process continues until the R² score of the model drops below the specified threshold. The function returns a list of indices of the most important features in descending order of importance. This list can be used to select the most relevant features for further analysis or modeling.

### Minimal Hitting Set (MHS)
The goal of this code is to find a minimal hitting set that covers causal constraints. It starts by building constraints from input data based on target values. Next, KMeans clustering identifies representative clusters of binary feature vectors. The centroids representing cluster centers are selected, and the Hitman hitting set solver is initialized with these constraints. The solver iteratively adds constraints to achieve a valid hitting set until all constraints are covered.
## How to..

### Install
```bash
pip install git+https://github.com/smba/gsnoop.git@main # install 
pip install --upgrade git+https://github.com/smba/gsnoop.git@main # upgrade
pip install --upgrade --force-reinstall --ignore-installed git+https://github.com/smba/gsnoop.git@main # alles neu
```

### Documentation

