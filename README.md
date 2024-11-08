# Logistic Regression with L1 and L2 Regularization

This repository provides a Python implementation of logistic regression with L1 (Lasso) and L2 (Ridge) regularization using NumPy. Regularization is an essential technique for preventing overfitting by penalizing complex models.

## Overview

Logistic regression is used for binary classification tasks. The cost function for logistic regression is modified by adding a regularization term to improve generalization:

### Cost Function with L2 Regularization (Ridge)
The cost function \( J(\theta) \) with L2 regularization is defined as:
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
$$ J(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(h_\theta(x_i)) + (1 - y_i) \log(1 - h_\theta(x_i)) \right] $$


#### Sigmoid Function
```latex
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
```
#### Gradient Update with L1 Regularization
```latex
\theta_j := \theta_j - \alpha \left[ \frac{1}{N} \sum_{i=1}^N (h_\theta(x_i) - y_i) x_{ij} + \frac{\lambda}{N} \text{sign}(\theta_j) \right]

```
