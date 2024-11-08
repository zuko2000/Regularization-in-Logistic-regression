# Logistic Regression with L1 Regularization

This repository provides a Python implementation of logistic regression with **L1 regularization** using NumPy. L1 regularization helps in feature selection by penalizing the absolute value of the coefficients, resulting in sparse solutions with some weights set to zero.

## Overview

Logistic regression is a statistical model used for binary classification. Regularization is added to prevent overfitting and enhance the generalization of the model. This implementation includes:

- **Sigmoid Function** for binary classification.
- **Cost Function** with L1 regularization to add a penalty to large weights.
- **Gradient Descent Algorithm** modified to include L1 regularization for parameter updates.

## Features

- Implementation of logistic regression with L1 regularization.
- Cost function computation with L1 regularization.
- Gradient descent optimization to learn the parameters.
- Customizable hyperparameters: learning rate, regularization strength, and number of iterations.

## Implementation Details

### Cost Function with L1 Regularization

The cost function with L1 regularization is defined as:

\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(h_\theta(x_i)) + (1 - y_i) \log(1 - h_\theta(x_i)) \right] + \frac{\lambda}{m} \sum_{j=1}^n |\theta_j|
\]

### Gradient Descent with L1 Regularization

During gradient descent, the weights are updated using:

\[
\theta_j = \theta_j - \text{learning rate} \times \left( \text{gradient} + \frac{\lambda}{m} \times \text{sign}(\theta_j) \right)
\]

where `sign()` returns the sign of \(\theta_j\).
