# Logistic Regression from Scratch — Breast Cancer Classification

A from-scratch implementation of binary logistic regression using NumPy on the Breast Cancer Wisconsin dataset. The goal of this project was to understand the mechanics behind logistic regression rather than rely on a black-box classifier.

The implementation includes:

- train/CV/test splitting
- feature normalization using training-set statistics
- sigmoid activation
- binary cross-entropy loss
- L2 regularization
- gradient descent
- cost logging
- learning-curve visualization
- numerical stability fixes

## Dataset

The project uses the breast cancer dataset from `sklearn.datasets`.

```text
0 → malignant
1 → benign
```

This encoding is important when interpreting evaluation metrics, especially recall and the confusion matrix.

## Model

The model first computes a linear score:

```math
z = Xw + b
```

Then it converts this score into a probability using the sigmoid function:

```math
\hat{y} = \frac{1}{1 + e^{-z}}
```

The model is trained using binary cross-entropy loss with optional L2 regularization.

## Numerical Stability

During training, the model can become extremely confident, producing probabilities close to `0` or `1`. This can cause `log(0)` and result in `nan` values.

To avoid this, probabilities are clipped before computing the loss:

```python
y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
```

The logits are also clipped before applying sigmoid to reduce overflow in `np.exp`.

## Evaluation

The next stage of the project is to evaluate the model using:

- accuracy
- precision
- recall
- F1 score
- confusion matrix
- train/CV/test comparison

Since this is a medical classification dataset, accuracy alone is not enough. The class-wise behavior of the model matters.

## Status

The core model and training loop are complete. Evaluation metrics and final cleanup are in progress.