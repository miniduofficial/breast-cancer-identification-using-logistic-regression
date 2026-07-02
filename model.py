import numpy as np

def sigmoid(z):
    sigmoid_out = 1/(1 + (np.exp(-z)))
    return (sigmoid_out)

def pred(X, w, b, sigmoid_fx = sigmoid):
    z = np.dot(X, w) + b
    y_prob = sigmoid_fx(z)
    y_pred = np.zeros(len(X),)
    for i in range(len(y_prob)):
        if y_prob[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return (y_pred, y_prob)

def reg_cost(y_prob, target, w,lambda_):
    loss = 0
    for i in range(len(y_prob)):
        loss += -(target[i]*(np.log(y_prob[i])) + (1-target[i])*(np.log(1-y_prob[i])))
    binary_cross_entropy_cost = loss/(len(y_prob)) + (lambda_/ (2*(len(y_prob))))*(np.sum(w**2))
    return (binary_cross_entropy_cost)

def reg_gradient(w, b, X, y, y_prob, lambda_):
    d_w = np.zeros(len(w))
    for i in range(len(d_w)):
        for j in range (len(X)):
            d_w[i] += (y_prob[j] - y[j])*X[j,i]
    d_w = d_w/(len(X)) + (lambda_/(len(X)))*w
    d_b = np.mean((y_prob - y))
    return (d_w, d_b)

def reg_grad_desc (w, b, X, y, num_iters, alpha,lambda_):
    for i in range(num_iters):
        y_pred, y_prob = pred(X, w, b)
        d_w, d_b = reg_gradient(w, b, X, y, y_prob, lambda_)
        w -= alpha*d_w
        b -= alpha*d_b
    return w, b

