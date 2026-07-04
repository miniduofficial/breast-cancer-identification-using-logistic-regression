##Project File that Handles Loading Data, Preprocessing, training and evaluating the model

import numpy as np
from sklearn.datasets import load_breast_cancer
from data_utils import *
from model import *
from visualize import learning_curve

#Loading the breast cancer dataset from scikit learn
data = load_breast_cancer()

# print(type(data.data))
# print(data.data.dtype)
# print(data.data.shape)

#Defining the feature matrix and target vector
X = data.data
y = data.target

#Splitting data
X_train, X_cv, X_test, y_train, y_cv, y_test = my_train_cv_test_split(X, y, 0.2, 0.2, random_state=42)

#Getting mean and standard deviation using X_train
mean, std = mean_and_std(X_train)

#Normalizing X_train, X_cv and X_test
X_train_norm = normalize(X_train, mean, std)
X_cv_norm = normalize(X_cv, mean, std)
X_test_norm = normalize(X_test, mean, std)

w = np.zeros(X_train_norm.shape[1])
b = 0

w, b, cost_log = reg_grad_desc(w, b, X_train_norm, y_train, 1000, 10.25, 0.01)

learning_curve_plot_request = input("Do you wish to see the learning curve? (Y/n)\n")
learning_curve_plot_request = learning_curve_plot_request.lower().strip()

if learning_curve_plot_request == "y":
    learning_curve(np.arange(len(cost_log)), cost_log)