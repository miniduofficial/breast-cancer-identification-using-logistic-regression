##Defining Helper Functions for Data Preprocessing

import numpy as np

#Randomizing and splitting data into train, cross validation and test splits
def my_train_cv_test_split (X, y, test_size, cv_size, random_state=None):
    m = len(y)

    idx = np.arange(m)
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)

    shuffled_X = X[idx]
    shuffled_y = y[idx]

    test_count = int(m*test_size)
    cv_count = int(m*cv_size)
    train_count = m - (test_count + cv_count)

    X_train = shuffled_X[:train_count]
    X_cv = shuffled_X[train_count:(cv_count + train_count)]
    X_test = shuffled_X[(cv_count + train_count):]

    y_train = shuffled_y[:train_count]
    y_cv = shuffled_y[train_count:(cv_count + train_count)]
    y_test = shuffled_y[(cv_count + train_count):]

    # print(X_train.shape, X_cv.shape, X_test.shape)
    # print(y_train.shape, y_cv.shape, y_test.shape)

    return(X_train, X_cv, X_test, y_train, y_cv, y_test)

#Computing mean
def mean_and_std (X_train):
    mean = sum(X_train)/(len(X_train))
    mean_np = np.mean(X_train, axis=0)
    std = (sum((X_train - mean)**2)/(len(X_train)))
    std = np.sqrt(std)
    std_np = np.std(X_train, axis=0)
    # print(mean, mean_np)
    # print(std, std_np)
    return (mean, std)

#Computing standard deviation
def normalize(X, mean, std):
    X_norm = (X - mean)/std
    return X_norm