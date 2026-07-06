import numpy as np
from model import sigmoid, pred
import shutil

terminal_width = shutil.get_terminal_size().columns

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
GOLD = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"

def color_higher_better(value):
    if value >= 0.90:
        return GREEN + f"{value:.6f}" + RESET
    elif value >= 0.75:
        return GOLD + f"{value:.6f}" + RESET
    else:
        return RED + f"{value:.6f}" + RESET


def color_lower_better(value):
    if value <= 0.10:
        return GREEN + f"{value:.6f}" + RESET
    elif value <= 0.25:
        return GOLD + f"{value:.6f}" + RESET
    else:
        return RED + f"{value:.6f}" + RESET

def accuracy (w, b, X, y_target):
    y_pred, y_prob = pred(X, w, b, sigmoid)
    count = 0
    for i in range (len(X)):
        if y_pred[i] == y_target[i]:
            count += 1
    accuracy = count/len(X)
    return accuracy

def confusion_matrix(w, b, X, y):
    y_pred, y_prob = pred(X, w, b, sigmoid)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for j in range(len(X)):
        if (y_pred[j] == 1) and (y[j] == 1):
            TP += 1
        elif (y_pred[j] == 0) and (y[j] == 0):
            TN += 1
        elif (y_pred[j] == 1) and (y[j] == 0):
            FP += 1
        elif (y_pred[j] == 0) and (y[j] == 1):
            FN += 1
    return TP, TN, FP, FN

def false_negative_rate(FN, TP):
    if FN + TP == 0:
        return 0
    return FN/(FN + TP)

def false_positive_rate(FP, TN):
    if FP + TN == 0:
        return 0
    return FP/(FP + TN)

def precision(TP, FP):
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def specificity(TN, FP):
    if TN + FP == 0:
        return 0
    return TN / (TN + FP)

def cv_eval (w, b, X_cv_norm, y_cv):
    TP, TN, FP, FN = confusion_matrix(w, b, X_cv_norm, y_cv)

    acc = accuracy(w, b, X_cv_norm, y_cv)
    fnr = false_negative_rate(FN, TP)
    fpr = false_positive_rate(FP, TN)
    prec = precision(TP, FP)
    rec = recall(TP, FN)
    spec = specificity(TN, FP)

    print("\n" + "─" * terminal_width)
    print("Logistic Regression Evaluation".center(terminal_width))
    print("─" * terminal_width)
    print(f"\n{DIM}Accuracy            :{RESET} {color_higher_better(acc)}")
    print(f"{DIM}False Negative Rate :{RESET} {color_lower_better(fnr)}")
    print(f"{DIM}False Positive Rate :{RESET} {color_lower_better(fpr)}")
    print(f"{DIM}Precision           :{RESET} {color_higher_better(prec)}")
    print(f"{DIM}Recall              :{RESET} {color_higher_better(rec)}")
    print(f"{DIM}Specificity         :{RESET} {color_higher_better(spec)}")