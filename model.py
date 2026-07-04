import numpy as np
import shutil

terminal_width = shutil.get_terminal_size().columns

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
GOLD = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"

def sigmoid(z):
    z = np.clip(z, -500, 500) #Clip logits to prevent np.exp(-z) from overflowing for large magnitude z values
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

    #Clipping to avoid nan error
    epsilon = 1e-15
    y_prob = np.clip(y_prob, epsilon, 1-epsilon)

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
    print("\n" + "─" * terminal_width)
    print("Logistic Regression Training".center(terminal_width))
    print("─" * terminal_width)
    cost_log = np.zeros(num_iters,)
    initial_cost = reg_cost(pred(X, w, b)[1], y, w, lambda_)
    print(f"Initial cost: {initial_cost:.6f}\n")
    for i in range(num_iters):
        y_pred, y_prob = pred(X, w, b)
        reg_cost_i = reg_cost(y_prob, y, w, lambda_)
        cost_log[i] = reg_cost_i
        if i % 50 == 0 or i ==num_iters-1:
            progress = (i + 1) / num_iters
            bar_width = min(40, terminal_width // 2)
            filled = int(progress * bar_width)

            bar = "█" * filled + "·" * (bar_width - filled)

            print(
                f"{DIM}Descent{RESET} [{BOLD}{bar}{RESET}] "
                f"{progress * 100:5.1f}% │ "
                f"{CYAN}J(w,b){RESET} {reg_cost_i:.6f}",
                end="\r"
            )
        d_w, d_b = reg_gradient(w, b, X, y, y_prob, lambda_)
        w -= alpha*d_w
        b -= alpha*d_b
    count = 0
    print("\n")
    for j in range(1, num_iters):
        if cost_log[j-1] >= cost_log[j]:
            count += 1

    print("\n" + "─" * terminal_width)
    print("Training Summary".center(terminal_width))
    print("" + "─" * terminal_width)

    print(f"Initial cost : {cost_log[0]:.6f}")
    print(f"Final cost   : {cost_log[-1]:.6f}")
    print(f"Reduction    : {cost_log[0] - cost_log[-1]:.6f}")
    if count == num_iters - 1:
        print(f"\n{GREEN}Status       : Monotonic descent achieved{RESET}")
    else:
        print(f"\n{RED}Status       : Cost did not decrease monotonically{RESET}")
    
    print("─" * terminal_width)
    return w, b, cost_log

