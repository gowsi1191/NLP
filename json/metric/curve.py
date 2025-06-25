import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input range
x = np.linspace(0, 1, 500)

# Reward/Penalty functions for each mode
def explicit_not(e, n, c):
    e_reward = 1.0 * sigmoid(30 * (e - 0.95))
    n_reward = 0.5 * (1 - sigmoid(50 * (n - 0.01)))
    c_penalty = 0.3 * sigmoid(20 * (c - 0.02))
    return e_reward, n_reward, c_penalty

def implicit_not(e, n, c):
    e_reward = 1.2 * sigmoid(35 * (e - 0.96))
    n_penalty = 0.7 * sigmoid(40 * (n - 0.03))
    c_penalty = 0.5 * sigmoid(40 * (c - 0.02))
    return e_reward, n_penalty, c_penalty

def comparative_not(e, n, c):
    n_reward = 0.3 * (1 - sigmoid(30 * (n - 0.17)))
    c_reward = 1.0 * sigmoid(18.0 * (c - 0.17))
    e_penalty = 0.8 * sigmoid(25 * (e - 0.70))
    return n_reward, c_reward, e_penalty

def prohibition_not(e, n, c):
    e_reward = 0.5 * sigmoid(25 * (e - 0.965))
    n_penalty = 0.2 * sigmoid(40 * (n - 0.015))
    c_penalty = 0.8 * sigmoid(40 * (c - 0.02))
    return e_reward, n_penalty, c_penalty

def scope_not(e, n, c):
    e_penalty = 0.6 * sigmoid(35 * (e - 0.90))
    n_penalty = 0.3 * sigmoid(30 * (n - 0.05))
    c_reward = 1.2 * sigmoid(25.0 * (c - 0.04))
    return e_penalty, n_penalty, c_reward

def general_linear(e, n, c):
    return n, c, n + c

# Mapping modes to functions
modes = {
    "explicit_NOT": explicit_not,
    "implicit_NOT": implicit_not,
    "comparative_NOT": comparative_not,
    "prohibition_NOT": prohibition_not,
    "scope_NOT": scope_not,
    "general_LINEAR": general_linear
}

# Plotting
for mode, func in modes.items():
    e = n = c = x
    y1, y2, y3 = func(e, n, c)
    plt.figure()
    plt.plot(x, y1, label="e/n")
    plt.plot(x, y2, label="n/c")
    plt.plot(x, y3, label="c/sum")
    plt.title(f"{mode} - Reward/Penalty Curves")
    plt.xlabel("Probability")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
