import numpy as np
import matplotlib.pyplot as plt

# Inverse sigmoid function
def sigmoid_e(x, k=8, midpoint=0.5):
    return 1 / (1 + np.exp(-k * (x - midpoint)))


def reverse_sigmoid(x, k=8, midpoint=0.5):
    return 1 / (1 + np.exp(k * (x - midpoint)))



def flipped_sigmoid_shape(x, k=8, midpoint=0.5):
    if x < midpoint:
        return 1 / (1 + np.exp(k * (x - midpoint)))  # mirrored sigmoid (like 'n')
    else:
        return 1 / (1 + np.exp(-k * (x - midpoint)))  # standard sigmoid (like 'u')


# Generate input values and corresponding outputs
x_vals = np.linspace(0, 1, 500)
y_vals = sigmoid_e(x_vals)


# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label="Inverse Sigmoid (k=8, midpoint=0.5)", color='orange')
plt.xlabel("Raw value")
plt.ylabel("Transformed score")
plt.title("Inverse Sigmoid Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
