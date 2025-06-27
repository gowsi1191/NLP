import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Input range
n_vals = np.linspace(0, 1, 100)
e_vals = np.linspace(0, 1, 100)
n_grid, e_grid = np.meshgrid(n_vals, e_vals)
c = 0.5  # constant c

# Define formulas
formulas = {
    "Sigmoid: σ(15n) + c - σ(15e)": lambda n, e: 1 / (1 + np.exp(-15 * n)) + c - 1 / (1 + np.exp(-15 * e)),
    "Sigmoid: σ(10n) + c - σ(10e)": lambda n, e: 1 / (1 + np.exp(-10 * n)) + c - 1 / (1 + np.exp(-10 * e)),
    "Ratio: (n/(e+0.01)) + c": lambda n, e: (n / (e + 0.01)) + c,
    "Threshold: (n>0.5) + c - (e>0.5)": lambda n, e: (n > 0.5).astype(float) + c - (e > 0.5).astype(float),
    "Sigmoid: σ(5n) + c - σ(5e)": lambda n, e: 1 / (1 + np.exp(-5 * n)) + c - 1 / (1 + np.exp(-5 * e)),
}

# Plot each formula
for name, func in formulas.items():
    z = func(n_grid, e_grid)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(n_grid, e_grid, z, cmap='viridis')
    ax.set_title(name)
    ax.set_xlabel('n')
    ax.set_ylabel('e')
    ax.set_zlabel('Score')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()
