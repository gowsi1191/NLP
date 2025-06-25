import numpy as np
import matplotlib.pyplot as plt

# Generate value ranges
e_vals = np.linspace(0, 1, 200)
n_vals = np.linspace(0, 1, 200)
c_vals = np.linspace(0, 0.05, 200)

# === Scoring Logic ===

# Entailment: flat at 1 till 0.05, slow decay to 0 by 0.6 (centered ~0.3)
e_reward = 0.8 * (1.0 - 1.0 / (1 + np.exp(-10 * (e_vals - 0.3))))

# Neutral: stays low till 0.65, rises steeply to 1 by 0.9 (centered ~0.75)
n_reward = 1.0 * (1.0 / (1 + np.exp(-30 * (n_vals - 0.75))))

# Contradiction: rises quickly from 0 to 0.9 until 0.018, then slowly toward 1
c_scaled = c_vals / 0.05  # Normalize to 0–1 range
c_penalty = 1.2 * (1.0 / (1 + np.exp(-25 * (c_scaled - 0.36))))  # center ~0.018/0.05

# === Plotting ===
plt.figure(figsize=(15, 4))

# Plot Entailment Reward
plt.subplot(1, 3, 1)
plt.plot(e_vals, e_reward)
plt.title("Entailment Reward Curve")
plt.xlabel("Entailment (e)")
plt.ylabel("Reward")
plt.grid(True)

# Plot Neutral Reward
plt.subplot(1, 3, 2)
plt.plot(n_vals, n_reward, color="orange")
plt.title("Neutral Reward Curve")
plt.xlabel("Neutral (n)")
plt.ylabel("Reward")
plt.grid(True)

# Plot Contradiction Penalty
plt.subplot(1, 3, 3)
plt.plot(c_vals, c_penalty, color="red")
plt.title("Contradiction Penalty Curve")
plt.xlabel("Contradiction (c)")
plt.ylabel("Penalty")
plt.grid(True)

plt.tight_layout()
plt.show()
