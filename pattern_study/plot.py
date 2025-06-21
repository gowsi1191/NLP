# import pandas as pd
# import json

# # Load data
# with open("implicit_enc.json", "r") as f:
#     data = json.load(f)

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Group by relevance and compute mean and variance
# summary_stats = df.groupby("relevance")[["e", "n", "c"]].agg(['mean', 'var'])

# # Flatten column names
# summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]

# # Print results
# print(summary_stats.round(4))
import numpy as np
import matplotlib.pyplot as plt

# Threshold and steepness values
threshold = 0.12
steepness = 15.0
penalty_steepness = 7.0

# Input values from 0 to 1
x = np.linspace(0, 1, 500)

# Sigmoid penalty for entailment
penalty = 1.0 * (1 / (1 + np.exp(-penalty_steepness * (x - threshold))))

# Sigmoid reward for contradiction
reward = 1.4 * (1 / (1 + np.exp(-steepness * (x - threshold))))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, penalty, label='Penalty (Entailment)', color='red')
plt.plot(x, reward, label='Reward (Contradiction)', color='green')
plt.axvline(threshold, color='gray', linestyle='--', label='Threshold = 0.12')
plt.title('Sigmoid Curves for Penalty and Reward')
plt.xlabel('Input Score (e or c)')
plt.ylabel('Scaled Sigmoid Output')
plt.legend()
plt.grid(True)
plt.show()
