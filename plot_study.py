import json
import matplotlib.pyplot as plt
import numpy as np

# Load your JSON file
with open("/Users/L020774/Movies/heu/NLP/evaluation_results_prajjwal1-albert-base-v2-mnli.json", "r") as f:
    data = json.load(f)

# Initialize value buckets
features = {"e": {}, "n": {}, "c": {}}
for rel in [1, 2, 4, 5]:
    for feat in features:
        features[feat][rel] = []

# Extract values
for example in data.values():
    for model_data in example.values():
        if "Roberta" in model_data:
            for doc in model_data["Roberta"]["ranking"]:
                rel = doc["relevance"]
                if rel in [1, 2, 4, 5]:
                    for feat in features:
                        features[feat][rel].append(doc[feat])

# Plot settings
bins = np.arange(0, 1.1, 0.1)
x = bins[:-1]
bar_width = 0.02
offsets = {-0.03: 1, -0.01: 2, 0.01: 4, 0.03: 5}
colors = {1: 'red', 2: 'orange', 4: 'blue', 5: 'green'}

# Create subplots for e, n, c
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
feature_names = ['e', 'n', 'c']

for ax, feat in zip(axes, feature_names):
    counts = {rel: np.histogram(features[feat][rel], bins=bins)[0] for rel in features[feat]}
    print(counts)
    for offset, rel in offsets.items():
        bar_positions = x + offset
        bar_heights = counts[rel]
        bars = ax.bar(bar_positions, bar_heights, width=bar_width, color=colors[rel], label=f"Rel {rel}")
        for xpos, height in zip(bar_positions, bar_heights):
            if height > 0:
                ax.text(xpos, height + 0.2, str(height), ha='center', va='bottom', fontsize=8, color=colors[rel])
    ax.set_title(f"Binned Distribution of {feat}")
    ax.set_xlabel(f"{feat} value range")
    ax.grid(axis='y')
    ax.set_xticks(bins)

axes[0].set_ylabel("Document count")
axes[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
