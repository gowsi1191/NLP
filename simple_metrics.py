import json
from collections import defaultdict

# Load the JSON file
with open("evaluation_results.json", "r") as file:
    data = json.load(file)

# Initialize containers
metrics_summary = {
    "BGE": defaultdict(list),
    "Roberta": defaultdict(list)
}

# Collect metrics
for doc_id, configs in data.items():
    for config_key, config_val in configs.items():
        for model in ["BGE", "Roberta"]:
            if model in config_val:
                metrics = config_val[model].get("metrics", {})
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_summary[model][metric].append(value)

# Compute mean
metrics_mean = {
    model: {
        metric: round(sum(values) / len(values), 4) if values else 0.0
        for metric, values in metric_set.items()
    }
    for model, metric_set in metrics_summary.items()
}

# Display output
for model, metrics in metrics_mean.items():
    print(f"\n{model} Metrics Mean:")
    for metric, mean in metrics.items():
        print(f"  {metric}: {mean}")

# Save to file
with open("metrics.json", "w") as outfile:
    json.dump(metrics_mean, outfile, indent=2)
