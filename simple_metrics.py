import json
import os
from collections import defaultdict

# List of evaluation files
files = [
    "evaluation_results_cross-encoder-nli-deberta-base.json",
    "evaluation_results_facebook-bart-large-mnli.json",
    "evaluation_results_microsoft-deberta-large-mnli.json",
    "evaluation_results_prajjwal1-albert-base-v2-mnli.json",
    "evaluation_results_pritamdeka-PubMedBERT-MNLI-MedNLI.json",
    "evaluation_results_roberta-large-mnli.json",
    "evaluation_results_typeform-distilbert-base-uncased-mnli.json"
]

# All metrics stored here
overall_summary = {}

# Loop through each file
for file_name in files:
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    with open(file_name, "r") as file:
        data = json.load(file)

    metrics_summary = defaultdict(lambda: defaultdict(list))

    # Extract metrics
    for _, configs in data.items():
        for _, config_val in configs.items():
            for model in config_val:
                if "metrics" in config_val[model]:
                    for metric, value in config_val[model]["metrics"].items():
                        if isinstance(value, (int, float)):
                            metrics_summary[model][metric].append(value)

    # Compute means
    metrics_mean = {
        model: {
            metric: round(sum(values) / len(values), 4) if values else 0.0
            for metric, values in metric_set.items()
        }
        for model, metric_set in metrics_summary.items()
    }

    # Store in overall summary
    overall_summary[file_name] = metrics_mean

    # Print summary
    print(f"\n📄 File: {file_name}")
    for model, metrics in metrics_mean.items():
        print(f"  🔹 Model: {model}")
        for metric, mean in metrics.items():
            print(f"    {metric}: {mean}")

# Save summary to file
with open("all_models_metrics_summary.json", "w") as outfile:
    json.dump(overall_summary, outfile, indent=2)
