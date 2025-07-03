import json
import os
from collections import defaultdict
from scipy.stats import ttest_rel

# Baseline to compare against
baseline_model = "BGE"

# List of evaluation files
files = [
    "evaluation_results_prajjwal1-albert-base-v2-mnli.json",
]

# All metrics stored here
overall_summary = {}

for file_name in files:
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    with open(file_name, "r") as file:
        data = json.load(file)

    metrics_summary = defaultdict(lambda: defaultdict(list))

    # Extract all metric values per model
    for _, configs in data.items():
        for _, config_val in configs.items():
            for model in config_val:
                if "metrics" in config_val[model]:
                    for metric, value in config_val[model]["metrics"].items():
                        if isinstance(value, (int, float)):
                            metrics_summary[model][metric].append(value)

    # Compute mean metrics
    metrics_mean = {
        model: {
            metric: round(sum(values) / len(values), 4) if values else 0.0
            for metric, values in metric_set.items()
        }
        for model, metric_set in metrics_summary.items()
    }

    # Compute p-values against the baseline
    p_values = {}
    for model in metrics_summary:
        if model == baseline_model or baseline_model not in metrics_summary:
            continue
        p_values[model] = {}
        for metric in metrics_summary[model]:
            if metric in metrics_summary[baseline_model]:
                try:
                    stat, pval = ttest_rel(
                        metrics_summary[model][metric],
                        metrics_summary[baseline_model][metric]
                    )
                    p_values[model][metric] = round(pval, 6)
                except Exception as e:
                    p_values[model][metric] = f"error: {str(e)}"

    # Store result
    overall_summary[file_name] = {
        "means": metrics_mean,
        "p_values_vs_BGE": p_values
    }

    # Print summary
    print(f"\n📄 File: {file_name}")
    for model, metrics in metrics_mean.items():
        print(f"  🔹 Model: {model}")
        for metric, mean in metrics.items():
            print(f"    {metric}: {mean}")
    if p_values:
        print("  🔸 P-values vs BGE:")
        for model, pvals in p_values.items():
            print(f"    {model}:")
            for metric, pval in pvals.items():
                print(f"      {metric}: {pval}")

# Save full output
with open("comp_150_20_with_pvalues.json", "w") as outfile:
    json.dump(overall_summary, outfile, indent=2)
