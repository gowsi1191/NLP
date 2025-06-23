import json
from pathlib import Path
from collections import defaultdict

# List of model evaluation file paths
model_files = [
    "enhanced_query_type_analysis_cross-encoder-nli-deberta-base.json",
    "enhanced_query_type_analysis_facebook-bart-large-mnli.json",
    "enhanced_query_type_analysis_microsoft-deberta-large-mnli.json",
    "enhanced_query_type_analysis_prajjwal1-albert-base-v2-mnli.json",
    "enhanced_query_type_analysis_pritamdeka-PubMedBERT-MNLI-MedNLI.json",
    "enhanced_query_type_analysis_roberta-large-mnli.json",
    "enhanced_query_type_analysis_typeform-distilbert-base-uncased-mnli.json"
]

combined_results = defaultdict(dict)
bge_captured = set()

for file_path in model_files:
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract full model identifier from the filename
    model_key = Path(file_path).stem.replace("enhanced_query_type_analysis_", "")

    for qtype, metrics in data.get("query_type_analysis", {}).items():
        # Save BGE only once per query type
        if "BGE" in metrics and qtype not in bge_captured:
            combined_results["BGE"][qtype] = metrics["BGE"]
            bge_captured.add(qtype)

        # Save model results using filename-based key
        combined_results[model_key][qtype] = metrics.get("Roberta", {})  # or any consistent model metric section

# Save combined results
output_path = "combined_query_type_metrics.json"
with open(output_path, "w") as f:
    json.dump(combined_results, f, indent=2)

print(f"Combined metrics saved to {output_path}")
