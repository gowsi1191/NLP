import json
from pathlib import Path
import pandas as pd

# List of JSON filenames
model_files = [
    "evaluation_results_cross-encoder-nli-deberta-base.json",
    "evaluation_results_facebook-bart-large-mnli.json",
    "evaluation_results_microsoft-deberta-large-mnli.json",
    "evaluation_results_prajjwal1-albert-base-v2-mnli.json",
    "evaluation_results_pritamdeka-PubMedBERT-MNLI-MedNLI.json",
    "evaluation_results_roberta-large-mnli.json",
    "evaluation_results_typeform-distilbert-base-uncased-mnli.json"
]

# Process each file
for file in model_files:
    model_name = Path(file).stem.replace("evaluation_results_", "")
    with open(file, "r") as f:
        evaluation_data = json.load(f)

    records = []
    for query_id, query_config in evaluation_data.items():
        for config_name, config_data in query_config.items():
            # if config_data.get("query_type") == "implicit_NOT":
            rankings = config_data.get("Roberta", {}).get("ranking", [])
            for doc in rankings:
                if all(k in doc for k in ("e", "n", "c", "relevance")):
                    records.append({
                            "query_id": config_data.get("query_type"),
                            "e": doc["e"],
                            "n": doc["n"],
                            "c": doc["c"],
                            "relevance": doc["relevance"]
                    })

    if records:
        df_extracted = pd.DataFrame(records)
        output_path = f"implicit_not_enc_relevance_{model_name}.json"
        df_extracted.to_json(output_path, orient="records", lines=False, indent=2)
        print(f"Saved: {output_path}")
