import json
from pathlib import Path
import pandas as pd

# Define the path to the uploaded JSON file
json_file_path = Path("evaluation_results.json")

# Load the JSON data
with open(json_file_path, "r") as f:
    evaluation_data = json.load(f)

# Extract (e, n, c, relevance) for implicit_NOT queries
records = []
for query_id, query_config in evaluation_data.items():
    for config_name, config_data in query_config.items():
        if config_data.get("query_type") == "implicit_NOT":
            rankings = config_data.get("Roberta", {}).get("ranking", [])
            for doc in rankings:
                if all(k in doc for k in ("e", "n", "c", "relevance")):
                    records.append({
                        "query_id": query_id,
                        "e": doc["e"],
                        "n": doc["n"],
                        "c": doc["c"],
                        "relevance": doc["relevance"]
                    })

# Convert to DataFrame
df_extracted = pd.DataFrame(records)

# Save the data to a new JSON file
output_path = "implicit_not_enc_relevance.json"
df_extracted.to_json(output_path, orient="records", lines=False, indent=2)

output_path  # Return the file path for download or reference
