import json
import pandas as pd
import numpy as np

# # Load the file
file_path = "evaluation_results_prajjwal1-albert-base-v2-mnli.json"
# file_path = "evaluation_results_roberta-large-mnli.json"

# file_path = "evaluation_results_facebook-bart-large-mnli.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Step 1: Normalize e/n/c per doc so e + n + c = 100
normalized_docs = []

for query_id, query_data in data.items():
    for config_key, config_data in query_data.items():
        if "Roberta" in config_data:
            query_type = config_data.get("query_type", "")
            for doc in config_data["Roberta"]["ranking"]:
                e, n, c = doc["e"], doc["n"], doc["c"]
                total = e + n + c
                if total == 0:
                    continue  # Skip broken values

                # Normalize to 100
                e_norm = (e / total) * 100
                n_norm = (n / total) * 100
                c_norm = (c / total) * 100

                # Mean & std for this document
                vals = [e_norm, n_norm, c_norm]
                mean_val = np.mean(vals)
                std_val = np.std(vals)

                normalized_docs.append({
                    "query_id": query_id,
                    "query_type": query_type,
                    "relevance": doc["relevance"],
                    "e_norm": e_norm,
                    "n_norm": n_norm,
                    "c_norm": c_norm,
                    "norm_mean": mean_val,
                    "norm_std": std_val
                })

# Step 2: Create DataFrame
df_norm = pd.DataFrame(normalized_docs)

# Step 3: Group by relevance and take mean/std of doc-wise mean/std
summary = df_norm.groupby("relevance").agg({
    "norm_mean": ["mean", "std"],
    "norm_std": ["mean", "std"],
    "e_norm": ["mean", "std"],
    "n_norm": ["mean", "std"],
    "c_norm": ["mean", "std"]
}).reset_index()

# Clean up column names
summary.columns = ['relevance', 'mean_of_means', 'std_of_means',
                   'mean_of_stds', 'std_of_stds',
                   'e_mean', 'e_std', 'n_mean', 'n_std', 'c_mean', 'c_std']

# Save and print
summary.to_csv("normalized_summary_by_relevance.csv", index=False)
print(summary)



# import json
# import pandas as pd

# # Load JSON file
# file_path = "evaluation_results_microsoft-deberta-large-mnli.json"
# with open(file_path, "r") as f:
#     data = json.load(f)

# # Normalize e/n/c and collect per-document rows
# normalized_rows = []

# for query_id, query_data in data.items():
#     for config_key, config_data in query_data.items():
#         if "Roberta" in config_data:
#             query_type = config_data.get("query_type", "")
#             for i, doc in enumerate(config_data["Roberta"]["ranking"]):
#                 e, n, c = doc["e"], doc["n"], doc["c"]
#                 total = e + n + c
#                 if total == 0:
#                     continue

#                 e_norm = (e / total) * 100
#                 n_norm = (n / total) * 100
#                 c_norm = (c / total) * 100

#                 normalized_rows.append({
#                     "query_id": query_id,
#                     "query_type": query_type,
#                     "doc_index": i + 1,
#                     "relevance": doc["relevance"],
#                     "e_normalized": round(e_norm, 2),
#                     "n_normalized": round(n_norm, 2),
#                     "c_normalized": round(c_norm, 2)
#                 })

# # Create DataFrame
# df = pd.DataFrame(normalized_rows)

# # Optional: Save to CSV
# df.to_csv("normalized_entailment_scores.csv", index=False)

# # Show first 20 rows in terminal
# print(df.head(20).to_string(index=False))
