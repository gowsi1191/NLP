# import pandas as pd
# import json

# # Load data
file_path = "evaluation_results_microsoft-deberta-large-mnli.json"
# with open(file_path, "r") as f:
#     data = json.load(f)

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Create 'query_type' from 'query_id' if not already present
# if 'query_type' not in df.columns and 'query_id' in df.columns:
#     df['query_type'] = df['query_id']

# # Group by query_type and relevance
# summary_stats = df.groupby(['query_type', 'relevance'])[['e', 'n', 'c']].agg(['mean', 'var'])

# # Flatten column names
# summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
# summary_stats_rounded = summary_stats.round(4)




import pandas as pd
import json
import matplotlib.pyplot as plt

# Load JSON file
file_path = "evaluation_results_prajjwal1-albert-base-v2-mnli.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Extract rows from nested structure
rows = []
for dataset_id, dataset_content in data.items():
    for config_key, config in dataset_content.items():
        query_type = config.get("query_type", "")
        roberta_docs = config.get("Roberta", {}).get("ranking", [])
        for doc in roberta_docs:
            if "e" in doc and "n" in doc and "c" in doc:
                rows.append({
                    "dataset_id": dataset_id,
                    "query_type": query_type,
                    "e": doc["e"],
                    "n": doc["n"],
                    "c": doc["c"],
                    "relevance": doc["relevance"]
                })

# Create DataFrame
df = pd.DataFrame(rows)

# Normalize e, n, c
def normalize(row):
    total = row['e'] + row['n'] + row['c']
    if total == 0:
        return pd.Series([0, 0, 0])
    return pd.Series([
        (row['e'] / total) * 100,
        (row['n'] / total) * 100,
        (row['c'] / total) * 100
    ])

df[['e_norm', 'n_norm', 'c_norm']] = df.apply(normalize, axis=1)

# Assign colors
color_map = {
    1: 'yellow',
    2: 'red',
    3: 'green',
    4: 'blue',
    5: 'black'
}
df['color'] = df['relevance'].map(lambda x: color_map.get(x, 'brown'))

# Expand with white gaps and dataset labels
expanded_rows = []
doc_id_counter = 1

for dataset_id, group in df.groupby("dataset_id"):
    group = group.reset_index(drop=True)
    for i, row in group.iterrows():
        expanded_rows.append({
            "dataset_id": dataset_id,
            "doc_id": doc_id_counter,
            "query_type": row["query_type"],
            "e_norm": row["e_norm"],
            "n_norm": row["n_norm"],
            "c_norm": row["c_norm"],
            "color": row["color"],
            "label_dataset": dataset_id if i < 7 else ""
        })
        doc_id_counter += 1
    # Add 2 blank points
    for _ in range(2):
        expanded_rows.append({
            "dataset_id": "",
            "doc_id": doc_id_counter,
            "query_type": "",
            "e_norm": 0,
            "n_norm": 0,
            "c_norm": 0,
            "color": "white",
            "label_dataset": ""
        })
        doc_id_counter += 1

plot_df = pd.DataFrame(expanded_rows)

# Plotting function
def plot_score(score_col, title, ylabel):
    plt.figure(figsize=(16, 5))
    plt.scatter(plot_df['doc_id'], plot_df[score_col], c=plot_df['color'], edgecolors='k')

    for _, row in plot_df.iterrows():
        if row.get("label_dataset"):
            plt.text(row['doc_id'], row[score_col] + 2, row['label_dataset'],
                     ha='center', fontsize=8, color='darkslategray')

    plt.title(title)
    plt.xlabel("Document ID")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot all three
plot_score("e_norm", "Normalized Entailment Score (e)", "e (%)")
plot_score("n_norm", "Normalized Neutral Score (n)", "n (%)")
plot_score("c_norm", "Normalized Contradiction Score (c)", "c (%)")





# import json
# import pandas as pd

# # Load JSON
# file_path = "evaluation_results_microsoft-deberta-large-mnli.json"
# with open(file_path, "r") as f:
#     data = json.load(f)

# # Flatten and normalize
# rows = []
# for dataset_id, dataset_content in data.items():
#     for _, config in dataset_content.items():
#         query_type = config.get("query_type", "")
#         for doc in config.get("Roberta", {}).get("ranking", []):
#             if "e" in doc and "n" in doc and "c" in doc:
#                 e, n, c = doc["e"], doc["n"], doc["c"]
#                 total = e + n + c
#                 if total > 0:
#                     e_norm = e / total * 100
#                     n_norm = n / total * 100
#                     c_norm = c / total * 100
#                 else:
#                     e_norm = n_norm = c_norm = 0
#                 rows.append({
#                     "dataset_id": dataset_id,
#                     "query_type": query_type,
#                     "e_norm": e_norm,
#                     "n_norm": n_norm,
#                     "c_norm": c_norm,
#                     "relevance": doc["relevance"]
#                 })

# df = pd.DataFrame(rows)

# # Per-dataset report
# print("==== Per Dataset Report ====")
# for dataset_id, group in df.groupby("dataset_id"):
#     print(f"\nDataset: {dataset_id}")
#     grouped = group.groupby("relevance")[["e_norm", "n_norm", "c_norm"]].mean()

#     # e
#     e_sorted = grouped["e_norm"].sort_values(ascending=False)
#     e_top, e_second = e_sorted.index[0], e_sorted.index[1] if len(e_sorted) > 1 else None
#     e_confidence = e_sorted.iloc[0] - e_sorted.iloc[1] if e_second else 0

#     # n
#     n_sorted = grouped["n_norm"].sort_values(ascending=False)
#     n_top, n_second = n_sorted.index[0], n_sorted.index[1] if len(n_sorted) > 1 else None
#     n_confidence = n_sorted.iloc[0] - n_sorted.iloc[1] if n_second else 0

#     # c
#     c_sorted = grouped["c_norm"].sort_values(ascending=False)
#     c_lowest, c_second = c_sorted.index[0], c_sorted.index[1] if len(c_sorted) > 1 else None
#     c_confidence = c_sorted.iloc[1] - c_sorted.iloc[0] if c_second else 0

#     print(f"Highest e → Relevance: {e_top} (avg e = {e_sorted.iloc[0]:.2f}%) | Confidence: {e_confidence:.2f}")
#     print(f"Highest n → Relevance: {n_top} (avg n = {n_sorted.iloc[0]:.2f}%) | Confidence: {n_confidence:.2f}")
#     print(f"Lowest c → Relevance: {c_lowest} (avg c = {c_sorted.iloc[0]:.2f}%) | Confidence: {c_confidence:.2f}")

# # Summary counts
# print("\n==== Total Document Count by Relevance ====")
# relevance_counts = df["relevance"].value_counts().sort_index()
# for rel, count in relevance_counts.items():
#     print(f"Relevance {rel}: {count} documents")
