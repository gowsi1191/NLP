import numpy as np
import pandas as pd
import json
import os
from scipy.stats import pearsonr
from collections import defaultdict

# --- Enhanced Normalization ---
def normalize_to_relevance(scores):
    sorted_indices = np.argsort(scores)[::-1]
    n = len(scores)
    ranks = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        tier = int((i / n) * 5)
        ranks[idx] = 5 - tier
    return np.clip(ranks, 1, 5)

# --- Formula Evaluation ---
def evaluate_formula(datasets, func, name):
    metrics = defaultdict(list)
    for data in datasets:
        rankings = data['ranking']
        true_relevance = np.array([r['relevance'] for r in rankings])
        raw_scores = np.array([func(r['n'], r['c'], r['e']) for r in rankings])
        pred_relevance = normalize_to_relevance(raw_scores)
        metrics['pearson'].append(pearsonr(pred_relevance, true_relevance)[0])
        metrics['p@1'].append(1 if pred_relevance[np.argmax(raw_scores)] == 5 else 0)
        metrics['mae'].append(np.mean(np.abs(pred_relevance - true_relevance)))
    return {
        'Formula': name,
        'Avg Pearson': np.mean(metrics['pearson']),
        'Avg P@1': np.mean(metrics['p@1']),
        'Avg MAE': np.mean(metrics['mae']),
        'Std Pearson': np.std(metrics['pearson'])
    }

# --- Grid Search Formula Generator ---
def grid_search_formulas(datasets):
    results = []
    steepness_values = np.linspace(5, 0.5, 50)
    bias_values = [0, 0.1, 0.2]
    weight_c_values = [0.0, 0.25, 0.5]

    for s in steepness_values:
        for b in bias_values:
            for wc in weight_c_values:
                name = f"Grid: σ({s}e) - σ({s}n) - {wc}σ({s}c) + {b}"
                func = lambda n, c, e, s=s, b=b, wc=wc: (
                    1/(1+np.exp(-s*e)) - 1/(1+np.exp(-s*n)) - wc/(1+np.exp(-s*c)) + b
                )
                results.append(evaluate_formula(datasets, func, name))
    return pd.DataFrame(results).sort_values('Avg P@1', ascending=False)

# --- Main ---
if __name__ == "__main__":
    file_list = ["evaluation_results_microsoft-deberta-large-mnli.json"]

    for file_path in file_list:
        if not os.path.exists(file_path):
            continue

        with open(file_path) as f:
            datasets = [
                config_data["Roberta"]
                for exp in json.load(f).values()
                for config_data in exp.values()
                if "Roberta" in config_data
            ]

        results = grid_search_formulas(datasets)
        print(f"\n📊 Grid Search Results for {os.path.basename(file_path)}:")
        print(results.head(10).to_string(index=False))

        out_path = f"gridsearch_results_{os.path.splitext(os.path.basename(file_path))[0]}.csv"
        results.to_csv(out_path, index=False)
