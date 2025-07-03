# main.py
import numpy as np
import json
import math
import os
from curve import sigmoid_e, smooth_join_score, power_root_curve, inverted_sigmoid, arctangent_scaled, smoothstep, smootherstep

# Add linear curve definition
def linear(x):
    return x

def normalize_to_relevance(scores):
    percentiles = np.percentile(scores, [80, 60, 40, 20])
    relevance = np.ones_like(scores, dtype=float)
    relevance[scores >= percentiles[0]] = 5
    relevance[(scores >= percentiles[1]) & (scores < percentiles[0])] = 4
    relevance[(scores >= percentiles[2]) & (scores < percentiles[1])] = 3
    relevance[(scores >= percentiles[3]) & (scores < percentiles[2])] = 2
    return relevance

def is_monotonic_descending(arr):
    return all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))

def calculate_mrr(predicted_relevance):
    for i, rel in enumerate(predicted_relevance):
        if rel == 5:
            return 1 / (i + 1)
    return 0.0

def evaluate_config(datasets, e_curve_fn, curve_name):
    steep_e, steep_n, steep_c = 0.5, 0.5, 1.0
    p1_flags, p2_flags, mrr_scores, consistency_flags = [], [], [], []
    diff_5_4, diff_4_3 = [], []
    all_scores = []

    for dataset_idx, data in enumerate(datasets[:150]):
        rankings = data['ranking']
        raw_scores = []
        e_scores, n_scores, c_scores = [], [], []

        for r in rankings:
            e_score = e_curve_fn(r['e'])
            n_score =1 - e_curve_fn(r['n'])
            c_score = 1 - e_curve_fn(r['c'])

            score =  0.5 * e_score+ 0.3 * n_score+ 0.2* c_score
            e_scores.append(e_score)
            n_scores.append(n_score)
            c_scores.append(c_score)
            raw_scores.append(score)

        pred_relevance = normalize_to_relevance(raw_scores)

        score_by_rel = {rel: [] for rel in range(1, 6)}
        for score, rel in zip(raw_scores, pred_relevance):
            score_by_rel[int(rel)].append(score)

        if score_by_rel[5] and score_by_rel[4]:
            diff_5_4.append(np.mean(score_by_rel[5]) - np.mean(score_by_rel[4]))
        if score_by_rel[4] and score_by_rel[3]:
            diff_4_3.append(np.mean(score_by_rel[4]) - np.mean(score_by_rel[3]))

        all_scores.extend(raw_scores)

        top_doc_index = np.argmax(raw_scores)
        top_doc_actual_relevance = rankings[top_doc_index]['relevance']
        p1 = 1 if top_doc_actual_relevance >= 4 else 0
        p1_flags.append(p1)

        top_2_docs = sorted(zip(raw_scores, rankings), key=lambda x: x[0], reverse=True)[:2]
        p2_hits = sum(1 for _, r in top_2_docs if r['relevance'] >= 4)
        p2 = p2_hits / 2
        p2_flags.append(p2)

        sorted_rels = [r['relevance'] for _, r in sorted(zip(raw_scores, rankings), key=lambda x: x[0], reverse=True)]
        mrr_scores.append(calculate_mrr(sorted_rels))
        consistency_flags.append(1 if is_monotonic_descending(pred_relevance) else 0)

    return {
        "curve": curve_name,
        "steepness": (steep_e, steep_n, steep_c),
        "P@1": np.mean(p1_flags),
        "P@2": np.mean(p2_flags),
        "MRR": np.mean(mrr_scores),
        "Consistency": np.mean(consistency_flags),
        "Δ(5→4)": np.mean(diff_5_4) if diff_5_4 else None,
        "Δ(4→3)": np.mean(diff_4_3) if diff_4_3 else None,
        "Max Score": np.max(all_scores) if all_scores else None,
        "Min Score": np.min(all_scores) if all_scores else None
    }

if __name__ == "__main__":
    base_path = "/Users/L020774/Movies/heu/NLP"
    file_list = [os.path.join(base_path, "evaluation_results_prajjwal1-albert-base-v2-mnli.json")]

    curve_parameters = {
        "sigmoid_e": [lambda x, k=k: sigmoid_e(x, k=k, midpoint=0.5) for k in [5, 8, 12]],
        "smooth_join_score": [lambda x: smooth_join_score(x)],
        "power_root_curve": [lambda x, exp=exp: power_root_curve(x, exponent=exp) for exp in [0.49, 0.5,0.51, 0.6]],
        "inverted_sigmoid": [lambda x, k=k: inverted_sigmoid(x, k=k, midpoint=0.5) for k in [5, 8, 12]],
        "arctangent_scaled": [lambda x, k=k: arctangent_scaled(x, k=k) for k in [5, 10, 15]],
        "smoothstep": [smoothstep],
        "smootherstep": [smootherstep],
        "linear": [linear]
    }

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path) as f:
            datasets = [
                config_data["Roberta"]
                for exp in json.load(f).values()
                for config_data in exp.values()
                if "Roberta" in config_data and "ranking" in config_data["Roberta"]
            ]

        for curve_name, funcs in curve_parameters.items():
            for idx, fn in enumerate(funcs):
                name_variant = f"{curve_name}_v{idx+1}"
                result = evaluate_config(datasets, fn, name_variant)
                print(f"\n🏆 Evaluation Summary for '{name_variant}':")
                print(
                    f"Steepness (e,n,c): {result['steepness']} | "
                    f"P@1: {result['P@1']:.3f}, P@2: {result['P@2']:.3f}, "
                    f"MRR: {result['MRR']:.3f}, Consistency: {result['Consistency']:.2f}, "
                    f"Δ(5→4): {result['Δ(5→4)']:.4f}, Δ(4→3): {result['Δ(4→3)']:.4f}, "
                    f"Max: {result['Max Score']:.4f}, Min: {result['Min Score']:.4f}"
                )