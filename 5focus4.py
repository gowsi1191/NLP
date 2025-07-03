import numpy as np
import json
import math
import os
import itertools
from curve import sigmoid_e, smooth_join_score, power_root_curve, inverted_sigmoid, arctangent_scaled, smoothstep, smootherstep, custom_e_curve

try:
    from jenkspy import jenks_breaks
except ImportError:
    jenks_breaks = None

# Add linear curve definition
def linear(x):
    return x

def normalize_to_relevance(scores, method='percentile'):
    if method == 'jenks' and jenks_breaks is not None:
        try:
            breaks = jenks_breaks(scores, n_classes=5)
            relevance = np.digitize(scores, breaks)
            return np.clip(relevance, 1, 5)
        except:
            method = 'percentile'  # Fallback

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

def evaluate_config(datasets, e_fn, n_fn, c_fn, name, weights=(1, 1, 1), norm_method='percentile',
                    k=None, sigmoid_mid=None, inverse_mid=None, low_thresh=None, mid_thresh=None, high_thresh=None):
    p1_flags, p2_flags, mrr_scores, consistency_flags = [], [], [], []
    diff_5_4, diff_4_3 = [], []
    all_scores = []

    for data in datasets[:150]:
        rankings = data['ranking']
        raw_scores = []

        for r in rankings:
            try:
                e_score = e_fn(r['e'])
                n_score = 1 - n_fn(r['n'])
                c_score = 1 - c_fn(r['c'])
            except Exception:
                continue

            score = weights[0] * e_score + weights[1] * n_score + weights[2] * c_score
            raw_scores.append(score)

        if not raw_scores:
            continue

        pred_relevance = normalize_to_relevance(raw_scores, method=norm_method)

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
        p2_flags.append(p2_hits / 2)

        sorted_rels = [r['relevance'] for _, r in sorted(zip(raw_scores, rankings), key=lambda x: x[0], reverse=True)]
        mrr_scores.append(calculate_mrr(sorted_rels))
        consistency_flags.append(1 if is_monotonic_descending(pred_relevance) else 0)

    return {
        "curve": name,
        "P@1": np.mean(p1_flags),
        "P@2": np.mean(p2_flags),
        "MRR": np.mean(mrr_scores),
        "Consistency": np.mean(consistency_flags),
        "Δ(5→4)": np.mean(diff_5_4) if diff_5_4 else None,
        "Δ(4→3)": np.mean(diff_4_3) if diff_4_3 else None,
        "Max Score": np.max(all_scores) if all_scores else None,
        "Min Score": np.min(all_scores) if all_scores else None,
        "Weights": weights,
        "Norm Method": norm_method,
        "params": {
            "k": k,
            "sigmoid_mid": sigmoid_mid,
            "inverse_mid": inverse_mid,
            "low_thresh": low_thresh,
            "mid_thresh": mid_thresh,
            "high_thresh": high_thresh
        }
    }

if __name__ == "__main__":
    base_path = "/Users/L020774/Movies/heu/NLP"
    file_list = [os.path.join(base_path, "evaluation_results_prajjwal1-albert-base-v2-mnli.json")]

    curve_variants = {
        "custom_e_curve": [
            (lambda x, k=k, sm=sm, im=im, lt=lt, mt=mt, ht=ht: custom_e_curve(
                x, k=k, sigmoid_mid=sm, inverse_mid=im,
                low_thresh=lt, mid_thresh=mt, high_thresh=ht
            ), {
                "k": k, "sigmoid_mid": sm, "inverse_mid": im,
                "low_thresh": lt, "mid_thresh": mt, "high_thresh": ht
            })
            for k in [ 8]
            for sm in [ 0.15]
            for im in [0.5]
            for lt in [0.15, 0.2, 0.25]
            for mt in [0.45, 0.5, 0.55]
            for ht in [0.75, 0.8, 0.85]
        ],
        # "smoothstep": [(smoothstep, {})],
        # "linear": [(linear, {})]
    }

    weight_combinations = [
        (1.0, 1.0, 1.0),
    ]

    if not os.path.exists(file_list[0]):
        print(f"File not found: {file_list[0]}")
    else:
        with open(file_list[0]) as f:
            datasets = [
                config_data["Roberta"]
                for exp in json.load(f).values()
                for config_data in exp.values()
                if "Roberta" in config_data and "ranking" in config_data["Roberta"]
            ]

        norm_methods = ['percentile', 'jenks'] if jenks_breaks else ['percentile']

        total_combinations = 0
        for e_name, e_fns in curve_variants.items():
            for n_name, n_fns in curve_variants.items():
                for c_name, c_fns in curve_variants.items():
                    total_combinations += len(e_fns) * len(n_fns) * len(c_fns)

        completed_combinations = 0
        last_print_percent = -1

        for norm_method in norm_methods:
            for weights in weight_combinations:
                for e_name, e_fns in curve_variants.items():
                    for n_name, n_fns in curve_variants.items():
                        for c_name, c_fns in curve_variants.items():
                            for e_idx, (e_fn, e_params) in enumerate(e_fns):
                                for n_idx, (n_fn, _) in enumerate(n_fns):
                                    for c_idx, (c_fn, _) in enumerate(c_fns):
                                        combo_name = (f"E:{e_name}_v{e_idx+1}|N:{n_name}_v{n_idx+1}|"
                                                      f"C:{c_name}_v{c_idx+1}|W:{weights}|NORM:{norm_method}")

                                        result = evaluate_config(
                                            datasets, e_fn, n_fn, c_fn, combo_name,
                                            weights=weights, norm_method=norm_method,
                                            **e_params
                                        )

                                        completed_combinations += 1
                                        percent_done = int((completed_combinations / total_combinations) * 100)
                                        if percent_done > last_print_percent:
                                            print(f"\nProgress: {completed_combinations}/{total_combinations} combinations completed ({percent_done}%)")
                                            last_print_percent = percent_done

                                        if result['P@1'] >= 0.52:
                                            print(f"\n\U0001F3C6 Evaluation Summary for '{combo_name}':")
                                            print(
                                                f"P@1: {result['P@1']:.3f}, P@2: {result['P@2']:.3f}, "
                                                f"MRR: {result['MRR']:.3f}, Consistency: {result['Consistency']:.2f}, "
                                                f"Δ(5→4): {result['Δ(5→4)']:.4f}, Δ(4→3): {result['Δ(4→3)']:.4f}, "
                                                f"Max: {result['Max Score']:.4f}, Min: {result['Min Score']:.4f}, "
                                                f"Weights: {result['Weights']}, Norm: {result['Norm Method']}"
                                            )
