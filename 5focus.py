import numpy as np
import json
import math
import os

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

def sigmoid_e(x, k=8, midpoint=0.5):
    return 1 / (1 + np.exp(-k * (x - midpoint)))

def smooth_join_score(x, join_x=0.6, join_y=0.4, exp1=0.6, exp2=0.5):
    if x < join_x:
        scale = join_y / (join_x ** exp1)
        return scale * (x ** exp1)
    else:
        norm_x = (x - join_x) / (1 - join_x)
        return join_y + (1 - join_y) * (norm_x ** exp2)


def evaluate_config(datasets):
    steep_e, steep_n, steep_c = 0.5, 0.5, 1.0
    p1_flags, p2_flags, mrr_scores, consistency_flags = [], [], [], []

    for dataset_idx, data in enumerate(datasets[:150]):
        rankings = data['ranking']
        raw_scores = []
        e_scores, n_scores, c_scores = [], [], []

        print(f"\n📘 Dataset {dataset_idx + 1} — Custom Sigmoid Transformation")

        for r in rankings:
            # e: 0–40 → 0–0.2, then 40–80 → 0.2–1, flat after 80

            # e_score = sigmoid_e(r['e'])
            e_score = smooth_join_score(r['e'])



            # n: 0–50 → 0–0.8, then 50–100 → 0.8–1
            if r['n'] < 50:
                n_score = (r['n'] / 50) * 0.8
            else:
                n_score = 0.8 + ((r['n'] - 50) / (100 - 50)) * 0.2
            n_score = min(n_score, 1.0)

            # c: 0–10 → 0–0.8, then 10–100 → 0.8–1.0
            if r['c'] < 0.12:
                c_score = (r['c'] / 12) * 0.8
            else:
                c_score = 0.8 + ((r['c'] - 0.2) / (100 - 0.2)) * 0.2
                # c_score = min(c_score, 1.0)

            print(r['e'])
            print(e_score)
            print('---------')
            # Score formula
            score = e_score
            # Append results
            e_scores.append(e_score)
            n_scores.append(n_score)
            c_scores.append(c_score)
            raw_scores.append(score)

        pred_relevance = normalize_to_relevance(raw_scores)

        for i, (e, n, c, score, r, pr) in enumerate(zip(e_scores, n_scores, c_scores, raw_scores, rankings, pred_relevance), 1):
            print(f"  Doc {i} | raw_e: {r['e']:.4f}, raw_n: {r['n']:.4f}, raw_c: {r['c']:.4f} | "
                  f"e: {e:.4f}, n: {n:.4f}, c: {c:.4f}, score: {score:.4f}, "
                  f"actual: {r['relevance']}, predicted: {int(pr)}")

        # Liberal P@1
        top_doc_index = np.argmax(raw_scores)
        top_doc_actual_relevance = rankings[top_doc_index]['relevance']
        p1 = 1 if top_doc_actual_relevance >= 4 else 0
        p1_flags.append(p1)

        # Liberal P@2
        # Liberal P@2 (fractional based on hits)
        top_2_docs = sorted(zip(raw_scores, rankings), key=lambda x: x[0], reverse=True)[:2]
        p2_hits = sum(1 for _, r in top_2_docs if r['relevance'] >= 4)
        p2 = p2_hits / 2  # 0.0, 0.5, or 1.0
        p2_flags.append(p2)


        # MRR and consistency
        sorted_rels = [r['relevance'] for _, r in sorted(zip(raw_scores, rankings), key=lambda x: x[0], reverse=True)]
        mrr_scores.append(calculate_mrr(sorted_rels))
        consistency_flags.append(1 if is_monotonic_descending(pred_relevance) else 0)

        # Per-dataset precision summary
        print(f"📊 P@1: {p1}, P@2: {p2}")

    return {
        "steepness": (steep_e, steep_n, steep_c),
        "P@1": np.mean(p1_flags),
        "P@2": np.mean(p2_flags),
        "MRR": np.mean(mrr_scores),
        "Consistency": np.mean(consistency_flags)
    }

if __name__ == "__main__":
    base_path = "/Users/L020774/Movies/heu/NLP"
    file_list = [os.path.join(base_path, "evaluation_results_prajjwal1-albert-base-v2-mnli.json")]

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

        print(f"\n📄 Evaluating {len(datasets)} datasets from {file_path}")
        result = evaluate_config(datasets)

        print("\n🏆 Evaluation Summary:")
        print(f"Steepness (e,n,c): {result['steepness']} | "
              f"P@1: {result['P@1']:.3f}, P@2: {result['P@2']:.3f}, "
              f"MRR: {result['MRR']:.3f}, Consistency: {result['Consistency']:.2f}")
