import numpy as np
import pandas as pd
import json
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- File list ---
file_list = [
    "evaluation_results_cross-encoder-nli-deberta-base.json",
    "evaluation_results_facebook-bart-large-mnli.json",
    "evaluation_results_microsoft-deberta-large-mnli.json",
    "evaluation_results_prajjwal1-albert-base-v2-mnli.json",
    "evaluation_results_pritamdeka-PubMedBERT-MNLI-MedNLI.json",
    "evaluation_results_roberta-large-mnli.json",
    "evaluation_results_typeform-distilbert-base-uncased-mnli.json"
]

# --- Generate formulas ---
def generate_formulas():
    formulas = []
    for n_weight in np.linspace(0.8, 1.5, 5):
        for c_weight in np.linspace(0.5, 1.5, 3):
            for e_weight in np.linspace(0.8, 1.5, 3):
                formulas.append({
                    'name': f"Linear: n*{n_weight:.1f} + c*{c_weight:.1f} - e*{e_weight:.1f}",
                    'func': lambda n, c, e, nw=n_weight, cw=c_weight, ew=e_weight: nw*n + cw*c - ew*e
                })
    formulas += [
        {'name': "Quadratic: n² + c - e²", 'func': lambda n, c, e: n**2 + c - e**2}
    ]
    for steepness in [5, 10, 15]:
        formulas.append({
            'name': f"Sigmoid: σ({steepness}n) + c - σ({steepness}e)",
            'func': lambda n, c, e, s=steepness: 1/(1+np.exp(-s*n)) + c - 1/(1+np.exp(-s*e))
        })
    for threshold in [0.3, 0.5, 0.7]:
        formulas.append({
            'name': f"Threshold: (n>{threshold}) + c - (e>{threshold})",
            'func': lambda n, c, e, t=threshold: (n>t)*1 + c - (e>t)*1
        })
    formulas += [
        {'name': "Interaction: n + (n*c) - e", 'func': lambda n, c, e: n + (n*c) - e},
        {'name': "SigmoidLinear: σ(5n + c - 3e)", 'func': lambda n, c, e: 1/(1+np.exp(-(5*n + c - 3*e)))},
        {'name': "Piecewise: n*1.5*(n>0.5) + c - e*1.2*(e>0.2)", 'func': lambda n, c, e: 1.5*n*(n>0.5) + c - 1.2*e*(e>0.2)},
        {'name': "Ratio: (n/(e+0.01)) + c", 'func': lambda n, c, e: (n/(e+0.01)) + c},
        {'name': "Exp: n + c*exp(-2*e)", 'func': lambda n, c, e: n + c*np.exp(-2*e)}
    ]
    return formulas

# --- Evaluate formulas ---
def evaluate_formulas(formulas, datasets):
    results = []
    for formula in formulas:
        correlations = []
        mae_scores = []
        for data in datasets:
            rankings = data['ranking']
            relevance = np.array([r['relevance'] for r in rankings])
            n_vals = [r['n'] for r in rankings]
            c_vals = [r['c'] for r in rankings]
            e_vals = [r['e'] for r in rankings]
            scores = np.array([formula['func'](n, c, e) for n, c, e in zip(n_vals, c_vals, e_vals)])
            # Normalize
            norm_scores = 1 + 4*(scores - np.min(scores))/(np.max(scores) - np.min(scores) + 1e-8)
            corr, _ = pearsonr(norm_scores, relevance)
            mae = np.mean(np.abs(norm_scores - relevance))
            correlations.append(corr)
            mae_scores.append(mae)
        results.append({
            'Formula': formula['name'],
            'Avg Pearson': np.mean(correlations),
            'Std Pearson': np.std(correlations),
            'Avg MAE': np.mean(mae_scores)
        })
    return pd.DataFrame(results).sort_values('Avg Pearson', ascending=False)

# --- Plot top formulas ---
def plot_top_formulas(results_df, top_n=5, file_prefix=""):
    plt.figure(figsize=(10, 6))
    top = results_df.head(top_n)
    x = range(len(top))
    plt.bar(x, top['Avg Pearson'], yerr=top['Std Pearson'], capsize=5)
    plt.xticks(x, top['Formula'], rotation=45, ha='right')
    plt.ylabel('Pearson Correlation')
    plt.title(f'Top {top_n} Formulas for {file_prefix}')
    plt.tight_layout()
    plt.savefig(f'top_formulas_{file_prefix}.png', dpi=300)
    plt.close()

# --- Main loop for files ---
if __name__ == "__main__":
    formulas = generate_formulas()

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "r") as file:
            raw_data = json.load(file)

        datasets = []
        for ex_data in raw_data.values():
            for config_data in ex_data.values():
                if "Roberta" in config_data:
                    roberta_data = config_data["Roberta"]
                    if "ranking" in roberta_data:
                        for r in roberta_data["ranking"]:
                            r["n"] = r.get("n", r["score"])
                            r["c"] = r.get("c", 0)
                            r["e"] = r.get("e", 0)
                        datasets.append(roberta_data)

        if not datasets:
            print(f"No usable data in {file_path}")
            continue

        results_df = evaluate_formulas(formulas, datasets)

        print(f"\n📄 File: {file_path}")
        print(results_df.head(10).to_string(index=False))

        file_tag = os.path.splitext(os.path.basename(file_path))[0]
        results_df.to_csv(f"formula_performance_{file_tag}.csv", index=False)
        plot_top_formulas(results_df, top_n=5, file_prefix=file_tag)
