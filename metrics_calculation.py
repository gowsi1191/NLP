from typing import List, Dict, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import time
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

class MetricsCalculator:
    def __init__(self):
        self.relevant_scores = {5,4}  # <- updated to strict criteria
        self.start_time = time.time()

    def _print_progress(self, current, total, message=""):
        elapsed = time.time() - self.start_time
        print(f"[{current}/{total}] {message} | Elapsed: {elapsed:.1f}s", end="\r")
        if current == total:
            print()

    def _save_cluster_plot(self, embeddings, labels, method_name, query_id, config_key):
        n_samples = len(embeddings)
        if n_samples < 2:
            print(f"Skipping cluster plot for {query_id} - need at least 2 samples")
            return None

        perplexity = min(30, n_samples - 1)

        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                                  c=labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title(f"Document Clusters - {method_name}\nQuery: {query_id}\nConfig: {config_key}")

            os.makedirs("cluster_plots", exist_ok=True)
            safe_config = config_key.replace(" ", "_").replace(".", "_")
            filename = f"cluster_plots/{query_id}_{method_name}_{safe_config}.png"
            plt.savefig(filename)
            plt.close()
            return filename
        except Exception as e:
            print(f"Failed to generate cluster plot for {query_id}: {str(e)}")
            return None

    def precision_at_k(self, ranked_list: List[int], relevance_labels: List[int], k: int) -> float:
        if len(ranked_list) < k:
            k = len(ranked_list)
        top_k = ranked_list[:k]
        relevant = sum(1 for doc in top_k if relevance_labels[doc] in self.relevant_scores)
        return relevant / k

    def mean_reciprocal_rank(self, ranked_list: List[int], relevance_labels: List[int]) -> float:
        for i, doc in enumerate(ranked_list):
            if relevance_labels[doc] in self.relevant_scores:
                return 1.0 / (i + 1)
        return 0.0

    def ndcg_at_k(self, ranked_list: List[int], relevance_labels: List[int], k: int) -> float:
        def dcg_at_k(scores: List[float], k: int) -> float:
            return sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores[:k]))

        actual_scores = [relevance_labels[doc] for doc in ranked_list]
        best_scores = sorted(relevance_labels, reverse=True)

        actual_dcg = dcg_at_k(actual_scores, k)
        best_dcg = dcg_at_k(best_scores, k)

        return actual_dcg / best_dcg if best_dcg > 0 else 0.0

    def calculate_all_metrics(self, ranked_list: List[int], relevance_labels: List[int]) -> Dict[str, float]:
        return {
            "P@1": self.precision_at_k(ranked_list, relevance_labels, 1),
            "P@2": self.precision_at_k(ranked_list, relevance_labels, 2),
            "P@3": self.precision_at_k(ranked_list, relevance_labels, 3),
            "MRR": self.mean_reciprocal_rank(ranked_list, relevance_labels),
            "NDCG@3": self.ndcg_at_k(ranked_list, relevance_labels, 3)
        }

    def evaluate_thresholds(self, queries: Dict[str, Any], relevance_data: Dict[str, List[int]], 
                            model_ops, base_threshold: float = 0.14, base_steepness: float = 15.0):
        PARAM_CONFIGS = {
            "default": {
                "thresholds": [0.12],
                "steepness": [15.0]
            },
            "implicit_NOT": {
                "thresholds": [0.12], 
                "steepness": [15]
            },
            "explicit_NOT": {
                "thresholds": [0.12],
                "steepness": [15.0]
            }
        }

        results = defaultdict(dict)
        total_queries = len(queries)
        print(f"Starting evaluation of {total_queries} queries")

        for query_idx, (query_id, query_data) in enumerate(queries.items(), 1):
            query_text = query_data['query_text']
            documents = query_data['documents']
            relevance_labels = relevance_data[query_id]
            query_type = query_data.get('query_type', 'default')

            config = PARAM_CONFIGS.get(query_type, PARAM_CONFIGS["default"])
            thresholds = config["thresholds"]
            steepness_vals = config["steepness"]
            total_configs = len(thresholds) * len(steepness_vals)

            self._print_progress(query_idx, total_queries, 
                                 f"Processing query {query_id} ({query_type}) with {total_configs} configs")

            bge_query_emb = model_ops.bge_model.encode(query_text, convert_to_tensor=True)
            bge_doc_embs = [model_ops.bge_model.encode(doc, convert_to_tensor=True) for doc in documents]
            bge_scores = [util.cos_sim(bge_query_emb, emb).item() for emb in bge_doc_embs]

            config_idx = 0
            for threshold in thresholds:
                for steepness in steepness_vals:
                    config_idx += 1
                    self._print_progress(config_idx, total_configs, 
                                         f"Config {threshold:.3f}/{steepness:.1f}")

                    logic_scores = []
                    entailment_scores = []
                    for doc in documents:
                        e, n, c = model_ops.get_entailment_scores(query_text, doc)
                        score = model_ops.calculate_score(e, n, c, threshold, steepness, mode=query_type)
                        logic_scores.append(score)
                        entailment_scores.append((e, n, c))

                    bge_rank = [
                        {"score": bge_scores[i], "relevance": relevance_labels[i]}
                        for i in sorted(range(len(bge_scores)), key=lambda i: -bge_scores[i])
                    ]
                    logic_rank = [
                        {
                            "score": logic_scores[i], 
                            "relevance": relevance_labels[i],
                            "e": entailment_scores[i][0],
                            "n": entailment_scores[i][1],
                            "c": entailment_scores[i][2]
                        }
                        for i in sorted(range(len(logic_scores)), key=lambda i: -logic_scores[i])
                    ]

                    bge_metrics = self.calculate_all_metrics(
                        sorted(range(len(bge_scores)), key=lambda i: -bge_scores[i]), 
                        relevance_labels
                    )
                    logic_metrics = self.calculate_all_metrics(
                        sorted(range(len(logic_scores)), key=lambda i: -logic_scores[i]),
                        relevance_labels
                    )

                    key = f"thresh_{threshold:.3f}_steep_{steepness:.1f}"
                    results[query_id][key] = {
                        "query_type": query_type,
                        "BGE": {
                            "ranking": bge_rank,
                            "metrics": bge_metrics
                        },
                        "Roberta": {
                            "ranking": logic_rank,
                            "metrics": logic_metrics
                        }
                    }

        print("\nEvaluation completed.")
        return results
